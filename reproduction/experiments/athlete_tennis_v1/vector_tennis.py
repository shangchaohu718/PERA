"""Independent official MuJoCo tennis states with shared static resources.

512 environments are 512 MjData instances, independent planners, histories,
ball audits and action states. A bounded CPU thread pool advances them; policy
inference is a single batch outside these threads. Only non-colliding display
markers are suppressed to keep the common MjModel immutable during stepping.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import time

from experiments.athlete_tennis_v1.distill import (
    CONTRACT,
    POLICY,
    Bridge,
    BallAudit,
    cfg,
    torch,
    np,
    mujoco,
    PerceptiveActorObservationInput,
    build_perceptive_actor_terms,
    TensorDict,
)
import athlete.scripts.play_student_onnx as native


def physics_digest(model):
    digest = hashlib.sha256()
    for name in (
        "body_mass",
        "body_inertia",
        "body_pos",
        "body_quat",
        "body_parentid",
        "jnt_axis",
        "jnt_pos",
        "dof_armature",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "pair_friction",
        "pair_solref",
        "pair_solimp",
        "jnt_range",
        "dof_damping",
        "dof_frictionloss",
    ):
        value = np.asarray(getattr(model, name))
        digest.update(name.encode())
        digest.update(value.tobytes())
    colliding = (model.geom_contype | model.geom_conaffinity) != 0
    for name in (
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_bodyid",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_contype",
        "geom_conaffinity",
    ):
        digest.update(name.encode())
        digest.update(np.asarray(getattr(model, name))[colliding].tobytes())
    digest.update(
        str(
            (
                model.opt.timestep,
                model.opt.integrator,
                model.opt.iterations,
                model.opt.ls_iterations,
            )
        ).encode()
    )
    digest.update(np.asarray(model.opt.gravity).tobytes())
    return digest.hexdigest()


class SharedSimulation(native._StandaloneStudentSimulation):
    def _update_manual_bounce_marker(self):
        assert not self.cfg.manual_mode

    def _update_landing_target_marker(self):
        # No observation uses these non-contact viewer marker geometries.
        pass


class PreparedAction:
    def __init__(self, teacher):
        self.teacher = teacher
        self.action = self.expected_observation = None

    def __getattr__(self, name):
        return getattr(self.teacher, name)

    def __call__(self, observation):
        assert self.action is not None
        # Guarantees the observation used for batched inference is exactly the
        # one the untouched upstream step would pass to its local policy.
        assert np.array_equal(observation, self.expected_observation)
        return self.action


class VectorTennis:
    def __init__(self, num_envs=512, workers=16, seed=2300000, progress=None):
        self.N, self.workers, self.next_seed = num_envs, workers, seed + num_envs
        self.teacher = native._StudentOnnxPolicy(POLICY)
        self.model = None
        original_model, original_policy = native._build_standalone_model, native._StudentOnnxPolicy
        original_print = native.__dict__.get("print")
        native.print = lambda *args, **kwargs: None

        def common_model(*args, **kwargs):
            if self.model is None:
                self.model = original_model(*args, **kwargs)
            return self.model

        native._build_standalone_model = common_model
        native._StudentOnnxPolicy = lambda path: self.teacher
        try:
            self.sims = []
            for i in range(num_envs):
                self.sims.append(SharedSimulation(cfg(seed + i)))
                if progress and ((i + 1) % 64 == 0 or i + 1 == num_envs):
                    progress(initialized_environments=i + 1)
        finally:
            native._build_standalone_model, native._StudentOnnxPolicy = (
                original_model,
                original_policy,
            )
        self._original_print = original_print
        self.bridge = Bridge(list(self.sims[0].joint_names), "cpu")
        self.audits = [BallAudit(sim) for sim in self.sims]
        self.controllers = [PreparedAction(self.teacher) for _ in self.sims]
        for sim, controller in zip(self.sims, self.controllers):
            assert not sim.cfg.manual_mode and sim.incoming_ball_plan is not None
            sim.policy = controller
        assert len({id(s.data) for s in self.sims}) == num_envs
        assert len({id(s.model) for s in self.sims}) == 1
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tennis_physics")
        self.last_action = np.zeros((num_envs, 29), dtype=np.float32)
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.resets, self.sampling_failures = 0, 0
        self.completed_records = []
        self.active = np.zeros(num_envs, dtype=bool)
        self.active_since = np.zeros(num_envs)
        self.stable_streak = np.zeros(num_envs, dtype=np.int64)
        self.last_intervened_cycle = np.full(num_envs, -1)
        self.last_trigger = np.full(num_envs, -np.inf)
        self.last_release = np.full(num_envs, -np.inf)
        self.assisted_steps = self.teacher_label_queries = self.transitions = 0
        self.assistance_allowed = np.arange(num_envs) % 2 == 1
        self.initial_distinct_ball_states = len(
            {tuple(s.data.qpos[s.ball_qpos_adr : s.ball_qpos_adr + 3]) for s in self.sims}
        )
        self.physics_sha256 = physics_digest(self.model)

    def observations(self):
        return np.stack([s.observation() for s in self.sims])

    def actor_terms(self, device):
        def tensor(values):
            return torch.tensor(np.asarray(values), dtype=torch.float32, device=device)

        gather = self.bridge.gather
        reference = self.bridge.reference.to(device).expand(self.N, -1, -1)
        terms = build_perceptive_actor_terms(
            PerceptiveActorObservationInput(
                reference_qpos=reference,
                robot_root_quat_wxyz=tensor([s.data.qpos[3:7] for s in self.sims]),
                robot_root_angvel_local=tensor(
                    [s.data.sensor("imu_ang_vel").data for s in self.sims]
                ),
                robot_dof_pos=tensor([s.data.qpos[s.joint_qpos_adr[gather]] for s in self.sims]),
                robot_dof_vel=tensor([s.data.qvel[s.joint_dof_adr[gather]] for s in self.sims]),
                last_action=tensor(self.last_action),
                default_dof_pos=self.bridge.default.to(device).expand(self.N, -1),
            )
        )
        return TensorDict({"unified/" + k: v for k, v in terms.items()}, batch_size=[self.N])

    def labels(self, native_actions):
        sim = self.sims[0]
        q = (sim.default_joint_pos + sim.action_scale * native_actions)[:, self.bridge.gather]
        return ((q - self.bridge.offset) / self.bridge.scale).astype(np.float32)

    def native_actions(self, holo_actions):
        sim = self.sims[0]
        q = (self.bridge.offset + self.bridge.scale * holo_actions)[:, self.bridge.inverse]
        return (q - sim.default_joint_pos) / sim.action_scale

    def correction(self, observation, student, teacher, max_alpha):
        names = self.sims[0].joint_names
        body = np.array(
            [
                i
                for i, n in enumerate(names)
                if not (
                    n.startswith("right_") and any(x in n for x in ("shoulder", "elbow", "wrist"))
                )
            ]
        )
        distance = np.linalg.norm(observation[:, 104:107] - observation[:, 128:131], axis=-1)
        time_s = np.array([s.data.time for s in self.sims])
        cycles = np.array([s.episode_index for s in self.sims])
        pelvis = np.stack([s.data.xpos[s.pelvis_body_id] for s in self.sims])
        ball = np.stack([s.data.xpos[s.ball_body_id] for s in self.sims])
        up = np.array([s.data.xmat[s.pelvis_body_id].reshape(3, 3)[2, 2] for s in self.sims])
        incoming = np.array([s.actual_hit_time_s is None for s in self.sims]) & (
            ball[:, 0] - pelvis[:, 0] > -0.75
        )
        rmse = np.sqrt(
            np.mean(
                (self.sims[0].action_scale[body] * (student[:, body] - teacher[:, body])) ** 2,
                axis=-1,
            )
        )
        stable = (pelvis[:, 2] > 0.62) & (up > np.cos(np.deg2rad(15)))
        alarm = (pelvis[:, 2] < 0.58) | (up < np.cos(np.deg2rad(20)))
        trigger = (
            (
                alarm
                | (
                    incoming
                    & (distance < 3.0)
                    & (rmse > 0.06)
                    & (cycles != self.last_intervened_cycle)
                )
            )
            & ~self.active
            & self.assistance_allowed
        )
        if max_alpha > 0:
            self.active[trigger] = True
            self.active_since[trigger] = time_s[trigger]
            self.last_trigger[trigger] = time_s[trigger]
            self.last_intervened_cycle[trigger] = cycles[trigger]
        else:
            self.active[:] = False
        ready = self.active & stable & ((rmse < 0.04) | ~incoming)
        self.stable_streak = np.where(ready, self.stable_streak + 1, 0)
        release = self.active & (time_s - self.active_since >= 0.4) & (self.stable_streak >= 5)
        self.active[release] = False
        self.last_release[release] = time_s[release]
        alpha = self.active.astype(np.float64) * max_alpha
        executed = student.copy()
        executed[:, body] = (
            student[:, body] * (1 - alpha[:, None]) + teacher[:, body] * alpha[:, None]
        )
        self.assisted_steps += int((alpha > 0).sum())
        correction = self.active | (time_s - self.last_release <= 0.6)
        return executed, {
            "distance": distance,
            "incoming": incoming,
            "correction": correction,
            "alpha": alpha,
            "body_target_RMSE_rad": rmse,
        }

    def _step_one(self, index):
        sim = self.sims[index]
        result = sim.step()
        self.steps[index] += 1
        audit = self.audits[index]
        audit.check(float(sim.data.time))
        return result

    def step(self, actions, observations):
        assert actions.shape == (self.N, 29)
        assert np.isfinite(actions).all()
        for i, controller in enumerate(self.controllers):
            controller.action = actions[i]
            controller.expected_observation = observations[i]
        reports = list(self.pool.map(self._step_one, range(self.N)))
        self.last_action = self.labels(actions)
        self.transitions += self.N
        reset_ids = []
        for i, (sim, audit, report) in enumerate(zip(self.sims, self.audits, reports)):
            if audit.first_unstable is not None:
                audit.end_cycle(False)
                self.completed_records.append(
                    {
                        "environment": i,
                        "failure": "physical_instability",
                        "cycles": [audit.records[-1]],
                    }
                )
                reset_ids.append(i)
            elif report.episode_complete:
                audit.end_cycle(True)
                self.completed_records.append(
                    {"environment": i, "failure": None, "cycles": [audit.records[-1]]}
                )
                audit.begin()
                try:
                    sim.start_next_strike_cycle()
                except RuntimeError as exc:
                    if str(exc) != "No over-net, once-bounced incoming ball sampled near G1.":
                        raise
                    self.sampling_failures += 1
                    reset_ids.append(i)
        for i in reset_ids:
            sim = self.sims[i]
            # Official reset, explicitly counted during training only.
            sim.reset()
            sim._ball_racket_contact = (
                native._StandaloneStudentSimulation._ball_racket_contact.__get__(sim)
            )
            sim._apply_explicit_ground_rebound = (
                native._StandaloneStudentSimulation._apply_explicit_ground_rebound.__get__(sim)
            )
            self.audits[i] = BallAudit(sim)
            self.last_action[i] = 0.0
            self.active[i] = False
            self.stable_streak[i] = 0
            self.last_intervened_cycle[i] = -1
            self.last_trigger[i] = self.last_release[i] = -np.inf
            self.resets += 1
        return reset_ids

    def reset_all(self):
        for i, sim in enumerate(self.sims):
            sim.reset()
            sim._ball_racket_contact = (
                native._StandaloneStudentSimulation._ball_racket_contact.__get__(sim)
            )
            sim._apply_explicit_ground_rebound = (
                native._StandaloneStudentSimulation._apply_explicit_ground_rebound.__get__(sim)
            )
            self.audits[i] = BallAudit(sim)
        self.last_action.fill(0.0)
        self.active.fill(False)
        self.stable_streak.fill(0)
        self.last_intervened_cycle.fill(-1)
        self.last_trigger.fill(-np.inf)
        self.last_release.fill(-np.inf)
        self.resets += self.N

    def close(self):
        self.pool.shutdown(wait=True)
        if self._original_print is None:
            native.__dict__.pop("print", None)
        else:
            native.print = self._original_print

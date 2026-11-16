"""512 independent native MuJoCo worlds with batched official observations.

Contact/task/IG terms are masked to zero in the released MuJoCo teacher builder.
We preserve those masks and call its original batch-capable feature function.
XBODY objectVelocity equals the native per-body Jacobian velocities; parity is
checked before collection. Each world retains independent q/dq/action history.
"""

import contextlib
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from experiments.ultra_box_v1.runtime import ROOT, FrozenTeacher, World, load_native
from modumotion_g1.perceptive_observations import (
    PerceptiveActorObservationInput,
    build_perceptive_actor_terms,
    sample_holomotion_reference,
)


def physics_digest(model):
    h = hashlib.sha256()
    for name in (
        "body_mass",
        "body_inertia",
        "body_pos",
        "body_quat",
        "body_parentid",
        "jnt_axis",
        "jnt_pos",
        "jnt_range",
        "dof_armature",
        "dof_damping",
        "dof_frictionloss",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_bodyid",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_contype",
        "geom_conaffinity",
        "geom_condim",
        "mesh_vert",
        "mesh_face",
    ):
        value = np.asarray(getattr(model, name))
        h.update(name.encode())
        h.update(value.tobytes())
    h.update(
        str(
            (
                model.opt.timestep,
                model.opt.integrator,
                model.opt.solver,
                model.opt.iterations,
                model.opt.ls_iterations,
            )
        ).encode()
    )
    h.update(np.asarray(model.opt.gravity).tobytes())
    return h.hexdigest()


class VectorULTRA:
    def __init__(
        self, motions, num_envs=512, workers=12, device="cuda:0", seed=52001, progress=None
    ):
        self.N = num_envs
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.templates = []
        for motion in motions:
            with contextlib.redirect_stdout(io.StringIO()):
                env = load_native(motion)
            self.templates.append(env)
        self.model = self.templates[0].model
        self.physics_sha256 = physics_digest(self.model)
        for template in self.templates:
            if physics_digest(template.model) != self.physics_sha256:
                raise ValueError("Reference templates changed native physics")
            template.model = self.model
            template.policy = self.templates[0].policy
        self.teacher = FrozenTeacher(self.templates[0], device)
        self.worlds = []
        for i in range(num_envs):
            self.worlds.append(World(self.templates[i % len(self.templates)]))
            if progress and ((i + 1) % 64 == 0 or i + 1 == num_envs):
                progress(initialized_environments=i + 1)
        assert len({id(w.data) for w in self.worlds}) == num_envs
        assert len({id(w.model) for w in self.worlds}) == 1
        self.body_ids = self.worlds[0].builder.body_ids
        self.qpos_ids = self.worlds[0].joint_qpos
        self.qvel_ids = self.worlds[0].joint_qvel
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ultra_physics")
        self.previous_q = torch.zeros(num_envs, 29)
        self.previous_dq = torch.zeros(num_envs, 29)
        self.transitions = self.resets = 0
        self.completed = []
        self.done = np.zeros(num_envs, dtype=bool)

    def observations(self, verify_indices=()):
        pose = np.stack([w.data.xpos[self.body_ids] for w in self.worlds]).astype(np.float32)
        quat = np.stack([w.data.xquat[self.body_ids] for w in self.worlds]).astype(np.float32)
        # Native obs uses xyzw; MuJoCo exposes wxyz.
        quat = quat[:, :, [1, 2, 3, 0]]
        velocities = np.empty((self.N, len(self.body_ids), 6), dtype=np.float64)
        for i, w in enumerate(self.worlds):
            # mj_step integrates qvel after computing cvel. Native observation
            # multiplies cached Jacobians by the new qvel; refresh only this
            # velocity cache to obtain exactly that value (no mj_forward/step).
            mujoco.mj_comVel(self.model, w.data)
            for j, bid in enumerate(self.body_ids):
                mujoco.mj_objectVelocity(
                    self.model, w.data, mujoco.mjtObj.mjOBJ_XBODY, int(bid), velocities[i, j], 0
                )
        velocity = torch.from_numpy(velocities.astype(np.float32))
        q = torch.from_numpy(
            np.stack([w.data.qpos[self.qpos_ids] for w in self.worlds]).astype(np.float32)
        )
        dq = torch.from_numpy(
            np.stack([w.data.qvel[self.qvel_ids] for w in self.worlds]).astype(np.float32)
        )
        actions = torch.cat([w.last_action for w in self.worlds])
        zero = torch.zeros(self.N, 29)
        contact = torch.zeros(self.N, 39)
        builder = self.worlds[0].builder
        halves = []
        for delta in (1, 16):
            ref = torch.stack(
                [
                    w.env.hoi_data[min(w.frame + delta, w.env.max_episode_length - 1)]
                    for w in self.worlds
                ]
            )
            obs, _ = builder.compute_humanoid_observations_max(
                torch.from_numpy(pose),
                torch.from_numpy(quat),
                velocity[:, :, 3:],
                velocity[:, :, :3],
                False,
                True,
                contact,
                ref,
                actions,
                q,
                dq,
                zero,
                self.previous_q if delta == 1 else q,
                self.previous_dq if delta == 1 else dq,
                student_obs=False,
            )
            halves.append(torch.cat((obs, torch.zeros(self.N, 21 + 234)), dim=-1))
        result = torch.cat(halves, dim=-1)
        assert result.shape == (self.N, 4052) and torch.isfinite(result).all()
        parity = []
        for i in verify_indices:
            w = self.worlds[i]
            w.builder.last_dof_pos = self.previous_q[i : i + 1].clone()
            w.builder.last_dof_vel = self.previous_dq[i : i + 1].clone()
            expected = w.observation()
            torch.testing.assert_close(result[i : i + 1], expected, atol=2e-6, rtol=2e-6)
            parity.append(float((result[i : i + 1] - expected).abs().max()))
        self.previous_q = q.clone()
        self.previous_dq = dq.clone()
        return result, parity

    def reset_all(self, training=False):
        for w in self.worlds:
            frame = int(self.rng.integers(max(1, int(w.original_frames * 0.25)))) if training else 0
            w.reset(frame)
            if training:
                w.data.qpos[w.joint_qpos] += self.rng.normal(0, 0.003, 29)
                mujoco.mj_forward(self.model, w.data)
        self.previous_q.zero_()
        self.previous_dq.zero_()
        self.done[:] = False
        self.resets += self.N

    def _step(self, item):
        i, action = item
        if not self.done[i]:
            self.worlds[i].step(action)

    def step(self, actions, auto_reset=True):
        actions = np.asarray(actions, dtype=np.float32)
        assert actions.shape == (self.N, 29) and np.isfinite(actions).all()
        list(self.pool.map(self._step, enumerate(actions)))
        self.transitions += int((~self.done).sum())
        reset = []
        for i, w in enumerate(self.worlds):
            if self.done[i]:
                continue
            if w.fallen or w.frame >= w.end_frame:
                self.completed.append(
                    {"environment": i, "motion": w.env.motion_path, **w.summary()}
                )
                if auto_reset:
                    frame = int(self.rng.integers(max(1, int(w.original_frames * 0.25))))
                    w.reset(frame)
                    w.data.qpos[w.joint_qpos] += self.rng.normal(0, 0.003, 29)
                    mujoco.mj_forward(self.model, w.data)
                    self.previous_q[i].zero_()
                    self.previous_dq[i].zero_()
                    reset.append(i)
                    self.resets += 1
                else:
                    self.done[i] = True
        return reset

    def close(self):
        self.pool.shutdown(wait=True)


class HoloBridge:
    def __init__(self, vector, device="cuda:0"):
        self.vector = vector
        self.device = device
        self.contract = json.loads((ROOT / "configs/g1_actor_io_contract.json").read_text())
        names = vector.worlds[0].joint_names
        self.gather = np.array([names.index(n) for n in self.contract["joint_names"]])
        self.inverse = np.argsort(self.gather)
        self.offset = torch.tensor(self.contract["action_offset"], device=device)
        self.scale = torch.tensor(self.contract["action_scale"], device=device)
        self.default = torch.tensor(self.contract["default_joint_pos"], device=device)
        pieces = []
        starts = []
        ends = []
        start = 0
        for env in vector.templates:
            raw = env.hoi_data[:, :630]
            ref = torch.cat(
                (raw[:, :3], raw[:, [6, 3, 4, 5]], raw[:, 13:42][:, self.gather]), dim=-1
            )
            pieces.append(ref)
            starts.append(start)
            ends.append(start + len(ref) - 1)
            start += len(ref)
        self.raw = torch.cat(pieces).to(device)
        self.starts = torch.tensor(starts, device=device)
        self.ends = torch.tensor(ends, device=device)
        self.clip = torch.arange(vector.N, device=device) % len(vector.templates)
        self.last = torch.zeros(vector.N, 29, device=device)

    def labels(self, native):
        physical = (native * 3.0)[:, self.gather]
        return (physical - self.offset) / self.scale

    def native(self, holo):
        return ((self.offset + self.scale * holo)[:, self.inverse] / 3.0).clamp(-1, 1)

    def terms(self):
        v = self.vector
        device = self.device

        def t(a):
            return torch.as_tensor(np.asarray(a), device=device, dtype=torch.float32)

        frame = torch.tensor([w.frame for w in v.worlds], device=device) + self.starts[self.clip]
        ref, times = sample_holomotion_reference(
            self.raw,
            frame,
            torch.zeros(v.N, device=device),
            self.starts[self.clip],
            self.ends[self.clip],
            reference_fps=60.0,
            actor_dt=0.02,
        )
        terms = build_perceptive_actor_terms(
            PerceptiveActorObservationInput(
                reference_qpos=ref,
                robot_root_quat_wxyz=t([w.data.qpos[3:7] for w in v.worlds]),
                robot_root_angvel_local=t([w.data.qvel[3:6] for w in v.worlds]),
                robot_dof_pos=t([w.data.qpos[w.joint_qpos[self.gather]] for w in v.worlds]),
                robot_dof_vel=t([w.data.qvel[w.joint_qvel[self.gather]] for w in v.worlds]),
                last_action=self.last,
                default_dof_pos=self.default,
                reference_sample_time=times,
            )
        )
        return TensorDict({"unified/" + k: value for k, value in terms.items()}, batch_size=[v.N])

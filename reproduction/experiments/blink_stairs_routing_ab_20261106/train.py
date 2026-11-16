"""Fresh competitive ninth-expert distillation in original MimicX stair physics."""

import argparse, json, time, hashlib, os, traceback, shutil
from pathlib import Path
import numpy as np, torch
from experiments.mimicx_all_tasks_v1.runtime import setup, module_digest
from experiments.blink_stairs_routing_ab_20261106.routing import (
    load_base,
    public_digest,
    initialize,
    features,
    FORMAT,
)
from experiments.blink_stairs_routing_ab_20261106.bridge import Bridge
from pera.paths import ROOT

R = ROOT
BASE = R / "checkpoints/base/best.pt"


def atomic(p, x):
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(x, indent=2, ensure_ascii=False))
    tmp.replace(p)


class Replay:
    def __init__(self, capacity=65536):
        self.capacity = capacity
        self.rows = 0
        self.pos = 0
        self.data = None

    def put(self, f, target, oldids, n=32):
        ids = torch.randperm(len(target), device=target.device)[: min(n, len(target))]
        raw = {**f, "target": target, "oldids": oldids}
        n = len(ids)
        if self.data is None:
            self.data = {
                k: torch.empty(self.capacity, *v.shape[1:], dtype=v.dtype) for k, v in raw.items()
            }
        slots = (torch.arange(n) + self.pos) % self.capacity
        for k, v in raw.items():
            self.data[k][slots] = v[ids].detach().cpu()
        self.pos = (self.pos + n) % self.capacity
        self.rows = min(self.capacity, self.rows + n)

    def sample(self, n=512):
        ids = torch.randint(self.rows, (n,))
        return {k: v[ids].to("cuda:0") for k, v in self.data.items()}


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("--mode", choices=["competitive", "fusion"], required=True)
    arg.add_argument("--output", type=Path, required=True)
    arg.add_argument("--num-envs", type=int, default=512)
    arg.add_argument("--preflight-only", action="store_true")
    arg.add_argument("--seed-updates", type=int, default=2000)
    arg.add_argument("--updates-per-round", type=int, default=1000)
    arg.add_argument("--rounds", type=int, default=7)
    arg.add_argument("--batch-size", type=int, default=512)
    a = arg.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(77201)
    torch.backends.cuda.matmul.allow_tf32 = True
    state = dict(
        phase="initializing",
        manager_pid=os.getpid(),
        robot="G1_29dof",
        task="MimicX stairs",
        num_envs=a.num_envs,
        public_checkpoint=str(BASE),
        public_checkpoint_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
        routing_mode=a.mode,
        method="teacher-action BC plus DAgger; paired comparison of competitive Top2 and continuous router fusion",
        world_model_used=False,
        PPO_used=False,
        actual_Adam_updates=0,
        teacher_label_queries=0,
        teacher_executed_transitions=0,
        student_executed_transitions=0,
        uses_vision=False,
        preflight_only=a.preflight_only,
        maximum_Adam_updates=a.seed_updates + a.rounds * a.updates_per_round,
    )

    def progress(phase, **kw):
        state.update(phase=phase, elapsed_seconds=round(time.time() - started, 2), **kw)
        atomic(out / "status.json", state)
        print(
            json.dumps(
                {
                    k: state.get(k)
                    for k in [
                        "phase",
                        "round",
                        "step",
                        "actual_Adam_updates",
                        "loss",
                        "elapsed_seconds",
                        "error",
                    ]
                }
            ),
            flush=True,
        )

    env = None
    try:
        progress("loading_native_teacher")
        env, teacher, row, paths = setup("stairs", 77301, a.num_envs)
        pi, cfg, robot = load_base(BASE, mode=a.mode)
        bridge = Bridge(env, pi, robot, paths["motion"], out)
        N = a.num_envs
        horizon = int(row["horizon"])
        assert env.num_envs == env.unwrapped.sim.data.qpos.shape[0] == N
        import mujoco

        mujoco.mj_saveModel(env.unwrapped.sim.mj_model, str(out / "native_scene.mjb"), None)
        params = list(pi.moe.plugin.parameters())
        assert len(params) == 4 and [n for n, p in pi.named_parameters() if p.requires_grad] == [
            "moe.plugin.gate_up",
            "moe.plugin.down",
            "moe.plugin.router.weight",
            "moe.plugin.router.bias",
        ]
        pub = public_digest(pi)
        tdigest = module_digest(teacher)
        state.update(
            public_sha256=pub,
            teacher_module_sha256=tdigest,
            teacher=str(paths["checkpoint"]),
            teacher_checkpoint_sha256=hashlib.sha256(paths["checkpoint"].read_bytes()).hexdigest(),
            trainable_parameters=sum(p.numel() for p in params),
            trainable_tensors=[n for n, p in pi.named_parameters() if p.requires_grad],
            horizon=horizon,
            reference_initialization=True,
            public_selection_bias_frozen=True,
            domain_selection_bias="old8 unchanged; competitive new offset calibrated once; fusion offset unused",
            fixed_merge_bias=0.0,
            learned_bias_constraint="clamp_max(0)",
            reference_uses_vision=False,
        )
        atomic(out / "plan.json", state)
        replay = Replay()

        def verify():
            assert public_digest(pi) == pub, "Public parameters or buffers changed"
            assert module_digest(teacher) == tdigest
            assert all(p.grad is None for p in pi.parameters() if not p.requires_grad)
            assert all(torch.isfinite(p).all() for p in params)

        def save(label, optimizer=None):
            verify()
            payload = dict(
                format=FORMAT,
                public_sha256=pub,
                public_checkpoint=str(BASE),
                public_checkpoint_sha256=state["public_checkpoint_sha256"],
                plugin={k: v.detach().cpu() for k, v in pi.moe.plugin.state_dict().items()},
                actual_Adam_updates=state["actual_Adam_updates"],
                initialization=state.get("initialization"),
                robot="g1",
                routing_mode=a.mode,
                routing="9-way Top2"
                if a.mode == "competitive"
                else "native public Top2 plus continuous anchor-router fusion",
                fixed_merge_bias=0.0,
                world_model_used=False,
                provenance=dict(
                    task="MimicX stairs",
                    teacher_sha256=state["teacher_checkpoint_sha256"],
                    method=state["method"],
                ),
            )
            if optimizer is not None:
                payload["optimizer"] = optimizer.state_dict()
            path = out / (label + ".domain.pt")
            tmp = path.with_suffix(".tmp")
            torch.save(payload, tmp)
            tmp.replace(path)
            state["latest_checkpoint"] = str(path)
            return path

        @torch.no_grad()
        def rollout(role, seed, label, collect=False, beta=0.0, limit=None, record=True):
            pi.moe.domain_enabled = role != "public"
            obs = bridge.reset(seed)
            alive = torch.ones(N, dtype=torch.bool, device="cuda:0")
            front = torch.zeros(N, dtype=torch.long, device="cuda:0")
            errors = []
            bodyerrors = []
            selected = weights = tokens = 0
            trace = []
            validtrace = []
            admissions = []
            contributions = []
            frameids = []
            t0 = time.time()
            local_student = local_teacher = 0
            for step in range(limit or horizon):
                h, g = bridge.observe()
                action = pi(h, g)
                f = features(pi, g)
                sel = pi.moe.last_selected
                wt = pi.moe.last_weight
                native_teacher = teacher(obs) if collect or role == "teacher" else None
                if native_teacher is not None:
                    if env.clip_actions is not None:
                        native_teacher = native_teacher.clamp(-env.clip_actions, env.clip_actions)
                    state["teacher_label_queries"] += N
                if collect:
                    replay.put(f, bridge.to_student(native_teacher), pi.moe.last_old_indices)
                if role == "teacher":
                    native = native_teacher
                    ch = torch.ones(N, 1, dtype=torch.bool, device="cuda:0")
                elif collect:
                    ch = torch.rand(N, 1, device="cuda:0") < beta
                    native = torch.where(ch, native_teacher, bridge.to_native(action))
                else:
                    native = bridge.to_native(action)
                    ch = torch.zeros(N, 1, dtype=torch.bool, device="cuda:0")
                selected += int((sel[:, 0] & alive).sum())
                weights += float(wt[:, 0][alive].sum())
                tokens += int(alive.sum())
                command = bridge.command
                e = (command.robot_anchor_pos_w - command.anchor_pos_w).norm(dim=-1)
                be = (command.robot_body_pos_w - command.body_pos_w).norm(dim=-1).mean(-1)
                if alive.any():
                    errors.append(float(e[alive].mean()))
                    bodyerrors.append(float(be[alive].mean()))
                if record:
                    trace.append(env.unwrapped.sim.data.qpos[0].cpu().numpy().copy())
                    validtrace.append(bool(alive[0]))
                    admissions.append(bool(sel[0]))
                    contributions.append(float(wt[0]))
                    frameids.append(int(command.time_steps[0]))
                obs, _, done, _ = env.step(native)
                alive &= ~done.bool()
                front += alive.long()
                bridge.stepped(native, done)
                local_teacher += int(ch.sum())
                local_student += N - int(ch.sum())
                if (step + 1) % 100 == 0:
                    progress(
                        "collecting" if collect else "evaluating_" + role,
                        step=step + 1,
                        evaluation=label,
                    )
            enderr = (bridge.command.robot_anchor_pos_w - bridge.command.anchor_pos_w).norm(dim=-1)
            result = dict(
                seed=seed,
                environments=N,
                steps=limit or horizon,
                complete=int(alive.sum()),
                arrived=int((alive & (enderr <= 0.25)).sum()),
                frontiers=front.cpu().tolist(),
                mean_valid_anchor_error_m=float(np.mean(errors)),
                mean_valid_body_error_m=float(np.mean(bodyerrors)),
                selection_fraction=selected / max(tokens, 1),
                mean_new_contribution=weights / max(tokens, 1),
                routing_mode=a.mode,
                selection_fraction_semantics="Top2 admission"
                if a.mode == "competitive"
                else "positive continuous fusion weight, not TopK admission",
                recorded_robot0_first_termination_frame=next(
                    (i for i, v in enumerate(validtrace) if not v), None
                ),
                teacher_executed_transitions=local_teacher,
                student_executed_transitions=local_student,
                seconds=time.time() - t0,
            )
            if collect:
                state["teacher_executed_transitions"] += local_teacher
                state["student_executed_transitions"] += local_student
            if record:
                np.savez_compressed(
                    out / (label + "_trajectory.npz"),
                    qpos=np.array(trace),
                    alive=np.array(validtrace),
                    selected=np.array(admissions),
                    contribution=np.array(contributions),
                    reference_frame=np.array(frameids),
                    fps=50,
                )
            atomic(out / (label + ".json"), result)
            verify()
            return result

        pi.moe.domain_enabled = False
        teacher_result = rollout("teacher", 77301, "teacher_gate", collect=True, beta=1.0)
        if teacher_result["complete"] / N < 0.9:
            raise RuntimeError("Native teacher fails gate; no student optimizer updates")
        bank = {k: v[: replay.rows].to("cuda:0") for k, v in replay.data.items()}
        init = initialize(pi, bank, bank["oldids"])
        state["initialization"] = init
        state["initial_ffn_sha256"] = hashlib.sha256(
            b"".join(
                p.detach().cpu().numpy().tobytes()
                for p in [pi.moe.plugin.gate_up, pi.moe.plugin.down]
            )
        ).hexdigest()
        save("initialized")
        atomic(out / "initialization.json", init)
        # Cached path must match complete Actor means and trainable gradients for identical histories.
        h, g = bridge.observe()
        full = pi(h, g)
        f = features(pi, g)
        pred, *_ = pi.moe.predict(f, pi.dec[-1])
        torch.testing.assert_close(pred, full, atol=2e-5, rtol=2e-5)
        parity_target = torch.ones_like(full) * 0.123
        full_grads = torch.autograd.grad(
            ((full - parity_target) * bridge.alpha).square().mean(), params
        )
        cache_grads = torch.autograd.grad(
            ((pred - parity_target) * bridge.alpha).square().mean(), params
        )
        for fg, cg in zip(full_grads, cache_grads):
            torch.testing.assert_close(fg, cg, atol=1e-7, rtol=1e-4)
        pi.moe.domain_enabled = False
        baseline = pi(h, g).detach()
        pi.moe.domain_enabled = True
        bias = pi.moe.plugin.router.bias.detach().clone()
        pi.moe.plugin.router.bias.data.fill_(-1e6)
        rejected = pi(h, g)
        assert not pi.moe.last_selected.any()
        torch.testing.assert_close(rejected, baseline, atol=0, rtol=0)
        pi.moe.plugin.router.bias.data.copy_(bias)
        batch = replay.sample(a.batch_size)
        pred, sel, weight, _ = pi.moe.predict(batch, pi.dec[-1])
        loss = ((pred - batch["target"]) * bridge.alpha).square().mean()
        loss.backward()
        grads = {
            n: float(p.grad.norm()) if p.grad is not None else None
            for n, p in pi.moe.plugin.named_parameters()
        }
        assert all(
            v is not None and np.isfinite(v) and v > 1e-8
            for n, v in grads.items()
            if n != "router.bias"
        ), grads
        assert float(weight.detach().mean()) > 0.01, "Admitted contribution too small"
        if pi.moe.plugin.router.bias.item() < 0:
            assert grads["router.bias"] > 0, grads
        state["preflight"] = dict(
            cached_action_matches_full=True,
            cached_gradients_match_full=True,
            forced_rejection_exact_public=True,
            gradients=grads,
            sampled_selection_fraction=float(sel.float().mean()),
            sampled_contribution=float(weight.detach().mean()),
            public_unchanged=True,
        )
        pi.zero_grad(set_to_none=True)
        teacher_seed = {k: v[: replay.rows].clone() for k, v in replay.data.items()}
        initial = rollout("student", 77311, "initialized_evaluation")
        public = rollout("public", 77311, "public_evaluation")
        pi.moe.domain_enabled = True
        if a.preflight_only:
            progress(
                "preflight_passed",
                teacher_gate=teacher_result,
                initial_evaluation=initial,
                public_evaluation=public,
                actual_Adam_updates=0,
            )
            return
        from experiments.blink_stairs_routing_ab_20261106.retention import make_bank

        progress("building_old_reference_retention_bank")
        retention = make_bank(pi, robot, bridge)
        torch.save(retention, out / "old_reference_feature_bank.pt")
        state["retention_training"] = dict(
            reference_clips=128,
            rows=len(retention["target"]),
            fraction_per_batch=0.25,
            persistent_teacher_fraction_per_batch=0.25,
            targets="unchanged base actions on reference-state histories; not physically executed states",
            new_base_parameters_trainable=False,
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": [pi.moe.plugin.gate_up, pi.moe.plugin.down], "lr": 1e-3},
                {"params": [pi.moe.plugin.router.weight, pi.moe.plugin.router.bias], "lr": 3e-5},
            ],
            weight_decay=0.0,
        )
        initial_values = {n: p.detach().clone() for n, p in pi.moe.plugin.named_parameters()}
        audit = []

        def update(count):
            for k in range(count):
                if state["actual_Adam_updates"] == 8000:
                    optimizer.param_groups[0]["lr"] = 2e-4
                    optimizer.param_groups[1]["lr"] = 3e-6
                b = replay.sample(a.batch_size)
                ns = a.batch_size // 4
                si = torch.randint(len(teacher_seed["target"]), (ns,))
                b = {
                    k: torch.cat((teacher_seed[k][si].to(v.device), v[ns:]), 0)
                    for k, v in b.items()
                }
                nr = a.batch_size // 4
                ri = torch.randint(len(retention["target"]), (nr,))
                b = {
                    k: torch.cat((v[:-nr], retention[k][ri].to(v.device)), 0) for k, v in b.items()
                }
                pred, sel, wt, _ = pi.moe.predict(b, pi.dec[-1])
                loss = ((pred - b["target"]) * bridge.alpha).square().mean()
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                state["actual_Adam_updates"] += 1
                if (k + 1) % 100 == 0:
                    progress(
                        "updating",
                        loss=float(loss.detach()),
                        replay_rows=replay.rows,
                        update_selection=float(sel.float().mean()),
                        update_contribution=float(wt.mean()),
                    )
            gradient = {
                n: float(p.grad.norm()) if p.grad is not None else None
                for n, p in pi.moe.plugin.named_parameters()
            }
            steps = sorted({int(s["step"]) for s in optimizer.state.values()})
            assert steps == [state["actual_Adam_updates"]]
            audit.append(
                dict(
                    actual_Adam_updates=state["actual_Adam_updates"],
                    gradients=gradient,
                    raw_bias=float(pi.moe.plugin.router.bias),
                    effective_bias=float(pi.moe.plugin.router.bias.clamp_max(0)),
                    loss=float(loss.detach()),
                )
            )
            atomic(out / "gradient_audit.json", audit)
            verify()

        update(a.seed_updates)
        bestpath = save("seed", optimizer)
        dev = rollout("student", 77401, "seed_development")
        best = (dev["complete"], dev["arrived"], -dev["mean_valid_body_error_m"])
        best_result = dev
        bestupdates = state["actual_Adam_updates"]
        history = []
        beta = [0.8, 0.6, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0]
        for ri in range(a.rounds):
            state["round"] = ri + 1
            collection = rollout(
                "student",
                77501 + ri,
                "round_%02d_collection" % (ri + 1),
                collect=True,
                beta=beta[min(ri, len(beta) - 1)],
                record=False,
            )
            update(a.updates_per_round)
            path = save("round_%02d" % (ri + 1), optimizer)
            dev = rollout("student", 77401, "round_%02d_development" % (ri + 1))
            rank = (dev["complete"], dev["arrived"], -dev["mean_valid_body_error_m"])
            if rank > best:
                best = rank
                bestpath = path
                best_result = dev
                bestupdates = state["actual_Adam_updates"]
            history.append(
                dict(
                    round=ri + 1,
                    beta=beta[min(ri, len(beta) - 1)],
                    actual_Adam_updates=state["actual_Adam_updates"],
                    development=dev,
                    collection=collection,
                )
            )
            atomic(out / "history.json", history)
            progress("round_completed", best_checkpoint=str(bestpath), best_updates=bestupdates)
            # Fixed paired budget: no adaptive early stopping.
        final_training = save("last", optimizer)
        payload = torch.load(final_training, map_location="cuda:0", weights_only=True)
        pi.moe.plugin.load_state_dict(payload["plugin"])
        shutil.copy2(bestpath, out / "best.domain.pt")
        shutil.copy2(final_training, out / "selected.domain.pt")
        final = {role: [] for role in ["teacher", "public", "student"]}
        for seed in [78601, 78611]:
            for role in final:
                final[role].append(rollout(role, seed, f"final_{role}_{seed}"))
        changed = {
            n: float((p - initial_values[n]).norm()) for n, p in pi.moe.plugin.named_parameters()
        }
        verify()
        passed = all(
            x["complete"] / N >= 0.95
            and x["arrived"] / N >= 0.9
            and x["mean_valid_body_error_m"] <= max(0.15, t["mean_valid_body_error_m"] * 1.25)
            for x, t in zip(final["student"], final["teacher"])
        )
        progress(
            "completed" if passed else "trial_not_accepted",
            task_acceptance=passed,
            teacher_gate=teacher_result,
            initial_evaluation=initial,
            public_evaluation=public,
            final=final,
            selected_updates=state["actual_Adam_updates"],
            selected_checkpoint=str(out / "selected.domain.pt"),
            development_best_updates=bestupdates,
            development_best_checkpoint=str(out / "best.domain.pt"),
            latest_training_checkpoint=str(final_training),
            new_parameter_delta_l2=changed,
            public_unchanged=True,
            teacher_unchanged=True,
            parameters_finite=True,
            retention_acceptance="pending separate native Blink128 evaluation",
            scope="Only 449 steps of the450-frame/50Hz reference. Surviving and reference-end proximity do not establish completing all15 steps, reaching the top, or standing still.",
            full_staircase_and_terminal_stability_validated=False,
            metric_definitions=dict(
                complete="alive through449 steps",
                arrived="alive and torso anchor within.25m of reference endpoint; NOT staircase top",
            ),
            final_seed_scope="new seeds78601/78611 for this paired pilot; same reference and fixed geometry, not held-out skills",
        )
        atomic(out / "report.json", state)
    except BaseException as exc:
        progress("failed", error=repr(exc))
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

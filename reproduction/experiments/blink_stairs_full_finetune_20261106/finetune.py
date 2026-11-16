"""Post-domain mixed rehearsal; jointly fine-tune the complete Actor."""

import argparse, hashlib, json, os, shutil, time, traceback
from pathlib import Path
import numpy as np
import torch
from experiments.mimicx_all_tasks_v1.runtime import setup, module_digest
from experiments.blink_stairs_full_finetune_20261106.model import (
    load_actor,
    teacher_base,
    public_digest,
    pack,
    unpack,
    digest,
    FORMAT,
    KEYS,
)
from experiments.blink_stairs_full_finetune_20261106.replay import make_old_reference
from blinkmimic.moe_policy import router_losses, CONTRACT
from experiments.blink_stairs_routing_ab_20261106.routing import load_base
from experiments.blink_stairs_routing_ab_20261106.bridge import Bridge
from experiments.blink_stairs_routing_ab_20261106.train import Replay

from pera.paths import ROOT

R = ROOT
BASE = R / "checkpoints/base/best.pt"


def atomic(p, x):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(x, indent=2, ensure_ascii=False))
    tmp.replace(p)


def sample(bank, n, keys):
    ids = torch.randint(len(bank["target"]), (n,))
    return {k: bank[k][ids].to("cuda:0") for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=Path, required=True)
    ap.add_argument("--old-bank", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(79101)
    torch.backends.cuda.matmul.allow_tf32 = True
    state = dict(
        phase="starting",
        pid=os.getpid(),
        robot="g1",
        source_domain=str(a.domain),
        source_domain_sha256=hashlib.sha256(a.domain.read_bytes()).hexdigest(),
        new_Adam_updates=0,
        maximum_new_Adam_updates=2000,
        public_checkpoint_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
        world_model=False,
        PPO=False,
    )

    def progress(phase, **kw):
        state.update(phase=phase, elapsed_seconds=time.time() - started, **kw)
        atomic(out / "status.json", state)
        print(
            json.dumps(
                {
                    k: state.get(k)
                    for k in [
                        "phase",
                        "new_Adam_updates",
                        "loss",
                        "evaluation",
                        "elapsed_seconds",
                        "error",
                    ]
                }
            ),
            flush=True,
        )

    env = None
    try:
        progress("loading")
        env, teacher, row, paths = setup("stairs", 79201, 512)
        pi, cfg, robot = load_actor(source=a.domain, trainable=True)
        old_teacher = teacher_base()
        source = torch.load(a.domain, map_location="cpu", weights_only=True)
        assert source["actual_Adam_updates"] == 9000
        state.update(source_Adam_updates=9000, routing_mode=pi.moe.routing_mode)
        bridge = Bridge(env, pi, robot, paths["motion"], out)
        N = 512
        horizon = int(row["horizon"])
        assert horizon == 449
        import mujoco

        mujoco.mj_saveModel(env.unwrapped.sim.mj_model, str(out / "native_scene.mjb"), None)
        pub = public_digest(pi)
        tdigest = module_digest(teacher)
        old_teacher_hash = digest(old_teacher)
        params = list(pi.parameters())
        names = [n for n, p in pi.named_parameters() if p.requires_grad]
        assert len(names) == len(list(pi.parameters()))
        buffers = {n: v.detach().clone() for n, v in pi.named_buffers()}
        old = torch.load(a.old_bank / "bank.pt", map_location="cpu", weights_only=True)
        old_report = json.loads((a.old_bank / "report.json").read_text())
        assert (
            old_report["source_domain_sha256"] == state["source_domain_sha256"]
            and old_report["public_sha256"] == pub
        )
        reference = make_old_reference(old_teacher, robot, bridge)
        keys = KEYS
        assert set(keys) <= set(old)
        order = torch.randperm(len(old["target"]))
        nv = max(512, len(order) // 10)
        validation = {k: old[k][order[:nv]] for k in keys}
        old = {k: old[k][order[nv:]] for k in keys}
        state.update(
            public_sha256=pub,
            teacher_module_sha256=tdigest,
            trainable_tensors=names,
            trainable_parameters=sum(p.numel() for p in params),
            mixture=dict(
                stair_teacher=0.25,
                stair_student_replay=0.25,
                old_actual_rollouts=0.25,
                old_reference=0.25,
            ),
            old_actual_train_rows=len(old["target"]),
            old_actual_validation_rows=nv,
            old_reference_rows=len(reference["target"]),
            public_actor_lr=3e-6,
            new_FFN_lr=3e-5,
            router_lr=1e-6,
            weight_decay=0.0,
            optimizer="Fresh AdamW for entire Actor; source weights retained; old and new router parameters at1e-6",
            public_actor_trainable=True,
            normalization_and_selection_buffers="preserved; all learnable parameters including RMSNorm remain trainable",
            input_cache="packed raw observations and goal only; full network recomputed for each update",
            router_auxiliary="Original K1 balance/inactive losses, weights.01/1, on old replay half; no forced new-expert quota",
            reference_fragment_only=True,
            full_staircase_validated=False,
        )
        atomic(out / "plan.json", state)
        replay = Replay()
        teacher_queries = 0

        def verify():
            assert module_digest(teacher) == tdigest and digest(old_teacher) == old_teacher_hash
            assert all(torch.equal(buffers[n], v) for n, v in pi.named_buffers())
            assert all(torch.isfinite(p).all() for p in params)
            assert all(not p.requires_grad and p.grad is None for p in old_teacher.parameters())

        @torch.no_grad()
        def rollout(role, seed, label, collect=False, beta=0.0, record=False):
            nonlocal teacher_queries
            pi.moe.domain_enabled = role != "public"
            obs = bridge.reset(seed)
            alive = torch.ones(N, dtype=torch.bool, device="cuda:0")
            front = torch.zeros(N, dtype=torch.long, device="cuda:0")
            err = []
            body = []
            contrib = tokens = 0
            trace = []
            valid = []
            selected = []
            weight = []
            frames = []
            for t in range(horizon):
                h, g = bridge.observe()
                act = pi(h, g)
                native_teacher = None
                if collect or role == "teacher":
                    native_teacher = teacher(obs)
                    teacher_queries += N
                    if env.clip_actions is not None:
                        native_teacher = native_teacher.clamp(-env.clip_actions, env.clip_actions)
                if collect:
                    target = bridge.to_student(native_teacher)
                    replay.put(pack(h, g, target), target, pi.moe.last_old_indices)
                if role == "teacher":
                    native = native_teacher
                elif collect and beta:
                    native = torch.where(
                        torch.rand(N, 1, device="cuda:0") < beta,
                        native_teacher,
                        bridge.to_native(act),
                    )
                else:
                    native = bridge.to_native(act)
                e = (bridge.command.robot_anchor_pos_w - bridge.command.anchor_pos_w).norm(dim=-1)
                be = (
                    (bridge.command.robot_body_pos_w - bridge.command.body_pos_w)
                    .norm(dim=-1)
                    .mean(-1)
                )
                if alive.any():
                    err.append(float(e[alive].mean()))
                    body.append(float(be[alive].mean()))
                contrib += float(pi.moe.last_weight[:, 0][alive].sum())
                tokens += int(alive.sum())
                if record:
                    trace.append(env.unwrapped.sim.data.qpos[0].cpu().numpy().copy())
                    valid.append(bool(alive[0]))
                    selected.append(bool(pi.moe.last_selected[0]))
                    weight.append(float(pi.moe.last_weight[0]))
                    frames.append(int(bridge.command.time_steps[0]))
                obs, _, done, _ = env.step(native)
                alive &= ~done.bool()
                front += alive.long()
                bridge.stepped(native, done)
                if t % 100 == 0:
                    progress("collecting" if collect else "evaluating", evaluation=label, frame=t)
            finalerr = (bridge.command.robot_anchor_pos_w - bridge.command.anchor_pos_w).norm(
                dim=-1
            )
            result = dict(
                seed=seed,
                environments=N,
                complete=int(alive.sum()),
                arrived=int((alive & (finalerr <= 0.25)).sum()),
                mean_valid_anchor_error_m=float(np.mean(err)),
                mean_valid_body_error_m=float(np.mean(body)),
                mean_new_contribution=contrib / max(tokens, 1),
                metric_scope="Survival449steps / proximity<=25cm to reference endpoint; not reaching staircase top",
                robot0_first_termination_frame=next(
                    (i for i, v in enumerate(valid) if not v), None
                ),
            )
            if record:
                np.savez_compressed(
                    out / f"{label}_trajectory.npz",
                    qpos=np.asarray(trace),
                    alive=np.asarray(valid),
                    selected=np.asarray(selected),
                    contribution=np.asarray(weight),
                    reference_frame=np.asarray(frames),
                    fps=50,
                )
            atomic(out / f"{label}.json", result)
            verify()
            return result

        teacher_gate = rollout("teacher", 79201, "teacher_gate", collect=True)
        assert teacher_gate["complete"] >= 500 and teacher_gate["arrived"] >= 500
        teacher_seed = {k: v[: replay.rows].clone() for k, v in replay.data.items() if k in keys}
        before = [rollout("student", s, f"before_student_{s}", record=True) for s in [79601, 79611]]
        baseline_dev = rollout("student", 79401, "baseline_development")
        # Refresh replay on combined-policy states; teacher only provides labels.
        rollout("student", 79301, "initial_student_collection", collect=True)

        def batch_forward(b):
            h, g = unpack(b)
            pred, stats = pi.action_and_aux(h, g)
            return pred, stats, h

        # Confirm compact raw inputs preserve outputs and the new whole graph
        # actually reaches all public and new parameter families.
        h, g = bridge.observe()
        actual = pi(h, g)
        packed = pack(h, g, actual)
        rh, rg = unpack(packed)
        reconstructed = pi(rh, rg)
        torch.testing.assert_close(actual, reconstructed, atol=2e-5, rtol=2e-5)
        frozen_source = load_base(BASE, domain=a.domain)[0]
        frozen_source.requires_grad_(False)
        with torch.no_grad():
            torch.testing.assert_close(pi(h, g), frozen_source(h, g), atol=2e-5, rtol=2e-5)
        del frozen_source
        stair_probe = sample(teacher_seed, 128, keys)
        old_probe = sample(old, 128, keys)
        probe = {k: torch.cat((stair_probe[k], old_probe[k])) for k in keys}
        pred, _, _ = batch_forward(probe)
        probe_loss = ((pred - probe["target"]) * bridge.alpha).square().mean()
        probe_loss.backward()
        families = {
            "goal_encoder": "enc.",
            "body_decoder": "dec.",
            "reference_encoder": "moe.router_embed.",
            "public_router": "moe.router.weight",
            "public_ffn_up": "moe.gate_up",
            "public_ffn_down": "moe.down",
            "shared_ffn": "moe.shared_",
            "new_ffn": "moe.plugin.gate_up",
            "new_router": "moe.plugin.router.weight",
        }
        task_gradients = {
            family: sum(
                float(p.grad.float().square().sum())
                for n, p in pi.named_parameters()
                if n.startswith(prefix) and p.grad is not None
            )
            ** 0.5
            for family, prefix in families.items()
        }
        assert all(np.isfinite(v) and v > 0 for v in task_gradients.values()), task_gradients
        pi.zero_grad(set_to_none=True)
        state["preflight"] = dict(
            compact_raw_input_matches_full=True,
            whole_graph_matches_original_source_outputs=True,
            all_actor_parameters_trainable=True,
            task_gradient_by_family=task_gradients,
        )
        routers = []
        new_ffn = []
        public = []
        for n, p in pi.named_parameters():
            if n.startswith(("moe.router", "moe.plugin.router")):
                routers.append(p)
            elif n in ["moe.plugin.gate_up", "moe.plugin.down"]:
                new_ffn.append(p)
            else:
                public.append(p)
        opt = torch.optim.AdamW(
            [
                {"params": public, "lr": 3e-6},
                {"params": new_ffn, "lr": 3e-5},
                {"params": routers, "lr": 1e-6},
            ],
            weight_decay=0.0,
        )
        assert sum(len(group["params"]) for group in opt.param_groups) == len(params)

        def save(label):
            verify()
            payload = dict(
                format=FORMAT,
                robot="g1",
                routing_mode=pi.moe.routing_mode,
                source_base_sha256=state["public_checkpoint_sha256"],
                source_domain=str(a.domain),
                source_domain_sha256=state["source_domain_sha256"],
                source_domain_Adam_updates=9000,
                additional_actor_Adam_updates=state["new_Adam_updates"],
                domain_lineage_Adam_updates=9000 + state["new_Adam_updates"],
                pi={k: v.detach().cpu() for k, v in pi.state_dict().items()},
                optimizer=opt.state_dict(),
                optimizer_step_semantics="fresh whole-Actor optimizer counts additional updates only",
                finetune=dict(
                    mixture=state["mixture"],
                    old_actual_bank=str(a.old_bank),
                    method="mixed BC+DAgger; all Actor weights trainable, teachers frozen",
                    world_model=False,
                    PPO=False,
                ),
            )
            p = out / f"{label}.actor.pt"
            tmp = p.with_suffix(".tmp")
            torch.save(payload, tmp)
            tmp.replace(p)
            return p

        @torch.no_grad()
        def old_loss():
            total = 0.0
            for s in range(0, len(validation["target"]), 512):
                b = {k: v[s : s + 512].to("cuda:0") for k, v in validation.items()}
                pred = batch_forward(b)[0]
                total += float(((pred - b["target"]) * bridge.alpha).square().sum())
            return total / (len(validation["target"]) * 29)

        initial_loss = old_loss()
        initial_path = save("before")
        candidates = [
            dict(
                new_Adam_updates=0,
                old_actual_validation_loss=initial_loss,
                development=baseline_dev,
                checkpoint=str(initial_path),
            )
        ]
        initial_params = {n: p.detach().clone() for n, p in pi.named_parameters()}
        audit = []
        for target in [500, 1000, 2000]:
            if state["new_Adam_updates"]:
                rollout(
                    "student",
                    79301 + state["new_Adam_updates"],
                    f"student_collection_{target}",
                    collect=True,
                )
            while state["new_Adam_updates"] < target:
                parts = [
                    sample(teacher_seed, 128, keys),
                    {k: v for k, v in replay.sample(128).items() if k in keys},
                    sample(old, 128, keys),
                    sample(reference, 128, keys),
                ]
                b = {k: torch.cat([p[k] for p in parts]) for k in keys}
                pred, stats, h = batch_forward(b)
                per = ((pred - b["target"]) * bridge.alpha).square().mean(-1)
                old_logits = pi.moe.logits(pi.obs_norm.router(h[256:, -1, 99:]))
                balance, inactive, *_ = router_losses(
                    old_logits, pi.moe.select(old_logits), old_logits + pi.moe.selection_bias
                )
                aux = (
                    CONTRACT["moe_load_balance"]["weight"] * balance
                    + CONTRACT["inactive_expert_margin_to_topk"]["weight"] * inactive
                )
                loss = per.mean() + aux
                assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                opt.step()
                state["new_Adam_updates"] += 1
                if state["new_Adam_updates"] % 100 == 0:
                    progress(
                        "updating",
                        loss=float(loss.detach()),
                        loss_parts=[float(v) for v in per.detach().reshape(4, 128).mean(-1)],
                        router_aux=float(aux.detach()),
                    )
            steps = sorted({int(s["step"]) for s in opt.state.values()})
            assert steps == [target]
            audit.append(
                dict(
                    new_Adam_updates=target,
                    optimizer_steps=steps,
                    gradients={
                        n: float(p.grad.norm()) if p.grad is not None else None
                        for n, p in pi.named_parameters()
                    },
                    raw_bias=float(pi.moe.plugin.router.bias),
                    effective_bias=float(pi.moe.plugin.router.bias.clamp_max(0)),
                )
            )
            atomic(out / "gradient_audit.json", audit)
            path = save(f"update_{target}")
            dev = rollout("student", 79401, f"development_{target}")
            val = old_loss()
            candidates.append(
                dict(
                    new_Adam_updates=target,
                    old_actual_validation_loss=val,
                    development=dev,
                    checkpoint=str(path),
                )
            )
            atomic(out / "candidates.json", candidates)

        def eligible(c):
            d = c["development"]
            return (
                d["complete"] >= baseline_dev["complete"] - 6
                and d["arrived"] >= baseline_dev["arrived"] - 16
                and d["mean_valid_body_error_m"]
                <= baseline_dev["mean_valid_body_error_m"] * 1.05 + 0.005
            )

        # Retain source as an explicit fallback; old validation is state-level,
        # not a substitute for the complete native128 physical check below.
        chosen = min(
            [c for c in candidates if eligible(c)],
            key=lambda c: (c["old_actual_validation_loss"], -c["development"]["arrived"]),
        )
        payload = torch.load(chosen["checkpoint"], map_location="cpu", weights_only=True)
        pi.load_state_dict(payload["pi"])
        shutil.copy2(chosen["checkpoint"], out / "selected.actor.pt")
        after = [rollout("student", s, f"after_student_{s}", record=True) for s in [79601, 79611]]
        teacher_final = rollout("teacher", 79601, "final_teacher_79601", record=True)
        deltas = {
            family: sum(
                float((p - initial_params[n]).float().square().sum())
                for n, p in pi.named_parameters()
                if n.startswith(prefix)
            )
            ** 0.5
            for family, prefix in families.items()
        }
        verify()
        progress(
            "completed",
            before_stairs=before,
            after_stairs=after,
            teacher_gate=teacher_gate,
            teacher_final=teacher_final,
            selected_new_Adam_updates=chosen["new_Adam_updates"],
            selected_domain_lineage_Adam_updates=9000 + chosen["new_Adam_updates"],
            optimizer_total_additional_updates=2000,
            selected_checkpoint=str(out / "selected.actor.pt"),
            selection="minimum held-out old actual-state action MSE among0/500/1000/2000 with no major development stair drop; source0 allowed as fallback",
            old_validation_baseline=initial_loss,
            old_validation_selected=chosen["old_actual_validation_loss"],
            candidates=candidates,
            selected_parameter_delta_l2=deltas,
            public_parameters_changed=public_digest(pi) != pub,
            public_file_unchanged=hashlib.sha256(BASE.read_bytes()).hexdigest()
            == state["public_checkpoint_sha256"],
            teacher_unchanged=True,
            old_teacher_unchanged=digest(old_teacher) == old_teacher_hash,
            teacher_label_queries=teacher_queries,
            old128_acceptance="pending full native physical evaluation",
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

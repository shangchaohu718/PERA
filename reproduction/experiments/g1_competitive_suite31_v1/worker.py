"""One native task; current MLP student, BC/DAgger, actual Top2 from rollout0."""

import argparse, json, hashlib, os, signal, time, traceback
from pathlib import Path

from pera.paths import ROOT


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    a = parser.parse_args()
    a.config = a.config.resolve()
    cfg = json.loads(a.config.read_text())
    task = next(t for t in cfg["tasks"] if t["id"] == a.task)
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=a.resume)
    started = time.monotonic()
    launcher = None
    if task["family"] in ["pmt", "ultra"]:
        os.chdir(ROOT / "third_party" / ("PMT" if task["family"] == "pmt" else "ULTRA"))
    if task["family"] in ["pmt", "kick"]:
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(
            {
                "headless": True,
                "enable_cameras": task["family"] == "pmt",
                "device": "cuda:0",
            }
        )
    import numpy as np, torch
    from .model import Actor, install_capture, base
    from .adapters import native

    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    install_capture()
    checkpoint = ROOT / cfg["base_checkpoint"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == cfg["base_sha256"]
    actor = Actor(checkpoint, task["context_dim"], task["id"])
    params = list(actor.pi.moe.plugin.parameters())
    assert len(params) == 4
    stopped = [False]
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    state = dict(
        status="initializing",
        task=task,
        base_checkpoint=str(checkpoint),
        base_sha256=cfg["base_sha256"],
        public_sha256=actor.public_sha256,
        world_model=False,
        PPO=False,
        routing=cfg["routing"],
        actual_Adam_updates=0,
        completed_rounds=0,
        task_accepted=False,
        retention_accepted=False,
        preflight_only=a.preflight_only,
        trainable_tensors=[n for n, p in actor.named_parameters() if p.requires_grad],
        maximum_Adam_updates=cfg["maximum_updates_per_task"],
    )

    def report(status, **kw):
        state.update(status=status, elapsed_s=round(time.monotonic() - started, 2), **kw)
        atomic(out / "report.json", state)
        print(
            json.dumps(
                {
                    k: state.get(k)
                    for k in [
                        "status",
                        "actual_Adam_updates",
                        "completed_rounds",
                        "loss",
                        "elapsed_s",
                        "error",
                    ]
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    adapter = None
    optimizer = None
    try:
        report("loading_original_native_task")
        adapter = native(task, actor, cfg, out)
        state["scope"] = adapter.scope

        def digest(module):
            h = hashlib.sha256()
            for name, x in sorted(module.state_dict().items()):
                h.update(name.encode())
                h.update(
                    x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                )
            return h.hexdigest()

        teacher_hashes = [digest(m) for m in adapter.modules]
        state["frozen_teacher_encoder_sha256"] = teacher_hashes
        state["legacy_evidence_plugin"] = task["legacy_evidence_plugin"]
        state["legacy_evidence_sha256"] = hashlib.sha256(
            (ROOT / task["legacy_evidence_plugin"]).read_bytes()
        ).hexdigest()
        actor.context_contract = dict(
            family=task["family"],
            dimensions=task["context_dim"],
            scope=adapter.scope,
            frozen_teacher_encoder_sha256=teacher_hashes,
        )

        def verify():
            assert base.public_digest(actor.pi) == actor.public_sha256, (
                "Frozen public parameters/buffers changed"
            )
            assert [digest(m) for m in adapter.modules] == teacher_hashes, "Teacher/encoder changed"
            assert all(p.grad is None for p in actor.pi.parameters() if not p.requires_grad)
            assert all(torch.isfinite(p).all() for p in params)

        def rollout(role, seed, collect=False, beta=0.0):
            actor.pi.moe.domain_enabled = role == "student"
            actor.context = None
            with torch.no_grad():
                r, data = adapter.run(role, seed, collect, beta)
            if not np.isfinite(r.get("success_fraction", float("nan"))):
                raise RuntimeError("Native evaluation returned an invalid success rate")
            verify()
            return r, data

        keys = [
            "x",
            "xn",
            "hidden",
            "native",
            "top1",
            "shared",
            "logit1",
            "choice2",
            "context",
            "skip",
            "target",
            "oldids",
        ]

        def trim(data, capacity=32768):
            if not data or not len(data["target"]):
                raise RuntimeError("Native task produced no teacher labels")
            ids = torch.randperm(len(data["target"]))[:capacity]
            return {k: data[k][ids].contiguous().cpu() for k in keys}

        def append(old, new):
            new = trim(new)
            return trim({k: torch.cat((old[k], new[k])) for k in keys}) if old is not None else new

        def device(data):
            return {k: v.to("cuda:0") for k, v in data.items()}

        def sample(data, n):
            ids = torch.randint(len(data["target"]), (n,))
            return {k: v[ids].to("cuda:0") for k, v in data.items()}

        def evaluate(seeds):
            results = [rollout("student", s)[0] for s in seeds]
            score = float(np.mean([r["success_fraction"] for r in results]))
            return results, score

        def task_gate(results):
            threshold = 0.8 if task["family"] == "tennis" else cfg["student_min_success_fraction"]
            for r in results:
                if r["success_fraction"] < threshold:
                    return False
                if task["family"] == "mimicx" and (
                    r.get("mean_body_error_m") is None
                    or r["mean_body_error_m"] > 0.15
                    or r["reference_survivors"] / r["environments"] < 0.95
                ):
                    return False
            return True

        report("teacher_physical_gate")
        teacher_eval = rollout("teacher", cfg["seed"] + 1)[0]
        state["teacher_gate"] = teacher_eval
        minimum = task.get(
            "teacher_min_success_fraction",
            0.6 if task["family"] == "tennis" else cfg["teacher_min_success_fraction"],
        )
        if teacher_eval["success_fraction"] < minimum:
            report(
                "teacher_failed",
                teacher_unchanged=True,
                public_unchanged=True,
                actual_Adam_updates=0,
            )
            return
        optimizer = torch.optim.AdamW(
            [
                {"params": [params[0], params[1]], "lr": cfg["ffn_lr"]},
                {"params": [params[2], params[3]], "lr": cfg["router_lr"]},
            ],
            weight_decay=0.0,
        )
        best_score = -1.0
        best_updates = 0
        replay = None
        teacher_seed = None
        retention = None
        initialization = None

        def save_boundary():
            verify()
            steps = sorted({int(s["step"]) for s in optimizer.state.values()})
            assert steps == [state["actual_Adam_updates"]]
            tmp = out / "last.domain.tmp"
            actor.save(tmp, state["actual_Adam_updates"], optimizer, initialization)
            tmp.replace(out / "last.domain.pt")
            payload = dict(
                plugin=actor.pi.moe.plugin.state_dict(),
                optimizer=optimizer.state_dict(),
                actual_Adam_updates=state["actual_Adam_updates"],
                completed_rounds=state["completed_rounds"],
                teacher_seed=teacher_seed,
                replay=replay,
                retention=retention,
                initialization=initialization,
                best_score=best_score,
                best_updates=best_updates,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
                base_sha256=cfg["base_sha256"],
                task=task["id"],
                config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),
                teacher_encoder_sha256=teacher_hashes,
            )
            tmp = out / "training_state.tmp"
            torch.save(payload, tmp)
            tmp.replace(out / "training_state.pt")
            state["optimizer_steps"] = steps

        if a.resume:
            ck = torch.load(
                out / "training_state.pt",
                map_location="cpu",
                weights_only=True,
            )
            if (
                ck["base_sha256"] != cfg["base_sha256"]
                or ck["task"] != task["id"]
                or ck["config_sha256"] != hashlib.sha256(a.config.read_bytes()).hexdigest()
                or ck["teacher_encoder_sha256"] != teacher_hashes
            ):
                raise ValueError("Resume base/task/config mismatch")
            actor.pi.moe.plugin.load_state_dict(ck["plugin"])
            optimizer.load_state_dict(ck["optimizer"])
            state.update(
                actual_Adam_updates=ck["actual_Adam_updates"],
                completed_rounds=ck["completed_rounds"],
            )
            (
                teacher_seed,
                replay,
                retention,
                initialization,
                best_score,
                best_updates,
            ) = [
                ck[k]
                for k in [
                    "teacher_seed",
                    "replay",
                    "retention",
                    "initialization",
                    "best_score",
                    "best_updates",
                ]
            ]
            torch.set_rng_state(ck["torch_rng"])
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        else:
            report("collecting_teacher_seed")
            _, raw = rollout("teacher", cfg["seed"] + 2, collect=True, beta=1.0)
            teacher_seed = trim(raw, 8192)
            replay = trim(raw)
            initialization = actor.initialize(device(teacher_seed), cfg["initial_admission_target"])
            state["initialization"] = initialization
            actor.pi.moe.domain_enabled = True
            # Actual native input: compare cached action/gradients and verify off path.
            h, g = actor.inputs(
                __import__(
                    "experiments.g1_competitive_suite31_v1.model",
                    fromlist=["CAPTURE"],
                ).CAPTURE["inputs"]
            )
            hist = actor.history.detach()
            actor.pi.moe.runtime_context = actor.cached["context"]
            full = actor.pi(hist, g)
            f = base.features(actor.pi, g)
            cached, *_ = actor.pi.moe.predict(f, actor.pi.dec[-1])
            torch.testing.assert_close(full, cached, atol=3e-5, rtol=3e-5)
            fg = torch.autograd.grad(full.square().mean(), params)
            cg = torch.autograd.grad(cached.square().mean(), params)
            for x, y in zip(fg, cg):
                torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-4)
            actor.pi.moe.domain_enabled = False
            public = actor.pi(hist, g).detach()
            bias = actor.pi.moe.plugin.router.bias.detach().clone()
            actor.pi.moe.domain_enabled = True
            actor.pi.moe.plugin.router.bias.data.fill_(-1e6)
            off = actor.pi(hist, g)
            torch.testing.assert_close(public, off, atol=0, rtol=0)
            actor.pi.moe.plugin.router.bias.data.copy_(bias)
            batch = sample(teacher_seed, cfg["batch_size"])
            pred, sel, wt, _ = actor.pi.moe.predict(batch, actor.pi.dec[-1])
            loss = ((pred - batch["target"]) * actor.pi.ref_alpha).square().mean()
            loss.backward()
            grads = {
                n: float(p.grad.norm()) if p.grad is not None else None
                for n, p in actor.pi.moe.plugin.named_parameters()
            }
            assert all(
                v is not None and np.isfinite(v) and v > 1e-8
                for k, v in grads.items()
                if k != "router.bias"
            ), grads
            state["preflight"] = dict(
                cached_actions_match=True,
                cached_gradients_match=True,
                forced_rejection_exact_public=True,
                initial_gradients=grads,
                mean_contribution=float(wt.mean()),
                public_unchanged=True,
            )
            actor.zero_grad(set_to_none=True)
            actor.save(out / "initialized.domain.pt", 0, None, initialization)
            if a.preflight_only:
                report("preflight_passed", actual_Adam_updates=0)
                return
            from blinkmimic.kinematics import Kinematics
            from .retention import make_bank

            actor.pi.moe.runtime_context = None
            bridge = type(
                "BankBridge",
                (),
                dict(
                    mdp=type(
                        "MDPStub",
                        (),
                        dict(kin=Kinematics(actor.robot, "cuda:0")),
                    )(),
                    q0=actor.pi.ref_q0,
                    device="cuda:0",
                ),
            )()
            bridge.mdp.xi_dim = actor.robot.xi_dim
            report("building_original128_reference_bank")
            retention = make_bank(actor.pi, actor.robot, bridge)
            retention["context"] = torch.zeros(len(retention["target"]), task["context_dim"])
            retention = trim(retention)

        def update(count):
            t0 = time.monotonic()
            for k in range(count):
                bs = cfg["batch_size"]
                ns = int(bs * cfg["persistent_teacher_fraction"])
                nr = int(bs * cfg["old_reference_fraction"])
                pieces = [
                    sample(teacher_seed, ns),
                    sample(replay, bs - ns - nr),
                    sample(retention, nr),
                ]
                b = {key: torch.cat([x[key] for x in pieces]) for key in keys}
                pred, selected, weight, _ = actor.pi.moe.predict(b, actor.pi.dec[-1])
                loss = ((pred - b["target"]) * actor.pi.ref_alpha).square().mean()
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                state["actual_Adam_updates"] += 1
                if (k + 1) % 100 == 0:
                    report(
                        "updating",
                        loss=float(loss.detach()),
                        updates_per_second=(k + 1) / (time.monotonic() - t0),
                        new_selection=float(selected.float().mean()),
                        new_contribution=float(weight.detach().mean()),
                    )
            verify()

        if state["actual_Adam_updates"] == 0:
            update(cfg["seed_updates"])
            save_boundary()
        for ri in range(state["completed_rounds"], cfg["rounds"]):
            if stopped[0] or (out.parent / "PAUSE").exists():
                report("paused", checkpoint=str(out / "last.domain.pt"))
                return
            report("collecting_student_states", round=ri + 1)
            _, raw = rollout(
                "student",
                cfg["seed"] + 100 + ri,
                collect=True,
                beta=cfg["teacher_probability"][ri],
            )
            replay = append(replay, raw)
            update(cfg["updates_per_round"])
            state["completed_rounds"] = ri + 1
            report("development_evaluation")
            dev, score = evaluate(cfg["validation_seeds"])
            if score > best_score:
                best_score = score
                best_updates = state["actual_Adam_updates"]
                actor.save(
                    out / "best.domain.pt",
                    best_updates,
                    optimizer,
                    initialization,
                )
            save_boundary()
            state.update(
                best_updates=best_updates,
                best_score=best_score,
                development=dev,
            )
            if ri >= 5 and task_gate(dev):
                break
        report("final_pure_student_evaluation")
        selected = torch.load(out / "best.domain.pt", map_location="cpu", weights_only=True)
        actor.pi.moe.plugin.load_state_dict(selected["plugin"])
        final, score = evaluate(cfg["final_seeds"])
        verify()
        report(
            "completed",
            selected_updates=best_updates,
            best_checkpoint=str(out / "best.domain.pt"),
            final=final,
            task_accepted=task_gate(final),
            public_unchanged=True,
            teacher_unchanged=True,
            parameters_finite=True,
            retention_accepted=False,
            retention_status="pending separate native128 evaluation",
            multiple_new_experts_jointly_validated=False,
        )
    except BaseException as exc:
        report("failed", error=repr(exc))
        traceback.print_exc()
        raise
    finally:
        if adapter is not None:
            adapter.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()

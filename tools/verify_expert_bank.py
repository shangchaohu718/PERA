"""CPU parity of a switched bank against each original single-expert Actor."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from blinkmimic import networks
from pera.context import ContextMoE, ContextPlugin
from pera.paths import ROOT
from pera.routing import CompetitiveMoE, Plugin, policy_from_args, public_digest
from pera.switching import SwitchedActor, file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/expert_bank.json")
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "checkpoints/composed/pera_30_experts.pt"
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261111)
    config = json.loads(args.config.read_text())
    bank = SwitchedActor.load_bundle(args.checkpoint)
    base = torch.load(ROOT / config["base"], map_location="cpu", weights_only=True)
    assert bank.base_sha256 == file_digest(ROOT / config["base"]) == config["base_sha256"]
    assert bank.expert_names == tuple(row["name"] for row in config["experts"])
    assert bank.active_expert is None and not any(bank.switches.values())
    initial_public = public_digest(bank.pi)
    assert all(not p.requires_grad for p in bank.parameters())
    rows, all_off = [], []
    with torch.no_grad():
        for precision in ("fp32", "bf16"):
            original, _ = policy_from_args(base["args"])
            original.load_state_dict(base["pi"], strict=True)
            original.eval().requires_grad_(False)
            networks.AMP_DTYPE = torch.bfloat16 if precision == "bf16" else None
            h = torch.randn(8, 10, 608) * 0.1
            g = torch.randn(8, 730) * 0.1
            bank.select_expert(None)
            native = original(h, g)
            actual = bank(h, g)
            torch.testing.assert_close(actual, native, atol=0, rtol=0)
            all_off.append(
                dict(precision=precision, max_abs_error=float((actual - native).abs().max()))
            )
            original.moe.__class__ = CompetitiveMoE
            original.moe.domain_enabled = True
            for row in config["experts"]:
                name = row["name"]
                payload = torch.load(ROOT / row["path"], map_location="cpu", weights_only=True)
                assert file_digest(ROOT / row["path"]) == row["sha256"]
                assert bank.expert_specs[name]["source_sha256"] == row["sha256"]
                spec = bank.expert_specs[name]
                if spec["kind"] == "context":
                    original.moe.__class__ = ContextMoE
                    original.moe.plugin = ContextPlugin(spec["context_dim"])
                else:
                    original.moe.__class__ = CompetitiveMoE
                    original.moe.plugin = Plugin()
                original.moe.plugin.load_state_dict(payload["plugin"], strict=True)
                for key, value in bank.pi.moe.plugins[name].state_dict().items():
                    torch.testing.assert_close(value, payload["plugin"][key], atol=0, rtol=0)
                original.moe.plugin.requires_grad_(False)
                bank.select_expert(name)
                context = torch.randn(8, spec["context_dim"]) if spec["context_dim"] else None
                original.moe.runtime_context = context
                expected = original(h, g)
                actual = bank(h, g, context)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                assert bool(actual.isfinite().all())
                torch.testing.assert_close(bank.pi.moe.last_selected, original.moe.last_selected)
                torch.testing.assert_close(
                    bank.pi.moe.last_weight, original.moe.last_weight, atol=0, rtol=0
                )
                cached = bank.predict_cached(bank.cached_features(g))[0]
                torch.testing.assert_close(cached, actual, atol=0, rtol=0)
                rows.append(
                    dict(
                        name=name,
                        precision=precision,
                        samples=len(h),
                        context_dim=spec["context_dim"],
                        source_sha256=row["sha256"],
                        max_abs_action_error=float((actual - expected).abs().max()),
                        max_abs_cached_error=float((cached - actual).abs().max()),
                        plugin_tensors_exact=True,
                        routing_exact=True,
                    )
                )
    bank.select_expert(bank.expert_names[0])
    try:
        bank.set_expert_enabled(bank.expert_names[1], True)
    except ValueError:
        pass
    else:
        raise AssertionError("Two enabled task experts were accepted")
    assert bank.active_expert == bank.expert_names[0]
    bank.select_expert(None)
    assert public_digest(bank.pi) == initial_public == bank.public_sha256
    assert len([n for n, _ in bank.pi.named_modules() if n == "moe.router_embed"]) == 1
    assert not torch.cuda.is_initialized()
    result = dict(
        schema="pera.switched_bank_check.v1",
        status="passed",
        checked_at=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(args.checkpoint.relative_to(ROOT))
        if args.checkpoint.is_relative_to(ROOT)
        else args.checkpoint.name,
        checkpoint_sha256=file_digest(args.checkpoint),
        bytes=args.checkpoint.stat().st_size,
        expert_count=len(bank.expert_names),
        max_active_experts=1,
        default_active_expert=None,
        public_actor_sha256=initial_public,
        all_parameters_frozen=True,
        all_off_parity=all_off,
        single_expert_parity=rows,
        second_enable_rejected_atomically=True,
        source_plugin_tensors_unchanged=True,
        CPU_only=True,
        CUDA_initialized=False,
        precision_scope="FP32 and bf16 AMP configurations on CPU; native CUDA autocast falls back to FP32 on CPU. Not a GPU bf16 parity test.",
        limitations=[
            "Synthetic native inputs/context; no physical rollout or new skill acceptance",
            "Frozen external task encoders and task resources remain separate",
            "Existing failed checkpoints retain their historical status",
        ],
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status=result["status"],
                expert_count=len(bank.expert_names),
                comparisons=len(rows),
                public_actor_sha256=initial_public,
                max_abs_action_error=max(r["max_abs_action_error"] for r in rows),
                CUDA_initialized=False,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

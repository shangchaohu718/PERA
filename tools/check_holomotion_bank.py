"""Compare a real single-file HoloMotion bank against each original domain file."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "integrations/HoloMotion")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from pera.holomotion_bank import FrozenHoloMotionBank, context_kwargs, file_digest
    from modumotion_g1.official import build_official_actor, sample_observations
    from modumotion_g1.domain_moe import (
        install_domains,
        load_domain,
        public_digest,
        set_active_domains,
        set_domain_context,
    )

    torch.set_num_threads(4)
    torch.manual_seed(7116)
    bank = FrozenHoloMotionBank.load(args.checkpoint, args.device, args.base_dir)
    config = OmegaConf.load(args.base_dir / "config.yaml")
    single = build_official_actor(config, args.device)
    state = load_file(
        str(args.base_dir / "model_14000/actor/model.safetensors"), device=args.device
    )
    single.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()}, strict=True)
    del state
    install_domains(single)
    single.eval().requires_grad_(False)
    observations = sample_observations(config, batch=2, device=args.device)
    observations.apply_(lambda value: value.add_(torch.randn_like(value) * 0.01))

    def reset(actor):
        actor.reset_kv_cache(2, args.device)
        actor.clear_env_cache(None)

    calls = {name: 0 for name in bank.domains}
    handles = []
    for name, domain in bank.domains.items():

        def record_call(module, inputs, output, name=name):
            calls[name] += 1

        for block in bank.actor.actor_module.layers:
            handles.append(block.domain_plugins[domain].register_forward_hook(record_call))
    reset(bank.actor)
    reset(single)
    public_error = 0.0
    with torch.no_grad():
        for _ in range(3):
            a = bank(observations)["actions"]
            b = single(observations, mode="inference")["actions"]
            public_error = max(public_error, float((a - b).abs().max()))
    assert public_error == 0 and sum(calls.values()) == 0
    results = []
    for name in bank.expert_names:
        row = bank.records[name]
        filename = args.native_root / row["source_relative_to_native_root"]
        assert file_digest(filename) == row["sha256"]
        domain = load_domain(single, filename, **context_kwargs(row))
        set_active_domains(single, [domain])
        contract = row.get("context")
        context = torch.randn(2, contract["dim"], device=args.device) * 0.05 if contract else None
        set_domain_context(single, {domain: context} if contract else None)
        reset(bank.actor)
        reset(single)
        before = dict(calls)
        maximum = 0.0
        with torch.no_grad():
            for _ in range(3):
                a = bank(observations, expert=name, context=context)["actions"]
                b = single(observations, mode="inference")["actions"]
                assert bool(a.isfinite().all())
                maximum = max(maximum, float((a - b).abs().max()))
        assert maximum == 0, name
        assert calls[name] - before[name] == 3 * len(bank.actor.actor_module.layers)
        assert all(calls[key] == before[key] for key in calls if key != name)
        results.append(
            dict(
                id=name, action_max_abs_error=maximum, sequence_steps=3, inactive_expert_FFN_calls=0
            )
        )
        set_active_domains(single, [])
        set_domain_context(single, None)
        for block in single.actor_module.layers:
            del block.domain_plugins[domain]
            del block.merge_bias[domain]
        print(f"PASS {name}: exact single-policy parity, other experts never evaluated", flush=True)
    for invalid in [list(bank.expert_names[:2]), "missing_expert"]:
        try:
            bank.select_expert(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid/multiple experts accepted")
    contextual = next(name for name, row in bank.records.items() if row.get("context"))
    try:
        bank(observations, expert=contextual)
    except ValueError:
        assert bank.active_expert is None
    else:
        raise AssertionError("Missing context accepted")
    reset(bank.actor)
    reset(single)
    with torch.no_grad():
        a = bank(observations)["actions"]
        b = single(observations, mode="inference")["actions"]
    assert torch.equal(a, b)
    assert bank.verify() == public_digest(single)
    for handle in handles:
        handle.remove()
    report = dict(
        status="passed",
        expert_count=len(bank.expert_names),
        base_family="holomotion_official_g1_v1.4",
        public_sha256=bank.public_sha256,
        checkpoint_sha256=file_digest(args.checkpoint),
        checkpoint_bytes=args.checkpoint.stat().st_size,
        contains_one_public_actor=True,
        all_parameters_frozen=True,
        public_off_action_max_abs_error=public_error,
        public_restored_after_switches=True,
        multiple_experts_rejected=True,
        missing_context_rejected_and_switch_cleared=True,
        comparisons=results,
        device=args.device,
        synthetic_action_parity_only=True,
        new_physics_skill_evaluation=False,
        training=False,
        uploaded=False,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, indent=2))


if __name__ == "__main__":
    main()

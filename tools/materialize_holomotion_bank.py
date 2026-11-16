"""Combine a released expert bank with upstream-downloaded official weights locally."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "integrations/HoloMotion")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT) or output.exists():
        parser.error(
            "Choose a new output outside the source release; this file contains official weights"
        )
    import torch
    from pera.holomotion_bank import FrozenHoloMotionBank, file_digest

    torch.set_num_threads(4)
    bank = FrozenHoloMotionBank.load(args.experts, "cpu", args.base_dir)
    bank.verify()
    payload = torch.load(args.experts, map_location="cpu", weights_only=True)
    if payload["contains_official_actor"]:
        parser.error("Input is already a full checkpoint")
    payload["public_state_dict"] = {
        name: value.detach().cpu()
        for name, value in bank.actor.state_dict().items()
        if ".domain_plugins." not in name
    }
    payload["contains_official_actor"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(
        json.dumps(
            dict(
                expert_count=len(bank.expert_names),
                bytes=output.stat().st_size,
                sha256=file_digest(output),
                local_use_only=True,
                uploaded=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

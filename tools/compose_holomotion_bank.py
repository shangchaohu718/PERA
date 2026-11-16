"""Compose a local single-Actor checkpoint and a separate expert-only bank."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "integrations/HoloMotion")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/holomotion_experts25.json")
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Full official Actor + domains; outside the release folder, local use only",
    )
    parser.add_argument(
        "--plugins-output",
        type=Path,
        required=True,
        help="All domain FFNs/routers in one file, without official Actor weights",
    )
    args = parser.parse_args()
    from pera.holomotion_bank import compose

    print(
        json.dumps(
            compose(
                args.manifest, args.native_root, args.base_dir, args.output, args.plugins_output
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

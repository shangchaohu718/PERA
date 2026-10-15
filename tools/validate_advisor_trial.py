"""Compare paired frozen MJLab evaluations before adopting an LLM coefficient proposal."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from unified_motion.llm.validation import compare_trials, TRACKING_METRICS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path, nargs='+')
    parser.add_argument('--candidate', required=True, type=Path, nargs='+')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-cost-ratio', type=float, default=1.25)
    parser.add_argument('--primary-metric', choices=TRACKING_METRICS, default='root_position_error_m',
                        help='choose before running trials; recorded in the admission report')
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    result = compare_trials(read(args.proposal), list(map(read, args.baseline)),
                            list(map(read, args.candidate)), max_cost_ratio=args.max_cost_ratio,
                            primary_metric=args.primary_metric)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    return 0 if result['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

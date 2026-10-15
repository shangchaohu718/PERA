"""Analyze existing JSONL/gzip training metrics through DeepSeek or GLM, or preview offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_motion.llm.client import ChatClient, ChatConfig
from unified_motion.llm.advisor import SYSTEM_PROMPT, make_request, propose, read_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('deepseek', 'glm'), required=True)
    parser.add_argument('--model', required=True, help='model enabled on your provider account')
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--runner-metadata', type=Path, help='fallback snapshot for old logs without per-iteration training_tuning')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window', type=int, default=100)
    parser.add_argument('--task-description', default='Reference motion tracking')
    parser.add_argument('--dry-run', action='store_true', help='write exact redacted prompt; no API key or simulator needed')
    parser.add_argument('--no-json-mode', action='store_true', help='for models without server JSON mode; local validation still required')
    args = parser.parse_args()
    records = read_metrics(args.metrics, args.window)
    settings = records[-1].get('training_tuning') if records else None
    if settings is None and args.runner_metadata is not None:
        settings = json.loads(args.runner_metadata.read_text(encoding='utf-8')).get('training_tuning')
    if settings is None:
        parser.error('need the latest training_tuning snapshot in metrics or --runner-metadata')
    config = ChatConfig(args.provider, args.model, json_mode=not args.no_json_mode)
    if args.dry_run:
        result = dict(mode='offline_preview_no_api_call', system=SYSTEM_PROMPT,
            user=json.loads(make_request(settings, records, args.task_description)))
    else:
        result = propose(ChatClient(config), settings, records, args.task_description)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), status=result.get('status', result.get('mode')))))


if __name__ == '__main__':
    main()

"""Replay the fifteen-scene suite with one expert-bank file and external official base."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = "898d1a02829f901fbad4ed07178df2b06778620b54e8bccf5e07ffcb4c31d32d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error("Choose a new output directory; prior results are retained")
    out.mkdir(parents=True)
    suite = json.loads((ROOT / "configs/reproduction_suite15.json").read_text())
    bank = (args.bank or ROOT / suite["expert_bank"]).resolve()
    summary = {"schema": "pera.suite_replay.v1", "status": "running", "groups": [], "scenes": {}}
    report_path = out / "report.json"

    def save():
        report_path.write_text(json.dumps(summary, indent=2) + "\n")

    save()
    started = time.monotonic()
    if args.video:
        print("Offscreen recording: no simulation window will open.", flush=True)
    try:
        for group in suite["groups"]:
            target = out / group["id"]
            command = [
                sys.executable,
                "-m",
                "pera",
                "demo",
                "--bank",
                str(bank),
                "--config",
                group["config"],
                "--device",
                args.device,
                "--seconds",
                str(group["seconds"]),
                "--repeat",
                "--check-isolation",
                "--output",
                str(target),
            ]
            if args.base_dir:
                command += ["--base-dir", str(args.base_dir.resolve())]
            if args.video:
                command += ["--video"]
            print(
                f"Running {group['id']}: {group['seconds']} simulated seconds; "
                f"log: {out / (group['id'] + '.log')}",
                flush=True,
            )
            with (out / (group["id"] + ".log")).open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads((target / "report.json").read_text())
            if (
                report["status"] != "completed"
                or report["teacher_action_queries"] != 0
                or report["public_sha256"] != PUBLIC
                or report["actor_instances"] != 1
                or report["resident_expert_count"] != 25
                or report["expert_fault_events"]
                or report["max_active_experts_per_inference"] != 1
                or not report["runtime_sources"]["checkout_only_project_sources"]
            ):
                raise RuntimeError(f"Policy/source contract failed in {group['id']}")
            isolation = report["isolation_check"]
            if (
                isolation["single_vs_shared_action_max_abs"] != 0
                or isolation["forward_vs_reverse_action_max_abs"] != 0
                or not isolation["distinct_attention_cache_storage"]
            ):
                raise RuntimeError(f"Temporal isolation failed in {group['id']}")
            summary["groups"].append(
                {
                    "id": group["id"],
                    "report": f"{group['id']}/report.json",
                    "simulated_seconds": report["simulated_seconds"],
                    "torch_peak_MiB": report["torch_cuda_peak_allocated_MiB"],
                }
            )
            for task in report["tasks"]:
                records = task["records"]
                if not records or any(row["tracking_lost"] for row in records):
                    raise RuntimeError(f"Tracking failed: {task['config']['id']}")
                errors = [r["root_error_m"] for r in records if r["root_error_m"] is not None]
                summary["scenes"][task["config"]["id"]] = {
                    "group": group["id"],
                    "control_steps": len(records),
                    "tracking_lost_frames": 0,
                    "root_error_max_m": max(errors) if errors else None,
                    "completed_reference_intervals": task["completed_reference_intervals"],
                }
            save()
            print(
                f"Completed {group['id']}; elapsed {time.monotonic() - started:.0f}s; "
                f"results: {target}",
                flush=True,
            )
        if len(summary["scenes"]) != suite["unique_scene_count"]:
            raise RuntimeError("Scene roster differs from the declared suite")
        summary["status"] = "passed"
        summary["scope"] = (
            "Deterministic scene-port replay; not statistical task success or training reproduction"
        )
    except Exception as error:
        summary.update(status="failed", error=str(error))
        save()
        raise
    save()
    print(f"PASS: {len(summary['scenes'])} scenes; report: {report_path}")


if __name__ == "__main__":
    main()

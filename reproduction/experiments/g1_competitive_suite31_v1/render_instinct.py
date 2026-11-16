"""Record the current MLP competitive expert in its original physical scene."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from pera.paths import ROOT


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def module_sha(module):
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", default="diveroll_lab_no_wall")
    parser.add_argument("--seed", type=int, default=970101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    cfg = json.loads((run / "config.json").read_text())
    taskcfg = next(t for t in cfg["tasks"] if t["id"] == args.task)
    if taskcfg["family"] != "instinct":
        raise ValueError("This recorder supports the original Instinct scenes only")
    used = {
        cfg["seed"] + 1,
        cfg["seed"] + 2,
        *cfg["validation_seeds"],
        *cfg["final_seeds"],
    }
    used.update(cfg["seed"] + 100 + i for i in range(cfg["rounds"]))
    if args.seed in used:
        raise ValueError("Video seed must differ from training and prior evaluation seeds")
    plugin = run / args.task / "best.domain.pt"
    checkpoint = ROOT / cfg["base_checkpoint"]
    if file_sha(checkpoint) != cfg["base_sha256"]:
        raise ValueError("Frozen base file changed")
    folder = ROOT / "data/instinct_task_dagger_v1" / args.task
    if args.task == "knee_climb_no_wall":
        folder = ROOT / "data/instinct_climb_dagger_v1"
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        if file_sha(folder / name) != digest:
            raise ValueError("Original motion/mesh changed: " + name)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    import imageio.v2 as imageio
    import numpy as np
    import torch
    from .model import Actor, FORMAT, base, install_capture

    install_capture()  # Dispatch before legacy modules bind their imports.
    from instinct_mj.scripts.supermimic.evaluate import EvalArgs, choose_camera
    from modumotion_g1.instinct_dagger import NativeTeacherTask
    from scripts.instinct_dagger import TEACHER

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    free, _ = torch.cuda.mem_get_info()
    if free < 4 * 2**30:
        raise RuntimeError("Need 4 GiB free for independent native video evaluation")
    torch.manual_seed(cfg["seed"])
    data = torch.load(plugin, map_location="cpu", weights_only=True)
    if (
        data["format"] != FORMAT
        or data["domain"] != args.task
        or data["base_sha256"] != cfg["base_sha256"]
    ):
        raise ValueError("Expert/base/task mismatch")
    actor = Actor(checkpoint, data["context_dim"], data["domain"])
    if actor.public_sha256 != data["public_sha256"]:
        raise ValueError("Frozen public tensors differ")
    actor.pi.moe.plugin.load_state_dict(data["plugin"])
    actor.requires_grad_(False).eval()
    actor.pi.moe.domain_enabled = True
    started = time.monotonic()
    native = NativeTeacherTask(folder, TEACHER, 1, "cuda:0", video=True)
    try:
        modules = [native.teacher_actor, native.teacher_norm, native.depth]
        hashes = [module_sha(m) for m in modules]
        if hashes != data["context_contract"]["frozen_teacher_encoder_sha256"]:
            raise ValueError("Native teacher/depth interface differs from training")
        native.reset(args.seed, actor)
        azimuth, distance = choose_camera(
            native.env,
            native.motion.buffer,
            0,
            EvalArgs(
                motion_folder=str(folder),
                video_distance=3.5,
                video_elevation=-35,
            ),
        )
        views = [(azimuth, distance, -35), (azimuth, max(4.0, distance), -70)]
        camera = native.env._offline_renderer._cam
        original_render = native.env.render

        def render_both():
            images = []
            for az, dist, elevation in views:
                camera.azimuth, camera.distance, camera.elevation = (
                    az,
                    dist,
                    elevation,
                )
                images.append(original_render().copy())
            return np.concatenate(images, axis=1)

        native.env.render = render_both
        with torch.no_grad():
            result, _, frames = native.rollout(
                actor, seed=args.seed, controller="student", video=True
            )
        if result["teacher_queries"] != 0 or result["teacher_execution_probability"] != 0:
            raise RuntimeError("Student-only video unexpectedly used teacher actions")
        if not frames:
            raise RuntimeError("No physical video frames recorded")
        # Include the final post-action pose, which the normal rollout omits.
        frames.append(render_both())
        fps = 1 / native.env.step_dt
        video = output / "student_two_views.mp4"
        imageio.mimwrite(
            video,
            frames,
            fps=fps,
            codec="libx264",
            quality=8,
            ffmpeg_log_level="error",
        )
        unchanged = base.public_digest(actor.pi) == actor.public_sha256
        teacher_unchanged = hashes == [module_sha(m) for m in modules]
        if not unchanged or not teacher_unchanged:
            raise RuntimeError("Frozen tensors changed during rendering")
        report = {
            "task": args.task,
            "robot": "g1",
            "architecture": "current MLP-MoE 8 public +1 true Top2",
            "plugin": str(plugin),
            "plugin_sha256": file_sha(plugin),
            "selected_Adam_updates": data["actual_Adam_updates"],
            "seed": args.seed,
            "seed_not_in_training_or_previous_evaluation": True,
            "episodes_recorded": 1,
            "successful_episodes_selected_by_retry": False,
            "student_eval": result,
            "same_episode_for_both_views": True,
            "views_azimuth_distance_elevation": views,
            "reference_ghost_rendered": False,
            "video": str(video),
            "fps": fps,
            "frames": len(frames),
            "public_unchanged": unchanged,
            "teacher_encoder_unchanged": teacher_unchanged,
            "elapsed_s": time.monotonic() - started,
            "scope": "One unseen-seed physical rollout in the fixed original scene; not broad generalization acceptance",
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in report.items() if k != "student_eval"}
                | {"successes": result["successful_episodes"]},
                indent=2,
            ),
            flush=True,
        )
    finally:
        native.close()


if __name__ == "__main__":
    main()

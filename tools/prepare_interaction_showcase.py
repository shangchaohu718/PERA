"""Export existing native scene/encoder resources. No downloads or training.

Run from a compatible existing native interpreter. --apply explicitly permits a
one-environment CUDA export of Instinct's frozen encoder and phase-zero state.
"""

import argparse
import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_xml(xml, output, mesh_dir):
    """Save referenced meshes, retaining their bytes and native XML fields."""
    output.mkdir(parents=True, exist_ok=True)
    tree = ET.fromstring(xml)
    compiler = tree.find("compiler")
    for node in tree.find("asset"):
        if not node.get("file"):
            continue
        source = Path(node.get("file"))
        source = source if source.is_absolute() else mesh_dir / source
        name = digest(source)[:16] + source.suffix
        target = output / "assets" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        node.set("file", name)
    compiler.set("meshdir", "assets")
    materials = {n.get("name") for n in tree.find("asset").findall("material")}
    for node in tree.iter():
        value = node.get("material")
        if value and value not in materials:
            candidates = [m for m in materials if m.endswith("/" + value)]
            if len(candidates) != 1:
                raise ValueError("Unresolved exported material")
            node.set("material", candidates[0])
    ET.ElementTree(tree).write(output / "scene.xml", encoding="unicode")


def objects(source, husky):
    sys.path[:0] = [
        str(ROOT / "src"),
        str(ROOT / "reproduction"),
        str(ROOT / "integrations/HoloMotion"),
        str(source / "third_party/ATHLETE/athlete/src"),
    ]
    import mujoco
    import modumotion_g1.husky_skateboarding as hs

    hs.HUSKY = husky
    out = ROOT / "data/showcase/skate_native"
    for wrists in (False, True):
        model = hs.scene_model(wrists)
        folder = out / ("g1_29" if wrists else "g1_23")
        folder.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveLastXML(str(folder / "raw.xml"), model)
        xml = (folder / "raw.xml").read_text()
        mesh_dir = (
            husky / "test_scene" / ET.fromstring(xml).find("compiler").get("meshdir")
        ).resolve()
        portable_xml(xml, folder, mesh_dir)
        (folder / "raw.xml").unlink()
        mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    (out / "ckpts").mkdir(exist_ok=True)
    shutil.copyfile(husky / "ckpts/test.onnx", out / "ckpts/test.onnx")
    from experiments.athlete_tennis_v1.compat import setup

    setup()
    from experiments.athlete_tennis_v1 import evaluate_official as official

    official.POLICY = (
        source / "third_party/ATHLETE/deploy/policies/gpu3_2_model45000/model_45000.onnx"
    )
    import athlete.scripts.play_student_onnx as native

    # Capture the native scene spec before its function discards it. This adds
    # no simulation or policy changes and leaves the upstream source untouched.
    captured = {}
    code = textwrap.dedent(inspect.getsource(native._build_standalone_model))
    anchor = "  model = spec.compile()"
    if code.count(anchor) != 1:
        raise ValueError("Native scene builder changed; inspect before export")
    code = code.replace(anchor, '  captured["spec"] = spec.copy()\n' + anchor)
    namespace = {**vars(native), "captured": captured}
    exec(compile(code, "<capture native scene spec>", "exec"), namespace)
    native._build_standalone_model = namespace["_build_standalone_model"]
    sim = native._StandaloneStudentSimulation(official.cfg(97001))
    out = ROOT / "data/showcase/tennis_native"
    meshes = source / "third_party/ATHLETE/robots/replay_unitree_description/meshes"
    portable_xml(captured["spec"].to_xml(), out / "scene", meshes)
    mujoco.MjModel.from_xml_path(str(out / "scene/scene.xml"))
    shutil.copyfile(official.POLICY, out / "model_45000.onnx")
    shutil.copyfile(
        source / "reports/athlete_tennis_v1/body_window_continue_10k_v1/context_normalization.json",
        out / "context_normalization.json",
    )
    (out / "metadata.json").write_text(
        json.dumps(
            dict(
                physics_dt=sim.model.opt.timestep,
                iterations=sim.model.opt.iterations,
                ls_iterations=sim.model.opt.ls_iterations,
                joint_names=list(sim.joint_names),
            ),
            indent=2,
        )
        + "\n"
    )


def vault(source, task_name="roll_vault_no_wall", output_name="vault_native", device="cuda:0"):
    # This isolated process uses the original task only for resource extraction.
    sys.path[:0] = [
        str(source),
        str(source / "HoloMotion"),
        str(source / "third_party/InstinctMJ/src"),
        str(source / "third_party/mjlab/src"),
        str(source / "third_party/instinct_rl"),
    ]
    import numpy as np
    import torch
    from modumotion_g1.instinct_dagger import NativeTeacherTask

    out = ROOT / "data/showcase" / output_name
    out.mkdir(parents=True, exist_ok=True)
    task = NativeTeacherTask(
        source / "data/instinct_task_dagger_v1" / task_name,
        source
        / "checkpoints/instinct_supermimic_v0.1/g1_perceptive_shadowing_supermimic_model_8000.pt",
        1,
        device,
    )
    try:
        obs = task.reset(7)
        task.env.scene.write(out / "scene")
        # The upstream exporter omits disk-backed robot meshes in some versions.
        mesh_dir = (
            source / "third_party/InstinctMJ/src/instinct_mj/assets/resources/unitree_g1/meshes"
        )
        tree = ET.parse(out / "scene/scene.xml").getroot()
        for node in tree.find("asset"):
            if node.get("file"):
                dest = out / "scene/assets" / node.get("file")
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.is_file():
                    continue
                shutil.copyfile(mesh_dir / node.get("file"), dest)
        torch.save(task.depth.state_dict(), out / "depth_state.pt")
        frame, phase, origin = task.motion.current_frame()
        reference = task.adapter.reference_from_motion(task.motion).clone()
        reference[:, :3] += origin[0]
        np.savez_compressed(
            out / "actor_reference.npz",
            qpos_holo=reference.cpu().numpy(),
            fps=np.array(task.motion.framerate),
        )
        camera = task.env.scene["camera"]
        np.savez_compressed(
            out / "camera_geometry.npz",
            directions=camera.ray_directions[0].cpu().numpy(),
            starts=camera.ray_starts[0].cpu().numpy(),
            offset_pos=camera._offset_pos[0].cpu().numpy(),
            offset_quat=camera._offset_quat[0].cpu().numpy(),
            native_depth=obs[:, task.depth_slice].cpu().numpy(),
        )
        np.savez_compressed(
            out / "initial_state.npz",
            qpos=task.env.sim.data.qpos[0].cpu().numpy(),
            qvel=task.env.sim.data.qvel[0].cpu().numpy(),
            depth=obs[:, task.depth_slice].cpu().numpy(),
            context=task.depth(obs[:, task.depth_slice]).detach().cpu().numpy(),
        )
        (out / "contract.json").write_text(
            json.dumps(
                dict(
                    joint_names=list(task.robot.joint_names),
                    sim_joint_signs=task.adapter.sim_joint_signs.tolist(),
                    to_holo=task.adapter.to_holo,
                    to_sim=task.adapter.to_sim,
                    depth_eps=task.depth.eps,
                    depth_shape=list(task.depth.depth_shape),
                    physics_dt=0.005,
                ),
                indent=2,
            )
            + "\n"
        )
        configs = []
        for item in camera.noise_pipeline:
            cfg = dataclasses.asdict(item)
            cfg.pop("func", None)
            cfg["device"] = "cpu"
            configs.append(dict(cls=type(item).__name__, kwargs=cfg))
        (out / "noise.json").write_text(json.dumps(configs, indent=2) + "\n")
    finally:
        task.env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--husky-root", type=Path)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--stage", choices=["objects", "vault"], help=argparse.SUPPRESS)
    args = p.parse_args()
    source = args.source_root.resolve()
    husky = (args.husky_root or source.parent / "humanoid_skateboarding").resolve()
    if args.stage:
        if not args.apply:
            p.error("--stage requires --apply")
        if args.stage == "objects":
            objects(source, husky)
        else:
            vault(source)
        return
    required = {
        "objects": [
            "skate_native/g1_29/scene.xml",
            "tennis_native/scene/scene.xml",
            "tennis_native/context_normalization.json",
        ],
        "vault": [
            "vault_native/scene/scene.xml",
            "vault_native/camera_geometry.npz",
            "vault_native/depth_state.pt",
            "vault_native/noise.json",
        ],
    }
    missing = [
        stage
        for stage, names in required.items()
        if any(not (ROOT / "data/showcase" / name).is_file() for name in names)
    ]
    print(
        json.dumps(
            dict(apply=args.apply, missing_stages=missing, downloads=False, training=False),
            indent=2,
        ),
        flush=True,
    )
    if not args.apply:
        return
    for stage in missing:
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--source-root",
                str(source),
                "--husky-root",
                str(husky),
                "--apply",
                "--stage",
                stage,
            ],
            check=True,
        )
    files = []
    for folder in ("skate_native", "tennis_native", "vault_native"):
        for file in sorted((ROOT / "data/showcase" / folder).rglob("*")):
            if file.is_file():
                files.append(
                    dict(
                        path=str(file.relative_to(ROOT)),
                        bytes=file.stat().st_size,
                        sha256=digest(file),
                    )
                )
    (ROOT / "resources/interaction_showcase_manifest.json").write_text(
        json.dumps(
            dict(
                schema="pera.interaction_resources.v1",
                stage="local_review_only_not_uploaded",
                files=files,
                note="Native source scene exports and frozen encoder resources; original asset and dataset terms apply",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

"""Export existing official-base object/obstacle demonstration resources."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

from prepare_interaction_showcase import ROOT, portable_xml, vault
from prepare_compact_showcase import export_rough, crop_rough


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def box(source):
    import mujoco
    import numpy as np
    import torch

    sys.path[:0] = [
        str(source),
        str(source / "HoloMotion"),
        str(source / "experiments/ultra_box_v1"),
        str(source / "third_party/ULTRA/ultra"),
    ]
    import runtime
    import sim2sim_teacher as native

    out = ROOT / "data/showcase/box_native"
    out.mkdir(parents=True, exist_ok=True)
    motion = source / "data/ultra_official_v1/sub10_largebox_046_072_103_077_100_100_100.pt"
    previous = Path.cwd()
    try:
        os.chdir(source / "third_party/ULTRA")
        # The generated native XML already exists. Do not rewrite upstream files.
        with patch.object(native, "merge_mjcf", lambda *a, **k: None):
            env = runtime.load_native(motion, device="cpu")
        world = runtime.World(env)
        mujoco.mj_saveLastXML(str(out / "raw.xml"), env.model)
        portable_xml(
            (out / "raw.xml").read_text(),
            out / "scene",
            source / "third_party/ULTRA/ultra/data/assets/g1/meshes",
        )
        (out / "raw.xml").unlink()
        np.savez_compressed(
            out / "initial_state.npz",
            qpos=world.data.qpos,
            qvel=world.data.qvel,
            ctrl=world.data.ctrl,
        )
        torch.save(
            dict(hoi_data=env.hoi_data.cpu(), object_points=env.object_points.cpu()),
            out / "observation_data.pt",
        )
        norm = (
            source
            / "reports/distillation_suite_v1/run_20261105_173125/ultra_largebox/attempt_1/frozen_context_normalizer.pt"
        )
        shutil.copy2(norm, out / "frozen_context_normalizer.pt")
        contract = dict(
            joint_names=world.joint_names,
            stiffness=env.stiffness.tolist(),
            damping=env.damping.tolist(),
            torque_limits=env.torque_limits.tolist(),
            control_dt=env.control_dt,
            physics_dt=env.sim_dt,
            reference_frames=world.original_frames,
            end_frame=world.end_frame,
            motion_source=str(motion.relative_to(source)),
            motion_sha256=sha(motion),
            normalizer_sha256=sha(norm),
            teacher_action_queries=0,
        )
        (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        test = mujoco.MjModel.from_xml_path(str(out / "scene/scene.xml"))
        for name in [
            "body_mass",
            "body_inertia",
            "dof_armature",
            "dof_damping",
            "geom_friction",
            "actuator_gainprm",
        ]:
            np.testing.assert_allclose(
                getattr(test, name), getattr(env.model, name), atol=1e-8, rtol=1e-5
            )
        print("Exported native dynamic box, original observation data and frozen normalizer.")
    finally:
        os.chdir(previous)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--stage", choices=["box", "vault_wall", "gaps"], required=True)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    source = a.source_root.resolve()
    if not a.apply:
        print("Plan: export existing local " + a.stage + " resources; no downloads or training.")
        return
    folders = dict(box="box_native", vault_wall="vault_wall_native", gaps="gaps_native")
    if a.stage == "box":
        box(source)
    elif a.stage == "vault_wall":
        vault(source, "roll_vault_wall", "vault_wall_native", a.device)
    else:
        export_rough(source, "gaps", a.device)
        crop_rough("gaps")
    folder = ROOT / "data/showcase" / folders[a.stage]
    rows = [
        dict(path=str(f.relative_to(ROOT)), bytes=f.stat().st_size, sha256=sha(f))
        for f in sorted(folder.rglob("*"))
        if f.is_file()
    ]
    manifest = ROOT / "resources" / f"{a.stage}_showcase_manifest.json"
    manifest.write_text(
        json.dumps(
            dict(schema="pera.showcase_resources.v1", stage="local_native_export", files=rows),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

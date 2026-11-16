"""Prepare existing MimicX resources and matched experts for the compact arena."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def export_rough(source, task_name="rough", device="cuda:0"):
    import sys, copy, json, dataclasses, inspect, shutil
    import numpy as np, torch

    R = ROOT
    S = source
    sys.path[:0] = [
        str(S),
        str(S / "HoloMotion"),
        str(S / "third_party/InstinctMJ/src"),
        str(S / "third_party/mjlab/src"),
        str(S / "third_party/instinct_rl"),
    ]
    import mjlab.actuator  # noqa: F401 -- initialize the upstream actuator namespace
    from modumotion_g1.parkour_remaining import configure_hfield_capacity

    configure_hfield_capacity()
    from modumotion_g1.parkour_distillation import ParkourTask

    out = R / "data/showcase" / f"{task_name}_native"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(7)
    if task_name == "rough":
        task = ParkourTask("rough", 1, device, seed=0)
    else:
        from modumotion_g1.parkour_remaining import RemainingParkourTask

        task = RemainingParkourTask(task_name, 1, device, seed=0, route_mode="skill")
    try:
        task.reset(22001 if task_name == "rough" else 51001)
        task.env.scene.write(out / "scene")
        depth = task.obs[:, task.depth_slice].reshape(1, 8, 18, 32).clone()
        enc = copy.deepcopy(task.depth).cpu().eval()
        torch.jit.trace(enc, depth.cpu()).save(str(out / "depth_encoder.ts"))
        torch.save(enc.state_dict(), out / "depth_state.pt")
        camera = task.env.scene["camera"]
        np.savez_compressed(
            out / "camera_geometry.npz",
            directions=camera.ray_directions[0].cpu().numpy(),
            starts=camera.ray_starts[0].cpu().numpy(),
            offset_pos=camera._offset_pos[0].cpu().numpy(),
            offset_quat=camera._offset_quat[0].cpu().numpy(),
        )
        np.savez_compressed(
            out / "initial_state.npz",
            qpos=task.env.sim.data.qpos[0].cpu().numpy(),
            qvel=task.env.sim.data.qvel[0].cpu().numpy(),
            depth=depth.cpu().numpy(),
            context=task.depth(depth.flatten(1)).detach().cpu().numpy(),
            route_goal=task.route_goal[0].cpu().numpy(),
            standing=task.robot.data.default_joint_pos[0].cpu().numpy(),
        )
        configs = []
        for item in camera.noise_pipeline:
            cfg = dataclasses.asdict(item)
            cfg.pop("func", None)
            cfg["device"] = "cpu"
            configs.append(dict(cls=type(item).__name__, kwargs=cfg))
        (out / "noise.json").write_text(json.dumps(configs, indent=2))
        a = task.adapter
        cfg = task.command.cfg
        fields = [
            "velocity_control_stiffness",
            "heading_control_stiffness",
            "only_positive_lin_vel_x",
            "target_dis_threshold",
            "lin_vel_threshold",
            "ang_vel_threshold",
        ]
        command_cfg = {k: getattr(cfg, k) for k in fields}
        command_cfg["ang_vel_z"] = list(cfg.ranges.ang_vel_z)
        contract = dict(
            joint_names=list(task.robot.joint_names),
            sim_joint_signs=a.sim_joint_signs.tolist(),
            to_holo=a.to_holo,
            to_sim=a.to_sim,
            encoder_sha256=task.encoder_sha,
            context_contract=task.contract(),
            command_cfg=command_cfg,
            initial_command=task.command.command[0].cpu().tolist(),
            max_command=task.command.max_command_b[0].cpu().tolist(),
            physics_dt=float(task.env.sim.mj_model.opt.timestep),
            depth_eps=task.depth.eps,
            depth_shape=list(task.depth_shape),
        )
        (out / "contract.json").write_text(json.dumps(contract, indent=2))
        import xml.etree.ElementTree as ET

        tree = ET.parse(out / "scene/scene.xml")
        meshdir = S / "third_party/InstinctMJ/src/instinct_mj/assets/resources/unitree_g1/meshes"
        for el in tree.findall(".//asset/mesh"):
            if el.get("file"):
                dest = out / "scene/assets" / el.get("file")
                if not dest.exists():
                    src = meshdir / el.get("file")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
        print("Exported native rough terrain, initial state and frozen encoder.")
        from instinct_mj.envs import mdp

        for module in [mdp]:
            fn = getattr(module, "delayed_visualizable_image", None)
            if fn:
                (out / "history_source.txt").write_text(inspect.getsource(fn))
    finally:
        task.env.close()


def crop_rough(task_name="rough"):
    import numpy as np, mujoco, json, xml.etree.ElementTree as ET

    folder = ROOT / "data/showcase" / f"{task_name}_native/scene"
    src = folder / "scene.xml"
    m = mujoco.MjModel.from_xml_path(str(src))
    tree = ET.parse(src)
    root = tree.getroot()
    terrain = root.find("./worldbody/body[@name='terrain']")
    initial = np.load(folder.parent / "initial_state.npz")["qpos"]
    candidates = [g for g in terrain.findall("geom") if g.get("type") == "hfield"]
    keep = min(candidates, key=lambda g: abs(float(g.get("pos").split()[0]) - initial[0]))
    hf = root.find("./asset/hfield[@name='" + keep.get("hfield") + "']")
    idx = m.geom(keep.get("name")).dataid[0]
    nrow, ncol = int(m.hfield_nrow[idx]), int(m.hfield_ncol[idx])
    a = int(m.hfield_adr[idx])
    arr = m.hfield_data[a : a + nrow * ncol].reshape(nrow, ncol)
    i, j = 48, 111
    col_start = 50 if task_name == "gaps" else 0
    sub = arr[i : j + 1, col_start:].copy()
    lo, hi = float(sub.min()), float(sub.max())
    size = m.hfield_size[idx].copy()
    xs = np.linspace(-size[0], size[0], ncol)
    ys = np.linspace(-size[1], size[1], nrow)
    size[0] = (xs[-1] - xs[col_start]) / 2
    size[1] = (ys[j] - ys[i]) / 2
    pos = np.fromstring(keep.get("pos"), sep=" ")
    pos[0] += (xs[-1] + xs[col_start]) / 2
    pos[1] += (ys[j] + ys[i]) / 2
    pos[2] += lo * size[2]
    size[2] *= hi - lo
    # Inline XML elevations are written top-to-bottom, opposite to hfield_data rows.
    hf.set("nrow", str(len(sub)))
    hf.set("ncol", str(sub.shape[1]))
    hf.set("size", " ".join(map(str, size)))
    hf.set("elevation", " ".join(map(str, sub[::-1].ravel())))
    keep.set("pos", " ".join(map(str, pos)))
    for el in list(terrain):
        if el is not keep:
            terrain.remove(el)
    for el in list(root.find("asset")):
        if el.tag == "hfield" and el is not hf:
            root.find("asset").remove(el)
    for el in list(root.findall("./worldbody/site")):
        root.find("worldbody").remove(el)
    for el in list(root.findall("./keyframe")):
        root.remove(el)
    dest = folder / "scene_compact.xml"
    tree.write(dest, encoding="unicode")
    n = mujoco.MjModel.from_xml_path(str(dest))
    md, nd = mujoco.MjData(m), mujoco.MjData(n)
    mujoco.mj_forward(m, md)
    mujoco.mj_forward(n, nd)

    def heights(model, data):
        out = []
        for x in np.linspace(
            float(m.geom(keep.get("name")).pos[0]) - 1.3 if task_name == "gaps" else 0.1,
            float(m.geom(keep.get("name")).pos[0]) + 3.9,
            31,
        ):
            for y in np.linspace(-1.4, 1.4, 15):
                hit = np.array([-1], np.int32)
                dist = mujoco.mj_ray(
                    model,
                    data,
                    np.array([x, y, 5.0]),
                    np.array([0.0, 0.0, -1.0]),
                    np.array([1, 0, 0, 0, 0, 0], np.uint8),
                    True,
                    -1,
                    hit,
                )
                out.append(5 - dist)
        return np.array(out)

    err = float(abs(heights(m, md) - heights(n, nd)).max())
    print("CROP_RAY_HEIGHT_MAX_ERROR", err)
    if err > 1e-6:
        raise ValueError("Crop changed terrain heights")
    (ROOT / "data/showcase" / f"{task_name}_native/crop_check.json").write_text(
        json.dumps(
            dict(
                source="scene/scene.xml",
                destination="scene/scene_compact.xml",
                kept_geom=keep.get("name"),
                corridor_bounds=[
                    float(xs[col_start] + m.geom(keep.get("name")).pos[0]),
                    float(m.geom(keep.get("name")).pos[0] + 4),
                    float(ys[i]),
                    float(ys[j]),
                ],
                ray_checks=465,
                ground_height_max_abs_error_m=err,
            ),
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    p.add_argument(
        "--rough",
        action="store_true",
        help="Export native rough terrain and frozen depth encoder using the existing native environment",
    )
    args = p.parse_args()
    source = args.source_root.resolve()
    if args.rough:
        if not args.apply:
            print(
                "Plan: reuse/export existing native rough scene and encoder, then crop unused terrain; no downloads or training."
            )
            return
        folder = ROOT / "data/showcase/rough_native"
        required = [
            "scene/scene.xml",
            "initial_state.npz",
            "contract.json",
            "camera_geometry.npz",
            "depth_encoder.ts",
            "noise.json",
        ]
        if not all((folder / name).is_file() for name in required):
            export_rough(source)
        crop_rough()
        files = [
            dict(path=str(p.relative_to(ROOT)), sha256=digest(p), bytes=p.stat().st_size)
            for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name not in ("pixel_check.npy", "history_source.txt")
        ]
        (ROOT / "resources/rough_showcase_manifest.json").write_text(
            json.dumps(
                dict(schema="pera.showcase_resources.v1", stage="local_native_export", files=files),
                indent=2,
            )
            + "\n"
        )
        return
    assets = source / "third_party/mimicx_release/MimicX-Assets"
    catalog = json.loads(
        (
            source
            / "third_party/mimicx_release/MimicX-Policies/releases/2026-10-14-demonstrations/manifest.json"
        ).read_text()
    )
    tasks, files = [], []
    labels = dict(
        stairs="Stair ascent", kungfu="Martial arts", dance="Dance", football="Ball-control motion"
    )
    for name, label in labels.items():
        row = next(v for v in catalog["policies"] if v["task"] == name)
        folder = Path("data/showcase") / name
        for key, filename in [("motion", "motion.npz"), ("scene", "scene.json")]:
            if row.get(key):
                src = assets / row[key]["path"]
                dest = ROOT / folder / filename
                sha = digest(src)
                if sha != row[key]["sha256"]:
                    raise ValueError(f"Unexpected resource hash: {name}/{key}")
                if dest.exists() and digest(dest) != sha:
                    raise ValueError(f"Existing resource differs: {dest}")
                if args.apply and not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                files.append(
                    dict(
                        source=str(src.relative_to(source)),
                        path=str(dest.relative_to(ROOT)),
                        sha256=sha,
                        bytes=src.stat().st_size,
                    )
                )
        tasks.append(
            dict(
                id="mimicx_" + name,
                expert="mimicx_" + name,
                label=label,
                motion=str(folder / "motion.npz"),
                scene=str(folder / "scene.json") if row.get("scene") else None,
                origin=[0, 0, 0],
                horizon=row["horizon"],
                context_dim=0,
            )
        )
    config = json.loads((ROOT / "configs/showcase_descent.json").read_text())
    normalizer = (
        source
        / "reports/distillation_suite_v1/run_20261105_173125/mimicx_kungfu/attempt_1/frozen_context_normalizer.pt"
    )
    destination = ROOT / "data/showcase/kungfu/frozen_context_normalizer.pt"
    if args.apply:
        shutil.copy2(normalizer, destination)
    files.append(
        dict(
            source=str(normalizer.relative_to(source)),
            path=str(destination.relative_to(ROOT)),
            sha256=digest(normalizer),
            bytes=normalizer.stat().st_size,
        )
    )
    config["tasks"] = tasks
    config["default_tasks"] = [x["id"] for x in tasks]
    config.pop("checkpoint", None)
    config.pop("archived_for_demonstrations", None)
    config["base_family"] = "holomotion_official_g1_v1.4"
    config["public_sha256"] = "898d1a02829f901fbad4ed07178df2b06778620b54e8bccf5e07ffcb4c31d32d"
    config["resource_manifest"] = "resources/compact_showcase_manifest.json"
    if args.apply:
        (ROOT / "configs/showcase_motion_regions.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )
        (ROOT / "resources/compact_showcase_manifest.json").write_text(
            json.dumps(
                dict(
                    schema="pera.showcase_resources.v1",
                    stage="local_review_only_not_uploaded",
                    files=files,
                ),
                indent=2,
            )
            + "\n"
        )
    print(
        json.dumps(
            dict(apply=args.apply, tasks=[x["id"] for x in tasks], files=len(files)), indent=2
        )
    )


if __name__ == "__main__":
    main()

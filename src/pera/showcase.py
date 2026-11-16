"""Several task regions and robots in one MuJoCo model, driven by one shared Actor."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MDP
from blinkmimic.motion import MotionLib
from blinkmimic.rotations import quat_to_mat
from pera.multi_robot import MultiRobotController
from pera.paths import ROOT
from pera.robot import get_robot
from pera.routing import public_digest
from pera.switching import SwitchedActor, file_digest

COLORS = [
    (0.20, 0.64, 0.89),
    (0.96, 0.63, 0.24),
    (0.37, 0.76, 0.51),
    (0.72, 0.49, 0.88),
    (0.88, 0.40, 0.49),
    (0.36, 0.76, 0.75),
]


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def numbers(value):
    return " ".join(str(float(v)) for v in value)


def check_resources(config):
    manifest = json.loads(path(config["resource_manifest"]).read_text())
    for row in manifest["files"]:
        p = path(row["path"])
        if not p.is_file() or p.stat().st_size != row["bytes"] or file_digest(p) != row["sha256"]:
            raise ValueError(f"Missing or changed showcase resource: {row['path']}")


def build_world(config, tasks):
    """Clone robot bodies/drives but share all mesh/material assets in one model."""
    template_path = path(config["robot_scene"])
    template = ET.parse(template_path).getroot()
    root = ET.Element("mujoco", model="PERA shared skill arena")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(template_path.parent / "assets"))
    ET.SubElement(
        root,
        "option",
        timestep=str(config["physics_dt"]),
        integrator="implicitfast",
        gravity="0 0 -9.81",
        iterations=str(config.get("solver_iterations", 10)),
        ls_iterations=str(config.get("solver_ls_iterations", 20)),
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1600", offheight="1000")
    ET.SubElement(visual, "quality", shadowsize="2048")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.55 0.55 0.55")
    for tag in ("default", "asset"):
        root.append(copy.deepcopy(template.find(tag)))
    # Exported defaults can retain an unqualified material name. Resolve it to
    # its already exported asset; this does not change a material or collider.
    materials = {node.get("name") for node in root.find("asset").findall("material")}
    for node in root.find("default").iter():
        value = node.get("material")
        if value and value not in materials and "robot/" + value in materials:
            node.set("material", "robot/" + value)
    asset = root.find("asset")
    ET.SubElement(
        asset,
        "texture",
        name="arena_grid",
        type="2d",
        builtin="checker",
        rgb1="0.20 0.25 0.30",
        rgb2="0.25 0.31 0.36",
        width="256",
        height="256",
    )
    ET.SubElement(
        asset,
        "material",
        name="arena_floor",
        texture="arena_grid",
        texrepeat="40 40",
        texuniform="true",
        reflectance="0.05",
    )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="sun", directional="true", pos="0 0 15", dir="0 0 -1")
    ET.SubElement(
        world,
        "geom",
        name="ground",
        type="plane",
        size="50 50 0.1",
        material="arena_floor",
        friction="0.8 0.005 0.0001",
        contype="1",
        conaffinity="1",
    )
    actuator = ET.SubElement(root, "actuator")
    contact = ET.SubElement(root, "contact")
    body = template.find("worldbody/body[@name='robot/pelvis']")
    if body is None:
        raise ValueError("Expected the exported native G1 pelvis body")
    for index, task in enumerate(tasks):
        prefix = task["id"] + "/"
        origin = np.asarray(task["origin"], float)
        clone = copy.deepcopy(body)
        for node in clone.iter():
            if node.get("name"):
                node.set("name", prefix + node.get("name"))
            for light in list(node.findall("light")):
                node.remove(light)
        clone.set("pos", numbers(np.fromstring(body.get("pos", "0 0 0"), sep=" ") + origin))
        world.append(clone)
        for node in template.find("actuator"):
            drive = copy.deepcopy(node)
            for key in ("name", "joint", "site", "tendon"):
                if drive.get(key):
                    drive.set(key, prefix + drive.get(key))
            actuator.append(drive)
        for node in template.find("contact") or []:
            item = copy.deepcopy(node)
            if any(
                value and not value.startswith("robot/")
                for key, value in node.attrib.items()
                if key in ("body1", "body2", "geom1", "geom2")
            ):
                continue
            for key in ("name", "body1", "body2", "geom1", "geom2"):
                if item.get(key):
                    item.set(key, prefix + item.get(key))
            contact.append(item)
        if task.get("scene"):
            fixtures = json.loads(path(task["scene"]).read_text())
            for box in fixtures["boxes"]:
                yaw = float(box.get("yaw", 0))
                ET.SubElement(
                    world,
                    "geom",
                    name=prefix + box["name"],
                    type="box",
                    pos=numbers(origin + box["center"]),
                    size=numbers(box["half_size"]),
                    quat=numbers([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]),
                    friction=numbers([box.get("friction", 0.8), 0.005, 0.0001]),
                    rgba=numbers(box.get("rgba", [0.5, 0.6, 0.65, 1])),
                    contype="1",
                    conaffinity="1",
                )
        color = [*COLORS[index % len(COLORS)], 1]
        # The zone outlines are explicitly noncolliding decorations.
        for side, (offset, size) in enumerate(
            [
                ([-1.5, 0, 0.003], [0.018, 2, 0.002]),
                ([5, 0, 0.003], [0.018, 2, 0.002]),
                ([1.75, -2, 0.003], [3.25, 0.018, 0.002]),
                ([1.75, 2, 0.003], [3.25, 0.018, 0.002]),
            ]
        ):
            ET.SubElement(
                world,
                "geom",
                name=prefix + f"outline_{side}",
                type="box",
                pos=numbers(origin + offset),
                size=numbers(size),
                rgba=numbers(color),
                contype="0",
                conaffinity="0",
                group="1",
            )
    xml = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml), xml


class SharedWorld:
    def __init__(self, config, tasks, actor, *, repeat=False):
        if any(t["context_dim"] for t in tasks):
            raise ValueError(
                "This first world port supports context-free tasks only; do not synthesize sensor features"
            )
        self.model, self.xml = build_world(config, tasks)
        self.data = mujoco.MjData(self.model)
        self.tasks, self.actor = tasks, actor
        self.device = next(actor.parameters()).device
        self.robot = get_robot("g1")
        self.kin = Kinematics(self.robot, self.device)
        self.origins = torch.tensor([t["origin"] for t in tasks], device=self.device)
        # The saved native entity uses the exact SDK MuJoCo joint tree order.
        native_names = [self.model.joint(i).name.split("/robot/")[-1] for i in range(1, 30)]
        if native_names != self.robot.joint_names:
            raise ValueError("Reference/native joint order differs; provide an explicit mapping")

        def load(filename):
            with np.load(filename, allow_pickle=False) as ref:
                if float(np.asarray(ref["fps"]).reshape(-1)[0]) != 50:
                    raise ValueError("Expected original 50 Hz reference")
                return (
                    ref["body_pos_w"][:, 0].copy(),
                    ref["body_quat_w"][:, 0].copy(),
                    ref["joint_pos"].copy(),
                    50.0,
                )

        self.motion = MotionLib(
            [str(path(t["motion"])) for t in tasks], self.kin, str(self.device), load
        )
        for i, task in enumerate(tasks):
            if not 0 < task["horizon"] <= int(self.motion.clip_end[i] - self.motion.clip_start[i]):
                raise ValueError("Reference horizon must fit the saved motion interval")
        self.mdp = MDP(
            self.kin, self.motion, actor.pi.ref_q0, self.robot.xi_dim, self.robot.sensing
        )
        self.controller = MultiRobotController(actor, [t["expert"] for t in tasks])
        self.xi = torch.zeros(len(tasks), self.robot.xi_dim, device=self.device)
        self.last_action = torch.zeros(len(tasks), 29, device=self.device)
        contract = json.loads((ROOT / "configs/g1_actor_io_contract.json").read_text())
        sdk_from_holo = [contract["joint_names"].index(n) for n in self.robot.joint_names]
        self.initial_previous_action = (
            torch.tensor(contract["action_offset"], device=self.device)[sdk_from_holo]
            - actor.pi.ref_q0
        ) / actor.pi.ref_alpha
        self.q_ids, self.v_ids, self.u_ids, self.root_q, self.root_v = [], [], [], [], []
        for task in tasks:
            prefix = task["id"] + "/robot/"
            joints = [self.model.joint(prefix + n).id for n in self.robot.joint_names]
            self.q_ids.append(self.model.jnt_qposadr[joints])
            self.v_ids.append(self.model.jnt_dofadr[joints])
            self.u_ids.append([self.model.actuator(prefix + n).id for n in self.robot.joint_names])
            free = self.model.joint(prefix + "floating_base_joint").id
            self.root_q.append(int(self.model.jnt_qposadr[free]))
            self.root_v.append(int(self.model.jnt_dofadr[free]))
        self.q_ids, self.v_ids, self.u_ids = map(np.asarray, (self.q_ids, self.v_ids, self.u_ids))
        self.decimation = round(config["control_dt"] / config["physics_dt"])
        if not np.isclose(self.decimation * config["physics_dt"], config["control_dt"]):
            raise ValueError("Control dt must be an integer number of physics steps")
        self.control_step = 0
        self.repeat = repeat
        self.episode_step = torch.zeros(len(tasks), dtype=torch.long, device=self.device)
        self.episode_id = np.ones(len(tasks), dtype=int)
        self.tracking_lost = np.zeros(len(tasks), dtype=bool)
        self.reset_events = []
        self.episodes = []
        self.statistics = [[] for _ in tasks]
        self.reset()

    def reset(self, ids=None):
        """Reset selected episodes only; other robots and the physics clock continue."""
        ids = list(range(len(self.tasks))) if ids is None else list(ids)
        if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= len(self.tasks) for i in ids):
            raise ValueError("Provide unique valid robot indices")
        f = self.motion.clip_start
        p, R, v, w, q, qd = self.motion.ref_state(f)
        from blinkmimic.rotations import mat_to_quat

        quat = mat_to_quat(R)
        for i in ids:
            a, b = self.root_q[i], self.root_v[i]
            if self.control_step:
                self.reset_events.append(
                    dict(
                        task=self.tasks[i]["id"],
                        control_step=self.control_step,
                        finished_episode=int(self.episode_id[i]),
                        reason="reference_boundary",
                        tracking_lost=bool(self.tracking_lost[i]),
                    )
                )
                self.episode_id[i] += 1
            self.data.qpos[a : a + 3] = (p[i] + self.origins[i]).cpu().numpy()
            self.data.qpos[a + 3 : a + 7] = quat[i].cpu().numpy()
            # Match native MotionCommand's reset, which uses the released
            # velocity arrays. Policy reference derivatives remain the original
            # central differences; these are separate contracts.
            with np.load(path(self.tasks[i]["motion"]), allow_pickle=False) as raw:
                self.data.qvel[b : b + 3] = raw["body_lin_vel_w"][0, 0]
                self.data.qvel[b + 3 : b + 6] = R[i].cpu().numpy().T @ raw["body_ang_vel_w"][0, 0]
                self.data.qvel[self.v_ids[i]] = raw["joint_vel"][0]
            self.data.qpos[self.q_ids[i]] = q[i].cpu().numpy()
            self.data.ctrl[self.u_ids[i]] = q[i].cpu().numpy()
            self.data.qacc_warmstart[b : b + 6] = 0
            self.data.qacc_warmstart[self.v_ids[i]] = 0
            self.episode_step[i] = 0
            self.tracking_lost[i] = False
        mujoco.mj_forward(self.model, self.data)
        self.last_action[ids] = self.initial_previous_action
        self.controller.reset(ids)

    def state(self):
        qpos = np.stack([self.data.qpos[a : a + 7] for a in self.root_q])
        qvel = np.stack([self.data.qvel[a : a + 6] for a in self.root_v])
        p = (
            torch.as_tensor(qpos[:, :3].copy(), device=self.device, dtype=torch.float32)
            - self.origins
        )
        R = quat_to_mat(
            torch.as_tensor(qpos[:, 3:].copy(), device=self.device, dtype=torch.float32)
        )
        velocity = torch.as_tensor(qvel.copy(), device=self.device, dtype=torch.float32)
        return (
            p,
            R,
            velocity[:, :3],
            (R @ velocity[:, 3:, None])[..., 0],
            torch.tensor(self.data.qpos[self.q_ids], device=self.device, dtype=torch.float32),
            torch.tensor(self.data.qvel[self.v_ids], device=self.device, dtype=torch.float32),
        )

    @torch.no_grad()
    def step(self):
        if self.repeat:
            ended = [
                i for i, t in enumerate(self.tasks) if int(self.episode_step[i]) >= t["horizon"]
            ]
            if ended:
                self.reset(ended)
        f = torch.minimum(self.motion.clip_start + self.episode_step, self.motion.clip_end)
        state = self.state()
        obs = self.mdp.obs(state, self.last_action, self.xi, f)
        action = self.controller.infer(obs, self.motion.goal(f, 10))
        targets = self.actor.pi.ref_q0 + self.actor.pi.ref_alpha * action
        self.data.ctrl[self.u_ids] = targets.cpu().numpy()
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        self.last_action.copy_(action)
        self.control_step += 1
        self.episode_step += 1
        f = torch.minimum(self.motion.clip_start + self.episode_step, self.motion.clip_end)
        state = self.state()
        root = (state[0] - self.motion.root_p[f]).norm(dim=-1)
        joint = (state[4] - self.motion.q[f]).square().mean(-1).sqrt()
        height = (state[0][:, 2] - self.motion.root_p[f, 2]).abs()
        tilt = (state[1][:, 2, 2] - self.motion.root_R[f, 2, 2]).abs()
        for i, task in enumerate(self.tasks):
            in_reference = int(self.episode_step[i]) <= task["horizon"]
            if in_reference and (float(height[i]) > 0.25 or float(tilt[i]) > 0.8):
                self.tracking_lost[i] = True
            if int(self.episode_step[i]) == task["horizon"]:
                self.episodes.append(
                    dict(
                        task=task["id"],
                        episode=int(self.episode_id[i]),
                        reference_time_completed=True,
                        tracking_lost=bool(self.tracking_lost[i]),
                        endpoint_root_error_m=float(root[i]),
                        task_success_accepted=False,
                    )
                )
            self.statistics[i].append(
                dict(
                    control_step=self.control_step,
                    episode=int(self.episode_id[i]),
                    episode_step=int(self.episode_step[i]),
                    tracking_lost=bool(self.tracking_lost[i]),
                    root_error_m=float(root[i]),
                    joint_rmse_rad=float(joint[i]),
                    reference_height_error_m=float(height[i]),
                    reference_tilt_error=float(tilt[i]),
                    in_reference_interval=in_reference,
                )
            )
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Nonfinite physical state")


def annotate(frame, world):
    from PIL import Image, ImageDraw, ImageFont

    im = Image.fromarray(frame)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    small = ImageFont.truetype("DejaVuSans.ttf", 16)
    draw.rectangle((0, 0, im.width, 64), fill=(15, 23, 32))
    draw.text(
        (20, 10),
        f"PERA  |  ONE ACTOR  /  ONE PHYSICS WORLD  /  {len(world.tasks)} ROBOTS",
        font=font,
        fill="white",
    )
    draw.text(
        (20, 38),
        f"Student actions only  |  Physics t = {world.data.time:.2f} s  |  "
        + (
            f"EPISODIC REPEAT / {len(world.reset_events)} boundary resets"
            if world.repeat
            else "NO RESETS"
        ),
        font=small,
        fill=(180, 197, 211),
    )
    y = im.height - 30 * len(world.tasks) - 10
    draw.rectangle((0, y - 5, im.width, im.height), fill=(15, 23, 32))
    for i, task in enumerate(world.tasks):
        color = tuple(int(v * 255) for v in COLORS[i % len(COLORS)])
        text = task["label"] + "  |  " + task["expert"]
        if world.statistics[i]:
            record = world.statistics[i][-1]
            text += f"  |  episode {record['episode']}  frame {record['episode_step']}/{task['horizon']}"
            text += f"  |  error {record['root_error_m']:.2f} m"
            if record["tracking_lost"]:
                text += "  |  TRACKING LOST"
            if record["episode"] > 1 and record["episode_step"] <= 25:
                text += "  |  RESET TO START"
            if not record["in_reference_interval"]:
                text += "  |  reference ended"
        draw.text((20, y + 30 * i), text, font=small, fill=color)
    return np.asarray(im)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/showcase.json")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Reset each robot at its reference boundary; label and count every reset",
    )
    args = parser.parse_args(argv)
    if not 0 < args.seconds <= 120:
        parser.error("Choose a bounded demonstration duration in (0,120] seconds")
    torch.set_num_threads(4)
    config = json.loads(args.config.read_text())
    from pera.demo_policy import enforce_demo_policy

    try:
        enforce_demo_policy(config)
    except ValueError as exc:
        parser.error(str(exc))
    check_resources(config)
    names = args.tasks or config["default_tasks"]
    if len(set(names)) != len(names):
        parser.error("Task names must be unique")
    table = {t["id"]: t for t in config["tasks"]}
    if not set(names) <= set(table):
        parser.error("Some tasks have not yet been ported to the shared world")
    tasks = [copy.deepcopy(table[name]) for name in names]
    # Compact the chosen regions; this is a fixed XY translation for the whole
    # robot/reference/fixture set, never a root correction during execution.
    columns = math.ceil(math.sqrt(len(tasks)))
    for i, task in enumerate(tasks):
        task["origin"] = [float(i % columns * 8), float(i // columns * 7), 0.0]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    actor = SwitchedActor.load_bundle(path(config["checkpoint"]), device=args.device)
    digest = public_digest(actor.pi)
    world = SharedWorld(config, tasks, actor, repeat=args.repeat)
    (out / "world.xml").write_text(world.xml)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    origins = np.array([t["origin"] for t in tasks])
    camera.lookat[:] = (origins.min(0) + origins.max(0)) / 2 + [1.5, 0, 0.5]
    camera.distance = max(10, float(np.ptp(origins, axis=0).max()) * 0.9 + 10)
    camera.azimuth, camera.elevation = 90, -50
    renderer = writer = viewer = None
    try:
        if args.video:
            import imageio.v2 as imageio

            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError(
                    "Local ffmpeg is required; no binary is downloaded automatically"
                )
            os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg
            renderer = mujoco.Renderer(world.model, height=896, width=1440)
            writer = imageio.get_writer(out / "overview.mp4", fps=25, codec="libx264", quality=8)
        if args.viewer:
            from mujoco import viewer as viewer_module

            viewer = viewer_module.launch_passive(world.model, world.data)
            viewer.cam.lookat[:], viewer.cam.distance = camera.lookat, camera.distance
            viewer.cam.azimuth, viewer.cam.elevation = camera.azimuth, camera.elevation
        total = math.ceil(args.seconds / config["control_dt"])
        for tick in range(total):
            stamp = time.monotonic()
            world.step()
            if writer is not None and tick % 2 == 0:
                renderer.update_scene(world.data, camera=camera)
                image = annotate(renderer.render(), world)
                writer.append_data(image)
                if tick == 0:
                    from PIL import Image

                    Image.fromarray(image).save(out / "preview.png")
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(max(0, config["control_dt"] - (time.monotonic() - stamp)))
        if writer is not None:
            writer.close()
            writer = None
        assert public_digest(actor.pi) == digest == actor.public_sha256
        summaries = []
        for task, records in zip(tasks, world.statistics):
            active = [row for row in records if row["in_reference_interval"]]
            first_episode = [row for row in active if row["episode"] == 1]
            first_loss = next(
                (
                    row["control_step"]
                    for row in first_episode
                    if row["reference_height_error_m"] > 0.25 or row["reference_tilt_error"] > 0.8
                ),
                None,
            )
            summaries.append(
                dict(
                    task=task["id"],
                    reference_frames=task["horizon"],
                    reference_steps_observed=len(active),
                    first_episode_reference_steps_observed=len(first_episode),
                    reference_time_completed=world.control_step >= task["horizon"],
                    completed_episodes=sum(e["task"] == task["id"] for e in world.episodes),
                    completed_episodes_with_tracking_loss=sum(
                        e["task"] == task["id"] and e["tracking_lost"] for e in world.episodes
                    ),
                    first_reference_height_or_tilt_tolerance_exceeded_step=first_loss,
                    mean_root_error_m=float(np.mean([row["root_error_m"] for row in active])),
                    first_episode_last_root_error_m=first_episode[-1]["root_error_m"],
                    last_in_reference_root_error_m=active[-1]["root_error_m"],
                    mean_joint_rmse_rad=float(np.mean([row["joint_rmse_rad"] for row in active])),
                    task_success_accepted=False,
                    acceptance="Not assessed by original native task gate",
                )
            )
        report = dict(
            schema="pera.shared_world_run.v1",
            status="completed",
            actor_instances=1,
            mujoco_model_instances=1,
            mujoco_data_instances=1,
            robot_count=len(tasks),
            installed_experts=len(actor.expert_names),
            policy_tensor_bytes=sum(
                t.numel() * t.element_size() for t in actor.pi.state_dict().values()
            ),
            shared_world_nq=world.model.nq,
            shared_world_nv=world.model.nv,
            shared_world_nu=world.model.nu,
            max_active_experts_per_inference=1,
            checkpoint_sha256=file_digest(path(config["checkpoint"])),
            public_sha256=digest,
            actual_simulated_seconds=float(world.data.time),
            control_steps=world.control_step,
            physics_steps=world.control_step * world.decimation,
            teacher_action_queries=0,
            root_state_overwrites_after_initialization=len(world.reset_events),
            resets_after_initialization=len(world.reset_events),
            reset_events=world.reset_events,
            episodes=world.episodes,
            playback_mode="episodic_reference_repeat" if args.repeat else "finite_no_reset",
            mujoco_version=mujoco.__version__,
            solver_iterations=world.model.opt.iterations,
            solver_ls_iterations=world.model.opt.ls_iterations,
            elapsed_s=time.monotonic() - started,
            device=args.device,
            CUDA_initialized=torch.cuda.is_initialized(),
            torch_cuda_peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20
            if torch.cuda.is_initialized()
            else 0,
            tasks=[
                dict(task=task, steps=records) for task, records in zip(tasks, world.statistics)
            ],
            last_dispatch=world.controller.last_dispatch,
            summary=summaries,
            video="overview.mp4" if args.video else None,
            limitations=[
                "First shared-world port of context-free tasks; other experts not yet ported",
                "Physical task acceptance must be re-established after scene porting",
                "Reference time completion and task success are distinct; repeat mode explicitly resets at reference boundaries",
                "Football/tennis here are motion-tracking tasks, not physical ball interaction",
            ],
        )
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("tasks", "last_dispatch")}, indent=2
            )
        )
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        if viewer is not None:
            viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

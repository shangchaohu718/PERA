#!/usr/bin/env python3
"""One-environment Isaac Lab pilot for the grouped G1 physics Teacher.

The official HoloMotion Actor remains frozen and drives the rollout.  This
wrapper hooks the upstream offline evaluator, reads real PhysX contact patches,
then performs a small bounded Teacher update on detached 64-step windows.
Nothing is downloaded and the upstream checkpoint is never modified.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from types import MethodType

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOLOMOTION_ROOT = Path(os.environ.get(
    "HOLOMOTION_ROOT", PROJECT_ROOT.parent / "HoloMotion")).resolve()
for path in (PROJECT_ROOT, HOLOMOTION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from unified_motion.envs.isaac_g1 import IsaacG1Bridge  # noqa: E402
from unified_motion.holomotion_original_algorithm import _term_widths  # noqa: E402
from unified_motion.privileged_teacher import (  # noqa: E402
    GROUPED_PHYSICS_LATENT,
    PrivilegedTeacherConfig,
    PrivilegedTransformerTeacher,
    teacher_supervision,
)
from unified_motion.robot_contract import get_robot_contract  # noqa: E402
from tools.train_g1_teacher_multimotion import (  # noqa: E402
    contact_sequences, future_contact_supervision, multistep_supervision,
)


SLOTS = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
FRICTION_RANGE = (0.3, 1.2)
TORSO_MASS_SCALE_RANGE = (0.85, 1.15)
OUTPUT_DIR = Path(os.environ.get(
    "G1_TEACHER_OUTPUT_DIR",
    PROJECT_ROOT / "reports" / "g1_teacher_isaac_pilot",
)).expanduser().resolve()


class StopPilotRollout(RuntimeError):
    """Internal bounded-rollout signal used by prediction evaluation."""


def _leaf(obs, name: str) -> torch.Tensor:
    value = obs.get(("unified", name))
    if value is None:
        value = obs.get(f"unified/{name}")
    if value is None:
        raise KeyError(f"missing official observation term {name}")
    return value


def _flat_critic(obs) -> torch.Tensor:
    _, _, current, future = _term_widths(29, 30)
    pieces = []
    for name, _ in current:
        pieces.append(_leaf(obs, name).reshape(obs.batch_size[0], -1))
    for name, _ in future:
        pieces.append(_leaf(obs, name).reshape(obs.batch_size[0], -1))
    result = torch.cat(pieces, dim=-1)
    if result.shape != (obs.batch_size[0], 1046):
        raise RuntimeError(f"official Critic tensor is {tuple(result.shape)}, not (*,1046)")
    return result


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors into a wxyz quaternion's local frame."""
    qw, qv = q[..., :1], q[..., 1:]
    return ((2.0 * qw.square() - 1.0) * v
            - 2.0 * qw * torch.linalg.cross(qv, v)
            + 2.0 * qv * (qv * v).sum(-1, keepdim=True))


class IsaacTeacherCollector:
    def __init__(self, algo) -> None:
        if algo.env.num_envs != 1:
            raise RuntimeError("the VRAM-bounded pilot requires exactly one environment")
        self.algo = algo
        manager_env = algo.env._env
        self.bridge = IsaacG1Bridge(
            manager_env,
            action_term_name="dof_pos",
            contact_body_names=dict(zip(
                ("left_foot", "right_foot", "left_wrist", "right_wrist"),
                SLOTS,
                strict=True,
            )),
        )
        self.robot = self.bridge.robot
        self.physics_parameters, self.physics_readback = self._randomize_physics()
        self.sensor = manager_env.scene.sensors["contact_forces"]
        self.physics_dt = float(manager_env.cfg.sim.dt)
        self.raw_view, self.ground_paths = self._make_raw_view()
        self.frames: list[dict[str, torch.Tensor]] = []
        self._pending = None
        self._next_is_start = True
        self.max_frames = int(os.environ.get("G1_TEACHER_MAX_FRAMES", "0"))

    @torch.no_grad()
    def _randomize_physics(self) -> tuple[torch.Tensor, dict]:
        """Apply and verify one exactly representable friction/mass pair."""
        enabled = os.environ.get("G1_TEACHER_RANDOMIZE_PHYSICS", "0") == "1"
        if not enabled:
            return torch.zeros(2), {
                "enabled": False,
                "friction": None,
                "torso_mass_scale": None,
            }
        seed = int(os.environ.get("G1_TEACHER_PHYSICS_SEED", "20261016"))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        sample = torch.rand(2, generator=generator)
        friction = FRICTION_RANGE[0] + float(sample[0]) * (
            FRICTION_RANGE[1] - FRICTION_RANGE[0])
        mass_scale = TORSO_MASS_SCALE_RANGE[0] + float(sample[1]) * (
            TORSO_MASS_SCALE_RANGE[1] - TORSO_MASS_SCALE_RANGE[0])

        view = self.robot.root_physx_view
        env_ids = torch.tensor([0], dtype=torch.long)
        materials = view.get_material_properties()
        materials[0, :, 0] = friction
        materials[0, :, 1] = friction
        view.set_material_properties(materials, env_ids)

        try:
            torso_id = tuple(self.robot.body_names).index("torso_link")
        except ValueError as error:
            raise RuntimeError("G1 torso_link is required for mass randomization") from error
        nominal_mass = float(self.robot.data.default_mass[0, torso_id].detach().cpu())
        masses = view.get_masses()
        masses[0, torso_id] = nominal_mass * mass_scale
        view.set_masses(masses, env_ids)
        inertias = view.get_inertias()
        nominal_inertia = self.robot.data.default_inertia[0, torso_id].detach().cpu()
        inertias[0, torso_id] = nominal_inertia * mass_scale
        view.set_inertias(inertias, env_ids)

        material_check = view.get_material_properties()
        mass_check = view.get_masses()
        actual_friction = float(material_check[0, :, 1].mean())
        actual_mass_scale = float(mass_check[0, torso_id]) / nominal_mass
        if abs(actual_friction - friction) > 1.0e-5 or abs(actual_mass_scale - mass_scale) > 1.0e-5:
            raise RuntimeError("PhysX friction/mass readback disagrees with requested randomization")
        normalized = torch.tensor((
            2.0 * (actual_friction - FRICTION_RANGE[0]) /
            (FRICTION_RANGE[1] - FRICTION_RANGE[0]) - 1.0,
            2.0 * (actual_mass_scale - TORSO_MASS_SCALE_RANGE[0]) /
            (TORSO_MASS_SCALE_RANGE[1] - TORSO_MASS_SCALE_RANGE[0]) - 1.0,
        ))
        return normalized, {
            "enabled": True,
            "seed": seed,
            "friction": actual_friction,
            "friction_range": FRICTION_RANGE,
            "torso_mass_scale": actual_mass_scale,
            "torso_mass_scale_range": TORSO_MASS_SCALE_RANGE,
            "normalized_parameters": normalized.tolist(),
        }

    def _make_raw_view(self):
        from pxr import UsdPhysics
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        ground_paths = [
            str(prim.GetPath()) for prim in stage.Traverse()
            if str(prim.GetPath()).startswith("/World/ground")
            and prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if not ground_paths:
            raise RuntimeError("no collision prim was found below /World/ground")
        prim_paths = list(self.sensor.body_physx_view.prim_paths)
        body_paths = []
        for name in SLOTS:
            matches = [path for path in prim_paths if path.endswith("/" + name)]
            if len(matches) != 1:
                raise RuntimeError(f"expected one PhysX body for {name}, got {matches}")
            body_paths.append(matches[0])
        view = self.sensor._physics_sim_view.create_rigid_contact_view(
            body_paths,
            filter_patterns=[ground_paths for _ in body_paths],
            max_contact_data_count=512,
        )
        if view.sensor_count != 4 or view.filter_count < 1:
            raise RuntimeError(
                f"invalid raw contact view: sensors={view.sensor_count}, "
                f"filters={view.filter_count}")
        return view, ground_paths

    def _core(self) -> torch.Tensor:
        state = self.bridge.actor_state()
        raw = torch.cat((self.bridge.base_linear_velocity_body(), state[..., :64]), -1)
        scale = raw.new_tensor(
            [5.0] * 3 + [10.0] * 3 + [1.0] * 3
            + [torch.pi] * 29 + [10.0] * 29)
        return raw / scale

    @torch.no_grad()
    def _contacts(self) -> tuple[torch.Tensor, torch.Tensor]:
        forces, points, normals, _, counts, starts = self.raw_view.get_contact_data(
            self.physics_dt)
        friction, _, friction_counts, friction_starts = self.raw_view.get_friction_data(
            self.physics_dt)
        forces = forces.reshape(-1)
        counts = counts.reshape(4, -1)
        starts = starts.reshape(4, -1)
        friction_counts = friction_counts.reshape(4, -1)
        friction_starts = friction_starts.reshape(4, -1)

        body_ids = self.bridge.indices.contact_body_ids
        root_pos = self.robot.data.root_pos_w[0]
        root_quat = self.robot.data.root_quat_w[0]
        body_pos = self.robot.data.body_pos_w[0].index_select(0, body_ids)
        body_lin = self.robot.data.body_link_lin_vel_w[0].index_select(0, body_ids)
        body_ang = self.robot.data.body_ang_vel_w[0].index_select(0, body_ids)
        slots = points.new_zeros(4, 14)
        found = torch.zeros(4, dtype=torch.bool, device=points.device)

        for slot in range(4):
            patch_indices = []
            for filter_id in range(counts.shape[1]):
                start = int(starts[slot, filter_id].item())
                count = int(counts[slot, filter_id].item())
                patch_indices.extend(range(start, start + count))
            if not patch_indices:
                continue
            ids = torch.tensor(patch_indices, device=points.device, dtype=torch.long)
            loads = forces.index_select(0, ids).clamp_min(0.0)
            total_load = loads.sum()
            if not bool(total_load > 1.0e-6):
                continue
            weight = loads / total_load
            point_w = (points.index_select(0, ids) * weight[:, None]).sum(0)
            normal_w = (normals.index_select(0, ids) * weight[:, None]).sum(0)
            normal_w = torch.nn.functional.normalize(normal_w, dim=0)
            if bool(normal_w[2] < 0):
                normal_w = -normal_w
            point_vel_w = body_lin[slot] + torch.linalg.cross(
                body_ang[slot], point_w - body_pos[slot])
            tangential_vel_w = point_vel_w - normal_w * (point_vel_w * normal_w).sum()

            friction_ids = []
            for filter_id in range(friction_counts.shape[1]):
                start = int(friction_starts[slot, filter_id].item())
                count = int(friction_counts[slot, filter_id].item())
                friction_ids.extend(range(start, start + count))
            tangent_force = (friction.index_select(
                0, torch.tensor(friction_ids, device=friction.device, dtype=torch.long)
            ).sum(0).norm() if friction_ids else total_load.new_zeros(()))

            slots[slot, :3] = _quat_rotate_inverse(root_quat, point_w - root_pos)
            slots[slot, 3:6] = _quat_rotate_inverse(root_quat, point_vel_w)
            slots[slot, 6:9] = _quat_rotate_inverse(root_quat, normal_w)
            slots[slot, 9] = total_load / 100.0
            slots[slot, 10] = 1.0
            slots[slot, 11] = tangent_force / 100.0
            slots[slot, 12] = tangential_vel_w.norm()
            slots[slot, 13] = (tangent_force / total_load.clamp_min(1.0e-6)).clamp_max(5.0)
            found[slot] = True
        return slots, found

    @torch.no_grad()
    def before_step(self, obs) -> None:
        contacts, found = self._contacts()
        self._pending = dict(
            # Long rollouts must not retain a linearly growing CUDA replay
            # buffer.  Only the sampled 64-step training window returns to GPU.
            critic=_flat_critic(obs)[0].detach().cpu().clone(),
            core=self._core()[0].detach().cpu().clone(),
            contacts=contacts.detach().cpu().clone(),
            found=found.detach().cpu().clone(),
            params=self.physics_parameters.clone(),
            start=torch.tensor(self._next_is_start),
        )

    @torch.no_grad()
    def after_step(self, next_obs, action: torch.Tensor, dones: torch.Tensor) -> None:
        if self._pending is None:
            raise RuntimeError("missing pre-step Teacher sample")
        self._pending["action"] = action[0].detach().cpu().clone()
        self._pending["next_core"] = self._core()[0].detach().cpu().clone()
        next_contacts, next_found = self._contacts()
        self._pending["next_contacts"] = next_contacts.detach().cpu().clone()
        self._pending["next_found"] = next_found.detach().cpu().clone()
        self._pending["next_contact_valid"] = (~dones.reshape(-1)[0]).detach().cpu().clone()
        self.frames.append(self._pending)
        self._pending = None
        self._next_is_start = bool(dones.reshape(-1)[0])
        if self.max_frames > 0 and len(self.frames) >= self.max_frames:
            raise StopPilotRollout

    def _stacked(self) -> dict[str, torch.Tensor]:
        return {key: torch.stack([frame[key] for frame in self.frames])
                for key in self.frames[0]}

    def save_replay(self) -> dict:
        """Persist a bounded CPU-only rollout for multi-motion training."""
        if not self.frames:
            raise RuntimeError("cannot save an empty G1 Teacher replay")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        replay_name = os.environ.get("G1_TEACHER_REPLAY_NAME", "rollout")
        source_motion_name = os.environ.get("G1_TEACHER_SOURCE_MOTION", replay_name)
        replay_path = OUTPUT_DIR / f"{replay_name}.pt"
        stacked = self._stacked()
        torch.save({
            "schema": 2,
            "robot": "g1_29dof",
            "motion_name": replay_name,
            "source_motion_name": source_motion_name,
            "frames": stacked,
            "contact_slots": SLOTS,
            "physics_dt": self.physics_dt,
            "physics_randomization": self.physics_readback,
            "start_frame": int(os.environ.get("G1_TEACHER_START_FRAME", "0")),
        }, replay_path)
        report = {
            "status": "passed",
            "mode": "collect",
            "motion_name": replay_name,
            "source_motion_name": source_motion_name,
            "frames": len(self.frames),
            "replay": str(replay_path),
            "physics_randomization": self.physics_readback,
            "start_frame": int(os.environ.get("G1_TEACHER_START_FRAME", "0")),
            "contact_rate_by_slot": dict(zip(
                SLOTS, stacked["found"].float().mean(0).tolist(), strict=True)),
        }
        (OUTPUT_DIR / f"{replay_name}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("G1_TEACHER_COLLECT=" + json.dumps(report, ensure_ascii=False))
        return report

    @staticmethod
    def _window(stacked, start, end, device):
        critic = stacked["critic"][start:end].unsqueeze(0).to(device)
        contact_history, history_found, contact, found, contact_valid = contact_sequences(
            stacked, start, end, device)
        length = end - start
        params = stacked["params"][start:end].unsqueeze(0).to(device)
        privilege = torch.cat((
            contact_history.flatten(-2), torch.zeros(1, length, 68, device=device), params,
        ), -1)
        starts = stacked["start"][start:end].unsqueeze(0).to(device).bool()
        starts[:, 0] = True
        return dict(
            inputs=torch.cat((critic, privilege), -1),
            contact_history=contact_history,
            history_found=history_found,
            contact=contact,
            found=found,
            contact_valid=contact_valid,
            params=params,
            starts=starts,
            actions=stacked["action"][start:end].unsqueeze(0).to(device),
            core=stacked["core"][start:end].unsqueeze(0).to(device),
            next_core=stacked["next_core"][start:end].unsqueeze(0).to(device),
        )

    @torch.no_grad()
    def evaluate(self, checkpoint_path: Path, *, sequence: int = 64) -> dict:
        from unified_motion.privileged_teacher import core_groups

        device = torch.device(self.algo.device)
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = PrivilegedTeacherConfig(**saved["teacher_config"])
        teacher = PrivilegedTransformerTeacher(
            get_robot_contract("g1_29dof"), 64, 2, config,
            latent_groups=GROUPED_PHYSICS_LATENT,
        ).to(device).eval()
        teacher.load_state_dict(saved["model"], strict=True)
        stacked = self._stacked()
        group_names = ("base_lin_vel", "base_ang_vel", "gravity", "joint_pos", "joint_vel")
        pred_sse = torch.zeros(5, device=device)
        persist_sse = torch.zeros(5, device=device)
        counts = torch.zeros(5, device=device)
        physics_sse = torch.zeros(4, device=device)
        physics_count = torch.zeros(4, device=device)
        tp = fp = fn = correct = total = 0.0
        mode_correct = mode_total = 0.0
        for start in range(0, len(self.frames), sequence):
            batch = self._window(
                stacked, start, min(start + sequence, len(self.frames)), device)
            output = teacher(
                batch["inputs"], batch["starts"], batch["actions"], batch["core"])
            target_delta = batch["next_core"] - batch["core"]
            for index, section in enumerate(core_groups(67)):
                predicted_error = output["delta_core"][..., section] - target_delta[..., section]
                persistence_error = -target_delta[..., section]
                pred_sse[index] += predicted_error.square().sum()
                persist_sse[index] += persistence_error.square().sum()
                counts[index] += predicted_error.numel()
            predicted_contact = output["contact_logits"].sigmoid() >= 0.5
            truth_contact = batch["found"]
            contact_mask = batch["contact_valid"].unsqueeze(-1).expand_as(truth_contact)
            tp += float((predicted_contact & truth_contact & contact_mask).sum())
            fp += float((predicted_contact & ~truth_contact & contact_mask).sum())
            fn += float((~predicted_contact & truth_contact & contact_mask).sum())
            correct += float(((predicted_contact == truth_contact) & contact_mask).sum())
            total += float(contact_mask.sum())
            physics_target = batch["contact"][..., (9, 11, 12, 13)]
            active = (truth_contact & contact_mask).unsqueeze(-1).expand_as(physics_target)
            physics_sse += torch.where(
                active, (output["contact_physics"] - physics_target).square(), 0.0
            ).sum((0, 1, 2))
            physics_count += active.sum((0, 1, 2))
            modes = torch.where(truth_contact, torch.where(
                batch["contact"][..., 12] > config.contact_slip_threshold_m_s,
                2, 1), 0)
            predicted_modes = output["contact_mode_logits"].argmax(-1)
            mode_correct += float(((predicted_modes == modes) & contact_mask).sum())
            mode_total += float(contact_mask.sum())
        pred_mse = pred_sse / counts.clamp_min(1)
        persist_mse = persist_sse / counts.clamp_min(1)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        report = {
            "status": "passed",
            "checkpoint": str(checkpoint_path),
            "frames": len(self.frames),
            "transition": {
                name: {
                    "teacher_rmse_normalized": float(pred_mse[i].sqrt()),
                    "persistence_rmse_normalized": float(persist_mse[i].sqrt()),
                    "mse_improvement_vs_persistence": float(
                        1.0 - pred_mse[i] / persist_mse[i].clamp_min(1.0e-12)),
                } for i, name in enumerate(group_names)
            },
            "next_contact_prediction": {
                "accuracy": correct / max(total, 1.0),
                "precision": precision,
                "recall": recall,
                "f1": 2.0 * precision * recall / max(precision + recall, 1.0e-12),
                "mode_accuracy": mode_correct / max(mode_total, 1.0),
                "note": "targets are t+1; contact inputs are strictly t-1 or older",
            },
            "active_contact_physics_rmse": dict(zip(
                ("normal_force_div100", "tangential_force_div100", "slip_speed_m_s", "friction_utilization"),
                (physics_sse / physics_count.clamp_min(1)).sqrt().tolist(),
                strict=True,
            )),
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "prediction_evaluation.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("G1_TEACHER_EVAL=" + json.dumps(report, ensure_ascii=False))
        return report

    def train(self, *, updates: int = 20, sequence: int = 64) -> dict:
        updates = int(os.environ.get("G1_TEACHER_UPDATES", updates))
        sequence = int(os.environ.get("G1_TEACHER_SEQUENCE", sequence))
        if updates < 1 or sequence < 2:
            raise ValueError("G1 Teacher updates>=1 and sequence>=2 are required")
        device = torch.device(self.algo.device)
        stacked = self._stacked()
        config = PrivilegedTeacherConfig(
            objective_version=5, critic_dim=1046, contact_slots=4,
            object_slots=4, width=128, heads=8, layers=4,
            context_steps=64, train_envs=1, learning_rate=1.0e-4,
            parameter_reconstruction_weight=0.1,
            future_contact_weight=0.25, contact_input_lag=1,
            latent_groups=GROUPED_PHYSICS_LATENT,
        )
        teacher = PrivilegedTransformerTeacher(
            get_robot_contract("g1_29dof"), 64, 2, config,
            latent_groups=GROUPED_PHYSICS_LATENT,
        ).to(device).train()
        optimizer = torch.optim.Adam(teacher.parameters(), lr=config.learning_rate)
        maximum_start = max(0, len(self.frames) - sequence)
        losses = []
        grad_norms = {}
        generator = torch.Generator(device="cpu").manual_seed(20261016)
        train_started = time.perf_counter()
        for update in range(updates):
            start = (0 if maximum_start == 0 else int(torch.randint(
                maximum_start + 1, (), generator=generator)))
            end = min(start + sequence, len(self.frames))
            batch = self._window(stacked, start, end, device)
            contact, found, params = batch["contact"], batch["found"], batch["params"]
            inputs, starts = batch["inputs"], batch["starts"]
            actions, core, next_core = batch["actions"], batch["core"], batch["next_core"]
            contact_physics = contact[..., (9, 11, 12, 13)]
            modes = torch.where(found, torch.where(
                contact[..., 12] > config.contact_slip_threshold_m_s, 2, 1), 0).long()
            output = teacher(inputs, starts, actions, core)
            loss = teacher_supervision(
                output, next_core, torch.ones_like(next_core, dtype=torch.bool),
                found.float(), batch["contact_valid"].unsqueeze(-1).expand_as(found), params,
                torch.ones(1, end - start, dtype=torch.bool, device=device),
                config, current_core=core, contact_physics=contact_physics,
                contact_modes=modes,
            )
            loss = loss + config.multistep_weight * multistep_supervision(
                teacher, output, {
                    "core": core, "next_core": next_core, "starts": starts,
                    "actions": actions,
                }, config)
            loss = loss + config.future_contact_weight * future_contact_supervision(
                teacher, output, batch, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if update == updates - 1:
                grad_norms = {
                    name: float(head[1].weight.grad.norm().detach())
                    for name, head in teacher.latent_head.items()
                }
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        train_seconds = time.perf_counter() - train_started

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = OUTPUT_DIR / "teacher_grouped_z64_real_isaac_pilot.pt"
        torch.save({
            "schema": 1,
            "real_isaac_pilot": True,
            "teacher_config": config.__dict__,
            "latent_groups": GROUPED_PHYSICS_LATENT,
            "model": teacher.state_dict(),
        }, checkpoint)
        contact_rate = stacked["found"].float().mean(0)
        report = {
            "status": "passed",
            "real_isaac_physics": True,
            "official_actor_checkpoint": str(
                HOLOMOTION_ROOT / "checkpoints/holomotion_v1.4/model_14000.pt"),
            "rollout_frames": len(self.frames),
            "updates": updates,
            "sequence": sequence,
            "train_seconds": train_seconds,
            "updates_per_second": updates / train_seconds,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "contact_rate_by_slot": dict(zip(SLOTS, contact_rate.tolist(), strict=True)),
            "group_grad_norms": grad_norms,
            "ground_collision_prims": self.ground_paths,
            "checkpoint": str(checkpoint),
            "vram_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            "scope": "one rollout pilot; no domain randomization and no object contacts",
        }
        (OUTPUT_DIR / "short_run.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("G1_TEACHER_PILOT=" + json.dumps(report, ensure_ascii=False))
        return report


def main() -> None:
    import hydra
    import holomotion.src.evaluation.eval_motion_tracking_single as evaluation

    original_get_class = evaluation.get_class

    def get_pilot_class(path):
        base = original_get_class(path)

        class PilotAlgorithm(base):
            def offline_evaluate_policy(self, dump_npzs=False):
                collector = IsaacTeacherCollector(self)
                original_rollout = self._rollout_forward
                motion_cmd = self.env._env.command_manager.get_term("ref_motion")
                original_setup = motion_cmd.setup_offline_eval_deterministic
                start_frame = int(os.environ.get("G1_TEACHER_START_FRAME", "0"))

                def setup_with_offset(this, apply_pending_swap=True):
                    result = original_setup(apply_pending_swap=apply_pending_swap)
                    if start_frame:
                        current = this._motion_cache.current_batch
                        minimum_length = int(current.lengths.min().item())
                        required = start_frame + max(1, collector.max_frames) + int(this.cfg.n_fut_frames)
                        if required >= minimum_length:
                            raise ValueError(
                                f"start frame {start_frame} with {collector.max_frames} samples "
                                f"exceeds clip length {minimum_length}")
                        active = min(int(this.num_envs), int(this._motion_cache.clip_count))
                        this._frame_indices[:active] = start_frame
                        this._start_frame_indices[:active] = start_frame
                        this._update_ref_motion_state_from_cache()
                    return result

                motion_cmd.setup_offline_eval_deterministic = MethodType(
                    setup_with_offset, motion_cmd)

                def wrapped(this, obs, **kwargs):
                    collector.before_step(obs)
                    next_obs = original_rollout(obs, **kwargs)
                    collector.after_step(
                        next_obs, this._last_rollout_actions, this._last_rollout_dones)
                    return next_obs

                self._rollout_forward = MethodType(wrapped, self)
                try:
                    result = super().offline_evaluate_policy(False)
                except StopPilotRollout:
                    result = {}
                finally:
                    self._rollout_forward = original_rollout
                    motion_cmd.setup_offline_eval_deterministic = original_setup
                mode = os.environ.get("G1_TEACHER_MODE", "train")
                eval_checkpoint = os.environ.get("G1_TEACHER_EVAL_CHECKPOINT")
                if mode == "collect":
                    collector.save_replay()
                elif eval_checkpoint:
                    collector.evaluate(Path(eval_checkpoint).expanduser().resolve())
                else:
                    collector.train()
                return result

        return PilotAlgorithm

    evaluation.get_class = get_pilot_class
    overrides = [value for value in sys.argv[1:]
                 if not value.startswith("--config-name=")]
    with hydra.initialize_config_dir(
        config_dir=str(HOLOMOTION_ROOT / "holomotion/config"),
        version_base=None,
    ):
        config = hydra.compose(
            config_name="evaluation/eval_isaaclab", overrides=overrides)
    evaluation.main.__wrapped__(config)


if __name__ == "__main__":
    main()

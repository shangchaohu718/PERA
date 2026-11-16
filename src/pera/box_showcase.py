"""Native ULTRA observation and torque adapter; no teacher network is used."""

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from pera.switching import file_digest


class BoxRuntime:
    def __init__(self, slot, folder):
        from runtime import World
        from utils.obs import MujocoObs

        self.slot = slot
        self.cfg = json.loads((folder / "contract.json").read_text())
        expected = slot.actor.policy["experts"][slot.row["expert"]]["context"]["encoder_sha256"]
        if file_digest(folder / "frozen_context_normalizer.pt") != expected:
            raise ValueError("Box normalizer does not match the expert")
        self.norm = torch.load(
            folder / "frozen_context_normalizer.pt", map_location=slot.device, weights_only=True
        )
        raw = torch.load(folder / "observation_data.pt", map_location="cpu", weights_only=True)
        self.raw = raw
        self.builder = MujocoObs(
            slot.model,
            "largebox",
            len(raw["hoi_data"]),
            raw["hoi_data"],
            raw["object_points"],
            history_step=10,
        )
        self.joint_names = self.cfg["joint_names"]
        self.holo_gather = np.array(
            [self.joint_names.index(n) for n in slot.adapter.pi_contract_names]
        )
        source = raw["hoi_data"][:, :630]
        self.reference = torch.cat(
            (source[:, :3], source[:, [6, 3, 4, 5]], source[:, 13:42][:, self.holo_gather]), -1
        ).to(slot.device)
        self.world = World.__new__(World)
        w = self.world
        w.model, w.data, w.builder = slot.model, slot.data, self.builder
        w.env = SimpleNamespace(control_dt=self.cfg["control_dt"])
        w.pelvis_bid = slot.model.body("pelvis").id
        w.object_bid = slot.model.body("largebox").id
        w.hand_bids = {slot.model.body(n).id for n in ["left_rubber_hand", "right_rubber_hand"]}
        w.original_frames = self.cfg["reference_frames"]
        self.reset()

    def reset(self):
        w = self.world
        self.builder.last_dof_pos = self.builder.last_dof_vel = None
        w.frame = 0
        w.last_action = torch.zeros(1, 29)
        w.last_torque = torch.zeros(1, 29)
        w.start_position = w.data.xpos[w.object_bid].copy()
        w.minimum_pelvis = float("inf")
        w.max_lift = w.max_displacement = w.max_lifted_displacement = 0.0
        w.hand_contact_frames = w.lifted_contact_frames = 0
        w.fallen = False
        w.object_state_writes_after_reset = 0
        self.target = np.zeros(29)
        self.last_holo = torch.zeros(1, 29, device=self.slot.device)

    def inputs(self, old):
        from modumotion_g1.perceptive_observations import (
            PerceptiveActorObservationInput,
            build_perceptive_actor_terms,
            sample_holomotion_reference,
        )

        s = self.slot
        old = self.last_holo
        w = self.world
        d = s.data
        device = s.device
        mujoco.mj_comVel(s.model, d)
        raw = self.builder._compute_observations(
            d, w.frame, w.last_action, w.last_torque, student_obs=False, episode_length=w.frame
        ).to(device)
        norm = self.norm
        context = ((raw - norm["mean"]) / (norm["variance"] + norm["epsilon"]).sqrt()).clamp(
            -norm["clip"], norm["clip"]
        )
        frame = torch.tensor([w.frame], device=device)
        ref, times = sample_holomotion_reference(
            self.reference,
            frame,
            torch.zeros(1, device=device),
            torch.zeros(1, device=device, dtype=torch.long),
            torch.tensor([len(self.reference) - 1], device=device),
            reference_fps=60.0,
            actor_dt=0.02,
        )

        def t(x):
            return torch.tensor(np.array(x), device=device, dtype=torch.float32)[None]

        terms = build_perceptive_actor_terms(
            PerceptiveActorObservationInput(
                reference_qpos=ref,
                robot_root_quat_wxyz=t(d.qpos[3:7]),
                robot_root_angvel_local=t(d.qvel[3:6]),
                robot_dof_pos=t(d.qpos[self.builder.dof_qpos_ids[self.holo_gather]]),
                robot_dof_vel=t(d.qvel[self.builder.dof_qvel_ids[self.holo_gather]]),
                last_action=old,
                default_dof_pos=s.adapter.default,
                reference_sample_time=times,
            )
        )
        return TensorDict({"unified/" + k: v for k, v in terms.items()}, batch_size=[1]), context

    def execute(self, action, raw_action=None):
        s = self.slot
        targets = (s.actor.pi.ref_q0 + s.actor.pi.ref_alpha * action).detach().cpu().numpy()
        if raw_action is None:
            native_order = np.array([s.sdk_names.index(n) for n in self.joint_names])
            native = (targets[native_order] / np.float32(3.0)).clip(-1, 1).astype(np.float32)
        else:
            physical = s.adapter.old_offset + s.adapter.old_scale * raw_action
            inverse = np.argsort(self.holo_gather)
            native = (
                (physical[inverse] / 3.0).clamp(-1, 1).detach().cpu().numpy().astype(np.float32)
            )
        self.last_holo = (
            torch.tensor(native, device=s.device)[self.holo_gather][None] * 3.0
            - s.adapter.old_offset
        ) / s.adapter.old_scale
        self.world.last_action = torch.from_numpy(native)[None]
        self.target = (native * np.float32(3.0)).astype(np.float64)
        physical_sdk = self.target[np.array([self.joint_names.index(n) for n in s.sdk_names])]
        return (
            torch.tensor(physical_sdk, device=s.device, dtype=torch.float32) - s.actor.pi.ref_q0
        ) / s.actor.pi.ref_alpha

    def torque(self, arena):
        s = self.slot
        q = arena.data.qpos[s.map.qmap[self.builder.dof_qpos_ids]].astype(np.float32)
        dq = arena.data.qvel[s.map.vmap[self.builder.dof_qvel_ids]].astype(np.float32)
        torque = np.clip(
            (self.target - q) * np.array(self.cfg["stiffness"])
            - dq * np.array(self.cfg["damping"]),
            -np.array(self.cfg["torque_limits"]),
            np.array(self.cfg["torque_limits"]),
        )
        arena.data.ctrl[s.map.umap] = torque
        self.world.last_torque = torch.tensor(torque.astype(np.float32))[None]

    def audit(self, arena=None):
        w = self.world
        previous = (w.hand_contact_frames, w.lifted_contact_frames, w.max_lifted_displacement)
        w.frame += 1
        w.audit()
        if arena is not None:
            body = self.slot.map.bmap
            obj = int(body[w.object_bid])
            hands = {int(body[b]) for b in w.hand_bids}
            contact = False
            for c in range(arena.data.ncon):
                con = arena.data.contact[c]
                a, b = (int(arena.model.geom_bodyid[g]) for g in (con.geom1, con.geom2))
                if (a == obj and b in hands) or (b == obj and a in hands):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(arena.model, arena.data, c, force)
                    contact |= np.linalg.norm(force[:3]) > 0.1
            pos = arena.data.xpos[obj] - self.slot.origin
            lifted = contact and pos[2] - w.start_position[2] > 0.1
            w.hand_contact_frames = previous[0] + int(contact)
            w.lifted_contact_frames = previous[1] + int(lifted)
            w.max_lifted_displacement = max(
                previous[2],
                float(np.linalg.norm(pos[:2] - w.start_position[:2])) if lifted else 0.0,
            )
        return self.world.summary()

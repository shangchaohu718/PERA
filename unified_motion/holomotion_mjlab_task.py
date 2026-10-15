"""Reference-aware mjlab task boundary for the original HoloMotion PPO."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
import math
from typing import Any

import torch

from .data.motion_sampler import MotionSampler
from .envs.mjlab_physics import (
    K1PhysicsBridge,
    PrivilegedFootContactTruth,
)
from .holomotion_mjlab import HoloMotionK1MJLabAdapter
from .holomotion_mjlab_rewards import (
    HOLOMOTION_AUX_CONTACT_SENSOR,
    K1_AUX_CONTACT_BODY_NAMES,
    K1_AUX_POSITION_BODY_NAMES,
    HoloMotionK1CriticObservationBuilder,
    HoloMotionMJLabState,
    _quat_apply_inverse,
    gravity_projection_distance,
    holomotion_k1_reward_terms,
)


@dataclass(frozen=True)
class HoloMotionAuxiliaryTargets:
    base_lin_vel_b: torch.Tensor
    keybody_contacts: torch.Tensor
    ref_keybody_rel_pos: torch.Tensor
    robot_keybody_rel_pos: torch.Tensor


@dataclass(frozen=True)
class HoloMotionMJLabObservation:
    actor: torch.Tensor
    critic: torch.Tensor
    auxiliary: HoloMotionAuxiliaryTargets


@dataclass(frozen=True)
class HoloMotionMJLabTransition:
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    terms: dict[str, torch.Tensor]
    gravity_projection_distance: torch.Tensor
    motion_finished: torch.Tensor
    physical_failure: torch.Tensor | None = None


@dataclass(frozen=True)
class HoloMotionFootRewardConfig:
    """Independent left/right physical penalties; no symmetry is imposed."""

    left_slip_weight: float = 0.0
    right_slip_weight: float = 0.0
    left_impact_weight: float = 0.0
    right_impact_weight: float = 0.0
    impact_threshold_n: float = 100.0

    def __post_init__(self) -> None:
        weights = (
            self.left_slip_weight, self.right_slip_weight,
            self.left_impact_weight, self.right_impact_weight,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or value < 0
            for value in weights
        ):
            raise ValueError("foot reward weights must be finite and non-negative")
        if (
            isinstance(self.impact_threshold_n, bool)
            or not isinstance(self.impact_threshold_n, (float, int))
            or not math.isfinite(self.impact_threshold_n)
            or self.impact_threshold_n <= 0
        ):
            raise ValueError("impact_threshold_n must be positive and finite")


@dataclass(frozen=True)
class HoloMotionPhysicsRandomizationConfig:
    """Slow physical variables that the privileged Teacher is allowed to see."""

    friction_range: tuple[float, float] = (0.2, 1.2)
    trunk_mass_scale_range: tuple[float, float] = (0.8, 1.2)

    def __post_init__(self) -> None:
        for name, bounds in (
            ("friction_range", self.friction_range),
            ("trunk_mass_scale_range", self.trunk_mass_scale_range),
        ):
            if (
                not isinstance(bounds, tuple) or len(bounds) != 2
                or any(isinstance(v, bool) or not isinstance(v, (float, int))
                       or not math.isfinite(v) or v <= 0 for v in bounds)
                or bounds[0] >= bounds[1]
            ):
                raise ValueError(f"{name} must be an increasing positive range")


class HoloMotionK1MJLabTask:
    """Own reference timing, reset state, rewards, and official termination."""

    auto_reset = False

    def __init__(
        self,
        adapter: HoloMotionK1MJLabAdapter,
        sampler: MotionSampler,
        foot_reward_config: HoloMotionFootRewardConfig | None = None,
        physics_randomization: HoloMotionPhysicsRandomizationConfig | None = None,
    ) -> None:
        if not isinstance(adapter, HoloMotionK1MJLabAdapter):
            raise TypeError("adapter must be HoloMotionK1MJLabAdapter")
        if not isinstance(sampler, MotionSampler):
            raise TypeError("sampler must be MotionSampler")
        if sampler.num_envs != adapter.bridge.num_envs:
            raise ValueError("sampler and mjlab environment counts differ")
        if sampler.library.device != adapter.bridge.device:
            raise ValueError("sampler and mjlab environment devices differ")
        if len(sampler.offsets_s) != adapter.contract.future_frames + 2:
            raise ValueError("HoloMotion requires current plus eleven future samples")
        if abs(sampler.step_dt - adapter.bridge.env.step_dt) > 1.0e-9:
            raise ValueError("sampler and mjlab control periods differ")
        if not sampler.library.body_names:
            raise ValueError(
                "HoloMotion rewards and critic require body reference data"
            )
        self.adapter = adapter
        self.bridge = adapter.bridge
        # The bridge exposes two intentionally separate reads: deployable
        # proprioceptive/kinematic input for the student and exact simulator
        # contact truth for Teacher/Critic/reward supervision.
        self.physics_bridge = K1PhysicsBridge(self.bridge)
        self.foot_reward_config = foot_reward_config or HoloMotionFootRewardConfig()
        if not isinstance(self.foot_reward_config, HoloMotionFootRewardConfig):
            raise TypeError("foot_reward_config must be HoloMotionFootRewardConfig")
        if physics_randomization is not None and not isinstance(
            physics_randomization, HoloMotionPhysicsRandomizationConfig
        ):
            raise TypeError("physics_randomization must be HoloMotionPhysicsRandomizationConfig")
        self.physics_randomization = physics_randomization
        self._physics_fields_prepared = False
        self._base_body_mass = None
        self._base_body_inertia = None
        self._trunk_body_id = int(self.bridge.entity.indexing.root_body_id)
        self.sampler = sampler
        self.critic_builder = HoloMotionK1CriticObservationBuilder(
            adapter.contract.joint_names
        )
        self.num_envs = self.bridge.num_envs
        self.device = self.bridge.device
        self._friction = torch.ones(self.num_envs, device=self.device)
        self._trunk_mass_scale = torch.ones(self.num_envs, device=self.device)
        robot_body_names = tuple(self.bridge.entity.body_names)
        reference_body_names = tuple(self.sampler.library.body_names)
        self._aux_robot_body_indices = torch.tensor(
            [robot_body_names.index(name) for name in K1_AUX_POSITION_BODY_NAMES],
            device=self.device,
            dtype=torch.long,
        )
        self._aux_reference_body_indices = torch.tensor(
            [
                reference_body_names.index(name)
                for name in K1_AUX_POSITION_BODY_NAMES
            ],
            device=self.device,
            dtype=torch.long,
        )
        self._reference_root_body_index = reference_body_names.index(
            self.sampler.library.robot.root_name
        )
        aux_sensor = self.bridge.env.scene.sensors.get(
            HOLOMOTION_AUX_CONTACT_SENSOR
        )
        if aux_sensor is None:
            raise RuntimeError("HoloMotion auxiliary contact sensor is unavailable")
        sensor_names = tuple(
            dict.fromkeys(slot.primary_name for slot in aux_sensor._slots)
        )
        if set(sensor_names) != set(K1_AUX_CONTACT_BODY_NAMES):
            raise RuntimeError(
                "HoloMotion auxiliary contact sensor body contract differs"
            )
        self._aux_contact_indices = torch.tensor(
            [sensor_names.index(name) for name in K1_AUX_CONTACT_BODY_NAMES],
            device=self.device,
            dtype=torch.long,
        )
        self._pending_reference: Any | None = None
        self._pending_body_reference: Any | None = None

    @property
    def physics_teacher_dim(self) -> int:
        return 2 if self.physics_randomization is not None else 0

    def _prepare_physics_randomization(self) -> None:
        if self._physics_fields_prepared:
            return
        from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

        def marker():
            pass

        marked = requires_model_fields(
            "geom_friction", "body_mass", "body_inertia",
            recompute=RecomputeLevel.set_const,
        )(marker)
        self.bridge.env.sim.expand_model_fields(tuple(marked.model_fields))
        self._base_body_mass = self.bridge.env.sim.get_default_field("body_mass").clone()
        self._base_body_inertia = self.bridge.env.sim.get_default_field("body_inertia").clone()
        self._physics_fields_prepared = True

    @torch.no_grad()
    def _randomize_physics(self, ids: torch.Tensor) -> None:
        if self.physics_randomization is None:
            return
        self._prepare_physics_randomization()
        cfg = self.physics_randomization
        count = ids.numel()
        friction = torch.empty(count, device=self.device).uniform_(*cfg.friction_range)
        mass_scale = torch.empty(count, device=self.device).uniform_(
            *cfg.trunk_mass_scale_range
        )
        sim = self.bridge.env.sim
        sim.model.geom_friction[ids, :, 0] = friction[:, None]
        root = self._trunk_body_id
        sim.model.body_mass[ids, root] = self._base_body_mass[root] * mass_scale
        sim.model.body_inertia[ids, root] = self._base_body_inertia[root] * mass_scale[:, None]
        from mjlab.managers.event_manager import RecomputeLevel
        sim.recompute_constants(RecomputeLevel.set_const)
        sim.forward();sim.sense()
        self._friction[ids] = friction
        self._trunk_mass_scale[ids] = mass_scale

    @torch.no_grad()
    def deployable_physics(self):
        """Student-only observation; never reads exact mjlab contacts."""
        return self.physics_bridge.read_deployable()

    @torch.no_grad()
    def privileged_foot_contact_truth(self) -> PrivilegedFootContactTruth:
        """Exact contact truth kept outside the Actor observation object."""
        return self.physics_bridge.privileged_contact_truth()

    @torch.no_grad()
    def physics_teacher_target(self) -> torch.Tensor:
        if self.physics_randomization is None:
            raise RuntimeError("slow-physics Teacher requires domain randomization")
        f0, f1 = self.physics_randomization.friction_range
        m0, m1 = self.physics_randomization.trunk_mass_scale_range
        friction = 2.0 * (self._friction - f0) / (f1 - f0) - 1.0
        mass = 2.0 * (self._trunk_mass_scale - m0) / (m1 - m0) - 1.0
        return torch.stack((friction, mass), dim=-1)

    @torch.no_grad()
    def privileged_physics_observation(self):
        """Training-only truth for transition labels and privileged Teacher inputs."""
        return self.physics_bridge.read()

    def _auxiliary_targets(
        self,
        state: HoloMotionMJLabState,
        body_reference: Any,
    ) -> HoloMotionAuxiliaryTargets:
        base_lin_vel_b = _quat_apply_inverse(
            state.root_quat_wxyz, state.root_lin_vel_w
        )
        robot_delta = state.body_pos_w.index_select(1, self._aux_robot_body_indices) - (
            state.root_pos_w[:, None, :]
        )
        robot_relative = _quat_apply_inverse(
            state.root_quat_wxyz[:, None, :].expand(
                -1, self._aux_robot_body_indices.numel(), -1
            ),
            robot_delta,
        )
        ref_root_pos = body_reference.pos_w[:, 0, self._reference_root_body_index]
        ref_root_quat = body_reference.quat_wxyz[
            :, 0, self._reference_root_body_index
        ]
        ref_delta = body_reference.pos_w[:, 0].index_select(
            1, self._aux_reference_body_indices
        ) - ref_root_pos[:, None, :]
        ref_relative = _quat_apply_inverse(
            ref_root_quat[:, None, :].expand(
                -1, self._aux_reference_body_indices.numel(), -1
            ),
            ref_delta,
        )
        sensor = self.bridge.env.scene.sensors.get(HOLOMOTION_AUX_CONTACT_SENSOR)
        if sensor is None or sensor.data.current_contact_time is None:
            raise RuntimeError("HoloMotion auxiliary contact sensor is unavailable")
        contacts = (
            sensor.data.current_contact_time.index_select(
                1, self._aux_contact_indices
            )
            > 0.0
        ).to(state.joint_pos.dtype)
        return HoloMotionAuxiliaryTargets(
            base_lin_vel_b=base_lin_vel_b,
            keybody_contacts=contacts,
            ref_keybody_rel_pos=ref_relative,
            robot_keybody_rel_pos=robot_relative,
        )

    def _ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        return self.bridge._ids(env_ids)

    @torch.no_grad()
    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        clip_ids: torch.Tensor | Sequence[int] | None = None,
        start_times_s: torch.Tensor | Sequence[float] | None = None,
        random_start: bool = False,
        min_remaining_s: float = 0.0,
    ) -> HoloMotionMJLabObservation:
        ids = self._ids(env_ids)
        self.adapter.reset(ids)
        self._randomize_physics(ids)
        data = self.bridge.entity.data
        self.sampler.reset(
            env_ids=ids,
            clip_ids=clip_ids,
            start_times_s=start_times_s,
            random_start=random_start,
            min_remaining_s=min_remaining_s,
            current_root_pos_w=data.root_link_pos_w.index_select(0, ids),
            current_root_quat_wxyz=data.root_link_quat_w.index_select(0, ids),
        )
        reference = self.sampler.reference_window(ids)
        root_pose = torch.cat(
            (reference.root_pos_w[:, 0], reference.root_quat_wxyz[:, 0]), -1
        )
        root_velocity = torch.cat(
            (reference.root_lin_vel_w[:, 0], reference.root_ang_vel_w[:, 0]), -1
        )
        joint_pos = self.bridge.robot.to_simulator_order(
            reference.joint_pos[:, 0], self.bridge.joint_names
        )
        joint_vel = self.bridge.robot.to_simulator_order(
            reference.joint_vel[:, 0], self.bridge.joint_names
        )
        self.bridge.entity.write_root_link_pose_to_sim(root_pose, env_ids=ids)
        self.bridge.entity.write_root_link_velocity_to_sim(root_velocity, env_ids=ids)
        self.bridge.entity.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
        self.bridge.env.sim.forward()
        self.bridge.env.sim.sense()
        self._pending_reference = None
        self._pending_body_reference = None
        return self.observe()

    @torch.no_grad()
    def critic_observation(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """Read selected final states without touching actor or pending references."""
        ids = self._ids(env_ids)
        action = self.adapter.last_action_onnx
        state = HoloMotionMJLabState.from_adapter(
            self.adapter, action=action, previous_action=action
        )
        state = replace(state, **{
            field.name: getattr(state, field.name).index_select(0, ids)
            for field in fields(state)
            if isinstance(getattr(state, field.name), torch.Tensor)
        })
        return self.critic_builder(
            state, self.sampler.reference_window(ids),
            self.sampler.body_reference_window(ids),
        )

    @torch.no_grad()
    def observe(self) -> HoloMotionMJLabObservation:
        if self._pending_reference is not None:
            raise RuntimeError("the previous observation must be consumed by step()")
        reference = self.sampler.reference_window()
        body_reference = self.sampler.body_reference_window()
        actor_observation = self.adapter.observation(reference)
        action = self.adapter.last_action_onnx
        state = HoloMotionMJLabState.from_adapter(
            self.adapter, action=action, previous_action=action
        )
        critic_observation = self.critic_builder(state, reference, body_reference)
        auxiliary = self._auxiliary_targets(state, body_reference)
        self._pending_reference = reference
        self._pending_body_reference = body_reference
        return HoloMotionMJLabObservation(
            actor_observation, critic_observation, auxiliary
        )

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> HoloMotionMJLabTransition:
        if self._pending_reference is None or self._pending_body_reference is None:
            raise RuntimeError("observe() must be called exactly once before step()")
        reference = self._pending_reference
        body_reference = self._pending_body_reference
        previous_action = self.adapter.last_action_onnx.clone()
        physics = self.adapter.step(action)
        state = HoloMotionMJLabState.from_adapter(
            self.adapter, action=action, previous_action=previous_action
        )
        terms = holomotion_k1_reward_terms(
            state, reference, body_reference, reference_index=1
        )
        contact = self.privileged_foot_contact_truth()
        active = contact.contact_found.to(action.dtype)
        left_slip = contact.slip_speed_m_s[:, 0] * active[:, 0]
        right_slip = contact.slip_speed_m_s[:, 1] * active[:, 1]
        threshold = float(self.foot_reward_config.impact_threshold_n)
        left_impact = torch.relu(contact.normal_force_n[:, 0] - threshold) / threshold
        right_impact = torch.relu(contact.normal_force_n[:, 1] - threshold) / threshold
        terms.update(
            left_foot_slip=left_slip,
            right_foot_slip=right_slip,
            left_foot_impact=left_impact,
            right_foot_impact=right_impact,
        )
        cfg = self.foot_reward_config
        terms["weighted_total"] = (
            terms["weighted_total"]
            - float(cfg.left_slip_weight) * left_slip
            - float(cfg.right_slip_weight) * right_slip
            - float(cfg.left_impact_weight) * left_impact
            - float(cfg.right_impact_weight) * right_impact
        )
        terms["known_weighted_total"] = terms["weighted_total"]
        gravity_distance = gravity_projection_distance(
            state.root_quat_wxyz, reference.root_quat_wxyz[:, 1]
        )
        self.sampler.advance()
        # 下一帧 observe 仍要求当前帧与全部未来参考有效；
        # 不能等当前时间真正越过 clip 终点才结束。
        motion_finished = self.sampler.finished(
            min_remaining_s=float(self.sampler.offsets_s[-1])
        )
        physical_failure = physics.terminated | (gravity_distance > 0.8)
        terminated = physical_failure | motion_finished
        truncated = physics.truncated & ~terminated
        # Diagnostics only: added after weighted_total is finalized. A failure
        # on the final reference frame is not counted as a successful completion.
        terms['physical_failure'] = physical_failure.to(action.dtype)
        terms['reference_completion'] = (motion_finished & ~physical_failure).to(action.dtype)
        self._pending_reference = None
        self._pending_body_reference = None
        return HoloMotionMJLabTransition(
            reward=terms["weighted_total"],
            terminated=terminated,
            truncated=truncated,
            terms=terms,
            gravity_projection_distance=gravity_distance,
            motion_finished=motion_finished,
            physical_failure=physical_failure,
        )

    def close(self) -> None:
        self.bridge.close()

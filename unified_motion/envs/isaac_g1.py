"""Isaac Lab boundary for the official HoloMotion G1 29-DoF asset.

The module intentionally has no top-level Isaac imports, so model/checkpoint
tools and CPU contract tests remain usable outside an Isaac Sim process.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..robot_contract import RobotContract, get_robot_contract

ISAAC_G1_ROBOT_ID = "g1_29dof"
ISAAC_G1_CONTACT_SLOTS = ("left_foot", "right_foot", "left_wrist", "right_wrist")


def isaac_g1_articulation_cfg(
    robot_config,
    *,
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    domain_rand_config=None,
):
    """Build the official HoloMotion G1 articulation inside Isaac Lab.

    ``robot_config`` is the resolved ``robot`` node from HoloMotion's
    ``29dof_training_isaaclab.yaml``.  Requiring it explicitly prevents this
    bridge from silently falling back to Isaac Lab's incompatible 37-DoF
    bundled G1 asset.
    """
    try:
        from holomotion.src.env.isaaclab_components.isaaclab_scene import SceneFunctions
    except ImportError as error:
        raise RuntimeError(
            "official HoloMotion G1 configuration must be created inside its Isaac Lab runtime"
        ) from error
    return SceneFunctions.build_robot_config(
        robot_config, domain_rand_config=domain_rand_config
    ).replace(prim_path=prim_path)


def _ids_tensor(ids, *, count: int, device: torch.device) -> torch.Tensor:
    if isinstance(ids, slice):
        return torch.arange(count, device=device, dtype=torch.long)[ids]
    result = torch.as_tensor(ids, device=device, dtype=torch.long)
    if result.ndim != 1 or result.numel() < 1:
        raise ValueError("Isaac action joint ids must be a non-empty vector")
    return result


def _unique_names(names: Sequence[str], *, label: str) -> tuple[str, ...]:
    result = tuple(names)
    if not result or any(not isinstance(name, str) or not name for name in result):
        raise ValueError(f"{label} must contain non-empty names")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate names")
    return result


@dataclass(frozen=True)
class IsaacG1IndexContract:
    """Resolved Isaac articulation/action/body indices with stable identities."""

    robot_joint_names: tuple[str, ...]
    action_joint_names: tuple[str, ...]
    action_joint_ids: torch.Tensor
    contact_body_names: tuple[str, ...] | None
    contact_body_ids: torch.Tensor | None


class IsaacG1Bridge:
    """Translate an Isaac ManagerBasedRLEnv G1 to the shared policy contract.

    The Actor sees policy-order joint offsets, velocities, and previous raw
    actions.  Actor outputs are mapped back into the action term's actual joint
    order before ``env.step``.  No assumption is made about USD column order.
    """

    robot_id = ISAAC_G1_ROBOT_ID

    def __init__(
        self,
        env,
        *,
        robot_name: str = "robot",
        action_term_name: str = "joint_pos",
        contact_body_names: Mapping[str, str] | None = None,
    ) -> None:
        self.env = env
        try:
            self.robot = env.scene[robot_name]
            self.action_term = env.action_manager.get_term(action_term_name)
        except (AttributeError, KeyError) as error:
            raise TypeError("expected an Isaac Lab manager environment with robot/action term") from error
        self.contract: RobotContract = get_robot_contract(self.robot_id)
        self.device = self.robot.data.joint_pos.device
        self.num_envs = int(self.robot.data.joint_pos.shape[0])
        robot_names = _unique_names(self.robot.joint_names, label="Isaac robot joints")
        self.contract.check_joint_set(robot_names, source="Isaac Lab G1 articulation")
        action_ids = _ids_tensor(
            self.action_term._joint_ids,
            count=len(robot_names),
            device=self.device,
        )
        action_names = tuple(robot_names[index] for index in action_ids.tolist())
        self.contract.check_joint_set(action_names, source="Isaac Lab G1 action term")
        body_names = None
        body_ids = None
        if contact_body_names is not None:
            if tuple(contact_body_names) != ISAAC_G1_CONTACT_SLOTS:
                raise ValueError(
                    f"contact bodies must preserve slot order {ISAAC_G1_CONTACT_SLOTS}"
                )
            available = _unique_names(self.robot.body_names, label="Isaac robot bodies")
            missing = [name for name in contact_body_names.values() if name not in available]
            if missing:
                raise ValueError(f"Isaac G1 contact bodies are missing {missing}")
            body_names = tuple(contact_body_names[slot] for slot in ISAAC_G1_CONTACT_SLOTS)
            body_ids = torch.tensor(
                [available.index(name) for name in body_names],
                device=self.device,
                dtype=torch.long,
            )
        self.indices = IsaacG1IndexContract(
            robot_names, action_names, action_ids, body_names, body_ids)

    def _policy_joints(self, value: torch.Tensor) -> torch.Tensor:
        return self.contract.to_policy_order(value, self.indices.robot_joint_names)

    def previous_action(self) -> torch.Tensor:
        value = self.env.action_manager.prev_action
        expected = (self.num_envs, self.contract.dof)
        if value.shape != expected:
            raise ValueError(f"Isaac previous action must have shape {expected}")
        return self.contract.to_policy_order(value, self.indices.action_joint_names)

    @torch.no_grad()
    def actor_state(self) -> torch.Tensor:
        """Return ``[B, 6+3N]`` in the shared deployable state schema."""
        data = self.robot.data
        joint_offset = data.joint_pos - data.default_joint_pos
        pieces = (
            data.root_ang_vel_b,
            data.projected_gravity_b,
            self._policy_joints(joint_offset),
            self._policy_joints(data.joint_vel),
            self.previous_action(),
        )
        result = torch.cat(pieces, dim=-1)
        expected = (self.num_envs, 6 + 3 * self.contract.dof)
        if result.shape != expected or not torch.isfinite(result).all():
            raise ValueError(f"invalid Isaac G1 actor state; expected {expected}")
        return result

    @torch.no_grad()
    def base_linear_velocity_body(self) -> torch.Tensor:
        value = self.robot.data.root_lin_vel_b
        if value.shape != (self.num_envs, 3) or not torch.isfinite(value).all():
            raise ValueError("invalid Isaac G1 root linear velocity")
        return value

    def action_to_isaac(self, action: torch.Tensor) -> torch.Tensor:
        expected = (self.num_envs, self.contract.dof)
        if action.shape != expected or not torch.isfinite(action).all():
            raise ValueError(f"G1 Actor action must be finite with shape {expected}")
        return self.contract.to_simulator_order(action, self.indices.action_joint_names)

    def step(self, action: torch.Tensor):
        return self.env.step(self.action_to_isaac(action))

    def contact_body_state(self) -> dict[str, torch.Tensor]:
        """Read the four explicitly configured end-effector bodies."""
        ids = self.indices.contact_body_ids
        if ids is None:
            raise RuntimeError("contact body names were not configured")
        data = self.robot.data
        return {
            "position_w": data.body_pos_w.index_select(1, ids),
            "quaternion_wxyz": data.body_quat_w.index_select(1, ids),
            "linear_velocity_w": data.body_link_lin_vel_w.index_select(1, ids),
            "angular_velocity_w": data.body_ang_vel_w.index_select(1, ids),
        }

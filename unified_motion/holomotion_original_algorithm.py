"""Load original trainable HoloMotion actor/critic checkpoints.

The network and PPO implementations stay owned by the audited, vendored
HoloMotion snapshot.  This module only reconstructs the resolved run configuration,
converts flat observations back to HoloMotion's TensorDict schema, and strictly
restores safetensors checkpoints.  Dimensions are derived from the resolved
robot config so official G1 29-DoF weights are not forced through the K1
22-DoF contract.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

ACTOR_CURRENT_TERM_WIDTHS = (
    ("actor_ref_gravity_projection_cur", 3),
    ("actor_ref_base_linvel_cur", 3),
    ("actor_ref_base_angvel_cur", 3),
    ("actor_ref_dof_pos_cur", 22),
    ("actor_ref_root_height_cur", 1),
    ("actor_ref_robot_yaw_error_sin_cos", 2),
    ("actor_projected_gravity", 3),
    ("actor_rel_robot_root_ang_vel", 3),
    ("actor_dof_pos", 22),
    ("actor_dof_vel", 22),
    ("actor_last_action", 22),
)

ACTOR_FUTURE_TERM_WIDTHS = (
    ("actor_ref_dof_pos_fut", 22),
    ("actor_ref_root_height_fut", 1),
    ("actor_ref_gravity_projection_fut", 3),
    ("actor_ref_base_linvel_fut", 3),
    ("actor_ref_base_angvel_fut", 3),
    ("actor_ref_future_yaw_delta_sin_cos", 2),
    ("actor_ref_future_root_ori_robot_frame_6d", 6),
)

CRITIC_CURRENT_TERM_WIDTHS = (
    ("critic_ref_dof_pos_cur", 22),
    ("critic_global_anchor_diff", 9),
    ("critic_ref_motion_cur_heading_aligned_root_pos", 3),
    ("critic_ref_motion_cur_heading_aligned_root_rot6d", 6),
    ("critic_ref_motion_cur_heading_aligned_root_lin_vel", 3),
    ("critic_ref_motion_cur_heading_aligned_root_ang_vel", 3),
    ("critic_rel_robot_root_lin_vel", 3),
    ("critic_rel_robot_root_ang_vel", 3),
    ("critic_global_robot_bodylink_lin_vel_flat", 69),
    ("critic_global_robot_bodylink_ang_vel_flat", 69),
    ("critic_root_rel_robot_bodylink_pos_flat", 69),
    ("critic_root_rel_robot_bodylink_rot_mat_flat", 138),
    ("critic_dof_pos", 22),
    ("critic_dof_vel", 22),
    ("critic_last_action", 22),
)

CRITIC_FUTURE_TERM_WIDTHS = (
    ("critic_ref_dof_pos_fut", 22),
    ("critic_ref_root_height_fut", 1),
    ("critic_ref_motion_fut_heading_aligned_root_pos", 3),
    ("critic_ref_motion_fut_heading_aligned_root_rot6d", 6),
    ("critic_ref_motion_fut_heading_aligned_root_lin_vel", 3),
    ("critic_ref_motion_fut_heading_aligned_root_ang_vel", 3),
)


def _split_terms(
    observation: torch.Tensor,
    *,
    current: tuple[tuple[str, int], ...],
    future: tuple[tuple[str, int], ...],
    expected_dim: int,
    future_steps: int = 10,
) -> dict[str, torch.Tensor]:
    if observation.ndim not in (2, 3) or observation.shape[-1] != expected_dim:
        raise ValueError(
            f"observation must have shape [B,{expected_dim}] or [B,T,{expected_dim}]"
        )
    if not torch.isfinite(observation).all().item():
        raise ValueError("observation contains NaN/Inf")
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for name, width in current:
        result[f"unified/{name}"] = observation[..., offset : offset + width]
        offset += width
    for name, width in future:
        size = future_steps * width
        result[f"unified/{name}"] = observation[..., offset : offset + size].reshape(
            *observation.shape[:-1], future_steps, width
        )
        offset += size
    if offset != expected_dim:
        raise RuntimeError(f"term schema consumes {offset}, expected {expected_dim}")
    return result


def _term_widths(dof: int, num_bodies: int):
    if type(dof) is not int or dof < 1:
        raise ValueError("dof must be a positive integer")
    if type(num_bodies) is not int or num_bodies < 1:
        raise ValueError("num_bodies must be a positive integer")
    actor_current = tuple(
        (name, dof if width == 22 else width)
        for name, width in ACTOR_CURRENT_TERM_WIDTHS
    )
    actor_future = tuple(
        (name, dof if width == 22 else width)
        for name, width in ACTOR_FUTURE_TERM_WIDTHS
    )
    critic_current = tuple(
        (name, (
            6 * num_bodies if width == 138 else
            3 * num_bodies if width == 69 else
            dof if width == 22 else width
        ))
        for name, width in CRITIC_CURRENT_TERM_WIDTHS
    )
    critic_future = tuple(
        (name, dof if width == 22 else width)
        for name, width in CRITIC_FUTURE_TERM_WIDTHS
    )
    return actor_current, actor_future, critic_current, critic_future


def original_holomotion_observation_dims(
    dof: int, num_bodies: int, *, future_steps: int = 10
) -> tuple[int, int]:
    """Return official flat Actor/Critic widths for a robot morphology."""
    if type(future_steps) is not int or future_steps < 1:
        raise ValueError("future_steps must be a positive integer")
    actor_current, actor_future, critic_current, critic_future = _term_widths(
        dof, num_bodies
    )
    actor_dim = sum(width for _, width in actor_current) + future_steps * sum(
        width for _, width in actor_future
    )
    critic_dim = sum(width for _, width in critic_current) + future_steps * sum(
        width for _, width in critic_future
    )
    return actor_dim, critic_dim


def flat_observations_to_tensordict(
    actor_observation: torch.Tensor,
    critic_observation: torch.Tensor,
    *,
    dof: int = 22,
    num_bodies: int = 23,
    future_steps: int = 10,
) -> Any:
    """Restore the TensorDict leaves consumed by the original wrappers."""
    if actor_observation.shape[:-1] != critic_observation.shape[:-1]:
        raise ValueError("actor and critic observation batch dimensions differ")
    if (
        actor_observation.device != critic_observation.device
        or actor_observation.dtype != critic_observation.dtype
    ):
        raise ValueError("actor and critic observations must share device/dtype")
    try:
        from tensordict import TensorDict
    except ImportError as error:
        raise RuntimeError(
            "the original HoloMotion algorithm requires tensordict"
        ) from error

    actor_current, actor_future, critic_current, critic_future = _term_widths(
        dof, num_bodies
    )
    actor_dim, critic_dim = original_holomotion_observation_dims(
        dof, num_bodies, future_steps=future_steps
    )
    leaves = _split_terms(
        actor_observation,
        current=actor_current,
        future=actor_future,
        expected_dim=actor_dim,
        future_steps=future_steps,
    )
    leaves.update(
        _split_terms(
            critic_observation,
            current=critic_current,
            future=critic_future,
            expected_dim=critic_dim,
            future_steps=future_steps,
        )
    )
    return TensorDict(leaves, batch_size=list(actor_observation.shape[:-1]))


@dataclass(frozen=True)
class OriginalHoloMotionK1Algorithm:
    actor: torch.nn.Module
    critic: torch.nn.Module
    config: Any
    run_dir: Path
    network_config: dict | None = None
    pretrained: bool = False


class _ConfigNode(dict):
    """Tiny attribute-access mapping used by HoloMotionPPOConfig.from_original."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def _config_node(value: Any) -> Any:
    if isinstance(value, dict):
        return _ConfigNode({key: _config_node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_config_node(item) for item in value]
    return value


def _merge_mapping(target: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_mapping(target[key], value)
        else:
            target[key] = value


def initialize_original_holomotion_k1_algorithm(
    *,
    holomotion_root: str | Path,
    config_path: str | Path,
    device: str | torch.device = "cpu",
) -> OriginalHoloMotionK1Algorithm:
    """Randomly initialize the audited HoloMotion K1 actor and critic.

    The compact project config contains deliberate overrides and PPO settings;
    observation schemas continue to come from the vendored, audited HoloMotion
    module config.  No checkpoint or pretrained tensor is read here.
    """
    root = Path(holomotion_root).expanduser().resolve()
    path = Path(config_path).expanduser().resolve()
    base_path = (
        root
        / "holomotion"
        / "config"
        / "modules"
        / "motion_tracking"
        / "motion_tracking_tf-moe.yaml"
    )
    missing = [str(candidate) for candidate in (base_path, path) if not candidate.is_file()]
    if missing:
        raise FileNotFoundError(f"missing HoloMotion initialization files: {missing}")
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(values) != {
        "actor_overrides", "critic_overrides", "ppo"
    }:
        raise ValueError(
            "HoloMotion config must contain exactly actor_overrides, "
            "critic_overrides, and ppo mappings"
        )
    if not all(isinstance(values[name], dict) for name in values):
        raise ValueError("HoloMotion config sections must be YAML mappings")
    modules = yaml.safe_load(base_path.read_text(encoding="utf-8"))["modules"]
    actor_cfg, critic_cfg = modules["actor"], modules["critic"]
    _merge_mapping(actor_cfg, values["actor_overrides"])
    _merge_mapping(critic_cfg, values["critic_overrides"])
    ppo_cfg = values["ppo"]
    for name in ("noise_std_type", "fix_sigma", "min_sigma", "max_sigma"):
        actor_cfg[name] = ppo_cfg[name]
    for name in (
        "aux_state_pred",
        "aux_router_command_recon",
        "aux_router_switch_penalty",
        "moe_load_balance",
        "inactive_expert_margin_to_topk",
        "selected_expert_margin_to_unselected",
    ):
        actor_cfg[name] = ppo_cfg[name]
    router = ppo_cfg["moe_router"]
    for name in (
        "routing_score_fn", "routing_scale", "use_dynamic_bias",
        "bias_update_rate", "expert_bias_clip",
    ):
        actor_cfg[name] = router[name]
    # The local bridge always exposes one current-state token and ten future
    # reference frames, independent of the upstream Hydra interpolation tree.
    for cfg in (actor_cfg, critic_cfg):
        cfg["obs_schema"]["flattened_obs"]["seq_len"] = 1
        cfg["obs_schema"]["flattened_obs_fut"]["seq_len"] = 10
    network_config = dict(
        actor=actor_cfg,
        critic=critic_cfg,
        init_noise_std=float(ppo_cfg["init_noise_std"]),
    )
    actor, critic = build_original_holomotion_networks(
        network_config, device=device, holomotion_root=root
    )
    config = _config_node({"algo": {"config": ppo_cfg}})
    return OriginalHoloMotionK1Algorithm(
        actor, critic, config, path.parent, network_config
    )


def build_original_holomotion_networks(
    network_config,
    *,
    device="cpu",
    holomotion_root=None,
    robot_dof: int = 22,
    num_bodies: int = 23,
):
    """Reconstruct the audited network classes from resolved, inert metadata."""
    root = (Path(holomotion_root) if holomotion_root is not None else
            Path(__file__).resolve().parents[1] / "third_party/HoloMotion/source")
    if not (root / "holomotion/src/modules/agent_modules.py").is_file():
        raise FileNotFoundError(root)
    if str(root.resolve()) not in sys.path:
        sys.path.insert(0, str(root.resolve()))
    from holomotion.src.modules.agent_modules import PPOCritic, PPOTFRefRouterActor
    actor_cfg, critic_cfg = network_config["actor"], network_config["critic"]
    if (actor_cfg["type"] != "ReferenceRoutedGroupedMoETransformerPolicy"
            or critic_cfg["type"] != "MLP"):
        raise ValueError("unsupported HoloMotion checkpoint network types")
    actor_dim, critic_dim = original_holomotion_observation_dims(
        robot_dof, num_bodies
    )
    sample = flat_observations_to_tensordict(
        torch.zeros(1, actor_dim, device=device),
        torch.zeros(1, critic_dim, device=device),
        dof=robot_dof,
        num_bodies=num_bodies,
    )
    actor = PPOTFRefRouterActor(actor_cfg["obs_schema"], actor_cfg, robot_dof,
        float(network_config["init_noise_std"]), obs_example=sample).to(device)
    critic = PPOCritic(critic_cfg["obs_schema"], critic_cfg, obs_example=sample).to(device)
    return actor, critic


def load_original_holomotion_k1_algorithm(
    *,
    holomotion_root: str | Path,
    run_dir: str | Path,
    checkpoint_name: str = "model_9999",
    device: str | torch.device = "cpu",
) -> OriginalHoloMotionK1Algorithm:
    """Backward-compatible entry point; morphology is read from run config."""
    return load_original_holomotion_algorithm(
        holomotion_root=holomotion_root,
        run_dir=run_dir,
        checkpoint_name=checkpoint_name,
        device=device,
    )


def load_original_holomotion_algorithm(
    *,
    holomotion_root: str | Path,
    run_dir: str | Path,
    checkpoint_name: str = "model_9999",
    device: str | torch.device = "cpu",
) -> OriginalHoloMotionK1Algorithm:
    """Strictly load an official K1 or G1 run using its morphology metadata."""
    root = Path(holomotion_root).expanduser().resolve()
    run = Path(run_dir).expanduser().resolve()
    config_path = run / "config.yaml"
    checkpoint_dir = run / checkpoint_name
    required = (
        root / "holomotion" / "src" / "modules" / "agent_modules.py",
        config_path,
        checkpoint_dir / "actor" / "model.safetensors",
        checkpoint_dir / "critic" / "model.safetensors",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing original HoloMotion files: {missing}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError(
            "the mjlab environment is missing original HoloMotion runtime dependencies"
        ) from error

    config = OmegaConf.load(config_path)
    robot_dof = int(config.robot.actions_dim)
    observation_dof = int(config.robot.dof_obs_size)
    num_bodies = int(config.robot.num_bodies)
    if robot_dof != observation_dof:
        raise ValueError(
            "official HoloMotion loader requires actions_dim == dof_obs_size"
        )
    actor_cfg = OmegaConf.to_container(config.modules.actor, resolve=True)
    critic_cfg = OmegaConf.to_container(config.modules.critic, resolve=True)
    algo_cfg = OmegaConf.to_container(config.algo.config, resolve=True)
    for name in ("noise_std_type", "fix_sigma", "min_sigma", "max_sigma"):
        actor_cfg[name] = algo_cfg[name]
    for name in (
        "aux_state_pred",
        "aux_router_command_recon",
        "aux_router_switch_penalty",
        "moe_load_balance",
        "inactive_expert_margin_to_topk",
        "selected_expert_margin_to_unselected",
    ):
        actor_cfg[name] = algo_cfg[name]

    target = torch.device(device)
    network_config = dict(actor=actor_cfg, critic=critic_cfg,
                          init_noise_std=float(algo_cfg["init_noise_std"]))
    actor, critic = build_original_holomotion_networks(
        network_config,
        device=target,
        holomotion_root=root,
        robot_dof=robot_dof,
        num_bodies=num_bodies,
    )

    for name, module in (("actor", actor), ("critic", critic)):
        state = load_file(
            str(checkpoint_dir / name / "model.safetensors"), device=str(target)
        )
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
        module.load_state_dict(state, strict=True)
    return OriginalHoloMotionK1Algorithm(actor, critic, config, run, network_config, pretrained=True)

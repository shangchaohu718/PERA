"""Strict reconstruction of the official G1 v1.4 Actor/Critic."""

from pathlib import Path

import torch
from holomotion.src.modules.agent_modules import PPOCritic, PPOTFRefRouterActor
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tensordict import TensorDict

from pera.paths import ROOT

CHECKPOINT = ROOT / "checkpoints/holomotion_v1.4"


def sample_observations(config, batch=1, device="cpu"):
    """Build official term shapes for synthetic network checks."""
    dof, bodies = int(config.robot.actions_dim), int(config.robot.num_bodies)
    leaves = {}
    widths = {
        "gravity_projection": 3,
        "projected_gravity": 3,
        "base_linvel": 3,
        "base_angvel": 3,
        "root_height": 1,
        "yaw_error_sin_cos": 2,
        "future_yaw_delta_sin_cos": 2,
        "future_root_ori_robot_frame_6d": 6,
        "global_anchor_diff": 9,
        "root_pos": 3,
        "root_rot6d": 6,
        "root_lin_vel": 3,
        "root_ang_vel": 3,
        "root_lin_vel_local": 3,
    }
    for module in (config.modules.actor, config.modules.critic):
        for group in module.obs_schema.values():
            for term in group.terms:
                if term in leaves:
                    continue
                name = term.split("/")[-1]
                if any(s in name for s in ("dof_pos", "dof_vel", "last_action")):
                    width = dof
                elif "bodylink" in name:
                    width = bodies * (6 if "rot_mat" in name else 3)
                elif "rel_robot_root" in name:
                    width = 3
                else:
                    matches = [v for k, v in widths.items() if k in name]
                    if not matches or len(set(matches)) != 1:
                        raise ValueError(f"Unknown observation term: {term}")
                    width = matches[0]
                seq = int(group.seq_len)
                shape = (batch, width) if seq == 1 else (batch, seq, width)
                leaves[term] = torch.zeros(shape, device=device)
    return TensorDict(leaves, batch_size=[batch])


def build_official_actor(config, device="cpu"):
    """Reconstruct the published Actor architecture without reading weights."""
    if int(config.robot.actions_dim) != 29:
        raise ValueError("Expected official G1 29-DoF checkpoint")
    actor_cfg = OmegaConf.to_container(config.modules.actor, resolve=True)
    algo_cfg = OmegaConf.to_container(config.algo.config, resolve=True)
    for name in (
        "noise_std_type",
        "fix_sigma",
        "min_sigma",
        "max_sigma",
        "aux_state_pred",
        "aux_router_command_recon",
        "aux_router_switch_penalty",
        "moe_load_balance",
        "inactive_expert_margin_to_topk",
        "selected_expert_margin_to_unselected",
    ):
        actor_cfg[name] = algo_cfg[name]
    sample = sample_observations(config, device=device)
    actor = PPOTFRefRouterActor(
        actor_cfg["obs_schema"],
        actor_cfg,
        29,
        float(algo_cfg["init_noise_std"]),
        obs_example=sample,
    ).to(device)
    return actor


def load_official(device="cuda", checkpoint=CHECKPOINT):
    config = OmegaConf.load(Path(checkpoint) / "config.yaml")
    actor = build_official_actor(config, device)
    critic_cfg = OmegaConf.to_container(config.modules.critic, resolve=True)
    sample = sample_observations(config, device=device)
    critic = PPOCritic(critic_cfg["obs_schema"], critic_cfg, obs_example=sample).to(device)
    for name, module in (("actor", actor), ("critic", critic)):
        state = load_file(
            str(Path(checkpoint) / f"model_14000/{name}/model.safetensors"),
            device=device,
        )
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        module.load_state_dict(state, strict=True)
        module.eval()
    return actor, critic, config

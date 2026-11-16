"""Isolated roundhouse rewards and frozen-public PPO/evaluation adapters."""

from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch
import torch


def foot_states(env, command_name="ref_motion", ref_prefix="ref_"):
    cmd = env.command_manager.get_term(command_name)
    ids = [cmd.robot.body_names.index(s + "_ankle_roll_link") for s in ("left", "right")]
    ref = cmd.get_ref_motion_bodylink_global_pos_immediate_next(prefix=ref_prefix)[:, ids]
    actual = cmd.robot.data.body_pos_w[:, ids]
    return cmd, ids, ref, actual


def roundhouse_swing_position(env, std=0.20, command_name="ref_motion", ref_prefix="ref_"):
    _, _, ref, actual = foot_states(env, command_name, ref_prefix)
    weights = ((ref[..., 2] - 0.20) / 0.20).clamp(0, 1)
    score = torch.exp(-(actual - ref).square().sum(-1) / std**2)
    return (weights * score).sum(-1) / weights.sum(-1).clamp_min(1)


def roundhouse_swing_height(env, std=0.15, command_name="ref_motion", ref_prefix="ref_"):
    _, _, ref, actual = foot_states(env, command_name, ref_prefix)
    weights = ((ref[..., 2] - 0.20) / 0.20).clamp(0, 1)
    score = torch.exp(-(actual[..., 2] - ref[..., 2]).square() / std**2)
    return (weights * score).sum(-1) / weights.sum(-1).clamp_min(1)


def roundhouse_stance_velocity(env, std=0.35, command_name="ref_motion", ref_prefix="ref_"):
    cmd, ids, ref, _ = foot_states(env, command_name, ref_prefix)
    target = cmd.get_ref_motion_bodylink_global_lin_vel_immediate_next(prefix=ref_prefix)[:, ids]
    actual = cmd.robot.data.body_lin_vel_w[:, ids]
    # Follow the reference support-foot velocity, allowing the intended pivot.
    weights = ((0.12 - ref[..., 2]) / 0.08).clamp(0, 1)
    score = torch.exp(-(actual - target).square().sum(-1) / std**2)
    return (weights * score).sum(-1) / weights.sum(-1).clamp_min(1)


def register_terms():
    from holomotion.src.env.isaaclab_components import isaaclab_rewards
    from modumotion_g1.stairs_ppo_rewards import register_stairs_ppo_terms

    register_stairs_ppo_terms()
    for func in (roundhouse_swing_position, roundhouse_swing_height, roundhouse_stance_velocity):
        setattr(isaaclab_rewards, func.__name__, func)


@contextmanager
def cpu_public_metadata(public_path):
    original = torch.load
    target = Path(public_path).resolve()

    def load(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).resolve() == target:
            kwargs.update(map_location="cpu", mmap=True)
        return original(path, *args, **kwargs)

    with patch("torch.load", side_effect=load):
        yield


from modumotion_g1.stairs_ppo_algo import StairsTrainPPO
from modumotion_g1.eval_algo import DomainEvalPPO


class RoundhousePPO(StairsTrainPPO):
    def load(self, path):
        with cpu_public_metadata(self.config.public_checkpoint):
            result = super().load(path)
        actor = self.accelerator.unwrap_model(self.actor)
        names = [n for n, p in actor.named_parameters() if p.requires_grad]
        assert len(names) == 4 and all(".domain_plugins.roundhouse." in n for n in names)
        (Path(self.log_dir) / "freeze_preflight.json").write_text(
            json.dumps(
                dict(
                    public_sha256=actor.public_sha256,
                    trainable_actor_tensors=names,
                    public_frozen=True,
                    only_domain_in_actor_optimizer=True,
                    action_noise_frozen=True,
                    public_normalization_frozen=True,
                    fixed_merge_bias=-0.5,
                    pure_ppo=True,
                    teacher_action_queries=0,
                    unused_public_optimizer_state_on_cpu=True,
                ),
                indent=2,
            )
            + "\n"
        )
        return result

    def _post_update_hook(self, loss_dict):
        super()._post_update_hook(loss_dict)
        actor = self.accelerator.unwrap_model(self.actor)
        weights = actor.actor_module.layers[0].last_group_weights
        if weights is not None:
            record = dict(
                iteration=int(self.current_learning_iteration),
                new_domain_weight_mean=float(weights[..., 1].mean()),
                new_domain_weight_max=float(weights[..., 1].max()),
            )
            with (Path(self.log_dir) / "routing.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")


class RoundhouseEvalPPO(DomainEvalPPO):
    def load(self, path):
        with cpu_public_metadata(path):
            return super().load(path)

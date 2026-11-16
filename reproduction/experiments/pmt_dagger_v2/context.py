"""Controlled observation ablation; privileged inputs are diagnostic only."""

import torch

BASE_DIM = 64
RICH_DIM = 326
FULL_DIM = 362
MODES = ("baseline", "rich", "oracle")


def observable_features(vision, command, history, velocity, anchor_estimate):
    values = [vision, command, history, velocity, anchor_estimate]
    for value, width in zip(values, [64, 128, 128, 3, 3], strict=True):
        if value.ndim != 2 or value.shape[-1] != width:
            raise ValueError(f"Invalid frozen feature shape: {value.shape}")
    return torch.cat(values, dim=-1)


def select_context(full, mode, normalization=None):
    if mode not in MODES or full.shape[-1] != FULL_DIM:
        raise ValueError("Invalid ablation mode/context dimension")
    # Keep the original 64 features unchanged for exact warm-start parity.
    result = torch.zeros_like(full)
    result[..., :BASE_DIM] = full[..., :BASE_DIM]
    stop = {"baseline": BASE_DIM, "rich": RICH_DIM, "oracle": FULL_DIM}[mode]
    if stop > BASE_DIM:
        extra = full[..., BASE_DIM:stop]
        if normalization is not None:
            mean = full.new_tensor(normalization["mean"])[BASE_DIM:stop]
            std = full.new_tensor(normalization["std"])[BASE_DIM:stop]
            extra = (extra - mean) / std.clamp_min(1e-3)
        result[..., BASE_DIM:stop] = extra
    return result


def teacher_schedule(beta, streak, successes, episodes, independent_seed):
    if episodes < 4 or successes / episodes < 0.75:
        return beta, []
    streak = [*streak, independent_seed]
    if len(set(streak)) >= 2:
        return max(0.0, round(beta - 0.2, 6)), []
    return beta, streak


class AugmentedTaskMixin:
    """Use only frozen, raw-reference PMT features for deployable context."""

    context_mode = "rich"
    context_normalization = None

    @torch.no_grad()
    def observable_context(self, obs):
        native = self.vision_observations(obs)
        _, command, velocity, anchor, vision = self.teacher._compute_actor_context(
            native, store_aux_outputs=False
        )
        _, history = self.teacher.encode_proprio_history(native)
        return observable_features(vision, command, history, velocity, anchor)

    @torch.no_grad()
    def full_diagnostic_context(self, obs):
        from modumotion_g1.perceptive_observations import _qrotate_inverse

        observable = self.observable_context(obs)
        position_error = _qrotate_inverse(
            self.robot.data.root_quat_w,
            self.motion.anchor_pos_w - self.motion.robot_anchor_pos_w,
        )
        privileged = torch.cat(
            [
                self.motion.joint_pos - self.raw.joint_pos,
                position_error,
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_pos_w[:, 2:3],
            ],
            dim=-1,
        )
        if privileged.shape[-1] != 36:
            raise ValueError("Invalid diagnostic privilege shape")
        return torch.cat([observable, privileged], dim=-1)

    @torch.no_grad()
    def context(self, obs):
        if self.context_mode == "oracle":
            raise ValueError("Oracle is forbidden for acting student context")
        observable = self.observable_context(obs)
        full = torch.cat(
            [
                observable,
                observable.new_zeros(observable.shape[0], 36),
            ],
            dim=-1,
        )
        return select_context(full, self.context_mode, self.context_normalization)

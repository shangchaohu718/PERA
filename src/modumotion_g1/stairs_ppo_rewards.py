"""World-frame stair rewards and a reference-boundary PPO termination.

These terms use the original Isaac motion command and robot state. They do
not query a teacher or alter the public Actor. Call registration only after
the Isaac application has started and before constructing the environment.
"""

import math

import torch


def _positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _last_reference_frame(command, end_margin_frames=1):
    if int(end_margin_frames) != end_margin_frames or end_margin_frames < 0:
        raise ValueError("end_margin_frames must be a nonnegative integer")
    lengths = command._motion_cache.lengths_for_indices(command._clip_indices)
    return (lengths - 1 - int(command.cfg.n_fut_frames) - end_margin_frames).clamp(min=0)


def stairs_root_path_tracking_exp(
    env,
    std: float = 0.4,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
):
    """Track reference world XY, preserving the fixed stair path and phase."""
    _positive(std, "std")
    command = env.command_manager.get_term(command_name)
    reference = command.get_ref_motion_root_global_pos_immediate_next(prefix=ref_prefix)
    delta = command.robot.data.root_pos_w[:, :2] - reference[:, :2]
    return torch.exp(-delta.square().sum(-1) / std**2)


def stairs_root_height_tracking_exp(
    env,
    std: float = 0.2,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
):
    """Track world root height, which relative body-pose rewards remove."""
    _positive(std, "std")
    command = env.command_manager.get_term(command_name)
    reference = command.get_ref_motion_root_global_pos_immediate_next(prefix=ref_prefix)
    delta = command.robot.data.root_pos_w[:, 2] - reference[:, 2]
    return torch.exp(-delta.square() / std**2)


def stairs_arrival(
    env,
    distance_threshold: float = 0.25,
    min_up_cos: float = 0.5,
    end_margin_frames: int = 1,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
):
    """Reward an upright arrival at the reference boundary before wrapping.

    Pair with stairs_reference_end to reset there. A randomly initialized
    late reference segment can earn this reward; full-ascent validation
    must still start from frame zero and measure the complete trajectory.
    """
    _positive(distance_threshold, "distance_threshold")
    if not math.isfinite(min_up_cos) or not -1 <= min_up_cos <= 1:
        raise ValueError("min_up_cos must be finite and in [-1, 1]")
    command = env.command_manager.get_term(command_name)
    reference = command.get_ref_motion_root_global_pos_immediate_next(prefix=ref_prefix)
    distance = torch.linalg.vector_norm(command.robot.data.root_pos_w - reference, dim=-1)
    quat = command.robot.data.root_quat_w  # WXYZ, original Isaac convention.
    up_cos = 1 - 2 * (quat[:, 1].square() + quat[:, 2].square())
    at_end = command._frame_indices >= _last_reference_frame(command, end_margin_frames)
    return (at_end & (distance <= distance_threshold) & (up_cos >= min_up_cos)).to(reference.dtype)


def stairs_reference_end(
    env,
    end_margin_frames: int = 1,
    command_name: str = "ref_motion",
):
    """End before command.compute can wrap and teleport the reference state.

    Register as a time_out=True termination, since a finite reference window
    truncates the task rather than representing a physical failure.
    """
    command = env.command_manager.get_term(command_name)
    return command._frame_indices >= _last_reference_frame(command, end_margin_frames)


def register_stairs_ppo_terms():
    """Register uniquely named terms with the original config resolvers."""
    import isaaclab.envs.mdp.terminations as native_terminations
    from holomotion.src.env.isaaclab_components import isaaclab_rewards

    for function in (
        stairs_root_path_tracking_exp,
        stairs_root_height_tracking_exp,
        stairs_arrival,
    ):
        setattr(isaaclab_rewards, function.__name__, function)
    # The original termination resolver accepts arbitrary callable names from
    # this fallback namespace, while custom globals require native ownership.
    native_terminations.stairs_reference_end = stairs_reference_end

"""One PMT placement with an explicit, audited reference-input choice."""

import torch

from experiments.pmt_dagger_v1.native import NativePMTTask
from experiments.pmt_dagger_v2.context import AugmentedTaskMixin


class SinglePlacementTask(NativePMTTask):
    def __init__(self, *args, clip_id=0, actor_reference="raw", **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_id = clip_id
        self.actor_reference = actor_reference
        if actor_reference not in ("raw", "optimized"):
            raise ValueError("Unknown Actor reference")
        if actor_reference == "optimized":
            store = self.motion.data_store
            clips = []
            for i, length in enumerate(store.motion_lengths.tolist()):
                clips.append(
                    torch.cat(
                        (
                            store._stacked_body_pos_w[i, :length, 0],
                            store._stacked_body_quat_w[i, :length, 0],
                            store._stacked_joint_pos[i, :length, self.to_holo],
                        ),
                        -1,
                    )
                )
            if not torch.equal(store.motion_lengths, self.raw.data_store.motion_lengths):
                raise ValueError("Raw and optimized reference timing differ")
            self.raw_qpos = torch.cat(clips).float().to(self.device)

    def reset(self, seed, actor=None, from_start=True):
        from pmt_tasks.mdp.commands.multi_motion_command import SamplingResult

        if not from_start:
            raise ValueError("This diagnostic starts at phase zero")
        self.motion.cfg.force_start_frame_zero = True
        self.motion.cfg.default_reset_joint_prob = 0.0
        self.resampled.zero_()
        self.completions.zero_()
        draw = self.motion._draw_samples

        def fixed_clip(ids):
            return SamplingResult(torch.full_like(ids, self.clip_id), torch.zeros_like(ids))

        self.motion._draw_samples = fixed_clip
        try:
            obs, _ = self.env.reset(seed=seed)
        finally:
            self.motion._draw_samples = draw
        if not self.raw.motion_ids.eq(self.clip_id).all():
            raise ValueError("Paired raw reference did not follow selected placement")
        self.last_action.zero_()
        self.alignment_history.zero_()
        if actor is not None:
            actor.reset_kv_cache(self.N, self.device)
            actor.clear_env_cache(None)
        return obs


class OptimizedPlacementTask(SinglePlacementTask):
    """Planned optimized command plus frozen, observable 326-D PMT features."""

    @torch.no_grad()
    def context(self, obs):
        return AugmentedTaskMixin.observable_context(self, obs)

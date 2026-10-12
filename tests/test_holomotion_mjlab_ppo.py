import pytest
import torch

from unified_motion.holomotion_mjlab_ppo import (
    HoloMotionMJLabPPO,
    build_episode_causal_mask,
    generalized_advantage_estimate,
)


def test_episode_causal_mask_cuts_history_after_done():
    dones = torch.tensor([[False, True, False, False]])
    mask = build_episode_causal_mask(dones)

    expected = torch.tensor(
        [[[True, False, False, False],
          [True, True, False, False],
          [False, False, True, False],
          [False, False, True, True]]]
    )
    assert torch.equal(mask, expected)


def test_gae_stops_bootstrap_at_episode_boundary():
    rewards = torch.tensor([[[1.0], [2.0], [3.0]]])
    values = torch.zeros_like(rewards)
    dones = torch.tensor([[[False], [True], [False]]])
    bootstrap = torch.tensor([[4.0]])

    advantage, returns = generalized_advantage_estimate(
        rewards, values, dones, bootstrap, gamma=1.0, lam=1.0
    )

    assert torch.equal(advantage, torch.tensor([[[3.0], [2.0], [7.0]]]))
    assert torch.equal(returns, advantage)


def test_episode_causal_mask_rejects_non_boolean_input():
    with pytest.raises(ValueError, match="boolean"):
        build_episode_causal_mask(torch.zeros(1, 2))


def test_analytic_kl_is_zero_for_identical_diagonal_gaussians():
    mean = torch.randn(2, 3, 22)
    sigma = torch.rand(2, 3, 22).clamp_min(0.1)

    value = HoloMotionMJLabPPO._analytic_kl(mean, sigma, mean, sigma)

    assert value == pytest.approx(0.0, abs=2.0e-6)

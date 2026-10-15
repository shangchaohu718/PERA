import math

import pytest
import torch

from tools.audit_teacher_multihorizon import (
    _valid_horizon,
    constant_velocity_core,
)
from tools.audit_teacher_gradients import gradient_report


def test_horizon_mask_rejects_every_window_crossing_done():
    starts = torch.zeros(2, 10, dtype=torch.bool)
    dones = torch.zeros_like(starts)
    dones[0, 5] = True
    valid = _valid_horizon(starts, dones, 4)
    assert valid.shape == (2, 7)
    assert not valid[0, 2:6].any()
    assert valid[1].all()


def test_constant_velocity_uses_teacher_normalization_and_body_rotation():
    dof = 2
    core = torch.zeros(1, 9 + 2 * dof)
    core[..., 6] = 1.0
    core[..., 5] = 0.1  # normalized 1 rad/s yaw
    core[..., 9 + dof:] = 0.2  # normalized 2 rad/s joint speed
    predicted = constant_velocity_core(core, horizon=5, dt=0.02, dof=dof)
    expected_q = 2.0 * 0.1 / math.pi
    assert torch.allclose(predicted[..., 9:9 + dof],
                          torch.full((1, dof), expected_q), atol=1e-6)
    assert torch.allclose(predicted[..., 6:9].norm(dim=-1), torch.ones(1), atol=1e-6)
    assert not torch.equal(predicted[..., 6:9], core[..., 6:9])


def test_gradient_report_uses_weighted_task_gradients_without_mutation():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    before = parameter.detach().clone()
    report = gradient_report({
        "h1/root_linear_velocity": parameter[0].square(),
        "h1/contact": (parameter[0] - parameter[1]).square(),
        "h4/root_linear_velocity": parameter[1].square(),
    }, [parameter])
    assert report["shared_gradient_norm_pre_clip"]["h1/root_linear_velocity"] == 2.0
    assert report["shared_gradient_norm_pre_clip"]["h1/contact"] == pytest.approx(2**1.5)
    assert report["aggregates"]["contact_vs_one_step_state_cosine"] == pytest.approx(-2**-.5)
    assert torch.equal(parameter, before)

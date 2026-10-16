from pathlib import Path

import torch

from tools.train_g1_teacher_multimotion import (
    future_contact_supervision, make_window, multistep_supervision,
)
from unified_motion.privileged_teacher import (
    GROUPED_PHYSICS_LATENT, PrivilegedTeacherConfig, PrivilegedTransformerTeacher,
)
from unified_motion.robot_contract import get_robot_contract


def test_randomized_physics_parameters_reach_teacher_input_suffix():
    steps = 5
    parameters = torch.tensor([[0.25, -0.75]]).repeat(steps, 1)
    replay = {
        "path": Path("synthetic.pt"),
        "motion_name": "synthetic",
        "source_motion_name": "synthetic",
        "frames": {
            "critic": torch.zeros(steps, 1046),
            "contacts": torch.arange(steps, dtype=torch.float32)[:, None, None].expand(
                -1, 4, 14).clone(),
            "found": torch.zeros(steps, 4, dtype=torch.bool),
            "params": parameters,
            "start": torch.zeros(steps, dtype=torch.bool),
            "action": torch.zeros(steps, 29),
            "core": torch.zeros(steps, 67),
            "next_core": torch.zeros(steps, 67),
        },
    }

    batch = make_window(replay, 1, 3, torch.device("cpu"))

    assert torch.equal(batch["params"], parameters[1:4].unsqueeze(0))
    assert torch.equal(batch["inputs"][..., -2:], batch["params"])
    assert batch["inputs"].shape == (1, 3, 1172)
    # At origin t=1, the input contains contact t-1=0 while supervision is t+1=2.
    assert torch.equal(batch["contact_history"][0, 0], replay["frames"]["contacts"][0])
    assert torch.equal(batch["contact"][0, 0], replay["frames"]["contacts"][2])
    assert not torch.equal(batch["inputs"][0, 0, 1046:1102], batch["contact"][0, 0].flatten())


def test_multistep_supervision_trains_four_and_eight_step_heads():
    torch.manual_seed(7)
    steps = 10
    replay = {
        "path": Path("synthetic.pt"), "motion_name": "synthetic",
        "source_motion_name": "synthetic", "frames": {
            "critic": torch.randn(steps, 1046),
            "contacts": torch.randn(steps, 4, 14),
            "found": torch.zeros(steps, 4, dtype=torch.bool),
            "params": torch.randn(steps, 2),
            "start": torch.zeros(steps, dtype=torch.bool),
            "action": torch.randn(steps, 29),
            "core": torch.randn(steps, 67),
            "next_core": torch.randn(steps, 67),
        },
    }
    batch = make_window(replay, 0, steps, torch.device("cpu"))
    cfg = PrivilegedTeacherConfig(
        objective_version=5, critic_dim=1046, contact_slots=4, object_slots=4,
        width=16, heads=2, layers=1, context_steps=10,
        future_contact_weight=.25, contact_input_lag=1,
        latent_groups=GROUPED_PHYSICS_LATENT)
    teacher = PrivilegedTransformerTeacher(
        get_robot_contract("g1_29dof"), 64, 2, cfg)
    output = teacher(
        batch["inputs"], batch["starts"], batch["actions"], batch["core"])
    (multistep_supervision(teacher, output, batch, cfg)
     + future_contact_supervision(teacher, output, batch, cfg)).backward()
    for horizon in (4, 8):
        grad = teacher.future_transitions[str(horizon)][-1].weight.grad
        assert grad is not None and torch.count_nonzero(grad).item()
        contact_grad = teacher.future_contact_heads[str(horizon)]["force"][-1].weight.grad
        assert contact_grad is not None and torch.count_nonzero(contact_grad).item()

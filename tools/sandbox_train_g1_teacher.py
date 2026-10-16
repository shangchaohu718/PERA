#!/usr/bin/env python3
"""Offline wiring/optimization smoke test for the G1 grouped physics Teacher.

This deliberately does not claim physical training: Isaac Lab is not started
and the fixed supervision batch is synthetic.  The real published HoloMotion
Actor is nevertheless strict-loaded, frozen, and used to produce every action
fed to the Teacher.  The command verifies the stage-1 ownership boundary and
that all four z64 groups can optimize before an expensive simulator run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified_motion.holomotion_original_algorithm import (  # noqa: E402
    flat_observations_to_tensordict,
    load_original_holomotion_algorithm,
)
from unified_motion.privileged_teacher import (  # noqa: E402
    GROUPED_PHYSICS_LATENT,
    PrivilegedTeacherConfig,
    PrivilegedTransformerTeacher,
    teacher_supervision,
)
from unified_motion.robot_contract import get_robot_contract  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holomotion-root", type=Path,
        default=PROJECT_ROOT.parent / "HoloMotion")
    parser.add_argument("--checkpoint-name", default="model_14000")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "g1_teacher_sandbox")
    return parser.parse_args()


def make_batch(actor, *, batch: int, sequence: int, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = batch * sequence
    actor_observation = .15 * torch.randn(count, 604, generator=generator)
    critic_observation = .15 * torch.randn(count, 1046, generator=generator)
    data = flat_observations_to_tensordict(
        actor_observation, critic_observation, dof=29, num_bodies=30)
    with torch.inference_mode():
        actions = actor(data, mode="inference", update_obs_norm=False)["mu"]
    actions = actions.reshape(batch, sequence, 29).detach()
    critic_observation = critic_observation.reshape(batch, sequence, 1046)

    contacts = torch.zeros(batch, sequence, 4, 14)
    contacts[..., :6] = .1 * torch.randn(
        batch, sequence, 4, 6, generator=generator)
    contacts[..., 8] = 1.0
    found = torch.rand(batch, sequence, 4, generator=generator) > .35
    contacts[..., 9] = found * (.1 + .8 * torch.rand(
        batch, sequence, 4, generator=generator))
    contacts[..., 10] = found
    contacts[..., 11] = found * .3 * torch.rand(
        batch, sequence, 4, generator=generator)
    contacts[..., 12] = found * .2 * torch.rand(
        batch, sequence, 4, generator=generator)
    contacts[..., 13] = torch.where(
        found, contacts[..., 11] / contacts[..., 9].clamp_min(.01), 0.)

    # Four absent object slots: 64 values followed by four explicit masks.
    objects_and_masks = torch.zeros(batch, sequence, 68)
    parameters = 2.0 * torch.rand(
        batch, sequence, 2, generator=generator) - 1.0
    privilege = torch.cat((
        contacts.flatten(-2), objects_and_masks, parameters), -1)
    teacher_input = torch.cat((critic_observation, privilege), -1)

    core = .1 * torch.randn(batch, sequence, 67, generator=generator)
    gravity = torch.randn(batch, sequence, 3, generator=generator)
    core[..., 6:9] = F.normalize(gravity, dim=-1)
    delta = torch.zeros_like(core)
    delta[..., :3] = .01 * actions[..., :3]
    delta[..., 3:6] = .01 * actions[..., 3:6]
    delta[..., 9:38] = .02 * actions
    delta[..., 38:67] = .04 * actions + .01 * parameters[..., :1]
    next_core = core + delta
    next_core[..., 6:9] = F.normalize(next_core[..., 6:9], dim=-1)

    contact_physics = contacts[..., (9, 11, 12, 13)]
    contact_modes = torch.where(
        found, torch.where(contacts[..., 12] > .05, 2, 1), 0).long()
    starts = torch.zeros(batch, sequence, dtype=torch.bool)
    starts[:, 0] = True
    return dict(
        teacher_input=teacher_input,
        starts=starts,
        actions=actions,
        core=core,
        next_core=next_core,
        contacts=found.float(),
        contact_mask=torch.ones_like(found),
        contact_physics=contact_physics,
        contact_modes=contact_modes,
        parameters=parameters,
        valid=torch.ones(batch, sequence, dtype=torch.bool),
        core_valid=torch.ones_like(core, dtype=torch.bool),
    )


def actor_versions(actor) -> tuple[int, ...]:
    return tuple(parameter._version for parameter in actor.parameters())


def main() -> None:
    args = arguments()
    if args.steps < 2 or args.batch < 1 or args.sequence < 2:
        raise ValueError("steps>=2, batch>=1 and sequence>=2 are required")
    torch.manual_seed(args.seed)
    root = args.holomotion_root.expanduser().resolve()
    official = load_original_holomotion_algorithm(
        holomotion_root=root,
        run_dir=root / "checkpoints" / "holomotion_v1.4",
        checkpoint_name=args.checkpoint_name,
        device="cpu",
    )
    official.actor.eval().requires_grad_(False)
    before_versions = actor_versions(official.actor)
    batch = make_batch(
        official.actor, batch=args.batch, sequence=args.sequence, seed=args.seed)

    config = PrivilegedTeacherConfig(
        objective_version=5,
        critic_dim=1046,
        contact_slots=4,
        object_slots=4,
        width=128,
        heads=8,
        layers=4,
        context_steps=64,
        train_envs=args.batch,
    )
    teacher = PrivilegedTransformerTeacher(
        get_robot_contract("g1_29dof"), 64, 2, config,
        latent_groups=GROUPED_PHYSICS_LATENT,
    ).train()
    optimizer = torch.optim.Adam(
        teacher.parameters(), lr=args.learning_rate)
    losses = []
    final_group_grad_norms = {}
    for step in range(args.steps):
        output = teacher(
            batch["teacher_input"], batch["starts"],
            batch["actions"], batch["core"])
        loss = teacher_supervision(
            output,
            batch["next_core"],
            batch["core_valid"],
            batch["contacts"],
            batch["contact_mask"],
            batch["parameters"],
            batch["valid"],
            config,
            current_core=batch["core"],
            contact_physics=batch["contact_physics"],
            contact_modes=batch["contact_modes"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == args.steps - 1:
            final_group_grad_norms = {
                name: float(head[1].weight.grad.norm())
                for name, head in teacher.latent_head.items()
            }
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    after_versions = actor_versions(official.actor)
    actor_frozen = (
        before_versions == after_versions
        and all(parameter.grad is None for parameter in official.actor.parameters())
        and all(not parameter.requires_grad for parameter in official.actor.parameters())
    )
    finite_gradients = (
        set(final_group_grad_norms) == {name for name, _ in GROUPED_PHYSICS_LATENT}
        and all(value > 0 and torch.isfinite(torch.tensor(value)).item()
                for value in final_group_grad_norms.values())
    )
    loss_decreased = losses[-1] < losses[0]
    if not actor_frozen or not finite_gradients or not loss_decreased:
        raise RuntimeError(
            "sandbox invariant failed: Actor must remain frozen, all groups "
            "must receive gradients, and the fixed-batch loss must decrease")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "teacher_grouped_z64_sandbox.pt"
    torch.save({
        "schema": 1,
        "sandbox_only": True,
        "official_checkpoint": args.checkpoint_name,
        "teacher_config": config.__dict__,
        "latent_groups": GROUPED_PHYSICS_LATENT,
        "model": teacher.state_dict(),
    }, checkpoint_path)
    report = {
        "status": "passed",
        "sandbox_only": True,
        "official_checkpoint": str(
            root / "checkpoints" / "holomotion_v1.4" / f"{args.checkpoint_name}.pt"),
        "steps": args.steps,
        "batch": args.batch,
        "sequence": args.sequence,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "relative_loss_change": losses[-1] / losses[0] - 1.0,
        "actor_frozen": actor_frozen,
        "group_grad_norms": final_group_grad_norms,
        "checkpoint": str(checkpoint_path),
        "limitation": "synthetic fixed batch; not an Isaac physics result",
    }
    report_path = output_dir / "short_run.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

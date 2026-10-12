"""CPU counterexamples for the current Physics algorithm, not robot-skill tests.

No production parameters/files are changed. Probes call the checked-out training
helpers and compare goal-blind/conditioned residuals on a matched toy task.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import math
from pathlib import Path
import platform
import statistics
import time
from types import SimpleNamespace

import torch

from unified_motion.algorithms.dynamics_history import DynamicsHistory, snapshot_physics
from unified_motion.algorithms.common import diagonal_gaussian_kl
from unified_motion.algorithms.dynamics_objective import identification_loss
from unified_motion.physics_confidence import PhysicsConfidence
from unified_motion.physics_update import PhysicsUpdateConfig
from unified_motion.envs.mjlab_physics import K1PhysicsBridge
from unified_motion.holomotion_physics_actor import HoloMotionPhysicsActorConfig, PhysicsActionResidual
from unified_motion.holomotion_physics_runtime import physics_policy_inputs
from unified_motion.models.physics import DynamicsModel, PhysicsObservation, _core_state
from unified_motion.robot_contract import get_robot_contract


def model(*, default_width=False):
    return DynamicsModel(get_robot_contract("k1_22dof"),
        physics_width=64 if default_width else 8, memory_width=64 if default_width else 8,
        latent_width=8 if default_width else 4, hidden_dim=128 if default_width else 16,
        system_id_dim=2, teacher_student=True, uncertainty=True).eval()


def frame(batch=1):
    body = torch.zeros(batch, 72)
    body[:, 5] = -1
    contact = torch.zeros(batch, 2, 12)
    return PhysicsObservation(body, torch.zeros(batch, 3), contact,
        torch.ones(batch, 2, dtype=torch.bool), torch.zeros(batch, 0, 16),
        torch.zeros(batch, 0, dtype=torch.bool))


def missing_velocity_innovation():
    """Call the actual deployable adapter on an ideal translating body."""
    bridge = K1PhysicsBridge.__new__(K1PhysicsBridge)
    bridge.num_envs, bridge.device, bridge.foot_body_ids = 1, torch.device("cpu"), [0, 1]
    body = frame().body_obs
    bridge.bridge = SimpleNamespace(observation=lambda: body)
    root_velocity = torch.tensor([[1., 0., 0., 0., 0., 0.]])
    bridge.entity = SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=torch.zeros(1, 3), root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]),
        root_link_vel_w=root_velocity,
        body_link_pose_w=torch.tensor([[[0., .1, -.5, 1., 0., 0., 0.],
                                       [0., -.1, -.5, 1., 0., 0., 0.]]]),
        body_link_vel_w=root_velocity[:, None].expand(-1, 2, -1)))
    observed = bridge.read_deployable()
    truth = snapshot_physics(observed)
    truth.base_lin_vel_b[:, 0] = 1.
    net = model()
    with torch.no_grad():
        for p in net.state_predictor.parameters():
            p.zero_()
        net.state_predictor[-1].bias[0] = 1.
        enc, memory = net.step(observed, episode_start=torch.ones(1, dtype=torch.bool), env_ids=(0,))
        prediction = net.predict_next(observed, torch.zeros(1, 22), enc.latent,
                                       physics_encoding=enc.physics_encoding())
        memory = net.remember_prediction(memory, prediction)
        next_enc, _ = net.step(observed, episode_start=torch.zeros(1, dtype=torch.bool),
                               env_ids=(0,), memory=memory)
    return dict(true_velocity=truth.base_lin_vel_b.tolist(),
        deployable_velocity=observed.base_lin_vel_b.tolist(),
        prediction_squared_error=float((prediction.next_core-_core_state(truth, net.robot)).square().sum()),
        reported_innovation=next_enc.prediction_error[:, :3].tolist())


def confidence_without_information():
    net = model()
    obs = frame(1024)
    starts = torch.ones(1024, dtype=torch.bool)
    with torch.no_grad():
        latent = net.step(obs, episode_start=starts, env_ids=tuple(range(1024)))[0].latent
    grid = (torch.arange(1024)+.5)/512.-1.
    target = torch.stack((grid, grid[torch.randperm(1024)]), dim=-1)
    optimizer = torch.optim.Adam(list(net.system_id_head.parameters())+
        list(net.system_id_logvar_head.parameters()), lr=.03)
    origin = "unified_motion/algorithms/dynamics_objective.py:identification_loss"
    for _ in range(400):
        terms, _, _ = identification_loss(net, latent, target, starts)
        loss = terms["system_identification"]
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        identified, logvar = net.identify_privileged_distribution(latent)
        calibration = PhysicsConfidence(2)
        calibration.update(identified, logvar, target, starts)
        gate = {}
        for age in (0, 31, 127):
            _, _, _, confidence = physics_policy_inputs(net, obs, starts, tuple(range(1024)),
                None, torch.full((1024,), age, dtype=torch.long), 128, calibration)
            gate[str(age)] = float(confidence.mean())
    return dict(origin=origin, identical_inputs=True, target_variance=float(target.var(unbiased=False)),
        mse=float((identified-target).square().mean()), estimated_variance=float(logvar.exp().mean()),
        confidence_by_evidence_steps=gate, legacy_prior_only_gate=math.exp(-1/6),
        note="Age is held fixed to isolate the gate; all labels share identical observed features.")


def detached_kl_gradient():
    origin = "unified_motion/algorithms/common.py:diagonal_gaussian_kl"
    mean = torch.ones(1, dtype=torch.float64, requires_grad=True)
    def value(mu):
        return diagonal_gaussian_kl(torch.zeros_like(mu), torch.ones_like(mu), mu, torch.ones_like(mu))
    current = value(mean)
    reported_gradient = torch.autograd.grad(current, mean)[0]
    eps = 1e-4
    finite_difference = (value(mean.detach()+eps)-value(mean.detach()-eps))/(2*eps)
    return dict(origin=origin, coefficient_default=0., kl_value=float(current.detach()),
        code_gradient=float(reported_gradient.item()),
        scalar_finite_difference=float(finite_difference), analytic_gradient=1.)


def residual_reference_blindness():
    results = []
    for seed in (11, 22, 33):
        row = dict(seed=seed)
        for conditioned in (False, True):
            torch.manual_seed(seed)
            residual = PhysicsActionResidual(HoloMotionPhysicsActorConfig(
                physics_feature_dim=28, hidden_dim=16, condition_on_base_action=conditioned))
            features = torch.ones(64, 28)
            commands = torch.tensor([.5, -.5]).repeat(32)[:, None].expand(-1, 22)
            target = commands*.4
            optimizer = torch.optim.Adam(residual.parameters(), lr=.01)
            for _ in range(200):
                loss = (residual(features, commands)-target).square().mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            with torch.no_grad():
                row["conditioned_mse" if conditioned else "blind_mse"] = float(
                    (residual(features, commands)-target).square().mean())
        results.append(row)
    return dict(three_seed_matched_toy_fit=results,
        note="Same-state opposite-goal supervised task; this tests expressivity, not robot skill.")


def teacher_only_gradients():
    net = model()
    target = torch.rand(64, 2)*2-1
    teacher = net.encode_privileged_teacher(target)
    reconstruction = (net.reconstruct_privileged(teacher)-target).square().mean()
    reconstruction.backward()
    groups = {}
    for name, parameter in net.named_parameters():
        if parameter.grad is not None and parameter.grad.abs().sum() > 0:
            group = name.split(".")[0]
            groups[group] = groups.get(group, 0) + parameter.numel()
    return dict(nonzero_gradient_groups=groups,
        note="Teacher coordinates optimize parameter reconstruction; no reward or action objective enters this path.")


def replay_cost():
    net = model(default_width=True)
    batch = 8
    obs = frame(batch)
    action = torch.zeros(batch, 22)
    rows = []
    for length in (32, 128, 512):
        history = DynamicsHistory(max_steps=length)
        for i in range(length):
            history.append(obs, torch.full((batch,), i == 0), action)
        history.rebuild(net, tuple(range(batch)))
        elapsed = []
        for _ in range(3):
            start = time.perf_counter()
            history.rebuild(net, tuple(range(batch)))
            elapsed.append(time.perf_counter()-start)
        rows.append(dict(history_steps=length, batch=batch,
                         rebuild_median_seconds=statistics.median(elapsed)))
    per_row_bytes = sum(getattr(obs, f.name)[0].numel()*getattr(obs, f.name).element_size()
                        for f in fields(obs)) + action[0].numel()*action.element_size()+1
    return dict(cpu_benchmarks=rows, raw_history_bytes_per_env_step=per_row_bytes,
        raw_history_gib_4096env_1000steps=per_row_bytes*4096*1000/(2**30),
        raw_history_gib_4096env_16384steps=per_row_bytes*4096*16384/(2**30),
        note="Storage arithmetic, not measured GPU memory; excludes tensor objects, rollouts, activations, models.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261012)
    report = dict(scope="CPU algorithm counterexamples; no real simulation or trained checkpoint",
        python=platform.python_version(), torch=torch.__version__, seed=20261012,
        missing_velocity_innovation=missing_velocity_innovation(),
        confidence_without_information=confidence_without_information(),
        detached_kl_gradient=detached_kl_gradient(),
        residual_reference_blindness=residual_reference_blindness(),
        teacher_only_gradients=teacher_only_gradients(), replay_cost=replay_cost(),
        default_schedule=dict(first_dynamics_update_step=192, minimum_episode_age_for_full_factor=128,
            gate_requires_heldout_calibration=True,
            ema_half_life_sim_steps=PhysicsUpdateConfig().ema_half_life_steps))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

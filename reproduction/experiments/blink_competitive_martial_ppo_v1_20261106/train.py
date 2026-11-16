"""Isolated Isaac PPO: frozen G1 eight-expert Actor + one competing expert.

The original martial Isaac reward functions/configuration are reused verbatim.
There is no world model, teacher, forced Top-k admission, or public update.
"""

import argparse
import json
import sys
import math
import os
import signal
import time
from pathlib import Path
from experiments.blink_competitive_martial_ppo_v1_20261106.common import (
    BASE,
    BASE_SHA,
    REPORT,
    ROOT,
    bootstrap,
    motion_library,
    sha,
    write_json,
)

bootstrap()
import torch
from torch import nn
from omegaconf import OmegaConf
from experiments.blink_competitive_martial_ppo_v1_20261106.routing import (
    FORMAT,
    load_base,
    public_digest,
    features,
    initialize,
)
from blinkmimic.mdp import MDP
from blinkmimic.rotations import quat_to_mat


def logprob(action, mean, std=0.1):
    return (
        -0.5 * ((action - mean) / std).square() - math.log(std) - 0.5 * math.log(2 * math.pi)
    ).sum(-1)


def cached_prediction(pi, cache, batch_size):
    # The public policy uses bf16. Keep the exact rollout matrix shape for
    # both the new FFN and frozen output head: changing GEMM shape can change
    # bf16 rounding enough to produce a false PPO likelihood ratio.
    n = len(cache["x"])
    assert n % batch_size == 0
    return torch.cat(
        [
            pi.moe.predict({k: v[i : i + batch_size] for k, v in cache.items()}, pi.dec[-1])[0]
            for i in range(0, n, batch_size)
        ]
    )


class Bridge:
    def __init__(self, env, pi, robot, motion, kin):
        self.env, self.pi, self.motion = env, pi, motion
        self.cmd = env._env.command_manager.get_term("ref_motion")
        self.robot = self.cmd.robot
        self.to_sdk = torch.tensor(
            [self.robot.joint_names.index(n) for n in robot.joint_names], device=env.device
        )
        self.to_sim = torch.tensor(
            [robot.joint_names.index(n) for n in self.robot.joint_names], device=env.device
        )
        batch = self.cmd._motion_cache.current_batch
        keys = list(batch.raw_motion_keys)
        assert sorted(keys) == sorted(motion.names), keys
        self.cache_to_clip = torch.tensor([motion.names.index(k) for k in keys], device=env.device)
        self.mdp = MDP(kin, motion, pi.ref_q0, 30)
        self.zero_xi = torch.zeros(env.num_envs, 30, device=env.device)
        self.previous = torch.zeros(env.num_envs, 29, device=env.device)
        self.history = None
        term = env._env.action_manager.get_term("dof_pos")
        self.action_term = term
        assert term.action_dim == 29
        # Explicitly match native q0 + alpha*a joint targets through the Isaac
        # action manager, with exact SDK<->Isaac permutations.
        self.alpha = pi.ref_alpha
        assert torch.allclose(
            torch.tensor(robot.default_joint_pos_list(), device=env.device), pi.ref_q0
        )
        assert torch.allclose(
            torch.tensor(robot.action_scale_list(), device=env.device), self.alpha
        )

    def frame(self, offset=0):
        clip = self.cache_to_clip[self.cmd._clip_indices]
        local = self.cmd._frame_indices + offset
        return self.motion.clip_start[clip] + torch.minimum(local, self.motion.clip_len[clip] - 1)

    def state(self):
        d = self.robot.data
        return (
            d.root_link_pos_w - self.env._env.scene.env_origins,
            quat_to_mat(d.root_link_quat_w),
            d.root_link_lin_vel_w,
            d.root_link_ang_vel_w,
            d.joint_pos[:, self.to_sdk],
            d.joint_vel[:, self.to_sdk],
        )

    def observation(self, offset=0):
        f = self.frame(offset)
        obs = self.mdp.obs(self.state(), self.previous, self.zero_xi, f)
        return obs, self.motion.goal(f, 10)

    def critic_input(self, obs, goal):
        return torch.cat((self.pi.obs_norm(obs), self.pi.goal_norm(goal)), -1).clamp(-10, 10)

    def push(self, obs, done=None):
        repeated = obs[:, None].expand(-1, self.pi.hist, -1)
        if self.history is None:
            self.history = repeated.clone()
        else:
            self.history = torch.cat((self.history[:, 1:], obs[:, None]), 1)
            if done is not None:
                self.history[done] = repeated[done]

    def action(self, native_action):
        self.previous = native_action.detach().clone()
        target = (self.pi.ref_q0 + self.alpha * native_action)[:, self.to_sim]
        return (target - self.action_term._offset) / self.action_term._scale

    def validate(self):
        f = self.frame()
        q = self.cmd.get_ref_motion_dof_pos_cur()[:, self.to_sdk]
        p = self.cmd.get_ref_motion_root_global_pos_cur() - self.env._env.scene.env_origins
        R = quat_to_mat(self.cmd.get_ref_motion_root_global_rot_quat_wxyz_cur())
        errors = dict(
            reference_q_max_abs=float((q - self.motion.q[f]).abs().max()),
            reference_p_max_abs=float((p - self.motion.root_p[f]).abs().max()),
            reference_R_max_abs=float((R - self.motion.root_R[f]).abs().max()),
        )
        assert max(errors.values()) < 1e-4, errors
        errors.update(
            sdk_joint_names=[self.robot.joint_names[int(i)] for i in self.to_sdk],
            fk_body_names=list(self.mdp.kin.body_names),
            sdk_from_sim=self.to_sdk.tolist(),
            sim_from_sdk=self.to_sim.tolist(),
            cache_keys=list(self.cmd._motion_cache.current_batch.raw_motion_keys),
        )
        return errors


@torch.no_grad()
def initialize_domain(pi, bridge, run):
    motion = bridge.motion
    banks, old = [], []
    for f in torch.arange(motion.num_frames, device=bridge.env.device).split(128):
        obs = bridge.mdp.obs(
            motion.ref_state(f),
            torch.zeros(len(f), 29, device=f.device),
            torch.zeros(len(f), 30, device=f.device),
            f,
        )
        goal = motion.goal(f, 10)
        pi.moe.domain_enabled = False
        pi(obs[:, None].expand(-1, pi.hist, -1), goal)
        banks.append(features(pi, goal))
        old.append(pi.moe.last_old_indices)
    bank = {k: torch.cat([v[k] for v in banks]) for k in banks[0]}
    old = torch.cat(old)
    trials = []
    for target in (0.4, 0.6, 0.8, 0.95):
        result = initialize(pi, bank, old, target=target)
        _, selected, weight, _ = pi.moe.compete(bank)
        per_clip = {}
        for i, name in enumerate(motion.names):
            mask = motion.clip_id == i
            per_clip[name] = dict(
                admission=float(selected[mask].float().mean()),
                contribution=float(weight[mask].mean()),
            )
        result["per_clip_reference_probe"] = per_clip
        trials.append(result)
        if min(x["admission"] for x in per_clip.values()) >= 0.1:
            break
    else:
        raise RuntimeError("Some martial clips never admit the initialized expert: " + str(trials))
    pi.moe.domain_enabled = True
    result["calibration_trials"] = trials[:-1]
    write_json(run / "initialization.json", result)
    return result


def parity_audit(pi, history, goal, cached, action):
    """Compare complete/cached Actor outputs, PPO likelihood and gradients."""
    h, g, a = history, goal, action
    f = cached
    full = pi(h, g)
    short = pi.moe.predict(f, pi.dec[-1])[0]
    torch.testing.assert_close(full, short, atol=2e-4, rtol=2e-4)
    old = logprob(a, full.detach())
    advantage = torch.linspace(-1, 1, len(a), device=a.device)
    parameters = list(pi.moe.plugin.parameters())

    def objective(mean):
        ratio = (logprob(a, mean) - old).exp()
        return -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()

    full_loss, cached_loss = objective(full), objective(short)
    fg = torch.autograd.grad(full_loss, parameters)
    cg = torch.autograd.grad(cached_loss, parameters)
    errors = {}
    for (name, _), x, y in zip(pi.moe.plugin.named_parameters(), fg, cg):
        torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-3)
        errors[name] = float((x - y).abs().max())
    bias = pi.moe.plugin.selection_bias.clone()
    with torch.no_grad():
        pi.moe.plugin.selection_bias.fill_(-1e9)
        rejected = pi(h, g)
        pi.moe.domain_enabled = False
        base = pi(h, g)
        pi.moe.domain_enabled = True
        pi.moe.plugin.selection_bias.copy_(bias)
    assert torch.equal(rejected, base), "Rejected expert changed the frozen base output"
    return dict(
        mean_max_error=float((full - short).abs().max()),
        logprob_max_error=float((logprob(a, full) - logprob(a, short)).abs().max()),
        objective_error=float((full_loss - cached_loss).abs()),
        gradient_max_errors=errors,
        rejected_expert_exact_base_parity=True,
    )


@torch.no_grad()
def evaluate(bridge, critic, mode, run, iteration):
    """Each of the seven whole clips, from frame zero, deterministic actions."""
    env, cmd, m, pi = bridge.env, bridge.cmd, bridge.motion, bridge.pi
    env.reset_all()
    ids = torch.arange(env.num_envs, device=env.device)
    inverse = {int(v): i for i, v in enumerate(bridge.cache_to_clip.tolist())}
    clips = ids % len(m.names)
    cmd._clip_indices[:] = torch.tensor([inverse[int(x)] for x in clips], device=env.device)
    cmd._frame_indices.zero_()
    cmd._start_frame_indices.zero_()
    cmd._reference_observation_cache = {}
    cmd.force_realign_offline_eval_no_perturb(ids)
    env._env.scene.write_data_to_sim()
    env._env.sim.forward()
    bridge.previous.zero_()
    bridge.history = None
    pi.moe.domain_enabled = mode != "base"
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    sums = torch.zeros(env.num_envs, 5, device=env.device)
    lengths = torch.zeros(env.num_envs, device=env.device)
    successes = torch.zeros(env.num_envs, device=env.device)
    for _ in range(int(m.clip_len.max())):
        obs, goal = bridge.observation()
        bridge.push(obs)
        mean = pi(bridge.history, goal)
        f = bridge.frame()
        actual = (
            bridge.robot.data.body_pos_w[:, cmd.sim2urdf_body_idx]
            - env._env.scene.env_origins[:, None]
        )
        reference = (
            cmd.get_ref_motion_bodylink_global_pos_cur()[:, cmd.sim2urdf_body_idx]
            - env._env.scene.env_origins[:, None]
        )
        error = (actual - reference).norm(dim=-1).mean(-1)
        stats = torch.stack(
            (
                torch.zeros_like(error),
                error,
                pi.moe.last_selected[:, 0].float(),
                pi.moe.last_weight[:, 0],
                torch.zeros_like(error),
            ),
            -1,
        )
        _, reward, done, timeout, _ = env.step(bridge.action(mean))
        stats[:, 0] = reward
        stats[:, 4] = (~timeout & done).float()
        sums += stats * active[:, None]
        lengths += active
        successes += (active & done & timeout).float()
        active &= ~done
        bridge.previous[done] = 0
        if not bool(active.any()):
            break
    records = {}
    for i, name in enumerate(m.names):
        mask = clips == i
        records[name] = dict(
            environments=int(mask.sum()),
            success=float(successes[mask].mean()),
            mean_lifetime_steps=float(lengths[mask].mean()),
            mean_reward=float((sums[mask, 0] / lengths[mask].clamp_min(1)).mean()),
            body_position_error_m=float((sums[mask, 1] / lengths[mask].clamp_min(1)).mean()),
            admission=float((sums[mask, 2] / lengths[mask].clamp_min(1)).mean()),
            contribution=float((sums[mask, 3] / lengths[mask].clamp_min(1)).mean()),
        )
    report = dict(
        iteration=iteration, mode=mode, phase_zero=True, training_clips=True, clips=records
    )
    write_json(run / f"eval_{mode}_{iteration:06d}.json", report)
    print(json.dumps({"evaluation": report}), flush=True)
    pi.moe.domain_enabled = True
    env.reset_all()
    bridge.previous.zero_()
    bridge.history = None
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--run-name", default="training")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--resume")
    args = p.parse_args()
    assert args.num_envs >= 8 and args.iterations > 0
    run = REPORT / args.run_name
    run.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    from experiments.blink_competitive_martial_ppo_v1_20261106.prepare import prepare

    c = OmegaConf.load(prepare())
    c.num_envs = args.num_envs
    c.env.config.num_envs = args.num_envs
    for motion_config in (
        c.robot.motion,
        c.env.config.robot.motion,
        c.env.config.commands.ref_motion.params.motion_lib_cfg,
    ):
        motion_config.dataloader.timeout = 0
        motion_config.world_frame_normalization = True
    # The MLP's actual 608/730 observations are built by Bridge. Avoid also
    # evaluating the unused legacy Transformer observation/critic groups.
    minimal_groups = dict(unified=dict(atomic_obs_list=[dict(actor_dof_pos=dict(func="dof_pos"))]))
    c.obs.obs_groups = minimal_groups
    c.env.config.obs.obs_groups = minimal_groups
    c.env.config.eval_log_dir = str(run / "eval_logs")
    OmegaConf.save(c, run / "config.yaml")
    write_json(
        run / "status.json", dict(stage="starting", pid=os.getpid(), target_rounds=args.iterations)
    )
    assert sha(BASE) == BASE_SHA
    pi, base_args, robot = load_base(BASE, "cuda:0")
    frozen_hash = public_digest(pi)
    m, kin, manifest = motion_library(robot, "cuda:0")
    # Start Isaac only after resolving all model/data contracts.
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        headless=True,
        enable_cameras=False,
        device="cuda:0",
        multi_gpu=False,
        kit_args="--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1",
    )
    os.chdir(ROOT / "integrations/HoloMotion")
    sys.path.insert(0, str(ROOT / "reproduction"))
    from roundhouse_impl import register_terms

    register_terms()
    from holomotion.src.env.motion_tracking import MotionTrackingEnv

    env = MotionTrackingEnv(c.env.config, device="cuda:0", log_dir=str(run), headless=True)
    env.reset_all()
    bridge = Bridge(env, pi, robot, m, kin)
    io_audit = bridge.validate()
    if args.resume:
        initialization = dict(
            method="resume_trained_expert_without_reinitialization",
            checkpoint=str(Path(args.resume).resolve()),
            checkpoint_sha256=sha(args.resume),
        )
        write_json(run / "initialization.json", initialization)
    else:
        initialization = initialize_domain(pi, bridge, run)
    parameters = list(pi.moe.plugin.parameters())
    assert len(parameters) == 4
    assert all(n.startswith("moe.plugin.") for n, v in pi.named_parameters() if v.requires_grad)
    critic = nn.Sequential(
        nn.Linear(1338, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 1)
    ).to(env.device)
    # obs608 + future730 =1338; critic normalization is frozen public input statistics.
    actor_opt = torch.optim.AdamW(parameters, lr=3e-5)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-4)
    start = 0
    actor_steps = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=env.device, weights_only=True)
        assert saved["format"] == FORMAT and saved["public_sha256"] == frozen_hash
        pi.moe.plugin.load_state_dict(saved["plugin"])
        critic.load_state_dict(saved["critic"])
        actor_opt.load_state_dict(saved["actor_optimizer"])
        critic_opt.load_state_dict(saved["critic_optimizer"])
        start, actor_steps = saved["iteration"], saved["actor_adam_updates"]
        assert saved["reference_names"] == m.names
        assert actor_steps == start * 24
        write_json(
            run / "resume.json",
            dict(
                source_checkpoint=str(Path(args.resume).resolve()),
                source_checkpoint_sha256=sha(args.resume),
                start_iteration=start,
                appended_iterations=args.iterations,
                target_iteration=start + args.iterations,
                initial_actor_adam_updates=actor_steps,
                actor_optimizer_restored=True,
                critic_optimizer_restored=True,
                expert_reinitialized=False,
                simulator_restarted=True,
                rng_restarted_seed=42,
            ),
        )
        c.algo.config.num_learning_iterations = start + args.iterations
        OmegaConf.save(c, run / "config.yaml")

    def save(iteration, filename="latest.pt"):
        assert public_digest(pi) == frozen_hash
        steps = [int(actor_opt.state[x].get("step", 0)) for x in parameters]
        assert len(set(steps)) == 1 and steps[0] == actor_steps, steps
        target = run / filename
        temporary = target.with_suffix(".pt.tmp")
        torch.save(
            dict(
                format=FORMAT,
                public_sha256=frozen_hash,
                base_checkpoint=str(BASE),
                base_checkpoint_sha256=BASE_SHA,
                iteration=iteration,
                actor_adam_updates=actor_steps,
                plugin=pi.moe.plugin.state_dict(),
                critic=critic.state_dict(),
                actor_optimizer=actor_opt.state_dict(),
                critic_optimizer=critic_opt.state_dict(),
                reference_names=m.names,
                method="PPO",
                world_model=False,
                gaussian_std=0.1,
            ),
            temporary,
        )
        os.replace(temporary, target)

    save(start, "initialized.pt")
    write_json(
        run / "freeze_preflight.json",
        dict(
            public_sha256=frozen_hash,
            trainable_names=[n for n, v in pi.named_parameters() if v.requires_grad],
            trainable_actor_parameters=sum(x.numel() for x in parameters),
            io_audit=io_audit,
            reward_config_unchanged=True,
            initialization=initialization,
            no_world_model=True,
        ),
    )
    if not args.skip_eval:
        evaluate(bridge, critic, "base", run, start)
        evaluate(bridge, critic, "initialized", run, start)
    interrupted = [False]

    def request_stop(*_):
        interrupted[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    terminal_values = torch.zeros(env.num_envs, device=env.device)
    original_reward_compute = env._env.reward_manager.compute

    @torch.no_grad()
    def reward_compute(dt):
        reward = original_reward_compute(dt)
        terminal_values.zero_()
        timeout = env._env.reset_time_outs & ~env._env.reset_terminated
        if bool(timeout.any()):
            obs, goal = bridge.observation(offset=1)
            terminal_values[timeout] = critic(bridge.critic_input(obs, goal))[timeout, 0]
        return reward

    env._env.reward_manager.compute = reward_compute
    begin = time.monotonic()
    actor_audited = False
    for iteration in range(start + 1, start + args.iterations + 1):
        storage = []
        for t in range(32):
            with torch.no_grad():
                obs, goal = bridge.observation()
                fresh = env._env.episode_length_buf == 0
                bridge.push(obs, fresh)
                mean = pi(bridge.history, goal)
                cache = features(pi, goal)
                ci = bridge.critic_input(obs, goal)
                value = critic(ci)[:, 0]
                action = mean + 0.1 * torch.randn_like(mean)
                lp = logprob(action, mean)
                admission = pi.moe.last_selected[:, 0].float()
                contribution = pi.moe.last_weight[:, 0]
                if not actor_audited:
                    audit_history, audit_goal = bridge.history.clone(), goal.clone()
                clip = m.clip_id[bridge.frame()]
                _, reward, done, timeout, infos = env.step(bridge.action(action))
                reward = reward.clone()
                done = done.clone()
                timeout = timeout.clone()
                bootstrap_reward = reward + 0.99 * terminal_values
                bridge.previous[done] = 0
            if not actor_audited:
                write_json(
                    run / "cached_full_parity.json",
                    parity_audit(pi, audit_history, audit_goal, cache, action),
                )
                actor_audited = True
            storage.append(
                dict(
                    cache=cache,
                    ci=ci,
                    action=action,
                    logprob=lp,
                    value=value,
                    reward=reward,
                    gae_reward=bootstrap_reward,
                    done=done,
                    timeout=timeout,
                    admission=admission,
                    contribution=contribution,
                    clip=clip,
                )
            )
        with torch.no_grad():
            obs, goal = bridge.observation()
            last_value = critic(bridge.critic_input(obs, goal))[:, 0]
            gae = torch.zeros_like(last_value)
            advantages = []
            for sample in reversed(storage):
                live = (~sample["done"]).float()
                delta = sample["gae_reward"] + 0.99 * last_value * live - sample["value"]
                gae = delta + 0.99 * 0.95 * live * gae
                advantages.append(gae)
                last_value = sample["value"]
            advantage = torch.stack(advantages[::-1]).flatten()
            returns = advantage + torch.stack([s["value"] for s in storage]).flatten()
            advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-8)
            flat = {
                k: torch.stack([s[k] for s in storage]).flatten(0, 1)
                for k in ("ci", "action", "logprob", "value")
            }
            caches = {
                k: torch.stack([s["cache"][k] for s in storage]).flatten(0, 1)
                for k in storage[0]["cache"]
            }
        gradients = {n: 0.0 for n, _ in pi.moe.plugin.named_parameters()}
        losses, kls = [], []
        count = len(advantage)
        for epoch in range(3):
            order = torch.randperm(count, device=env.device)
            for ids in order.tensor_split(8):
                f = {k: v[ids] for k, v in caches.items()}
                prediction = cached_prediction(pi, f, args.num_envs)
                new_lp = logprob(flat["action"][ids], prediction)
                if actor_steps == start * 24:
                    initial_error = float((new_lp.detach() - flat["logprob"][ids]).abs().max())
                    write_json(
                        run / "initial_minibatch_likelihood_parity.json",
                        dict(
                            max_logprob_error=initial_error,
                            rollout_batch=args.num_envs,
                            minibatch=len(ids),
                            fixed_gemm_chunk=args.num_envs,
                        ),
                    )
                    assert initial_error < 2e-3, (
                        "PPO starts with incorrect likelihood ratio: " + str(initial_error)
                    )
                ratio = (new_lp - flat["logprob"][ids]).exp()
                loss = -torch.minimum(
                    ratio * advantage[ids], ratio.clamp(0.8, 1.2) * advantage[ids]
                ).mean()
                assert torch.isfinite(loss), "Nonfinite PPO Actor objective"
                actor_opt.zero_grad(set_to_none=True)
                loss.backward()
                for name, parameter in pi.moe.plugin.named_parameters():
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                    gradients[name] = max(gradients[name], float(parameter.grad.norm()))
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                actor_opt.step()
                actor_steps += 1
                # Project the nonpositive bias rather than leaving a raw
                # positive parameter permanently behind clamp's zero gradient.
                with torch.no_grad():
                    pi.moe.plugin.router.bias.clamp_(max=0)
                assert all(
                    v.grad is None
                    for n, v in pi.named_parameters()
                    if not n.startswith("moe.plugin.")
                )
                v = critic(flat["ci"][ids])[:, 0]
                value_loss = (v - returns[ids]).square().mean()
                assert torch.isfinite(value_loss)
                critic_opt.zero_grad(set_to_none=True)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_opt.step()
                losses.append(float(loss.detach()))
                kls.append(float(((ratio - 1) - (new_lp - flat["logprob"][ids])).mean().detach()))
        assert all(torch.isfinite(v).all() for v in parameters)
        record = dict(
            iteration=iteration,
            target_iteration=start + args.iterations,
            actor_adam_updates=actor_steps,
            critic_adam_updates=actor_steps,
            physical_transitions=iteration * 32 * args.num_envs,
            mean_reward=float(torch.stack([s["reward"] for s in storage]).mean()),
            done_fraction=float(torch.stack([s["done"] for s in storage]).float().mean()),
            admission=float(torch.stack([s["admission"] for s in storage]).mean()),
            contribution=float(torch.stack([s["contribution"] for s in storage]).mean()),
            actor_loss=sum(losses) / len(losses),
            approximate_kl=sum(kls) / len(kls),
            gradients=gradients,
            raw_bias=float(pi.moe.plugin.router.bias),
            elapsed_seconds=time.monotonic() - begin,
            peak_torch_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
        )
        for i, name in enumerate(m.names):
            mask = torch.stack([s["clip"] for s in storage]) == i
            if bool(mask.any()):
                record.setdefault("per_clip", {})[name] = dict(
                    samples=int(mask.sum()),
                    admission=float(torch.stack([s["admission"] for s in storage])[mask].mean()),
                    reward=float(torch.stack([s["reward"] for s in storage])[mask].mean()),
                )
        with (run / "metrics.jsonl").open("a") as out:
            out.write(json.dumps(record, allow_nan=False) + "\n")
        if iteration <= start + 2:
            assert record["admission"] > 0, "No new expert admitted in real rollout"
            assert record["contribution"] > 1e-5, (
                "Admitted expert has negligible action contribution"
            )
            assert (
                gradients["gate_up"] > 1e-8
                and gradients["down"] > 1e-8
                and gradients["router.weight"] > 1e-8
            ), gradients
        write_json(run / "status.json", dict(stage="training", pid=os.getpid(), **record))
        if iteration <= start + 2 or iteration % 5 == 0:
            print(json.dumps(record, allow_nan=False), flush=True)
        if iteration % 50 == 0 or iteration == start + args.iterations or interrupted[0]:
            save(iteration)
            save(iteration, f"expert_{iteration:06d}.pt")
        if not args.skip_eval and iteration % 250 == 0:
            env._env.reward_manager.compute = original_reward_compute
            evaluate(bridge, critic, "trained", run, iteration)
            env._env.reward_manager.compute = reward_compute
        if interrupted[0]:
            break
    env._env.reward_manager.compute = original_reward_compute
    if not args.skip_eval and iteration % 250 != 0:
        evaluate(bridge, critic, "trained", run, iteration)
    save(iteration)
    write_json(
        run / "status.json",
        dict(
            stage="stopped" if interrupted[0] else "complete",
            pid=os.getpid(),
            iteration=iteration,
            actor_adam_updates=actor_steps,
            public_sha256=public_digest(pi),
            expected_public_sha256=frozen_hash,
            elapsed_seconds=time.monotonic() - begin,
        ),
    )
    env._env.close()
    launcher.app.close()


if __name__ == "__main__":
    main()

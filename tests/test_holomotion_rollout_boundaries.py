"""Numerical and task-boundary regressions; no simulator or checkpoint required."""
from types import SimpleNamespace

import pytest
import torch

from unified_motion.holomotion_mjlab_ppo import (
    HoloMotionMJLabPPO, HoloMotionPPOConfig, generalized_advantage_estimate,
)
from unified_motion.holomotion_mjlab_task import HoloMotionK1MJLabTask
from unified_motion.holomotion_mjlab_rewards import HoloMotionMJLabState
from unified_motion.holomotion_physics_training import HoloMotionPhysicsMJLabPPO
from unified_motion.models.physics import DynamicsModel, PhysicsObservation
from unified_motion.envs.mjlab_physics import PrivilegedFootContactTruth
from unified_motion.robot_contract import get_robot_contract


def test_timeout_bootstraps_final_value_but_does_not_cross_episode():
    rewards = torch.tensor([[[1.], [2.], [100.]]])
    values = torch.tensor([[[3.], [4.], [50.]]])
    dones = torch.tensor([[[False], [True], [False]]])
    # At t=1 V(final)=10, while the reset episode has V=50.
    advantage, returns = generalized_advantage_estimate(
        rewards, values, dones, torch.tensor([[60.]]), gamma=.9, lam=.8,
        truncated=dones, timeout_values=torch.tensor([[[0.], [10.], [0.]]]),
    )
    # delta_1=2+.9*10-4=7; delta_0=1+.9*4-3=1.6.
    expected = torch.tensor([[[1.6 + .9*.8*7], [7.], [104.]]])
    torch.testing.assert_close(advantage, expected)
    torch.testing.assert_close(returns, expected + values)


def test_last_step_timeout_and_true_termination_have_different_returns():
    rewards = torch.ones(2, 1, 1)
    dones = torch.ones(2, 1, 1, dtype=torch.bool)
    truncated = torch.tensor([[[True]], [[False]]])
    _, returns = generalized_advantage_estimate(
        rewards, torch.zeros_like(rewards), dones, torch.full((2, 1), 999.),
        gamma=.5, lam=.9, truncated=truncated,
        timeout_values=torch.full_like(rewards, 8.),
    )
    torch.testing.assert_close(returns[:, 0, 0], torch.tensor([5., 1.]))


@pytest.mark.parametrize("invalid", ["missing_values", "not_done", "dtype", "nonfinite"])
def test_gae_rejects_invalid_timeout_contract(invalid):
    rewards = torch.ones(1, 1, 1)
    dones = torch.ones_like(rewards, dtype=torch.bool)
    truncated = dones.clone()
    timeout_values = rewards.clone()
    if invalid == "missing_values":
        timeout_values = None
    elif invalid == "not_done":
        dones.zero_()
    elif invalid == "dtype":
        truncated = truncated.float()
    else:
        timeout_values.fill_(float("nan"))
    with pytest.raises(ValueError):
        generalized_advantage_estimate(
            rewards, rewards, dones, torch.ones(1, 1), gamma=.99, lam=.95,
            truncated=truncated, timeout_values=timeout_values,
        )


def test_singleton_advantage_normalization_is_finite():
    from unified_motion.holomotion_mjlab_ppo import normalize_advantages
    assert torch.equal(normalize_advantages(torch.tensor([[[5.]]])), torch.zeros(1, 1, 1))


class _HistoryActor(torch.nn.Module):
    """History-sensitive actor: sampling and sequence replay have known answers."""
    def reset_kv_cache(self, num_envs, device):
        self.history = torch.zeros(num_envs, device=device)

    def clear_env_cache(self, ids):
        self.history[ids] = 0

    def forward(self, data, *, mode, **kwargs):
        current = data["unified/actor_ref_gravity_projection_cur"][..., 0]
        if mode == "sampling":
            self.history += current
            mu = self.history[:, None].expand(-1, 22).clone()
        else:
            mask = kwargs["attn_mask"].to(current.dtype)
            mu = (mask @ current.unsqueeze(-1)).expand(-1, -1, 22).clone()
        return {"actions": mu, "actions_log_prob": mu[..., :1],
                "mu": mu, "sigma": torch.ones_like(mu)}


class _Critic(torch.nn.Module):
    def forward(self, data, **kwargs):
        return {"values": data["unified/critic_ref_dof_pos_cur"][..., :1]}


class _Task:
    def __init__(self, *, boundaries=False, num_envs=3):
        self.device = torch.device("cpu")
        self.num_envs = num_envs
        self.state = torch.zeros(num_envs)
        self.sampler = SimpleNamespace(clip_ids=torch.arange(num_envs), offsets_s=[.2])
        self.boundaries = boundaries
        self.steps = 0
        self.full_resets = 0
        self.timeout_reads = []
        self.pending = False

    def reset(self, env_ids=None, **kwargs):
        if env_ids is None:
            self.state.zero_()
            self.full_resets += 1
        else:
            self.state[env_ids] = 100.
        self.pending = False
        return self.observe()

    def observe(self):
        assert not self.pending, "observe must be consumed by step"
        self.pending = True
        batch = self.num_envs
        return SimpleNamespace(
            actor=(self.state + 1)[:, None].expand(-1, 506).clone(),
            critic=self.state[:, None].expand(-1, 843).clone(),
            auxiliary=SimpleNamespace(base_lin_vel_b=torch.zeros(batch, 3),
                keybody_contacts=torch.zeros(batch, 10),
                ref_keybody_rel_pos=torch.zeros(batch, 6, 3),
                robot_keybody_rel_pos=torch.zeros(batch, 6, 3)),
        )

    def step(self, action):
        assert self.pending
        self.pending = False
        self.steps += 1
        self.state += 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        if self.boundaries and self.steps == 1:
            truncated[0] = True
            terminated[1] = True
        return SimpleNamespace(reward=torch.ones(self.num_envs),
            terminated=terminated, truncated=truncated, terms={})

    def critic_observation(self, ids):
        assert not self.pending
        self.timeout_reads.append(ids.tolist())
        # The terminated row simulates a reference clip that is no longer valid.
        assert ids.tolist() == [0]
        return self.state[ids, None].expand(-1, 843).clone()

    def deployable_physics(self):
        batch = self.num_envs
        body = torch.zeros(batch, 72)
        body[:, 5] = -1
        return PhysicsObservation(body, torch.zeros(batch, 3), torch.zeros(batch, 2, 12),
            torch.ones(batch, 2, dtype=torch.bool), torch.zeros(batch, 0, 16),
            torch.zeros(batch, 0, dtype=torch.bool))

    privileged_physics_observation = deployable_physics

    def privileged_teacher_observation(self):
        frame = self.deployable_physics()
        batch = self.num_envs
        truth = PrivilegedFootContactTruth(
            point_position_b=frame.contact_features[..., :3],
            point_velocity_b=frame.contact_features[..., 3:6],
            normal_b=frame.contact_features[..., 6:9],
            normal_force_n=frame.contact_features[..., 9],
            contact_found=frame.contact_features[..., 10].bool(),
            tangential_force_n=torch.zeros(batch, 2),
            slip_speed_m_s=torch.zeros(batch, 2),
            friction_utilization=torch.zeros(batch, 2),
        )
        return frame, truth

    def physics_teacher_target(self):
        return torch.zeros(self.num_envs, 2)


def _trainer(physics, **task_kwargs):
    cls = HoloMotionPhysicsMJLabPPO if physics else HoloMotionMJLabPPO
    trainer = cls.__new__(cls)
    trainer.task = _Task(**task_kwargs)
    trainer.actor = _HistoryActor()
    trainer.critic = _Critic()
    trainer.config = HoloMotionPPOConfig(gamma=.5, lam=1., normalize_advantage=False)
    trainer._observation = None
    if physics:
        from unified_motion.algorithms.dynamics_history import DynamicsHistory
        trainer._dynamics_history = DynamicsHistory()
        trainer._dynamics_rollouts = []
        trainer.privileged_teacher = None
        trainer.dynamics = DynamicsModel(get_robot_contract("k1_22dof"),
            physics_width=4, memory_width=4, latent_width=2, hidden_dim=8,
            system_id_dim=2, teacher_student=True, uncertainty=True)
        trainer.actor_dynamics = trainer.dynamics
        trainer.confidence_warmup_steps = 2
        from unified_motion.physics_confidence import PhysicsConfidence
        trainer.confidence = PhysicsConfidence(2)
        trainer._physics_memory = None
        trainer._physics_episode_start = None
        trainer._physics_age = None
    return trainer


@pytest.mark.parametrize("physics", [False, True])
def test_collect_uses_pre_reset_timeout_value_and_real_terminal_mask(physics):
    trainer = _trainer(physics, boundaries=True)
    rollout = trainer.collect(2)
    assert trainer.task.timeout_reads == [[0]]
    # t=0: timeout gives 1+.5*V(final=1); real termination gives 1.
    torch.testing.assert_close(rollout.returns[:2, 0, 0], torch.tensor([1.5, 1.]))
    assert rollout.timeout_values[0, 0, 0] == 1
    assert rollout.critic_observation[0, 1, 0] == 100
    # Neither boundary cuts or resets the continuing third environment.
    assert trainer.task.state[2] == 2
    assert trainer.task.full_resets == 1


@pytest.mark.parametrize("physics", [False, True])
def test_collect_cache_matches_replay_on_second_rollout_without_env_reset(physics):
    from unified_motion.holomotion_mjlab_ppo import build_episode_causal_mask
    from unified_motion.holomotion_original_algorithm import flat_observations_to_tensordict
    trainer = _trainer(physics)
    trainer.collect(3)
    rollout = trainer.collect(2)
    data = flat_observations_to_tensordict(rollout.actor_observation, rollout.critic_observation)
    replay = trainer.actor(data, mode="sequence_logp",
        attn_mask=build_episode_causal_mask(rollout.dones.squeeze(-1)))
    torch.testing.assert_close(replay["mu"], rollout.old_mu)
    torch.testing.assert_close(replay["actions_log_prob"], rollout.old_log_prob)
    assert trainer.task.full_resets == 1
    assert trainer.task.steps == 5
    if physics:
        assert trainer._physics_age.tolist() == [5, 5, 5]


@pytest.mark.parametrize("physics", [False, True])
def test_collect_single_sample_normalization(physics):
    trainer = _trainer(physics, num_envs=1)
    trainer.config = HoloMotionPPOConfig()
    rollout = trainer.collect(1)
    assert torch.isfinite(rollout.advantages).all()
    assert rollout.advantages.item() == 0


def test_critic_only_read_selects_rows_without_advancing_task(monkeypatch):
    from test_holomotion_mjlab_rewards import _inputs
    state, _, _ = _inputs()
    state.joint_pos[1].fill_(7)
    task = HoloMotionK1MJLabTask.__new__(HoloMotionK1MJLabTask)
    task.adapter = SimpleNamespace(last_action_onnx=state.action)
    task._ids = lambda ids: ids
    pending = object()
    task._pending_reference = pending
    task._pending_body_reference = pending
    task.sampler = SimpleNamespace(reference_window=lambda ids: ids,
                                   body_reference_window=lambda ids: ids)
    monkeypatch.setattr(HoloMotionMJLabState, "from_adapter", lambda *args, **kwargs: state)
    def build(selected, ref, body_ref):
        assert ref.tolist() == body_ref.tolist() == [1]
        assert selected.joint_names == state.joint_names
        assert selected.soft_joint_pos_limits.shape == (1, 22, 2)
        return selected.joint_pos
    task.critic_builder = build
    torch.testing.assert_close(task.critic_observation(torch.tensor([1])), torch.full((1, 22), 7.))
    assert task._pending_reference is task._pending_body_reference is pending


def _real_trainer(physics, monkeypatch, *, boundaries=True):
    """Use real upstream nets, with smaller widths and synthetic task states."""
    from pathlib import Path
    import yaml
    from unified_motion.holomotion_original_algorithm import flat_observations_to_tensordict
    from unified_motion.holomotion_mjlab_ppo import build_episode_causal_mask

    root = Path(__file__).resolve().parents[1] / "third_party/HoloMotion/source"
    monkeypatch.syspath_prepend(str(root))
    from holomotion.src.modules.agent_modules import PPOTFRefRouterActor, PPOCritic

    config = yaml.safe_load((root / "holomotion/config/modules/motion_tracking/motion_tracking_tf-moe.yaml").read_text(encoding="utf-8"))["modules"]
    actor_cfg, critic_cfg = config["actor"], config["critic"]
    actor_cfg.update(d_model=16, n_heads=4, n_kv_heads=2, n_layers=1,
        obs_embed_mlp_hidden=32, router_embed_mlp_hidden=32,
        num_fine_experts=4, num_shared_experts=1, top_k=2,
        routing_score_fn="softmax", routing_scale=1., use_dynamic_bias=False,
        bias_update_rate=.001, expert_bias_clip=1., noise_std_type="log",
        fix_sigma=False, min_sigma=.01, max_sigma=1.)
    critic_cfg["layer_config"]["hidden_dims"] = [32, 32]
    for cfg in (actor_cfg, critic_cfg):
        cfg["obs_schema"]["flattened_obs"]["seq_len"] = 1
        cfg["obs_schema"]["flattened_obs_fut"]["seq_len"] = 10
    torch.manual_seed(321)
    sample = flat_observations_to_tensordict(torch.zeros(3, 506), torch.zeros(3, 843))
    actor = PPOTFRefRouterActor(actor_cfg["obs_schema"], actor_cfg, 22, .2, obs_example=sample)
    critic = PPOCritic(critic_cfg["obs_schema"], critic_cfg, obs_example=sample)
    algorithm = SimpleNamespace(actor=actor, critic=critic,
        network_config=dict(actor=actor_cfg, critic=critic_cfg, init_noise_std=.2))
    task = _Task(boundaries=boundaries)
    task.bridge = SimpleNamespace(robot=get_robot_contract("k1_22dof"))
    task.physics_teacher_dim = 2
    ppo_cfg = HoloMotionPPOConfig(num_learning_epochs=1, num_mini_batches=1,
        aux_state_pred_enabled=False, inactive_expert_margin_enabled=False,
        desired_kl=None)
    if physics:
        dynamics = DynamicsModel(task.bridge.robot, physics_width=4, memory_width=4,
            latent_width=2, hidden_dim=8, system_id_dim=2, teacher_student=True,
            uncertainty=True)
        from unified_motion.physics_update import PhysicsUpdateConfig
        trainer = HoloMotionPhysicsMJLabPPO(algorithm, task, dynamics, config=ppo_cfg,
            update_config=PhysicsUpdateConfig(freeze_base_actor=True))
    else:
        trainer = HoloMotionMJLabPPO(algorithm, task, config=ppo_cfg)
    return trainer


@pytest.mark.parametrize("physics", [False, True])
def test_vendored_actor_replays_second_rollout_and_updates(physics, monkeypatch):
    from unified_motion.holomotion_original_algorithm import flat_observations_to_tensordict
    from unified_motion.holomotion_mjlab_ppo import build_episode_causal_mask
    trainer = _real_trainer(physics, monkeypatch)
    for _ in range(2):
        rollout = trainer.collect(3)
        data = flat_observations_to_tensordict(rollout.actor_observation, rollout.critic_observation)
        with torch.no_grad():
            replay = trainer._actor_forward(data, mode="sequence_logp", actions=rollout.actions,
                attn_mask=build_episode_causal_mask(rollout.dones.squeeze(-1)),
                update_obs_norm=False, physics_features=rollout.physics_features,
                physics_confidence=rollout.physics_confidence)
        torch.testing.assert_close(replay["mu"], rollout.old_mu, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(replay["actions_log_prob"], rollout.old_log_prob,
                                   atol=2e-4, rtol=2e-5)
        metrics = trainer.update(rollout)
        assert metrics.num_updates == 1
        assert metrics.ratio_mean == pytest.approx(1., abs=2e-5)
        assert torch.isfinite(torch.tensor([metrics.actor_loss, metrics.critic_loss])).all()
    assert trainer.task.full_resets == 1

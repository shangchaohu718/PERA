"""Seven real Actor implementations on one task/PPO/Physics/runtime contract."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from test_holomotion_rollout_boundaries import _Task, _real_trainer
from unified_motion.holomotion_mjlab_ppo import HoloMotionPPOConfig, build_episode_causal_mask
from unified_motion.holomotion_original_algorithm import flat_observations_to_tensordict
from unified_motion.holomotion_physics_runtime import FrozenPhysicsPolicy, state_fingerprint
from unified_motion.k1_policy_backends import ACTOR_BACKENDS, K1ActorConfig, K1CustomActor
from unified_motion.k1_policy_runtime import (
    FrozenK1BasePolicy, load_k1_checkpoint, frozen_k1_policy, resume_k1_trainer, evaluate_k1_policy,
)
from unified_motion.k1_training import create_custom_k1_algorithm, create_k1_trainer
from unified_motion.physics_confidence import PhysicsConfidenceConfig
from unified_motion.robot_contract import get_robot_contract


def test_shipped_actor_yaml_runs_all_custom_backends():
    path = Path(__file__).resolve().parents[1] / 'configs/actors/k1_small.yaml'
    options = yaml.safe_load(path.read_text(encoding='utf-8'))
    for backend in ACTOR_BACKENDS[1:]:
        actor = create_custom_k1_algorithm(K1ActorConfig(backbone=backend, **options)).actor
        data = flat_observations_to_tensordict(torch.randn(2, 3, 506), torch.zeros(2, 3, 843))
        with torch.no_grad():
            output = actor(data, mode='sequence_logp', actions=torch.zeros(2, 3, 22),
                           attn_mask=build_episode_causal_mask(torch.zeros(2, 3, dtype=torch.bool)))
        assert output['mu'].shape == (2, 3, 22), backend
        assert torch.isfinite(output['mu']).all(), backend
        assert torch.isfinite(output['actions_log_prob']).all(), backend


def task():
    result = _Task(boundaries=True)
    result.bridge = SimpleNamespace(robot=get_robot_contract("k1_22dof"))
    result.physics_teacher_dim = 2
    return result


def trainer(backend, physics, monkeypatch):
    if backend == "holomotion":
        source = _real_trainer(False, monkeypatch)
        algorithm = SimpleNamespace(actor=source.actor, critic=source.critic,
                                    network_config=source._network_config)
    else:
        algorithm = create_custom_k1_algorithm(K1ActorConfig(backbone=backend, d_model=16,
            hidden_dim=24, n_layers=1, context_len=4, init_std=.2))
    cfg = HoloMotionPPOConfig(num_learning_epochs=1, num_mini_batches=2, desired_kl=None,
        aux_state_pred_enabled=False, inactive_expert_margin_enabled=False)
    options = dict(dynamics_config=dict(physics_width=4, memory_width=4, latent_width=2, hidden_dim=8),
        dynamics_burn_in_steps=2, dynamics_train_steps=4, dynamics_epochs=1,
        confidence_config=PhysicsConfidenceConfig(mode="ungated")) if physics else {}
    return create_k1_trainer(algorithm, task(), physics=physics, ppo_config=cfg, **options)


def assert_replay(trainer, rollout):
    data = flat_observations_to_tensordict(rollout.actor_observation, rollout.critic_observation)
    with torch.no_grad():
        output = trainer._actor_forward(data, mode="sequence_logp", actions=rollout.actions,
            attn_mask=build_episode_causal_mask(rollout.dones.squeeze(-1)),
            physics_features=rollout.physics_features, physics_confidence=rollout.physics_confidence,
            update_obs_norm=False)
    torch.testing.assert_close(output["mu"], rollout.old_mu, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(output["actions_log_prob"], rollout.old_log_prob, atol=2e-4, rtol=2e-5)


@pytest.mark.parametrize("backend", ACTOR_BACKENDS)
@pytest.mark.parametrize("physics", [False, True, 'transformer'])
def test_every_backend_collects_updates_resumes_and_freezes(backend, physics, monkeypatch, tmp_path):
    torch.manual_seed(72)
    engine = trainer(backend, physics, monkeypatch)
    actor = engine.actor.base_actor if physics else engine.actor
    initial = state_fingerprint(actor)
    for _ in range(2):
        rollout = engine.collect(3)
        assert_replay(engine, rollout)
        metrics = engine.update(rollout)
        # Later minibatches see updated weights; exact old-policy parity is
        # checked above, before any optimizer step.
        assert metrics.num_updates == 2
        assert torch.isfinite(torch.tensor([metrics.actor_loss, metrics.critic_loss])).all()
        if physics:
            dynamics = engine.update_dynamics(rollout)
    if physics:
        assert dynamics is not None
        assert not engine.update_config.freeze_base_actor  # All fixtures initialize fresh weights.
    assert state_fingerprint(actor) != initial
    assert engine.task.full_resets == 1 and engine.task.steps == 6
    path = engine.save_checkpoint(tmp_path/"complete.pt", iteration=2)
    # New runtime loads without sidecars, preserves actual selected architecture.
    alone = tmp_path/"standalone.pt"
    alone.write_bytes(path.read_bytes())
    loaded = load_k1_checkpoint(alone)
    assert state_fingerprint(loaded.actor) == state_fingerprint(engine.actor)
    if backend != "holomotion":
        base = loaded.actor.base_actor if physics else loaded.actor
        assert isinstance(base, K1CustomActor) and base.config.backbone == backend
    reference = (FrozenPhysicsPolicy(engine.actor, engine.actor_dynamics,
        confidence_warmup_steps=engine.confidence_warmup_steps, rollout_steps=3, calibration=engine.confidence)
        if physics else FrozenK1BasePolicy(engine.actor, rollout_steps=3))
    report = evaluate_k1_policy(reference, task(), steps=11)
    actual = evaluate_k1_policy(frozen_k1_policy(loaded), task(), steps=11)
    assert report == actual and report["weights_unchanged"]
    resumed = resume_k1_trainer(alone, task())
    assert resumed._observation is None
    assert state_fingerprint(resumed.actor, resumed.critic) == state_fingerprint(engine.actor, engine.critic)
    for optimizer in (resumed.actor_optimizer, resumed.critic_optimizer):
        assert optimizer.state
    rollout = resumed.collect(6)
    assert_replay(resumed, rollout)
    resumed.update(rollout)
    if physics:
        assert resumed.update_dynamics(rollout) is not None
    assert resumed.task.full_resets == 1


@pytest.mark.parametrize("backend", ACTOR_BACKENDS[1:])
def test_all_observation_fields_reach_custom_actor_without_critic_leakage(backend):
    actor = create_custom_k1_algorithm(K1ActorConfig(backbone=backend, d_model=16,
        hidden_dim=24, n_layers=1)).actor
    observations = torch.randn(2,3,506, requires_grad=True)
    privileged = torch.randn(2,3,843, requires_grad=True)
    data = flat_observations_to_tensordict(observations, privileged)
    output = actor(data, mode="sequence_logp", actions=torch.zeros(2,3,22),
                   attn_mask=build_episode_causal_mask(torch.zeros(2,3,dtype=torch.bool)))
    output["mu"].square().sum().backward()
    assert (observations.grad.abs().sum((0,1)) > 0).all()
    assert privileged.grad is None


def test_selected_cores_are_existing_implementations():
    from unified_motion.models.physics_policy import DenseReferenceTransformer
    from unified_motion.models.transformer import ReferenceConditionedTransformer
    from unified_motion.models.recurrent import RecurrentReferenceBackbone
    from unified_motion.models.lru import LRUCore
    expected = dict(mlp=torch.nn.Sequential, transformer=DenseReferenceTransformer,
                    moe=ReferenceConditionedTransformer, gru=RecurrentReferenceBackbone,
                    lstm=RecurrentReferenceBackbone, lru=RecurrentReferenceBackbone)
    for name, cls in expected.items():
        actor = create_custom_k1_algorithm(K1ActorConfig(backbone=name, d_model=16,hidden_dim=24,n_layers=1)).actor
        assert isinstance(actor.backbone, cls)
        if name == "lru":
            assert isinstance(actor.backbone.core, LRUCore)


def test_backend_mismatch_and_physics_to_plain_resume_rejected(tmp_path, monkeypatch):
    engine = trainer("lru", True, monkeypatch)
    path = engine.save_checkpoint(tmp_path/"lru.pt", iteration=0)
    plain = trainer("lru", False, monkeypatch)
    with pytest.raises(ValueError, match="Physics trainer"):
        plain.load_optimizer_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    payload["physics_runtime_config"]["network_config"]["backend"] = "gru"
    torch.save(payload, tmp_path/"wrong.pt")
    with pytest.raises(ValueError, match="backend declarations"):
        load_k1_checkpoint(tmp_path/"wrong.pt")


@pytest.mark.parametrize("backend", ACTOR_BACKENDS[1:])
def test_custom_cli_does_not_require_holomotion_weights(backend, monkeypatch):
    from tools.train_holomotion_k1_mjlab import parse_args
    monkeypatch.setattr("sys.argv", ["train_k1_mjlab.py", "--actor", backend, "--onnx", "contract.onnx",
        "--asset-root", "assets", "--motion", "motion.npz", "--output-dir", "new_run", "--iterations", "2"])
    config = parse_args()
    assert config.actor == backend and config.run_dir is None


def test_holomotion_cli_accepts_random_initialization_config(monkeypatch):
    from tools.train_holomotion_k1_mjlab import parse_args
    monkeypatch.setattr("sys.argv", ["train_k1_mjlab.py", "--actor", "holomotion",
        "--holomotion-config", "holomotion.yaml", "--onnx", "contract.onnx",
        "--asset-root", "assets", "--motion", "motion.npz", "--output-dir", "new_run",
        "--iterations", "2", "--physics"])
    config = parse_args()
    assert config.run_dir is None
    assert config.holomotion_config == Path("holomotion.yaml")
    assert config.physics


def test_shipped_holomotion_config_builds_16_expert_network_metadata(monkeypatch):
    from unified_motion import holomotion_original_algorithm as original
    root = Path(__file__).resolve().parents[1]
    captured = {}
    actor, critic = object(), object()
    def build(network_config, **kwargs):
        captured.update(network_config)
        return actor, critic
    monkeypatch.setattr(original, "build_original_holomotion_networks", build)
    result = original.initialize_original_holomotion_k1_algorithm(
        holomotion_root=root/"third_party/HoloMotion/source",
        config_path=root/"configs/actors/holomotion_k1_16e.yaml")
    assert result.actor is actor and result.critic is critic
    assert captured["actor"]["num_fine_experts"] == 16
    assert captured["actor"]["d_model"] == 256
    assert captured["critic"]["layer_config"]["hidden_dims"] == [1024, 1024, 512]
    assert captured["actor"]["obs_schema"]["flattened_obs_fut"]["seq_len"] == 10
    assert result.config.algo.config.init_noise_std == 1.0


@pytest.mark.parametrize("physics", [False, True])
def test_cli_runs_and_resumes_with_explicit_mock_simulator(tmp_path, monkeypatch, physics):
    """Exercise real main/runner/checkpoints; hardware and task observations are synthetic."""
    import json
    import sys
    from dataclasses import asdict
    from types import ModuleType
    from test_holomotion_mjlab import _contract
    from tools import train_holomotion_k1_mjlab as cli
    contract_file = tmp_path/"contract.json"
    contract_file.write_text(json.dumps(asdict(_contract())), encoding="utf-8")
    config_file = tmp_path/"actor.yaml"
    config_file.write_text("d_model: 16\nhidden_dim: 24\nn_layers: 1\n", encoding="utf-8")
    # Keep the production library's torch intact; only main's CUDA boundary is replaced.
    monkeypatch.setattr(cli, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, manual_seed_all=lambda seed: None),
        device=lambda value: torch.device("cpu"), manual_seed=torch.manual_seed, load=torch.load))
    env_module = ModuleType("mjlab.envs")
    env_module.ManagerBasedRlEnv = lambda cfg, device: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "mjlab.envs", env_module)
    monkeypatch.setattr(cli, "make_holomotion_k1_mjlab_cfg", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(cli, "K1MJLabBridge", lambda *args, **kwargs: SimpleNamespace(robot=get_robot_contract("k1_22dof")))
    monkeypatch.setattr(cli, "HoloMotionK1MJLabAdapter", lambda bridge, contract: SimpleNamespace(bridge=bridge, contract=contract))
    monkeypatch.setattr(cli, "MotionLibrary", SimpleNamespace(from_named_mjlab_files=lambda *args, **kwargs:
        SimpleNamespace(clip_names=("a", "b", "c"))))
    monkeypatch.setattr(cli, "MotionSampler", lambda library, **kwargs: SimpleNamespace(library=library,
        clip_ids=torch.arange(3), offsets_s=[.2], step_dt=.02))
    tasks = []
    def make_task(adapter, sampler, foot_reward_config, physics_randomization):
        result = task()
        result.adapter, result.sampler = adapter, sampler
        result.foot_reward_config = foot_reward_config or cli.HoloMotionFootRewardConfig()
        result.physics_randomization = physics_randomization
        result.close = lambda: None
        tasks.append(result)
        return result
    monkeypatch.setattr(cli, "HoloMotionK1MJLabTask", make_task)
    motion_path = tmp_path/'synthetic_motion.npz'
    motion_path.write_bytes(b'MOCK: MotionLibrary is replaced for this CLI wiring test')
    common = ["--asset-root", "assets", "--motion", str(motion_path), "--num-envs", "3", "--iterations", "2"]
    args = ["train_k1_mjlab.py", "--actor", "lru", "--actor-config", str(config_file),
        "--task-contract", str(contract_file), "--output-dir", str(tmp_path/"first"),
        "--steps-per-env", "3", *common]
    if physics:
        args += ["--physics-teacher" if physics == 'transformer' else "--physics", "--physics-burn-in-steps", "1", "--physics-train-steps", "2",
            "--physics-dynamics-epochs", "1", "--physics-width", "4", "--physics-memory-width", "4",
            "--physics-latent-width", "2", "--physics-hidden-dim", "8", "--physics-scene-encoder", "gnn"]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    path = tmp_path/"first/model_1.pt"
    assert path.is_file() and tasks[-1].steps == 6
    metadata = json.loads((tmp_path/"first/mjlab_runner.json").read_text(encoding="utf-8"))
    assert metadata["actor_backend"] == "lru" and metadata["physics_enabled"] == bool(physics)
    monkeypatch.setattr(sys, "argv", ["train_k1_mjlab.py", "--resume", str(path),
        "--output-dir", str(tmp_path/"resumed"), *common])
    cli.main()
    assert (tmp_path/"resumed/model_3.pt").is_file()
    assert tasks[-1].steps == 6  # Saved rollout length (3), not CLI default 32.
    if physics:
        loaded = load_k1_checkpoint(tmp_path/"resumed/model_3.pt")
        assert loaded.dynamics.encoder.scene_encoder == "gnn"
        if physics == 'transformer':
            assert loaded.payload['privileged_teacher']['updates'] == 4


def test_resume_does_not_silently_override_saved_training_settings(monkeypatch):
    from tools.train_holomotion_k1_mjlab import parse_args
    monkeypatch.setattr("sys.argv", ["train_k1_mjlab.py", "--resume", "model.pt", "--steps-per-env", "99",
        "--asset-root", "assets", "--motion", "motion.npz", "--output-dir", "new_run", "--iterations", "2"])
    with pytest.raises(SystemExit):
        parse_args()

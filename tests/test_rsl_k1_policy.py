import pytest
import torch

from unified_motion.rsl_k1_policy import (
    RSL_EMPIRICAL_NORMALIZATION_EPS,
    RSL_K1_BUNDLE_SCHEMA,
    load_frozen_rsl_k1_bundle,
    load_rsl_k1_checkpoint,
    save_frozen_rsl_k1_bundle,
    verify_frozen_rsl_k1_provenance,
)


def _source_payload():
    torch.manual_seed(7)
    actor = torch.nn.Sequential(
        torch.nn.Linear(125, 512), torch.nn.ELU(),
        torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 22),
    )
    state = {f"actor.{key}": value for key, value in actor.state_dict().items()}
    state["std"] = torch.ones(22)
    return {
        "model_state_dict": state,
        "obs_norm_state_dict": {
            "_mean": torch.randn(1, 125), "_std": torch.rand(1, 125) + .1,
            "_var": torch.ones(1, 125), "count": torch.tensor(100),
        },
        "optimizer_state_dict": {}, "privileged_obs_norm_state_dict": {},
        "iter": 19, "infos": None,
    }, actor


def test_native_rsl_policy_matches_reference_and_is_frozen(tmp_path):
    payload, actor = _source_payload()
    checkpoint = tmp_path / "model_19.pt"
    torch.save(payload, checkpoint)
    policy = load_rsl_k1_checkpoint(checkpoint)
    observation = torch.randn(8, 125)
    expected = actor((observation - payload["obs_norm_state_dict"]["_mean"]) /
                     (payload["obs_norm_state_dict"]["_std"] + RSL_EMPIRICAL_NORMALIZATION_EPS))
    assert torch.equal(policy.act(observation), expected)
    assert not policy.training
    assert all(not parameter.requires_grad for parameter in policy.parameters())


def test_safe_bundle_round_trip(tmp_path):
    payload, _ = _source_payload()
    checkpoint = tmp_path / "model_19.pt"
    bundle = tmp_path / "frozen.pt"
    torch.save(payload, checkpoint)
    source = load_rsl_k1_checkpoint(checkpoint)
    save_frozen_rsl_k1_bundle(source, bundle)
    frozen = load_frozen_rsl_k1_bundle(bundle)
    observation = torch.randn(3, 125)
    assert torch.equal(source.act(observation), frozen.act(observation))
    raw = torch.load(bundle, weights_only=True)
    assert raw["policy_schema"] == RSL_K1_BUNDLE_SCHEMA
    assert "optimizer_state_dict" not in raw and "critic_state_dict" not in raw


def test_explicit_checkpoint_identity_and_provenance_are_enforced(tmp_path):
    payload, _ = _source_payload()
    checkpoint = tmp_path / "ambiguous_model_3.pt"
    other_checkpoint = tmp_path / "other_model_3.pt"
    bundle = tmp_path / "frozen.pt"
    torch.save(payload, checkpoint)
    payload["iter"] = 20
    torch.save(payload, other_checkpoint)
    source = load_rsl_k1_checkpoint(checkpoint, checkpoint_id="globally_unique_run_model3")
    save_frozen_rsl_k1_bundle(source, bundle)
    frozen = load_frozen_rsl_k1_bundle(bundle)

    assert frozen.metadata.checkpoint_id == "globally_unique_run_model3"
    assert verify_frozen_rsl_k1_provenance(
        frozen, checkpoint=checkpoint, checkpoint_id="globally_unique_run_model3")
    with pytest.raises(ValueError, match="checkpoint_id"):
        verify_frozen_rsl_k1_provenance(
            frozen, checkpoint=checkpoint, checkpoint_id="model_3")
    with pytest.raises(ValueError, match="source_sha256"):
        verify_frozen_rsl_k1_provenance(
            frozen, checkpoint=other_checkpoint,
            checkpoint_id="globally_unique_run_model3")


def test_rejects_empty_explicit_checkpoint_identity(tmp_path):
    payload, _ = _source_payload()
    checkpoint = tmp_path / "model_19.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="non-empty"):
        load_rsl_k1_checkpoint(checkpoint, checkpoint_id="  ")


def test_rejects_wrong_observation_and_actor_contract(tmp_path):
    payload, _ = _source_payload()
    checkpoint = tmp_path / "bad.pt"
    del payload["model_state_dict"]["actor.6.bias"]
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="architecture"):
        load_rsl_k1_checkpoint(checkpoint)

    payload, _ = _source_payload()
    torch.save(payload, checkpoint)
    policy = load_rsl_k1_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="125"):
        policy.act(torch.randn(2, 506))

import torch
from torch import nn
from torch.distributions import Normal

from unified_motion.holomotion_physics_actor import (
    HoloMotionPhysicsActor,
    HoloMotionPhysicsActorConfig,
    dynamics_actor_features,
    teacher_actor_features,
)


class TinyTensorDict:
    def __init__(self, values=None, **kwargs):
        self.values = {} if values is None else dict(values)
        self.values.update(kwargs)

    def get(self, name):
        return self.values[name]

    def set(self, name, value):
        self.values[name] = value
        return self


class FakeActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.linspace(-0.2, 0.2, 4))
        self.actor_module = nn.Identity()

    def reset_kv_cache(self, *args, **kwargs):
        return None

    def clear_env_cache(self, *args, **kwargs):
        return None

    def forward(self, obs_td, actions=None, mode="sampling", attn_mask=None, **kwargs):
        lead = obs_td.get("lead")
        mu = self.bias.expand(*lead, 4)
        sigma = torch.full_like(mu, 0.3)
        if mode == "sampling":
            selected = Normal(mu, sigma).sample()
        elif mode == "inference":
            selected = mu
        else:
            selected = actions
        out = TinyTensorDict(mu=mu, sigma=sigma, actions=selected)
        if mode != "inference":
            dist = Normal(mu, sigma)
            out.set("actions_log_prob", dist.log_prob(selected).sum(-1, keepdim=True))
            out.set("entropy", dist.entropy().sum(-1, keepdim=True))
        return out


def build_actor():
    return HoloMotionPhysicsActor(
        FakeActor(),
        HoloMotionPhysicsActorConfig(
            physics_feature_dim=6, hidden_dim=8, action_dim=4, residual_limit=0.2
        ),
    )


def test_zero_initialized_adapter_preserves_sample_and_distribution_exactly():
    actor = build_actor()
    data = TinyTensorDict(lead=(3,))
    features = torch.randn(3, 6)
    torch.manual_seed(12)
    expected = actor.base_actor(data, mode="sampling")
    torch.manual_seed(12)
    actual = actor(data, mode="sampling", physics_features=features)

    for name in ("mu", "sigma", "actions", "actions_log_prob", "entropy"):
        torch.testing.assert_close(actual.get(name), expected.get(name), rtol=0, atol=0)


def test_sequence_log_prob_uses_corrected_mean_and_preserves_shape():
    actor = build_actor()
    with torch.no_grad():
        actor.physics_residual.net[-1].bias.fill_(0.1)
    data = TinyTensorDict(lead=(2, 3))
    actions = torch.randn(2, 3, 4)
    features = torch.randn(2, 3, 6)
    output = actor(
        data,
        actions=actions,
        mode="sequence_logp",
        physics_features=features,
    )
    expected = Normal(output.get("mu"), output.get("sigma")).log_prob(actions).sum(
        -1, keepdim=True
    )
    assert output.get("mu").shape == (2, 3, 4)
    assert output.get("actions_log_prob").shape == (2, 3, 1)
    assert output.get("entropy").shape == (2, 3, 1)
    torch.testing.assert_close(output.get("actions_log_prob"), expected)


def test_confidence_gates_physics_residual_without_changing_base_policy():
    actor = build_actor()
    with torch.no_grad():
        actor.physics_residual.net[-1].bias.fill_(0.1)
    data = TinyTensorDict(lead=(2,))
    features = torch.randn(2, 6)
    base = actor.base_actor(data, mode="inference").get("mu")
    uncertain = actor(
        data, mode="inference", physics_features=features,
        physics_confidence=torch.zeros(2, 1),
    ).get("mu")
    confident = actor(
        data, mode="inference", physics_features=features,
        physics_confidence=torch.ones(2, 1),
    ).get("mu")
    torch.testing.assert_close(uncertain, base)
    assert not torch.equal(confident, base)


def test_dynamics_actor_features_concatenates_only_deployable_outputs():
    context = torch.randn(2, 3, 5)
    latent = torch.randn(2, 3, 2)
    contacts = torch.randn(2, 3, 2, 5)
    output = dynamics_actor_features(context, latent, contacts)
    assert output.shape == (2, 3, 17)
    torch.testing.assert_close(output[..., :5], context)
    torch.testing.assert_close(output[..., 5:7], latent)
    torch.testing.assert_close(output[..., 7:12], contacts[...,0,:])
    torch.testing.assert_close(output[..., 12:17], contacts[...,1,:])


def test_teacher_actor_features_use_physical_predictions_not_private_latent():
    core = torch.randn(2, 3, 17)
    logits = torch.randn(2, 3, 4)
    output = teacher_actor_features(
        {"next_core": core, "contact_logits": logits, "latent": torch.randn(2, 3, 99)},
        dof=4, contact_slots=2)
    assert output.shape == (2, 3, 19)
    torch.testing.assert_close(output[..., :17], core)
    torch.testing.assert_close(output[..., 17:], logits[..., :2].sigmoid())

"""Exact cached public features for DAgger on the one-layer frozen Actor.

The sole domain FFN is downstream of the only attention cache. Its inputs,
public FFN and reference-router features cannot change during training.
Each label still supervises the complete mixed Actor's final action.
"""

import torch

from modumotion_g1.domain_moe import domain_parameters


class FrozenFeatureDagger:
    def __init__(self, actor, domain):
        if getattr(actor, "domain_mode", None) == "soft_residual":
            raise ValueError("Feature DAgger expects competitive anchor mode")
        if len(actor.actor_module.layers) != 1:
            raise ValueError("Feature replay requires the validated one-layer Actor")
        self.actor = actor
        self.block = actor.actor_module.layers[0]
        self.domain = domain
        if self.block.active_domains != (domain,):
            raise ValueError("Train exactly one active domain")
        allowed = {id(p) for p in domain_parameters(actor)}
        if any(p.requires_grad and id(p) not in allowed for p in actor.parameters()):
            raise ValueError("Public Actor must be frozen")
        self.features = {}
        self.handle = self.block.norm2.register_forward_pre_hook(self._residual)
        self.block.dagger_feature_sink = self._features

    def _residual(self, module, inputs):
        self.features["residual"] = inputs[0].detach().squeeze(1)

    def _features(self, x, router_x, native, anchor, shared):
        self.features.update(
            x=x.detach().squeeze(1),
            router_x=router_x.detach().squeeze(1),
            native=native.detach().squeeze(1),
            anchor=anchor.detach().squeeze(1),
            shared=shared.detach().squeeze(1),
        )

    def predict(self, batch):
        plugin = self.block.domain_plugins[self.domain]
        context = batch.get("context")
        weight = (
            (
                plugin.route(batch["router_x"], context).float()
                - batch["anchor"]
                + self.block.merge_bias[self.domain]
            )
            .sigmoid()
            .to(batch["x"].dtype)
        )
        domain = batch["shared"] + self.block.routing_scale * plugin(batch["x"], context)
        mixed = (1 - weight) * batch["native"] + weight * domain
        policy = self.actor.actor_module
        return policy.action_mu_head(policy.norm_f(batch["residual"] + mixed))

    def snapshot(self, target, actual_mu, context=None):
        if context is not None:
            self.features["context"] = context.detach()
        with torch.no_grad():
            replay_mu = self.predict(self.features)
            torch.testing.assert_close(replay_mu, actual_mu, atol=2e-5, rtol=2e-5)
        return {
            **{k: v.detach().cpu().clone() for k, v in self.features.items()},
            "target": target.detach().cpu().clone(),
        }

    def close(self):
        self.handle.remove()
        self.block.dagger_feature_sink = None

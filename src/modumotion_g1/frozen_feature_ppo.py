"""Exact frozen-feature acceleration for one-layer domain PPO updates.

Rollouts still execute the complete Actor. Only repeated sequence-logp
evaluation reuses detached public features. There are no teacher actions or
imitation objectives here; the caller retains its ordinary PPO update.
"""

import math

import torch
from tensordict import TensorDict

from modumotion_g1.domain_moe import FrozenPublicActor, domain_parameters

FEATURE_KEY = "ppo_frozen_features"
FEATURE_NAMES = ("residual", "x", "router_x", "native", "anchor", "shared")
AUX_OUTPUT_NAMES = {
    "base_lin_vel_loc": "aux_base_lin_vel_loc",
    "base_lin_vel_log_std": "aux_base_lin_vel_log_std",
    "root_height_loc": "aux_root_height_loc",
    "root_height_log_std": "aux_root_height_log_std",
    "keybody_contact_logits": "aux_keybody_contact_logits",
    "ref_keybody_rel_pos": "aux_ref_keybody_rel_pos",
    "robot_keybody_rel_pos": "aux_robot_keybody_rel_pos",
}


class FrozenFeaturePPOActor(FrozenPublicActor):
    """Keep rollout behavior and state-dict keys, specialize cached logp."""

    def forward(
        self,
        obs_td,
        actions=None,
        mode="sampling",
        attn_mask=None,
        *,
        update_obs_norm=True,
        forward_chunk_size=0,
        past_key_values=None,
        current_pos=None,
    ):
        cache = self._ppo_feature_cache
        cache.validate_runtime()
        if forward_chunk_size:
            raise ValueError("Frozen-feature PPO does not support chunking")
        if past_key_values is not None:
            raise ValueError("Frozen-feature PPO does not support ONNX inputs")
        if mode == "sequence_logp" and FEATURE_KEY in obs_td.keys():
            return cache.sequence_forward(obs_td, actions, attn_mask)
        if mode == "sequence_logp":
            # Explicit full-path fallback is retained for parity checks.
            return super().forward(
                obs_td,
                actions=actions,
                mode=mode,
                attn_mask=attn_mask,
                update_obs_norm=False,
            )
        cache.begin_capture()
        try:
            result = super().forward(
                obs_td,
                actions=actions,
                mode=mode,
                attn_mask=attn_mask,
                update_obs_norm=False,
                current_pos=current_pos,
            )
            cache.finish_capture(obs_td, result)
            return result
        finally:
            cache._capturing = False


class FrozenFeaturePPOCache:
    """Attach after domain installation and before rollout-storage creation.

    Use ``augment_observation(obs, template=True)`` for the storage template,
    and ``augment_observation(obs)`` in the trainer's transition builder after
    the full Actor forward. Both return shallow observation copies. The
    added nested TensorDict is ignored by the original observation schema.
    """

    def __init__(self, domain, *, verify_capture=True):
        self.domain = domain
        self.verify_capture = bool(verify_capture)
        self.actor = None
        self.block = None
        self.latest = None
        self._pending = {}
        self._capturing = False
        self._handle = None
        self._original_class = None
        self._trainable_ids = None
        self._sink = self._capture_features
        self.capture_count = 0
        self.cached_sequence_count = 0
        self.last_capture_max_mu_error = None

    def attach_actor(self, actor):
        if self.actor is not None:
            raise ValueError("Feature cache is already attached")
        if type(actor) is not FrozenPublicActor:
            raise TypeError("Attach to the installed FrozenPublicActor")
        if len(actor.actor_module.layers) != 1:
            raise ValueError("Frozen-feature PPO requires exactly one layer")
        block = actor.actor_module.layers[0]
        if block.active_domains != (self.domain,):
            raise ValueError("Train exactly one active domain")
        if getattr(block, "dagger_feature_sink", None) is not None:
            raise ValueError("Another feature sink is already installed")
        if any(
            getattr(actor, name, False)
            for name in (
                "aux_router_command_recon_enabled",
                "aux_router_switch_penalty_enabled",
                "aux_router_future_recon_enabled",
            )
        ):
            raise ValueError("Router auxiliary predictions are unsupported")
        trainable = {id(p) for p in domain_parameters(actor)}
        expected = {id(p) for p in block.domain_plugins[self.domain].parameters()}
        if trainable != expected:
            raise ValueError("Train only all parameters of the active domain")
        if any(p.dtype != torch.float32 for p in actor.parameters()):
            raise ValueError("Frozen-feature PPO requires float32 parameters")
        self.actor, self.block = actor, block
        self._trainable_ids = expected
        try:
            self.validate_runtime()
        except Exception:
            self.actor = self.block = None
            raise
        self._original_class = type(actor)
        self._handle = block.norm2.register_forward_pre_hook(self._capture_residual)
        block.dagger_feature_sink = self._sink
        actor._ppo_feature_cache = self
        actor.__class__ = FrozenFeaturePPOActor
        return actor

    def validate_runtime(self):
        if self.actor is None:
            raise RuntimeError("Attach the feature cache first")
        if any(module.training for module in self.actor.modules()):
            raise ValueError("Frozen Actor and dropout must stay in eval mode")
        if self.block.active_domains != (self.domain,):
            raise ValueError("The active domain changed after attachment")
        if {id(p) for p in self.actor.parameters() if p.requires_grad} != self._trainable_ids:
            raise ValueError("Trainable parameters changed after attachment")
        device = next(self.actor.parameters()).device
        if torch.is_autocast_enabled(device.type):
            raise ValueError("Frozen-feature PPO requires autocast disabled")

    def begin_capture(self):
        self.latest = None
        self._pending = {}
        self._capturing = True

    def _capture_residual(self, module, inputs):
        if self._capturing:
            value = inputs[0]
            if value.ndim != 3 or value.shape[1] != 1:
                raise ValueError("Capture requires a complete single-step batch")
            self._pending["residual"] = value[:, 0].detach()

    def _capture_features(self, x, router_x, native, anchor, shared):
        if self._capturing:
            for name, value in zip(
                ("x", "router_x", "native", "anchor", "shared"),
                (x, router_x, native, anchor, shared),
                strict=True,
            ):
                self._pending[name] = value[:, 0].detach()

    def finish_capture(self, obs_td, result):
        if set(self._pending) != set(FEATURE_NAMES):
            raise RuntimeError("The full Actor did not capture every feature")
        self.latest = TensorDict(
            {key: value.clone() for key, value in self._pending.items()},
            batch_size=obs_td.batch_size,
        )
        if any(value.dtype != torch.float32 for value in self.latest.values()):
            raise ValueError("Captured features must be float32")
        if self.verify_capture:
            with torch.no_grad():
                predicted, _ = self.predict(self.latest)
                actual = result.get("mu")
                torch.testing.assert_close(predicted, actual, atol=2e-5, rtol=2e-5)
                self.last_capture_max_mu_error = float((predicted - actual).abs().max())
        result.set(FEATURE_KEY, self.latest)
        self.capture_count += 1

    def augment_observation(self, obs_td, *, template=False):
        if not isinstance(obs_td, TensorDict) or obs_td.batch_dims != 1:
            raise ValueError("Augment per-step TensorDict observations [N]")
        self.validate_runtime()
        if template:
            parameter = next(self.actor.parameters())
            features = TensorDict(
                {
                    name: torch.zeros(
                        *obs_td.batch_size,
                        1 if name == "anchor" else self.block.d_model,
                        device=parameter.device,
                        dtype=torch.float32,
                    )
                    for name in FEATURE_NAMES
                },
                batch_size=obs_td.batch_size,
            )
        else:
            if self.latest is None:
                raise RuntimeError("No complete Actor step has been captured")
            if self.latest.batch_size != obs_td.batch_size:
                raise ValueError("Captured and observation batches differ")
            features = self.latest.clone()
        augmented = obs_td.clone(recurse=False)
        augmented.set(FEATURE_KEY, features)
        return augmented

    def predict(self, features):
        """Recompute the complete mixed action with domain gradients."""
        self.validate_runtime()
        if set(features.keys()) != set(FEATURE_NAMES):
            raise ValueError("Incomplete frozen feature batch")
        if any(value.dtype != torch.float32 or value.requires_grad for value in features.values()):
            raise ValueError("Use detached float32 public features")
        plugin = self.block.domain_plugins[self.domain]
        weight = (
            plugin.router(features["router_x"]).float()
            - features["anchor"]
            + self.block.merge_bias[self.domain]
        ).sigmoid()
        domain = features["shared"] + self.block.routing_scale * plugin(features["x"])
        mixed = (1 - weight) * features["native"] + weight * domain
        hidden = features["residual"] + mixed
        policy = self.actor.actor_module
        return policy.action_mu_head(policy.norm_f(hidden)), hidden

    def sequence_forward(self, obs_td, actions, attn_mask):
        if obs_td.batch_dims != 2 or actions is None:
            raise ValueError("Sequence logp requires [B,T] observations/actions")
        batch, steps = obs_td.batch_size
        if steps > self.actor.actor_module.max_ctx_len:
            raise ValueError("Sequence length exceeds the public KV context")
        if attn_mask is not None and attn_mask.shape != (batch, steps, steps):
            raise ValueError("Attention mask must have shape [B,T,T]")
        if actions.dtype != torch.float32:
            raise ValueError("PPO actions must be float32")
        mu, hidden = self.predict(obs_td.get(FEATURE_KEY))
        if actions.shape != mu.shape:
            raise ValueError("PPO actions and policy means have different shapes")
        sigma_vector = self.actor._sigma_from_params().clamp(
            self.actor.min_sigma, self.actor.max_sigma
        )
        sigma = sigma_vector[None, None, :].expand_as(mu)
        # Identical formulas, epsilons and output shapes to PPOTFActor.
        logp = -0.5 * (
            ((actions - mu) ** 2) / (sigma * sigma + 1e-8)
            + 2.0 * torch.log(sigma + 1e-8)
            + math.log(2.0 * math.pi)
        ).sum(-1, keepdim=True)
        entropy = (0.5 + 0.5 * math.log(2.0 * math.pi) + torch.log(sigma + 1e-8)).sum(
            -1, keepdim=True
        )
        output = obs_td.clone(recurse=False)
        for key, value in (
            ("mu", mu),
            ("sigma", sigma),
            ("actions", actions),
            ("actions_log_prob", logp),
            ("entropy", entropy),
        ):
            output.set(key, value)
        if self.actor.aux_state_pred_enabled:
            auxiliary = self.actor.actor_module.predict_aux_from_pre_moe(hidden)
            for source, destination in AUX_OUTPUT_NAMES.items():
                output.set(destination, auxiliary[source])
        self.cached_sequence_count += 1
        return output

    def close(self):
        if self.actor is None:
            return
        if self._handle is not None:
            self._handle.remove()
        if self.block.dagger_feature_sink == self._sink:
            self.block.dagger_feature_sink = None
        self.actor.__class__ = self._original_class
        del self.actor._ppo_feature_cache
        self.actor = self.block = self.latest = None
        self._trainable_ids = None
        self._pending = {}
        self._capturing = False

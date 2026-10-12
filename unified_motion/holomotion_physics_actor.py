"""Physics-conditioned residual adapter for the original HoloMotion actor.

The upstream actor remains the owner of the Transformer/MoE policy.  This
module only shifts its Gaussian mean with a small deployable dynamics feature
vector.  The last layer is initialized to zero, so loading the adapter around
an existing HoloMotion checkpoint preserves its policy distribution exactly.

The privileged teacher is intentionally not part of this actor.  It trains the
deployable :class:`DynamicsModel` latent through distillation and is removed at
deployment time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal


@dataclass(frozen=True)
class HoloMotionPhysicsActorConfig:
    """Dimensions of the physics residual branch.

    For the K1 foot task this is normally ``3 * physics_width + latent_width``:
    global context, student latent, ordered left-foot token, ordered right-foot
    token. Prediction error stays inside the dynamics GRU.
    """

    physics_feature_dim: int = 200
    hidden_dim: int = 128
    action_dim: int = 22
    residual_limit: float = 0.25

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (self.physics_feature_dim, self.hidden_dim, self.action_dim)
        ):
            raise ValueError("physics actor dimensions must be positive integers")
        if (
            isinstance(self.residual_limit, bool)
            or not isinstance(self.residual_limit, (float, int))
            or not 0.0 < float(self.residual_limit) <= 1.0
        ):
            raise ValueError("residual_limit must be in (0, 1]")


class PhysicsActionResidual(nn.Module):
    """Map deployable physics features to a bounded action-mean correction."""

    def __init__(self, config: HoloMotionPhysicsActorConfig) -> None:
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.LayerNorm(config.physics_feature_dim),
            nn.Linear(config.physics_feature_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )
        # Exact checkpoint compatibility at initialization: both weight and
        # bias must be zero, not merely small.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, physics_features: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(physics_features, torch.Tensor)
            or physics_features.ndim not in (2, 3)
            or physics_features.shape[-1] != self.config.physics_feature_dim
        ):
            raise ValueError(
                "physics_features must have shape [B,D] or [B,T,D] with "
                f"D={self.config.physics_feature_dim}"
            )
        if not torch.isfinite(physics_features).all().item():
            raise ValueError("physics_features contains NaN/Inf")
        return float(self.config.residual_limit) * torch.tanh(self.net(physics_features))


class HoloMotionPhysicsActor(nn.Module):
    """Wrap an original HoloMotion actor with a zero-init physics residual.

    The wrapper accepts the same arguments as the upstream actor plus the
    required keyword ``physics_features``.  During sampling it shifts the
    already sampled action by the same residual.  This preserves the original
    Gaussian noise sample and therefore gives bit-identical actions while the
    residual head is still zero.
    """

    def __init__(
        self,
        base_actor: nn.Module,
        config: HoloMotionPhysicsActorConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base_actor, nn.Module):
            raise TypeError("base_actor must be a torch module")
        self.base_actor = base_actor
        self.config = config or HoloMotionPhysicsActorConfig()
        self.physics_residual = PhysicsActionResidual(self.config)

    @property
    def actor_module(self) -> Any:
        """Expose the upstream MoE module used by existing PPO diagnostics."""
        return self.base_actor.actor_module

    def reset_kv_cache(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_actor.reset_kv_cache(*args, **kwargs)

    def clear_env_cache(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_actor.clear_env_cache(*args, **kwargs)

    def forward(
        self,
        obs_td: Any,
        actions: torch.Tensor | None = None,
        mode: str = "sampling",
        attn_mask: torch.Tensor | None = None,
        *,
        physics_features: torch.Tensor,
        physics_confidence: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        output = self.base_actor(
            obs_td,
            actions=actions,
            mode=mode,
            attn_mask=attn_mask,
            **kwargs,
        )
        mu = output.get("mu")
        sigma = output.get("sigma")
        residual = self.physics_residual(physics_features)
        if physics_confidence is not None:
            expected = residual.shape[:-1] + (1,)
            if physics_confidence.shape != expected:
                raise ValueError(f"physics_confidence must have shape {expected}")
            if ((physics_confidence < 0) | (physics_confidence > 1)).any().item():
                raise ValueError("physics_confidence must be in [0,1]")
            residual = residual * physics_confidence
        if residual.shape != mu.shape:
            raise ValueError(
                f"physics residual shape {tuple(residual.shape)} differs from "
                f"HoloMotion mean {tuple(mu.shape)}"
            )
        corrected_mu = mu + residual
        output.set("mu", corrected_mu)

        if mode == "inference":
            output.set("actions", corrected_mu)
            return output

        if mode == "sampling":
            # The base action is mu + sigma*epsilon. Shifting it by residual is
            # exactly a sample from N(mu+residual, sigma) without a second RNG draw.
            corrected_actions = output.get("actions") + residual
        else:
            if actions is None:
                raise ValueError(f"actions must be supplied for actor mode {mode!r}")
            corrected_actions = actions

        distribution = Normal(corrected_mu, sigma)
        output.set("actions", corrected_actions)
        # Keep the upstream scalar-event convention [..., 1].  Dropping this
        # axis would make PPO subtract [B,T] from stored [B,T,1] and silently
        # broadcast the ratio to [B,T,T].
        output.set(
            "actions_log_prob",
            distribution.log_prob(corrected_actions).sum(-1, keepdim=True),
        )
        output.set("entropy", distribution.entropy().sum(-1, keepdim=True))
        return output


def dynamics_actor_features(
    context: torch.Tensor,
    latent: torch.Tensor,
    contact_tokens: torch.Tensor,
) -> torch.Tensor:
    """Keep fixed left/right foot identity in the deployable residual input."""
    if context.shape[:-1] != latent.shape[:-1]:
        raise ValueError("dynamics context and latent batch axes differ")
    if contact_tokens.shape[:-2] != context.shape[:-1] or contact_tokens.shape[-2] != 2:
        raise ValueError("K1 physics residual requires fixed [left,right] contact tokens")
    if any(value.device != context.device or value.dtype != context.dtype
           for value in (latent, contact_tokens)):
        raise ValueError("dynamics features device/dtype differ")
    # Flattening a fixed slot axis is intentional here: unlike set mean/max,
    # weights connected to the left token are independent of right-token weights.
    return torch.cat((context, latent, contact_tokens.flatten(-2)), dim=-1)

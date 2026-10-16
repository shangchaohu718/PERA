"""Interchangeable Actors for the SAME K1 motion-tracking observation/task.

The custom backends reuse this project's temporal cores. Their input adapter
consumes all 506 HoloMotion observations without fabricating missing reference
velocities. The older ControlGoal-based policies remain a separate input schema.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.distributions import Normal

from .holomotion_original_algorithm import (
    ACTOR_CURRENT_TERM_WIDTHS, ACTOR_FUTURE_TERM_WIDTHS,
    CRITIC_CURRENT_TERM_WIDTHS, CRITIC_FUTURE_TERM_WIDTHS,
    build_original_holomotion_networks,
)
from .models.encoder import EncodedMotion
from .models.physics_policy import PhysicsPolicyConfig, build_reference_backbone
from .models.physics import _mlp

ACTOR_BACKENDS = ("holomotion", "mlp", "transformer", "moe", "gru", "lstm", "lru")


@dataclass(frozen=True)
class K1ActorConfig:
    backbone: str = "transformer"
    d_model: int = 128
    hidden_dim: int = 256
    n_layers: int = 2
    context_len: int = 32
    num_experts: int = 4
    init_std: float = 1.
    recurrent_hidden_dim: int | None = None
    lru_state_dim: int | None = None
    lru_scan: str = "parallel"
    moe_balance_coef: float = .01
    teacher_latent_dim: int = 0
    critic_privileged_dim: int = 0

    def __post_init__(self):
        if self.backbone == "dense":
            object.__setattr__(self, "backbone", "transformer")
        if self.backbone not in ACTOR_BACKENDS[1:]:
            raise ValueError("custom K1 backbone must be mlp/transformer/moe/gru/lstm/lru")
        if not math.isfinite(self.moe_balance_coef) or self.moe_balance_coef < 0:
            raise ValueError("moe_balance_coef must be finite and non-negative")
        if type(self.teacher_latent_dim) is not int or self.teacher_latent_dim < 0:
            raise ValueError("teacher_latent_dim must be a non-negative integer")
        if type(self.critic_privileged_dim) is not int or self.critic_privileged_dim < 0:
            raise ValueError("critic_privileged_dim must be a non-negative integer")
        if self.backbone != "lru" and (self.lru_state_dim is not None or self.lru_scan != "parallel"):
            raise ValueError("lru settings require the lru backbone")
        self.policy_config()  # Reuse the existing core's dimension/type validation.

    def policy_config(self):
        return PhysicsPolicyConfig(backbone="dense" if self.backbone == "mlp" else self.backbone,
            d_model=self.d_model, input_hidden_dim=self.hidden_dim, n_layers=self.n_layers,
            n_heads=4, n_kv_heads=2, context_len=self.context_len,
            dense_hidden_dim=self.hidden_dim, num_fine_experts=self.num_experts,
            num_shared_experts=1, top_k=2, expert_hidden_dim=self.hidden_dim,
            router_hidden_dim=self.d_model, init_std=self.init_std,
            recurrent_hidden_dim=self.recurrent_hidden_dim,
            lru_state_dim=self.lru_state_dim, lru_scan=self.lru_scan)


def _terms(data, terms):
    return torch.cat([data.get("unified/"+name) for name, _ in terms], dim=-1)


class K1CustomActor(nn.Module):
    """Existing custom cores behind the sampling/sequence API used by K1 PPO.

    Actor memory resets at each rollout, identically during collection, PPO and
    frozen evaluation. Dynamics episode memory is independent and continuous.
    """
    def __init__(self, config: K1ActorConfig):
        super().__init__()
        self.config = config
        width = config.d_model
        self.current_encoder = nn.Sequential(nn.Linear(106, width), nn.LayerNorm(width))
        self.reference_encoder = nn.Linear(40, width)
        self.reference_position = nn.Parameter(torch.zeros(10, width))
        nn.init.normal_(self.reference_position, std=.01)
        mlp_input = width + 10*(width+1) + (width if config.teacher_latent_dim else 0)
        self.backbone = (_mlp(mlp_input, config.hidden_dim, width)
                         if config.backbone == "mlp" else build_reference_backbone(config.policy_config()))
        self.mean_head = nn.Linear(width, 22)
        nn.init.normal_(self.mean_head.weight, std=.01)
        nn.init.zeros_(self.mean_head.bias)
        self.log_std = nn.Parameter(torch.full((22,), math.log(config.init_std)))
        # Robot/reference/physics are encoded independently.  For the MLP, z is
        # kept as an explicit third branch all the way into the policy trunk;
        # temporal backbones use a learned concatenation fusion for the state
        # token instead of the old ambiguous element-wise addition.
        self.teacher_latent_encoder = (
            nn.Sequential(nn.LayerNorm(config.teacher_latent_dim),
                          nn.Linear(config.teacher_latent_dim, width, bias=False))
            if config.teacher_latent_dim else None
        )
        self.teacher_state_fusion = (
            nn.Sequential(nn.Linear(2*width, width), nn.LayerNorm(width))
            if config.teacher_latent_dim and config.backbone != "mlp" else None
        )
        self._cache = None
        self._starts = None

    @property
    def actor_module(self):
        return self.backbone

    def reset_kv_cache(self, num_envs, device):
        self._cache = None
        self._starts = torch.ones(num_envs, dtype=torch.bool, device=device)

    def clear_env_cache(self, ids):
        if self._starts is None:
            raise RuntimeError("reset Actor runtime before clearing rows")
        self._starts[ids] = True

    def _encode(self, data, teacher_latent=None):
        current = _terms(data, ACTOR_CURRENT_TERM_WIDTHS)
        future = _terms(data, ACTOR_FUTURE_TERM_WIDTHS)
        if current.shape[-1] != 106 or future.shape != current.shape[:-1]+(10, 40):
            raise ValueError("custom Actor requires the complete K1 506 observation contract")
        if not torch.isfinite(current).all() or not torch.isfinite(future).all():
            raise ValueError("non-finite K1 Actor observations")
        state = self.current_encoder(current)
        if self.teacher_latent_encoder is None:
            if teacher_latent is not None:
                raise ValueError("Actor was not configured for a Teacher latent observation")
        else:
            expected = current.shape[:-1] + (self.config.teacher_latent_dim,)
            if (not isinstance(teacher_latent, torch.Tensor)
                    or teacher_latent.shape != expected
                    or teacher_latent.device != current.device
                    or teacher_latent.dtype != current.dtype
                    or not torch.isfinite(teacher_latent).all()):
                raise ValueError(f"teacher_latent must be finite with shape {expected}")
            physics = self.teacher_latent_encoder(teacher_latent)
            if self.teacher_state_fusion is not None:
                state = self.teacher_state_fusion(torch.cat((state, physics), -1))
        refs = self.reference_encoder(future)+self.reference_position
        return (EncodedMotion("k1_22dof", state, refs,
                    torch.ones(refs.shape[:-1], device=refs.device, dtype=torch.bool)),
                physics if self.teacher_latent_encoder is not None else None)

    def forward(self, data, actions=None, mode="sampling", attn_mask=None,
                update_obs_norm=False, *, teacher_latent=None):
        if update_obs_norm:
            raise ValueError("custom K1 inputs use the fixed task scaling; no running normalization")
        if mode not in ("sampling", "inference", "sequence_logp"):
            raise ValueError("unknown K1 Actor mode")
        encoded, physics = self._encode(data, teacher_latent)
        lead = encoded.state_token.shape[:-1]
        if mode == "sequence_logp":
            if len(lead) != 2:
                raise ValueError("PPO replay requires [B,T] observations")
            b, t = lead
            if attn_mask is None or attn_mask.shape != (b,t,t) or attn_mask.dtype != torch.bool:
                raise ValueError("PPO replay requires the episode causal mask")
            starts = torch.ones((b,t), dtype=torch.bool, device=encoded.state_token.device)
            starts[:, 1:] = ~attn_mask[:, torch.arange(1,t), torch.arange(t-1)]
            segments = starts.long().cumsum(1)
            expected = ((segments[:, :, None] == segments[:, None, :]) &
                        torch.ones(t,t,device=starts.device,dtype=torch.bool).tril())
            if not torch.equal(expected, attn_mask):
                raise ValueError("mask must describe causal, contiguous episodes")
            if self.config.backbone == "mlp":
                features = self.backbone(torch.cat((encoded.flatten_for_mlp(), physics), -1)
                                         if physics is not None else encoded.flatten_for_mlp())
                balance = features.new_zeros(())
            else:
                output = self.backbone(encoded, valid_mask=torch.ones_like(starts), episode_start=starts)
                features, balance = output.features, output.balance_loss
        else:
            if len(lead) != 1 or self._starts is None or self._starts.shape != lead:
                raise ValueError("reset Actor runtime for this batch before sampling")
            if self.training:
                raise RuntimeError("Actor sampling requires eval mode")
            if self.config.backbone == "mlp":
                features = self.backbone(torch.cat((encoded.flatten_for_mlp(), physics), -1)
                                         if physics is not None else encoded.flatten_for_mlp())
            else:
                output, self._cache = self.backbone.step(encoded, env_ids=tuple(range(lead[0])),
                    cache=self._cache, episode_start=self._starts)
                features = output.features
            self._starts.zero_()
            balance = features.new_zeros(())
        mu = self.mean_head(features)
        sigma = self.log_std.exp().expand_as(mu)
        distribution = Normal(mu, sigma)
        if mode == "sequence_logp":
            if actions is None or actions.shape != mu.shape:
                raise ValueError("PPO actions must match Actor output")
        else:
            actions = mu if mode == "inference" else distribution.sample()
        data.set("mu", mu); data.set("sigma", sigma); data.set("actions", actions)
        data.set("actions_log_prob", distribution.log_prob(actions).sum(-1, keepdim=True))
        data.set("entropy", distribution.entropy().sum(-1, keepdim=True))
        data.set("actor_aux_loss", (balance*self.config.moe_balance_coef).expand(*lead, 1))
        return data


class K1CustomCritic(nn.Module):
    def __init__(self, hidden_dim, privileged_dim=0):
        super().__init__()
        self.privileged_dim = privileged_dim
        self.net = _mlp(843+privileged_dim, hidden_dim, 1)

    def forward(self, data, update_obs_norm=False, *, critic_privileged=None):
        current = _terms(data, CRITIC_CURRENT_TERM_WIDTHS)
        future = torch.cat([data.get("unified/"+name).flatten(-2)
                            for name, _ in CRITIC_FUTURE_TERM_WIDTHS], dim=-1)
        inputs = torch.cat((current, future), dim=-1)
        if self.privileged_dim:
            expected = inputs.shape[:-1] + (self.privileged_dim,)
            if (not isinstance(critic_privileged, torch.Tensor)
                    or critic_privileged.shape != expected
                    or critic_privileged.device != inputs.device
                    or critic_privileged.dtype != inputs.dtype
                    or not torch.isfinite(critic_privileged).all()):
                raise ValueError(f"critic_privileged must be finite with shape {expected}")
            inputs = torch.cat((inputs, critic_privileged), -1)
        elif critic_privileged is not None:
            raise ValueError("Critic was not configured for privileged physics")
        data.set("values", self.net(inputs))
        return data


def custom_network_config(config):
    privileged = bool(config.critic_privileged_dim)
    return dict(schema=2 if privileged else 1, backend=config.backbone,
                observation_schema=("holomotion_k1_506_843_plus_physics_v2" if privileged
                                    else "holomotion_k1_506_843_v1"),
                custom_actor=asdict(config))


def build_k1_networks(network_config, *, device="cpu", holomotion_root=None):
    """One checkpoint factory; untagged historical configs are HoloMotion."""
    backend = network_config.get("backend", "holomotion")
    if backend == "holomotion":
        return build_original_holomotion_networks(network_config, device=device, holomotion_root=holomotion_root)
    schema = network_config.get("schema")
    expected = ("holomotion_k1_506_843_plus_physics_v2" if schema == 2
                else "holomotion_k1_506_843_v1")
    if schema not in (1, 2) or network_config.get("observation_schema") != expected:
        raise ValueError("unknown custom K1 network schema")
    config = K1ActorConfig(**network_config["custom_actor"])
    if backend != config.backbone:
        raise ValueError("checkpoint Actor backend declarations disagree")
    if (schema == 2) != bool(config.critic_privileged_dim):
        raise ValueError("network schema and Critic privilege declaration disagree")
    return (K1CustomActor(config).to(device).eval(),
            K1CustomCritic(config.hidden_dim, config.critic_privileged_dim).to(device).eval())

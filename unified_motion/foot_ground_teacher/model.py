"""Lightweight shared-foot causal Transformer for next-step physics."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FootGroundTeacherConfig:
    student_observation_dim: int
    privileged_current_dim: int
    foot_current_dim: int
    dof: int = 22
    reference_intent_dim: int = 0
    latent_dim: int = 0
    latent_group_dims: tuple[int, int, int] = (0, 0, 0)
    width: int = 128
    foot_width: int = 48
    heads: int = 4
    layers: int = 3
    context_steps: int = 32
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("student_observation_dim", "privileged_current_dim",
                     "foot_current_dim", "dof", "width", "foot_width",
                     "heads", "layers", "context_steps"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("reference_intent_dim", "latent_dim"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.latent_group_dims, (tuple, list)) or \
                len(self.latent_group_dims) != 3 or \
                any(not isinstance(value, int) or value < 0
                    for value in self.latent_group_dims):
            raise ValueError("latent_group_dims must contain three non-negative integers")
        grouped = any(self.latent_group_dims)
        if grouped and (any(value == 0 for value in self.latent_group_dims)
                        or sum(self.latent_group_dims) != self.latent_dim):
            raise ValueError("grouped contact/force/dynamics dimensions must be positive "
                             "and sum to latent_dim")
        if self.width % self.heads or self.width % 2:
            raise ValueError("width must be even and divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")


class FootGroundTeacher(nn.Module):
    """Predicts physical labels at ``t+1`` from information available at ``t``.

    Left and right rows pass through the exact same ``foot_encoder`` and
    ``contact_head``. A one-hot side identifier and foot-local features retain
    identity without duplicating a large network. ``privileged_current`` may
    contain simulator truth at t, but callers must never place t+1 labels in it.
    """

    CONTACT_OUTPUT_WIDTH = 1 + 4 + 3 + 3 + 1 + 2 + 1 + 1 + 2 + 1
    CONTACT_STATE_OUTPUT_WIDTH = 1 + 4 + 3 + 3 + 1 + 2 + 1
    FORCE_OUTPUT_WIDTH = 1 + 2 + 1

    def __init__(self, config: FootGroundTeacherConfig):
        super().__init__()
        self.config = config
        cfg = config
        self.foot_encoder = nn.Sequential(
            nn.LayerNorm(cfg.foot_current_dim + 2),
            nn.Linear(cfg.foot_current_dim + 2, cfg.foot_width), nn.SiLU(),
            nn.Linear(cfg.foot_width, cfg.foot_width), nn.SiLU(),
        )
        global_dim = (cfg.student_observation_dim + cfg.privileged_current_dim
                      + cfg.reference_intent_dim
                      + cfg.dof + 2 * cfg.foot_width)
        self.input_projection = nn.Sequential(nn.LayerNorm(global_dim),
                                               nn.Linear(global_dim, cfg.width))
        layer = nn.TransformerEncoderLayer(
            cfg.width, cfg.heads, 4 * cfg.width, dropout=cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, cfg.layers, enable_nested_tensor=False)
        # latent_dim=0 is the exact V1/V2 checkpoint-compatible architecture.
        # A positive value creates an explicit actor-facing bottleneck.  Every
        # physical decoder then consumes z (plus only a fixed foot-side ID),
        # never the 128-D Transformer hidden state or raw foot features.
        self.latent_head = nn.Identity() if cfg.latent_dim == 0 else nn.Sequential(
            nn.LayerNorm(cfg.width), nn.Linear(cfg.width, cfg.latent_dim), nn.Tanh())
        self.grouped_latent = any(cfg.latent_group_dims)
        decoder_width = cfg.width if cfg.latent_dim == 0 else cfg.latent_dim
        contact_decoder_width = (cfg.latent_group_dims[0]
                                 if self.grouped_latent else decoder_width)
        contact_input_width = (decoder_width + cfg.foot_width + 2
                               if cfg.latent_dim == 0 else decoder_width + 2)
        if self.grouped_latent:
            contact_input_width = contact_decoder_width + 2
        self.contact_head = nn.Sequential(
            nn.LayerNorm(contact_input_width),
            nn.Linear(contact_input_width, cfg.width), nn.SiLU(),
            nn.Linear(cfg.width, (self.CONTACT_STATE_OUTPUT_WIDTH
                                  if self.grouped_latent else self.CONTACT_OUTPUT_WIDTH)),
        )
        if self.grouped_latent:
            force_width = cfg.latent_group_dims[1]
            self.force_head = nn.Sequential(
                nn.LayerNorm(force_width + 2),
                nn.Linear(force_width + 2, cfg.width), nn.SiLU(),
                nn.Linear(cfg.width, self.FORCE_OUTPUT_WIDTH),
            )
        else:
            self.force_head = None
        load_width = cfg.latent_group_dims[1] if self.grouped_latent else decoder_width
        dynamics_width = cfg.latent_group_dims[2] if self.grouped_latent else decoder_width
        self.load_head = nn.Sequential(
            nn.LayerNorm(load_width), nn.Linear(load_width, 2))
        self.dynamics_head = nn.Sequential(
            nn.LayerNorm(dynamics_width), nn.Linear(dynamics_width, cfg.width), nn.SiLU(),
            nn.Linear(cfg.width, 2 * cfg.dof + 3),
        )

    def shared_parameters(self):
        """Parameters used to audit contact/dynamics gradient interference."""
        yield from self.input_projection.parameters()
        yield from self.transformer.parameters()
        yield from self.latent_head.parameters()

    def decode_physics(self, latent_z: torch.Tensor,
                       foot_embedding: torch.Tensor | None = None,
                       ) -> dict[str, torch.Tensor]:
        """Decode all physical heads from an explicit z.

        New bottleneck models require only ``z``; the side identifier is fixed
        structure created here. ``foot_embedding`` exists solely for exact
        legacy checkpoint compatibility when ``latent_dim == 0``.
        """
        cfg = self.config
        expected_width = cfg.width if cfg.latent_dim == 0 else cfg.latent_dim
        if latent_z.ndim != 3 or latent_z.shape[-1] != expected_width:
            raise ValueError(f"latent_z must be [B,T,{expected_width}]")
        lead = latent_z.shape[:2]
        side = torch.eye(2, device=latent_z.device, dtype=latent_z.dtype)
        side = side.view(1, 1, 2, 2).expand(*lead, 2, 2)
        if self.grouped_latent:
            contact_z, force_z, dynamics_z = torch.split(
                latent_z, self.config.latent_group_dims, dim=-1)
            repeated = contact_z.unsqueeze(-2).expand(
                *lead, 2, self.config.latent_group_dims[0])
        else:
            contact_z = force_z = dynamics_z = latent_z
            repeated = latent_z.unsqueeze(-2).expand(*lead, 2, expected_width)
        if cfg.latent_dim == 0:
            if foot_embedding is None or foot_embedding.shape != lead + (2, cfg.foot_width):
                raise ValueError("legacy decoder requires [B,T,2,foot_width] embedding")
            contact_input = torch.cat((repeated, foot_embedding, side), -1)
        else:
            if foot_embedding is not None:
                raise ValueError("explicit-z decoder does not accept raw foot embedding")
            contact_input = torch.cat((repeated, side), -1)
        contact = self.contact_head(contact_input)
        cursor = 0

        def take(width):
            nonlocal cursor
            result = contact[..., cursor:cursor + width]
            cursor += width
            return result

        output = {
            "contact_logits": take(1).squeeze(-1),
            "contact_mode_logits": take(4),
            "contact_point_local": take(3),
            "contact_normal": F.normalize(take(3), dim=-1, eps=1e-8),
            "relative_velocity_normal": take(1).squeeze(-1),
            "relative_velocity_tangent": take(2),
            "slip_speed": F.softplus(take(1).squeeze(-1)),
        }
        expected_contact_width = (self.CONTACT_STATE_OUTPUT_WIDTH
                                  if self.grouped_latent else self.CONTACT_OUTPUT_WIDTH)
        if self.grouped_latent:
            repeated_force = force_z.unsqueeze(-2).expand(
                *lead, 2, self.config.latent_group_dims[1])
            force = self.force_head(torch.cat((repeated_force, side), -1))
            force_cursor = 0

            def take_force(width):
                nonlocal force_cursor
                result = force[..., force_cursor:force_cursor + width]
                force_cursor += width
                return result

            output.update(
                normal_force=F.softplus(take_force(1).squeeze(-1)),
                tangential_force=take_force(2),
                contact_impulse=F.softplus(take_force(1).squeeze(-1)),
            )
            if force_cursor != self.FORCE_OUTPUT_WIDTH:
                raise RuntimeError("force head layout mismatch")
        else:
            output.update(
                normal_force=F.softplus(take(1).squeeze(-1)),
                tangential_force=take(2),
                contact_impulse=F.softplus(take(1).squeeze(-1)),
            )
        output["foot_load_ratio"] = self.load_head(force_z).softmax(-1)
        if cursor != expected_contact_width:
            raise RuntimeError("contact head layout mismatch")
        dynamics = self.dynamics_head(dynamics_z)
        output.update(
            delta_joint_velocity=dynamics[..., :cfg.dof],
            delta_root_angular_velocity=dynamics[..., cfg.dof:cfg.dof + 3],
            joint_velocity_residual=dynamics[..., cfg.dof + 3:],
        )
        return output

    def _mask(self, episode_start: torch.Tensor) -> torch.Tensor:
        batch, steps = episode_start.shape
        index = torch.arange(steps, device=episode_start.device)
        segment = episode_start.long().cumsum(1)
        distance = index[:, None] - index[None, :]
        allowed = ((segment[:, :, None] == segment[:, None, :])
                   & (distance >= 0) & (distance < self.config.context_steps))
        return (~allowed).repeat_interleave(self.config.heads, dim=0)

    def forward(self, student_observation: torch.Tensor,
                privileged_current: torch.Tensor, executed_action: torch.Tensor,
                foot_current: torch.Tensor, episode_start: torch.Tensor,
                reference_intent: torch.Tensor | None = None,
                ) -> dict[str, torch.Tensor]:
        cfg = self.config
        if student_observation.ndim != 3:
            raise ValueError("Teacher inputs must be [B,T,D]")
        lead = student_observation.shape[:2]
        expected = (
            (student_observation, lead + (cfg.student_observation_dim,)),
            (privileged_current, lead + (cfg.privileged_current_dim,)),
            (executed_action, lead + (cfg.dof,)),
            (foot_current, lead + (2, cfg.foot_current_dim)),
            (episode_start, lead),
        )
        if any(value.shape != shape for value, shape in expected):
            raise ValueError("Teacher input shape mismatch")
        if episode_start.dtype != torch.bool:
            raise TypeError("episode_start must be boolean")
        if cfg.reference_intent_dim:
            expected_reference = lead + (cfg.reference_intent_dim,)
            if reference_intent is None or reference_intent.shape != expected_reference:
                raise ValueError(f"reference_intent must have shape {expected_reference}")
        elif reference_intent is None:
            reference_intent = student_observation.new_zeros(lead + (0,))
        elif reference_intent.shape != lead + (0,):
            raise ValueError("reference_intent was supplied to a model with dimension zero")
        side = torch.eye(2, device=foot_current.device, dtype=foot_current.dtype)
        side = side.view(1, 1, 2, 2).expand(*lead, 2, 2)
        foot = self.foot_encoder(torch.cat((foot_current, side), -1))
        combined = torch.cat((student_observation, privileged_current, reference_intent,
                              executed_action, foot.flatten(-2)), -1)
        hidden = self.input_projection(combined)
        position = torch.arange(lead[1], device=hidden.device, dtype=hidden.dtype)
        frequency = torch.exp(torch.arange(0, cfg.width, 2, device=hidden.device,
                                           dtype=hidden.dtype)
                              * (-math.log(10000.0) / cfg.width))
        phase = position[:, None] * frequency[None]
        encoding = torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)
        if encoding.shape[-1] < cfg.width:
            encoding = F.pad(encoding, (0, cfg.width - encoding.shape[-1]))
        hidden = self.transformer(hidden + encoding[None], mask=self._mask(episode_start))
        latent_z = self.latent_head(hidden)
        output = self.decode_physics(
            latent_z, foot_embedding=(foot if cfg.latent_dim == 0 else None))
        output["latent_z"] = latent_z
        if self.grouped_latent:
            (output["latent_z_contact"], output["latent_z_force"],
             output["latent_z_dynamics"]) = torch.split(
                 latent_z, cfg.latent_group_dims, dim=-1)
        return output

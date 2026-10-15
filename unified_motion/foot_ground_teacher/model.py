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
        if not isinstance(self.reference_intent_dim, int) or self.reference_intent_dim < 0:
            raise ValueError("reference_intent_dim must be a non-negative integer")
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
        self.contact_head = nn.Sequential(
            nn.LayerNorm(cfg.width + cfg.foot_width + 2),
            nn.Linear(cfg.width + cfg.foot_width + 2, cfg.width), nn.SiLU(),
            nn.Linear(cfg.width, self.CONTACT_OUTPUT_WIDTH),
        )
        self.load_head = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, 2))
        self.dynamics_head = nn.Sequential(
            nn.LayerNorm(cfg.width), nn.Linear(cfg.width, cfg.width), nn.SiLU(),
            nn.Linear(cfg.width, 2 * cfg.dof + 3),
        )

    def shared_parameters(self):
        """Parameters used to audit contact/dynamics gradient interference."""
        yield from self.input_projection.parameters()
        yield from self.transformer.parameters()

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
        repeated = hidden.unsqueeze(-2).expand(*lead, 2, cfg.width)
        contact = self.contact_head(torch.cat((repeated, foot, side), -1))
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
            "normal_force": F.softplus(take(1).squeeze(-1)),
            "tangential_force": take(2),
            "contact_impulse": F.softplus(take(1).squeeze(-1)),
            "foot_load_ratio": self.load_head(hidden).softmax(-1),
        }
        if cursor != self.CONTACT_OUTPUT_WIDTH:
            raise RuntimeError("contact head layout mismatch")
        dynamics = self.dynamics_head(hidden)
        output.update(
            delta_joint_velocity=dynamics[..., :cfg.dof],
            delta_root_angular_velocity=dynamics[..., cfg.dof:cfg.dof + 3],
            joint_velocity_residual=dynamics[..., cfg.dof + 3:],
        )
        return output

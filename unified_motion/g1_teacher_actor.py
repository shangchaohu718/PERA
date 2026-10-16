"""Two-stage privileged-physics Teacher and Actor models for Unitree G1.

This module is deliberately independent from the original HoloMotion K1
506/843 observation contract.  G1 has a different action width, so reusing the
K1 action head or flat observation schema would silently associate values with
the wrong joints.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Independent, Normal

from .privileged_teacher import (
    GROUPED_PHYSICS_LATENT,
    PrivilegedTeacherConfig,
    PrivilegedTransformerTeacher,
)
from .robot_contract import RobotContract, get_robot_contract

# Schema 4 adds the routed contact8/force16/kinematics16/dynamics24 z64.
G1_TEACHER_ACTOR_SCHEMA = 4
G1_ROBOT_IDS = (
    "g1_23dof", "g1_29dof", "g1_43dof", "g1_37dof_isaac",
)
TEACHER_TRAINING = "teacher_training"
TEACHER_GUIDED_ACTOR = "teacher_guided_actor"


@dataclass(frozen=True)
class G1TeacherActorConfig:
    """Robot-derived dimensions shared by the two G1 training stages."""

    # Match the official HoloMotion v1.4 G1 checkpoint/action contract by
    # default.  The 37-DoF Isaac Lab asset is a different mechanism and must
    # never be selected implicitly.
    robot_id: str = "g1_29dof"
    latent_width: int = 64
    actor_width: int = 256
    actor_hidden_dim: int = 256
    reference_steps: int = 10
    init_std: float = 1.0
    system_id_dim: int = 2

    def __post_init__(self) -> None:
        if self.robot_id not in G1_ROBOT_IDS:
            raise ValueError(f"robot_id must be one of {G1_ROBOT_IDS}")
        for name in (
            "latent_width", "actor_width", "actor_hidden_dim",
            "reference_steps", "system_id_dim",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.init_std, bool)
                or not isinstance(self.init_std, (int, float))
                or not math.isfinite(self.init_std) or self.init_std <= 0):
            raise ValueError("init_std must be positive and finite")
        if self.latent_width != sum(width for _, width in GROUPED_PHYSICS_LATENT):
            raise ValueError("G1 Teacher latent must use grouped z64")

    @property
    def robot(self) -> RobotContract:
        return get_robot_contract(self.robot_id)

    @property
    def state_dim(self) -> int:
        # body_obs: root angular velocity, projected gravity, q, qd, last action
        return 6 + 3 * self.robot.dof

    @property
    def reference_dim(self) -> int:
        # Shared reference contract used by ReferenceObservationBuilder.
        return 16 + 2 * self.robot.dof


@dataclass(frozen=True)
class G1ActorOutput:
    mean: torch.Tensor
    std: torch.Tensor
    joint_names: tuple[str, ...]

    def distribution(self) -> Independent:
        return Independent(Normal(self.mean, self.std), 1)


class G1TeacherConditionedActor(nn.Module):
    """G1 Actor whose third input branch is a frozen Teacher latent."""

    def __init__(self, config: G1TeacherActorConfig) -> None:
        super().__init__()
        self.config = config
        width = config.actor_width
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, width),
            nn.SiLU(),
        )
        self.reference_encoder = nn.Sequential(
            nn.LayerNorm(config.reference_dim),
            nn.Linear(config.reference_dim, width),
            nn.SiLU(),
        )
        self.teacher_encoder = nn.Sequential(
            nn.LayerNorm(config.latent_width),
            nn.Linear(config.latent_width, width, bias=False),
            nn.SiLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(3 * width, config.actor_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.actor_hidden_dim, width),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(width, config.robot.dof)
        nn.init.normal_(self.mean_head.weight, std=0.01)
        nn.init.zeros_(self.mean_head.bias)
        self.log_std = nn.Parameter(torch.full(
            (config.robot.dof,), math.log(config.init_std)))

    def forward(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
        reference_mask: torch.Tensor,
        teacher_latent: torch.Tensor,
    ) -> G1ActorOutput:
        lead = state.shape[:-1]
        cfg = self.config
        if state.shape[-1:] != (cfg.state_dim,):
            raise ValueError(f"G1 state must end in {cfg.state_dim} values")
        if reference.shape != lead + (cfg.reference_steps, cfg.reference_dim):
            raise ValueError("G1 reference shape disagrees with the configured horizon")
        if reference_mask.shape != lead + (cfg.reference_steps,) or reference_mask.dtype != torch.bool:
            raise ValueError("G1 reference mask must be boolean and match the reference horizon")
        if teacher_latent.shape != lead + (cfg.latent_width,):
            raise ValueError("Teacher latent shape disagrees with the G1 Actor contract")
        tensors = (state, reference, teacher_latent)
        if (any(value.device != state.device or value.dtype != state.dtype for value in tensors)
                or any(not torch.isfinite(value).all() for value in tensors)):
            raise ValueError("G1 Actor inputs must be finite and share device/dtype")
        encoded_reference = self.reference_encoder(reference)
        mask = reference_mask.unsqueeze(-1)
        reference_summary = torch.where(mask, encoded_reference, 0.0).sum(-2)
        reference_summary = reference_summary / mask.sum(-2).clamp_min(1)
        features = self.trunk(torch.cat((
            self.state_encoder(state),
            reference_summary,
            self.teacher_encoder(teacher_latent.detach()),
        ), -1))
        mean = self.mean_head(features)
        std = self.log_std.exp().expand_as(mean)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("G1 Actor produced non-finite distribution parameters")
        return G1ActorOutput(mean, std, cfg.robot.joint_names)


class G1PrivilegedValueCritic(nn.Module):
    """Independent value network over the exact Teacher privilege tensor.

    Teacher and Critic receive identical information, but do not share
    parameters or gradients.  The Critic therefore cannot reshape the
    Teacher's physical latent through its value loss.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        width: int,
        heads: int,
        layers: int,
        context_steps: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        self.heads = heads
        self.context_steps = context_steps
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, width)
        )
        layer = nn.TransformerEncoderLayer(
            width, heads, 2 * width, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, layers, enable_nested_tensor=False
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 1)
        )

    def forward(
        self,
        privileged_history: torch.Tensor,
        episode_start: torch.Tensor,
    ) -> torch.Tensor:
        if (
            privileged_history.ndim != 3
            or privileged_history.shape[-1] != self.input_dim
            or episode_start.shape != privileged_history.shape[:2]
            or episode_start.dtype != torch.bool
        ):
            raise ValueError(
                "G1 Critic requires [B,T,D] shared privilege and boolean [B,T] starts"
            )
        if not torch.isfinite(privileged_history).all():
            raise ValueError("G1 Critic privilege contains NaN/Inf")
        _, steps, _ = privileged_history.shape
        if steps < 1:
            raise ValueError("G1 Critic history must contain at least one step")
        index = torch.arange(steps, device=privileged_history.device)
        segment = episode_start.long().cumsum(1)
        distance = index[:, None] - index[None, :]
        allowed = (
            (segment[:, :, None] == segment[:, None, :])
            & (distance >= 0)
            & (distance < self.context_steps)
        )
        blocked = (~allowed).repeat_interleave(self.heads, dim=0)
        last_start = torch.where(episode_start, index[None], 0).cummax(1).values
        age = index[None] - last_start
        frequency = torch.exp(
            torch.arange(
                0, self.width, 2,
                device=privileged_history.device,
                dtype=privileged_history.dtype,
            ) * (-math.log(10000.0) / self.width)
        )
        phase = age.unsqueeze(-1) * frequency
        position = torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)
        hidden = self.encoder(
            self.input_projection(privileged_history) + position,
            mask=blocked,
        )
        values = self.value_head(hidden)
        if values.shape != privileged_history.shape[:2] + (1,):
            raise RuntimeError("G1 Critic produced an invalid value shape")
        return values


class G1TeacherActorModel(nn.Module):
    """Phase-safe owner of the G1 Teacher and its guided Actor.

    Stage 1 (``teacher_training``) exposes the Teacher forward pass and freezes
    Actor/Critic.  Stage 2 freezes/evaluates the Teacher, detaches its latent,
    and trains independent Actor and Critic networks.
    """

    def __init__(
        self,
        config: G1TeacherActorConfig | None = None,
        teacher_config: PrivilegedTeacherConfig | None = None,
        *,
        phase: str = TEACHER_TRAINING,
    ) -> None:
        super().__init__()
        self.config = config or G1TeacherActorConfig()
        self.teacher_config = teacher_config or PrivilegedTeacherConfig(
            objective_version=5, critic_dim=1046, width=128, heads=8, layers=4,
            context_steps=64, train_envs=64, contact_slots=4)
        if self.teacher_config.objective_version != 5:
            raise ValueError(
                "G1 asymmetric AC requires the shared Teacher/Critic v5 contract"
            )
        self.teacher = PrivilegedTransformerTeacher(
            self.config.robot,
            self.config.latent_width,
            self.config.system_id_dim,
            self.teacher_config,
            latent_groups=GROUPED_PHYSICS_LATENT,
        )
        self.teacher_ema = copy.deepcopy(self.teacher).requires_grad_(False).eval()
        shared_dim = self.teacher_config.input_dim(
            self.config.robot.dof, self.config.system_id_dim
        )
        self.critic = G1PrivilegedValueCritic(
            shared_dim,
            width=self.teacher_config.width,
            heads=self.teacher_config.heads,
            layers=self.teacher_config.layers,
            context_steps=self.teacher_config.context_steps,
        )
        self.actor = G1TeacherConditionedActor(self.config)
        self.phase = ""
        self.set_phase(phase)

    def set_phase(self, phase: str) -> None:
        if phase not in (TEACHER_TRAINING, TEACHER_GUIDED_ACTOR):
            raise ValueError("unknown G1 training phase")
        self.phase = phase
        teacher_training = phase == TEACHER_TRAINING
        self.teacher.requires_grad_(teacher_training)
        self.teacher_ema.requires_grad_(False)
        self.critic.requires_grad_(not teacher_training)
        self.actor.requires_grad_(not teacher_training)
        self.teacher.train(teacher_training)
        self.teacher_ema.eval()
        self.critic.train(not teacher_training)
        self.actor.train(not teacher_training)

    def train(self, mode: bool = True):
        # Keep the frozen half in eval mode even when an outer trainer calls
        # model.train().  The Teacher has no dropout today, but this is part of
        # the phase contract rather than an implementation accident.
        super().train(mode)
        if self.phase == TEACHER_TRAINING:
            self.teacher.train(mode)
            self.critic.eval()
            self.actor.eval()
        else:
            self.teacher.eval()
            self.critic.train(mode)
            self.actor.train(mode)
        self.teacher_ema.eval()
        return self

    def trainable_parameters(self):
        modules = (
            (self.teacher,)
            if self.phase == TEACHER_TRAINING
            else (self.actor, self.critic)
        )
        return (
            parameter
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        )

    def teacher_forward(
        self,
        privileged_history: torch.Tensor,
        episode_start: torch.Tensor,
        action: torch.Tensor,
        current_core: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.phase != TEACHER_TRAINING:
            raise RuntimeError("Teacher updates are disabled during Actor training")
        return self.teacher(privileged_history, episode_start, action, current_core)

    @torch.no_grad()
    def update_teacher_ema(self) -> None:
        """Publish one smoothed Teacher update after the stage-1 optimizer step."""
        if self.phase != TEACHER_TRAINING:
            raise RuntimeError("Teacher EMA updates are disabled during Actor training")
        decay = self.teacher_config.ema_decay
        for ema, online in zip(
            self.teacher_ema.parameters(), self.teacher.parameters(), strict=True
        ):
            ema.mul_(decay).add_(online, alpha=1.0 - decay)
        for ema, online in zip(
            self.teacher_ema.buffers(), self.teacher.buffers(), strict=True
        ):
            ema.copy_(online)

    @torch.no_grad()
    def frozen_teacher_latent(
        self,
        privileged_history: torch.Tensor,
        episode_start: torch.Tensor,
    ) -> torch.Tensor:
        if self.phase != TEACHER_GUIDED_ACTOR:
            raise RuntimeError("frozen Teacher latents are only used in Actor training")
        self.teacher_ema.eval()
        return self.teacher_ema.encode(privileged_history, episode_start).detach()

    def actor_forward(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
        reference_mask: torch.Tensor,
        privileged_history: torch.Tensor,
        episode_start: torch.Tensor,
    ) -> G1ActorOutput:
        if self.phase != TEACHER_GUIDED_ACTOR:
            raise RuntimeError("Actor updates require the teacher_guided_actor phase")
        latent = self.frozen_teacher_latent(privileged_history, episode_start)
        return self.actor(state, reference, reference_mask, latent)

    def critic_forward(
        self,
        privileged_history: torch.Tensor,
        episode_start: torch.Tensor,
    ) -> torch.Tensor:
        if self.phase != TEACHER_GUIDED_ACTOR:
            raise RuntimeError("Critic updates require the teacher_guided_actor phase")
        return self.critic(privileged_history, episode_start)

    def actor_critic_forward(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
        reference_mask: torch.Tensor,
        privileged_history: torch.Tensor,
        episode_start: torch.Tensor,
    ) -> tuple[G1ActorOutput, torch.Tensor]:
        """Evaluate policy and value from one canonical privilege tensor."""
        actor = self.actor_forward(
            state, reference, reference_mask,
            privileged_history, episode_start,
        )
        values = self.critic_forward(privileged_history, episode_start)
        return actor, values

    def checkpoint(self) -> dict[str, Any]:
        return {
            "schema": G1_TEACHER_ACTOR_SCHEMA,
            "phase": self.phase,
            "model_config": asdict(self.config),
            "teacher_config": asdict(self.teacher_config),
            "model": self.state_dict(),
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping[str, Any],
        *,
        phase: str | None = None,
        map_location: str | torch.device = "cpu",
    ) -> G1TeacherActorModel:
        if not isinstance(payload, Mapping) or payload.get("schema") != G1_TEACHER_ACTOR_SCHEMA:
            raise ValueError("not a supported G1 Teacher/Actor checkpoint")
        required = {"phase", "model_config", "teacher_config", "model"}
        if required - payload.keys():
            raise ValueError("incomplete G1 Teacher/Actor checkpoint")
        model = cls(
            G1TeacherActorConfig(**payload["model_config"]),
            PrivilegedTeacherConfig(**payload["teacher_config"]),
            phase=phase or payload["phase"],
        ).to(map_location)
        model.load_state_dict(payload["model"], strict=True)
        model.set_phase(phase or payload["phase"])
        return model

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        phase: str | None = None,
        map_location: str | torch.device = "cpu",
    ) -> G1TeacherActorModel:
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
        return cls.from_checkpoint(payload, phase=phase, map_location=map_location)

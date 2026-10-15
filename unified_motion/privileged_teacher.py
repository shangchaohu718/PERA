"""Training-only causal Transformer teacher; deployment never imports its inputs."""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .models.fixed_normalization import body_observation, contact_observation, object_observation
from .models.physics import _safe_unit


@dataclass(frozen=True)
class PrivilegedTeacherConfig:
    objective_version: int = 2
    critic_dim: int = 843
    contact_slots: int = 2
    object_slots: int = 4
    width: int = 64
    heads: int = 4
    layers: int = 2
    context_steps: int = 32
    train_envs: int = 64
    learning_rate: float = 1e-4
    ema_decay: float = .98
    warmup_updates: int = 20
    validation_fraction: float = .25
    minimum_validation_samples: int = 16
    minimum_prediction_gain: float = .05
    minimum_action_gain: float = .01
    prediction_horizons: tuple[int, ...] = (1, 4, 8)
    multistep_weight: float = .5
    delta_whitening: bool = False
    delta_whitening_std_floor: float = 1e-3
    delta_whitening_statistics: dict | None = None

    def __post_init__(self):
        if type(self.objective_version) is not int or self.objective_version not in (1, 2, 3):
            raise ValueError('unsupported Teacher objective version')
        for name in ('critic_dim', 'contact_slots', 'width', 'heads', 'layers',
                     'context_steps', 'train_envs', 'warmup_updates', 'minimum_validation_samples'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.object_slots) is not int or self.object_slots < 0:
            raise ValueError('object_slots must be a non-negative integer')
        if self.width % self.heads or self.width % 2:
            raise ValueError('teacher width must be even and divisible by heads')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('teacher learning_rate must be positive and finite')
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError('teacher ema_decay must be in [0,1)')
        for name in ('validation_fraction', 'minimum_prediction_gain', 'minimum_action_gain'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < .5:
                raise ValueError(f'{name} must be in (0,.5)')
        horizons = tuple(self.prediction_horizons)
        if (not horizons or horizons[0] != 1 or any(type(v) is not int or v < 1 for v in horizons)
                or tuple(sorted(set(horizons))) != horizons):
            raise ValueError('prediction_horizons must be unique increasing positive integers starting at 1')
        object.__setattr__(self, 'prediction_horizons', horizons)
        if (isinstance(self.multistep_weight, bool) or not isinstance(self.multistep_weight, (int, float))
                or not math.isfinite(self.multistep_weight) or self.multistep_weight < 0):
            raise ValueError('multistep_weight must be finite and non-negative')
        if self.objective_version < 3 and horizons != (1, 4, 8):
            raise ValueError('custom prediction horizons require Teacher objective v3')
        if type(self.delta_whitening) is not bool:
            raise TypeError('delta_whitening must be boolean')
        if (isinstance(self.delta_whitening_std_floor, bool)
                or not isinstance(self.delta_whitening_std_floor, (int, float))
                or not math.isfinite(self.delta_whitening_std_floor)
                or self.delta_whitening_std_floor <= 0):
            raise ValueError('delta_whitening_std_floor must be positive and finite')
        if not self.delta_whitening:
            if self.delta_whitening_statistics is not None:
                raise ValueError('delta whitening statistics require delta_whitening=true')
            return
        statistics = self.delta_whitening_statistics
        if not isinstance(statistics, Mapping) or set(statistics) != {str(h) for h in horizons}:
            raise ValueError('delta whitening statistics must exactly cover prediction_horizons')
        canonical = {}
        for horizon in horizons:
            value = statistics[str(horizon)]
            if not isinstance(value, Mapping) or set(value) != {'mean', 'std'}:
                raise ValueError('each delta whitening statistic needs mean and std')
            mean, std = value['mean'], value['std']
            if (not isinstance(mean, Sequence) or isinstance(mean, (str, bytes))
                    or not isinstance(std, Sequence) or isinstance(std, (str, bytes))
                    or not mean or len(mean) != len(std)):
                raise ValueError('delta whitening mean/std must be equal non-empty sequences')
            mean, std = tuple(float(v) for v in mean), tuple(float(v) for v in std)
            if (not all(math.isfinite(v) for v in mean + std)
                    or any(v < self.delta_whitening_std_floor for v in std)):
                raise ValueError('delta whitening statistics must be finite and respect std floor')
            canonical[str(horizon)] = {'mean': mean, 'std': std}
        object.__setattr__(self, 'delta_whitening_statistics', canonical)

    @classmethod
    def from_saved(cls, values):
        # The first saved Teacher used latent MSE and had no version tag.
        return cls(**{'objective_version': 1, **values})

    def input_dim(self, dof, system_id_dim):
        return (self.critic_dim + 6 + 3*dof + 3 + 9 + 2*dof
                + 13*self.contact_slots + 17*self.object_slots + system_id_dim)


def core_scale(robot, like):
    return like.new_tensor([5.]*3 + [10.]*3 + [1.]*3
                           + [math.pi]*robot.dof + [10.]*robot.dof)


def normalized_core(frame, robot):
    core = torch.cat((frame.base_lin_vel_b, frame.body_obs[..., :6+2*robot.dof]), -1)
    return core / core_scale(robot, core)


def teacher_inputs(critic, frame, parameters, robot, config):
    """Pack current truth, including available object slots, before executing a_t.

    Future reference commands already in the Critic contract are known goals;
    future measured state, rewards and next-state labels are never inputs.
    Missing object/contact slots are zero padded and explicitly masked.
    """
    lead = critic.shape[:-1]
    if critic.shape[-1] != config.critic_dim or parameters.shape[:-1] != lead:
        raise ValueError('teacher Critic/parameter shape mismatch')
    if frame.body_obs.shape != lead + (6+3*robot.dof,):
        raise ValueError('teacher body observation shape mismatch')
    pieces = [critic.detach(), body_observation(frame.body_obs, robot),
              frame.base_lin_vel_b/5., frame.core_valid.to(critic.dtype)]
    for features, mask, slots, width, normalizer in (
        (frame.contact_features, frame.contact_mask, config.contact_slots, 12,
         lambda v: contact_observation(v, robot)),
        (frame.object_features, frame.object_mask, config.object_slots, 16, object_observation),
    ):
        if features.shape[:-2] != lead or features.shape[-1] != width or mask.shape != features.shape[:-1]:
            raise ValueError('teacher scene shape mismatch')
        count = features.shape[-2]
        if count > slots:
            raise ValueError('teacher scene exceeds configured slots; never silently drop truth')
        values = normalizer(torch.where(mask.unsqueeze(-1), features, 0.))
        values = F.pad(values, (0, 0, 0, slots-count))
        available = F.pad(mask, (0, slots-count)).to(critic.dtype)
        pieces.extend((values.flatten(-2), available))
    pieces.append(parameters.detach())
    result = torch.cat(pieces, -1).detach()
    if result.shape[-1] != config.input_dim(robot.dof, parameters.shape[-1]) or not torch.isfinite(result).all():
        raise ValueError('invalid privileged teacher input')
    return result


class PrivilegedTransformerTeacher(nn.Module):
    """Finite-window causal attention with explicit episode barriers and no dropout."""
    def __init__(self, robot, latent_width, system_id_dim, config=None):
        super().__init__()
        self.config = config or PrivilegedTeacherConfig()
        self.robot, self.system_id_dim = robot, system_id_dim
        cfg = self.config
        self.input_projection = nn.Sequential(
            nn.LayerNorm(cfg.input_dim(robot.dof, system_id_dim)),
            nn.Linear(cfg.input_dim(robot.dof, system_id_dim), cfg.width))
        layer = nn.TransformerEncoderLayer(cfg.width, cfg.heads, 2*cfg.width,
            dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.latent_head = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, latent_width), nn.Tanh())
        core_dim = 9+2*robot.dof
        self.transition = nn.Sequential(nn.Linear(core_dim+latent_width+robot.dof, cfg.width),
                                        nn.SiLU(), nn.Linear(cfg.width, core_dim+cfg.contact_slots))
        self.future_transitions = nn.ModuleDict()
        if cfg.objective_version >= 3:
            for horizon in cfg.prediction_horizons:
                if horizon > 1:
                    self.future_transitions[str(horizon)] = nn.Sequential(
                        nn.Linear(core_dim+latent_width+horizon*robot.dof, cfg.width),
                        nn.SiLU(), nn.Linear(cfg.width, core_dim))
        self.identification = nn.Linear(latent_width, system_id_dim)
        self._delta_horizon_index = {}
        if cfg.delta_whitening:
            means, stds = [], []
            for index, horizon in enumerate(cfg.prediction_horizons):
                statistics = cfg.delta_whitening_statistics[str(horizon)]
                if len(statistics['mean']) != core_dim:
                    raise ValueError('delta whitening statistic width disagrees with robot contract')
                self._delta_horizon_index[horizon] = index
                means.append(statistics['mean']); stds.append(statistics['std'])
            self.register_buffer('delta_whitening_mean', torch.tensor(means, dtype=torch.float32))
            self.register_buffer('delta_whitening_std', torch.tensor(stds, dtype=torch.float32))

    def delta_statistics(self, horizon, like):
        """Return fixed train-split statistics in normalized-core units."""
        if not self.config.delta_whitening:
            return torch.zeros_like(like), torch.ones_like(like)
        if horizon not in self._delta_horizon_index:
            raise ValueError('missing delta whitening horizon')
        index = self._delta_horizon_index[horizon]
        return (self.delta_whitening_mean[index].to(like),
                self.delta_whitening_std[index].to(like))

    def decode_standardized_delta(self, value, *, horizon):
        if not self.config.delta_whitening:
            return value
        mean, std = self.delta_statistics(horizon, value)
        return mean + std * value

    def forward(self, inputs, episode_start, actions, current_core):
        if inputs.ndim != 3 or episode_start.shape != inputs.shape[:2] or episode_start.dtype != torch.bool:
            raise ValueError('teacher needs [B,T,D] inputs and boolean [B,T] episode starts')
        batch, steps, _ = inputs.shape
        if steps < 1 or actions.shape != (batch, steps, self.robot.dof):
            raise ValueError('teacher action shape mismatch')
        if current_core.shape != (batch, steps, 9+2*self.robot.dof):
            raise ValueError('teacher core shape mismatch')
        cfg = self.config
        index = torch.arange(steps, device=inputs.device)
        segment = episode_start.long().cumsum(1)
        distance = index[:, None]-index[None, :]
        allowed = ((segment[:, :, None] == segment[:, None, :])
                   & (distance >= 0) & (distance < cfg.context_steps))
        blocked = (~allowed).repeat_interleave(cfg.heads, dim=0)
        last_start = torch.where(episode_start, index[None], 0).cummax(1).values
        age = index[None]-last_start
        frequency = torch.exp(torch.arange(0, cfg.width, 2, device=inputs.device,
                                           dtype=inputs.dtype)*(-math.log(10000.)/cfg.width))
        phase = age.unsqueeze(-1)*frequency
        position = torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)
        hidden = self.encoder(self.input_projection(inputs)+position, mask=blocked)
        latent = self.latent_head(hidden)
        return self.decode(latent, actions, current_core)

    def decode(self, latent, actions, current_core):
        """Reuse a causal latent to test the effect of changing executed actions."""
        output = self.transition(torch.cat((current_core.detach(), latent, actions.detach()), -1))
        core_dim = current_core.shape[-1]
        predicted_delta = self.decode_standardized_delta(output[..., :core_dim], horizon=1)
        next_core = current_core.detach()+predicted_delta
        if self.config.objective_version >= 2:
            # Match the deployable Student's physical output manifold, including
            # its finite fallback for a degenerate direction. V1 stays compatible.
            gravity = _safe_unit(next_core[..., 6:9], current_core[..., 6:9].detach())
            next_core = torch.cat((next_core[..., :6], gravity, next_core[..., 9:]), -1)
        delta_core = next_core-current_core.detach()
        return dict(latent=latent, next_core=next_core, delta_core=delta_core,
                    contact_logits=output[..., core_dim:], system_id=self.identification(latent))

    def predict_future_core(self, latent, action_window, current_core, *, horizon):
        """Predict a state delta after an observed sequence of executed actions."""
        if horizon not in self.config.prediction_horizons or horizon <= 1:
            raise ValueError('future Teacher prediction requires a configured horizon > 1')
        if action_window.shape != latent.shape[:-1] + (horizon, self.robot.dof):
            raise ValueError('Teacher future action window has incompatible shape')
        if current_core.shape != latent.shape[:-1] + (9+2*self.robot.dof,):
            raise ValueError('Teacher future current_core has incompatible shape')
        standardized_delta = self.future_transitions[str(horizon)](torch.cat(
            (current_core.detach(), latent, action_window.detach().flatten(-2)), -1))
        delta = self.decode_standardized_delta(standardized_delta, horizon=horizon)
        result = current_core.detach()+delta
        gravity = _safe_unit(result[..., 6:9], current_core[..., 6:9].detach())
        return torch.cat((result[..., :6], gravity, result[..., 9:]), -1)


def teacher_supervision(output, next_core, core_valid, contacts, contact_mask, parameters, valid,
                        config=None, *, current_core=None):
    """Measured transition targets train the Teacher, independently of the Student."""
    def masked(value, mask):
        return torch.where(mask, value, 0.).sum()/mask.sum().clamp_min(1)
    config = config or PrivilegedTeacherConfig()
    mask = core_valid & valid.unsqueeze(-1)
    if config.objective_version >= 3:
        if current_core is None or current_core.shape != next_core.shape:
            raise ValueError('Teacher objective v3 requires matching current_core')
        target = next_core.detach()-current_core.detach()
        delta_error = output['delta_core']-target
        if config.delta_whitening:
            std = next_core.new_tensor(config.delta_whitening_statistics['1']['std'])
            delta_error = delta_error/std
        error = torch.where(mask, delta_error, 0.).square()
    else:
        error = torch.where(mask, output['next_core']-next_core.detach(), 0.).square()
    if config.objective_version == 1:
        core = masked(error, mask)
    else:
        core = torch.stack([masked(error[..., section], mask[..., section])
                            for section in core_groups(next_core.shape[-1])]).mean()
    contact_mask = contact_mask & valid.unsqueeze(-1)
    contact = masked(F.binary_cross_entropy_with_logits(
        torch.where(contact_mask, output['contact_logits'], 0.),
        torch.where(contact_mask, contacts.detach(), 0.), reduction='none'), contact_mask)
    # Exact parameters remain privileged INPUTS for transition prediction. V2 does
    # not reward copying those same parameters to an identification output.
    if config.objective_version == 1:
        identification = masked((output['system_id']-parameters.detach()).square().mean(-1), valid)
        return core+contact+identification
    return core+contact


def core_groups(width):
    dof = (width-9)//2
    return (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 9+dof), slice(9+dof, width))


@torch.no_grad()
def assess_teacher(teacher, student_core, student_contact_logits, shuffled, truth, current,
                   core_valid, contacts, current_contacts, contact_valid, config):
    """Admit physical outputs, never latent coordinates, on reserved transitions.

    Each group must beat both the current Student and state/contact persistence.
    Action permutation is diagnostic only: delayed/inertial dynamics can be
    predictable before the current command has any effect. These are diagnostics
    on a temporal validation split, not evidence of generalization to new domains.
    """
    admitted, errors = [], []
    for section in core_groups(truth.shape[-1]):
        mask = core_valid[..., section]
        count = int(mask.any(-1).sum())
        denom = mask.sum().clamp_min(1)
        loss = lambda value: torch.where(mask, (value[..., section]-truth[..., section]).square(), 0.).sum()/denom
        errors.append((loss(teacher['next_core']), loss(student_core), loss(current), loss(shuffled['next_core']), count))
    count = contacts.shape[-1]
    def contact_loss(logits):
        value = F.binary_cross_entropy_with_logits(logits[..., :count], contacts, reduction='none')
        return torch.where(contact_valid, value, 0.).sum()/contact_valid.sum().clamp_min(1)
    # Finite logits for a strong persistence baseline, including contact switches.
    persist_logits = torch.logit(current_contacts.clamp(.01, .99))
    errors.append((contact_loss(teacher['contact_logits']), contact_loss(student_contact_logits),
                   contact_loss(persist_logits), contact_loss(shuffled['contact_logits']),
                   int(contact_valid.any(-1).sum())))
    details = []
    for t, s, p, shuffled_loss, count in errors:
        finite = bool(torch.isfinite(torch.stack((t, s, p))).all())
        useful = finite and count >= config.minimum_validation_samples
        useful = useful and bool(t < torch.minimum(s, p)*(1-config.minimum_prediction_gain))
        action_sensitive = (finite and bool(torch.isfinite(shuffled_loss))
                            and bool(shuffled_loss > t*(1+config.minimum_action_gain)+1e-8))
        admitted.append(useful)
        details.append(dict(teacher=float(t), student=float(s), persistence=float(p),
                            shuffled=float(shuffled_loss), samples=count, admitted=useful,
                            action_sensitive=action_sensitive))
    return truth.new_tensor(admitted), details


def predictive_distillation(student_core, student_contact_logits, teacher, core_valid,
                            contact_valid, transition_valid, admitted):
    """Fixed physical outputs, stopped Teacher targets, equal group weighting."""
    terms = []
    for i, section in enumerate(core_groups(student_core.shape[-1])):
        mask = core_valid[..., section] & transition_valid.unsqueeze(-1) & admitted[i].bool()
        error = torch.where(mask, student_core[..., section]-teacher['next_core'][..., section].detach(), 0.)
        terms.append(error.square().sum()/mask.sum().clamp_min(1))
    count = student_contact_logits.shape[-1]
    mask = contact_valid & transition_valid.unsqueeze(-1) & admitted[-1].bool()
    target = torch.where(mask, teacher['contact_logits'][..., :count].detach(), 0.).sigmoid()
    error = F.binary_cross_entropy_with_logits(torch.where(mask, student_contact_logits, 0.), target, reduction='none')
    terms.append(torch.where(mask, error, 0.).sum()/mask.sum().clamp_min(1))
    return torch.stack(terms).mean()

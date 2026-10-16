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


GROUPED_PHYSICS_LATENT = (
    ('contact', 8),
    ('force', 16),
    ('kinematics', 16),
    ('dynamics', 24),
)


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
    interaction_weight: float = .25
    contact_mode_weight: float = .1
    parameter_reconstruction_weight: float = 0.
    future_contact_weight: float = 0.
    contact_input_lag: int = 0
    contact_slip_threshold_m_s: float = .05
    delta_whitening: bool = False
    delta_whitening_std_floor: float = 1e-3
    delta_whitening_statistics: dict | None = None
    # Empty preserves every legacy checkpoint.  The only supported grouped
    # contract is the semantically routed z64 used by K1/G1 physics Teachers.
    latent_groups: tuple[tuple[str, int], ...] = ()

    def __post_init__(self):
        if type(self.objective_version) is not int or self.objective_version not in (1, 2, 3, 4, 5):
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
        for name in ('interaction_weight', 'contact_mode_weight',
                     'parameter_reconstruction_weight', 'future_contact_weight'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f'{name} must be finite and non-negative')
        if type(self.contact_input_lag) is not int or self.contact_input_lag < 0:
            raise ValueError('contact_input_lag must be a non-negative integer')
        if (isinstance(self.contact_slip_threshold_m_s, bool)
                or not isinstance(self.contact_slip_threshold_m_s, (int, float))
                or not math.isfinite(self.contact_slip_threshold_m_s)
                or self.contact_slip_threshold_m_s <= 0):
            raise ValueError('contact_slip_threshold_m_s must be positive and finite')
        if self.objective_version < 3 and horizons != (1, 4, 8):
            raise ValueError('custom prediction horizons require Teacher objective v3')
        if type(self.delta_whitening) is not bool:
            raise TypeError('delta_whitening must be boolean')
        try:
            latent_groups = tuple((str(name), int(width))
                                  for name, width in self.latent_groups)
        except (TypeError, ValueError) as error:
            raise ValueError('latent_groups must contain name/width pairs') from error
        if latent_groups and latent_groups != GROUPED_PHYSICS_LATENT:
            raise ValueError(
                'latent_groups must be contact8/force16/kinematics16/dynamics24')
        object.__setattr__(self, 'latent_groups', latent_groups)
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
        if self.objective_version >= 5:
            # Exactly the asymmetric Critic information contract: the original
            # HoloMotion Critic tensor plus the shared physical privilege.
            return self.critic_dim + self.critic_privileged_dim(system_id_dim)
        if self.objective_version >= 4:
            # Current proprioception (which contains the last executed action),
            # base velocity, exact per-contact physics, optional object slots,
            # and slow simulator parameters.  Reference/goal data is excluded.
            return 6 + 3*dof + 3 + 14*self.contact_slots + 17*self.object_slots + system_id_dim
        return (self.critic_dim + 6 + 3*dof + 3 + 9 + 2*dof
                + 13*self.contact_slots + 17*self.object_slots + system_id_dim)

    def critic_privileged_dim(self, system_id_dim):
        """Exact physical values appended only to the asymmetric Critic."""
        return 14*self.contact_slots + 17*self.object_slots + system_id_dim


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


def teacher_physics_inputs(frame, contact_truth, parameters, robot, config):
    """Pack the v4 Teacher's strictly physical, causal observation.

    Unlike :func:`teacher_inputs`, this contract does not copy the 843-D
    HoloMotion Critic input and therefore cannot leak current/future reference
    targets into ``z``.  Contact slots keep stable left/right identities and
    object slots are padded with an explicit availability mask.
    """
    if config.objective_version < 4:
        raise ValueError('clean physical Teacher inputs require objective v4')
    lead = frame.body_obs.shape[:-1]
    if frame.body_obs.shape[-1] != 6 + 3*robot.dof or parameters.shape[:-1] != lead:
        raise ValueError('physical Teacher body/parameter shape mismatch')
    contact = contact_truth.teacher_tensor()
    if (contact.shape[:-1] != lead
            or contact.shape[-1] != 14*config.contact_slots):
        raise ValueError('physical Teacher contact slots disagree with configuration')
    if (frame.object_features.shape[:-2] != lead
            or frame.object_features.shape[-1] != 16
            or frame.object_mask.shape != frame.object_features.shape[:-1]):
        raise ValueError('physical Teacher object shape mismatch')
    count = frame.object_features.shape[-2]
    if count > config.object_slots:
        raise ValueError('physical Teacher scene exceeds configured object slots')
    objects = object_observation(torch.where(
        frame.object_mask.unsqueeze(-1), frame.object_features, 0.))
    objects = F.pad(objects, (0, 0, 0, config.object_slots-count))
    available = F.pad(frame.object_mask, (0, config.object_slots-count)).to(frame.body_obs.dtype)
    result = torch.cat((
        body_observation(frame.body_obs, robot),
        frame.base_lin_vel_b/5.,
        contact,
        objects.flatten(-2),
        available,
        parameters.detach(),
    ), -1).detach()
    expected = config.input_dim(robot.dof, parameters.shape[-1])
    if result.shape[-1] != expected or not torch.isfinite(result).all():
        raise ValueError('invalid physical Teacher input')
    return result


def teacher_critic_privilege(frame, contact_truth, parameters, config):
    """Build Critic-only physics without duplicating proprioception/reference."""
    lead = frame.body_obs.shape[:-1]
    contact = contact_truth.teacher_tensor()
    if (contact.shape != lead + (14*config.contact_slots,)
            or parameters.shape[:-1] != lead):
        raise ValueError('Critic physical privilege shape mismatch')
    count = frame.object_features.shape[-2]
    if count > config.object_slots:
        raise ValueError('Critic scene exceeds configured object slots')
    objects = object_observation(torch.where(
        frame.object_mask.unsqueeze(-1), frame.object_features, 0.))
    objects = F.pad(objects, (0, 0, 0, config.object_slots-count))
    available = F.pad(frame.object_mask, (0, config.object_slots-count)).to(frame.body_obs.dtype)
    result = torch.cat((contact, objects.flatten(-2), available, parameters.detach()), -1).detach()
    if (result.shape[-1] != config.critic_privileged_dim(parameters.shape[-1])
            or not torch.isfinite(result).all()):
        raise ValueError('invalid Critic physical privilege')
    return result


def teacher_shared_critic_inputs(critic, frame, contact_truth, parameters, config):
    """The v5 Teacher Encoder and Critic consume the same information tensor.

    ``critic`` is the unmodified HoloMotion Critic tensor.  The appended suffix
    is built by the same helper passed to the asymmetric Critic, preventing the
    Teacher and Critic privilege contracts from drifting apart.
    """
    if config.objective_version < 5:
        raise ValueError('shared Teacher/Critic inputs require objective v5')
    if critic.shape[-1] != config.critic_dim:
        raise ValueError('shared Teacher/Critic base observation width mismatch')
    privilege = teacher_critic_privilege(
        frame, contact_truth, parameters, config)
    if privilege.shape[:-1] != critic.shape[:-1]:
        raise ValueError('shared Teacher/Critic leading shape mismatch')
    result = torch.cat((critic.detach(), privilege), -1)
    if (result.shape[-1] != config.input_dim(0, parameters.shape[-1])
            or not torch.isfinite(result).all()):
        raise ValueError('invalid shared Teacher/Critic input')
    return result.detach()


def teacher_contact_targets(contact_truth, config):
    """Return normalized continuous contact targets and 3-way mode labels.

    The continuous order is normal load, tangential load, slip speed and
    friction utilization.  Modes are separated/sticking/sliding and preserve
    the fixed left/right contact axis.
    """
    values = contact_truth.teacher_tensor().unflatten(-1, (config.contact_slots, 14))
    physics = values[..., (9, 11, 12, 13)]
    found = contact_truth.contact_found.bool()
    if found.shape != physics.shape[:-1]:
        raise ValueError('contact target shape mismatch')
    sliding = contact_truth.slip_speed_m_s > config.contact_slip_threshold_m_s
    mode = torch.where(found, torch.where(sliding, 2, 1), 0).long()
    return physics.detach(), mode.detach()


class PrivilegedTransformerTeacher(nn.Module):
    """Finite-window causal attention with explicit episode barriers and no dropout."""
    def __init__(self, robot, latent_width, system_id_dim, config=None, *, latent_groups=None):
        super().__init__()
        self.config = config or PrivilegedTeacherConfig()
        self.robot, self.system_id_dim = robot, system_id_dim
        cfg = self.config
        configured_groups = cfg.latent_groups
        requested_groups = (configured_groups if latent_groups is None
                            else tuple(latent_groups))
        if configured_groups and latent_groups is not None \
                and tuple(latent_groups) != configured_groups:
            raise ValueError('constructor latent_groups disagree with Teacher config')
        self.latent_groups = requested_groups
        if self.latent_groups:
            if (self.latent_groups != GROUPED_PHYSICS_LATENT
                    or sum(width for _, width in self.latent_groups) != latent_width):
                raise ValueError(
                    'grouped physics latent must be contact8/force16/kinematics16/dynamics24')
        offset = 0
        self.latent_slices = {}
        for name, width in self.latent_groups:
            self.latent_slices[name] = slice(offset, offset + width)
            offset += width
        self.state_reference_projection = None
        self.physics_projection = None
        if cfg.objective_version >= 5:
            # Preserve equal representational capacity for the much smaller
            # physical suffix instead of letting one flat 1172->128 layer
            # choose primarily from the 1046-D HoloMotion Critic block.
            physics_dim = cfg.critic_privileged_dim(system_id_dim)
            self.state_reference_projection = nn.Sequential(
                nn.LayerNorm(cfg.critic_dim),
                nn.Linear(cfg.critic_dim, cfg.width),
                nn.SiLU(),
            )
            self.physics_projection = nn.Sequential(
                nn.LayerNorm(physics_dim),
                nn.Linear(physics_dim, cfg.width),
                nn.SiLU(),
            )
            # 1046->128 and 126->128 become 256->128 for the production G1
            # configuration.  ``cfg.width`` keeps the same contract testable
            # with smaller models.
            self.input_projection = nn.Sequential(
                nn.LayerNorm(2 * cfg.width),
                nn.Linear(2 * cfg.width, cfg.width),
                nn.SiLU(),
            )
        else:
            self.input_projection = nn.Sequential(
                nn.LayerNorm(cfg.input_dim(robot.dof, system_id_dim)),
                nn.Linear(cfg.input_dim(robot.dof, system_id_dim), cfg.width))
        layer = nn.TransformerEncoderLayer(cfg.width, cfg.heads, 2*cfg.width,
            dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        if self.latent_groups:
            self.latent_head = nn.ModuleDict({
                name: nn.Sequential(
                    nn.LayerNorm(cfg.width), nn.Linear(cfg.width, width), nn.Tanh())
                for name, width in self.latent_groups
            })
        else:
            self.latent_head = nn.Sequential(
                nn.LayerNorm(cfg.width), nn.Linear(cfg.width, latent_width), nn.Tanh())
        core_dim = 9+2*robot.dof
        transition_latent_width = (40 if self.latent_groups else latent_width)
        transition_output_width = core_dim if self.latent_groups else core_dim+cfg.contact_slots
        self.transition = nn.Sequential(
            nn.Linear(core_dim+transition_latent_width+robot.dof, cfg.width),
            nn.SiLU(), nn.Linear(cfg.width, transition_output_width))
        self.contact_head = self.force_head = self.kinematics_head = self.contact_mode_head = None
        self.interaction = None
        if self.latent_groups:
            common = core_dim + robot.dof
            self.contact_head = nn.Sequential(
                nn.Linear(common + 8, cfg.width), nn.SiLU(),
                nn.Linear(cfg.width, cfg.contact_slots))
            # Per slot: normal load, tangential load, friction utilization.
            self.force_head = nn.Sequential(
                nn.Linear(common + 16, cfg.width), nn.SiLU(),
                nn.Linear(cfg.width, cfg.contact_slots * 3))
            # Per slot: slip speed. Direction remains present in the privileged
            # input and can be added as an explicit target once Isaac exposes it.
            self.kinematics_head = nn.Sequential(
                nn.Linear(common + 16, cfg.width), nn.SiLU(),
                nn.Linear(cfg.width, cfg.contact_slots))
            self.contact_mode_head = nn.Sequential(
                nn.Linear(common + 8 + 16, cfg.width), nn.SiLU(),
                nn.Linear(cfg.width, cfg.contact_slots * 3))
        elif cfg.objective_version >= 4:
            self.interaction = nn.Sequential(
                nn.Linear(core_dim+latent_width+robot.dof, cfg.width),
                nn.SiLU(),
                # 4 continuous values + 3 mode logits for every contact slot.
                nn.Linear(cfg.width, cfg.contact_slots*7),
            )
        self.future_transitions = nn.ModuleDict()
        self.future_contact_heads = nn.ModuleDict()
        if cfg.objective_version >= 3:
            for horizon in cfg.prediction_horizons:
                if horizon > 1:
                    self.future_transitions[str(horizon)] = nn.Sequential(
                        nn.Linear(core_dim+transition_latent_width+horizon*robot.dof, cfg.width),
                        nn.SiLU(), nn.Linear(cfg.width, core_dim))
                    if cfg.future_contact_weight and self.latent_groups:
                        action_width = horizon * robot.dof
                        self.future_contact_heads[str(horizon)] = nn.ModuleDict({
                            'contact': nn.Sequential(
                                nn.Linear(core_dim + 8 + action_width, cfg.width), nn.SiLU(),
                                nn.Linear(cfg.width, cfg.contact_slots)),
                            'force': nn.Sequential(
                                nn.Linear(core_dim + 16 + action_width, cfg.width), nn.SiLU(),
                                nn.Linear(cfg.width, cfg.contact_slots * 3)),
                            'slip': nn.Sequential(
                                nn.Linear(core_dim + 16 + action_width, cfg.width), nn.SiLU(),
                                nn.Linear(cfg.width, cfg.contact_slots)),
                            'mode': nn.Sequential(
                                nn.Linear(core_dim + 8 + 16 + action_width, cfg.width), nn.SiLU(),
                                nn.Linear(cfg.width, cfg.contact_slots * 3)),
                        })
        identification_width = 24 if self.latent_groups else latent_width
        self.identification = nn.Linear(identification_width, system_id_dim)
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

    def encode(self, inputs, episode_start):
        """Encode causal privileged history without requiring the current action.

        The latent describes information available before choosing ``a_t``.  Keeping
        this operation separate from :meth:`decode` prevents an Actor-observation
        experiment from accidentally conditioning ``z_t`` on its own output.
        """
        expected_dim = self.config.input_dim(self.robot.dof, self.system_id_dim)
        if (inputs.ndim != 3 or inputs.shape[-1] != expected_dim
                or episode_start.shape != inputs.shape[:2]
                or episode_start.dtype != torch.bool):
            raise ValueError('teacher needs [B,T,D] inputs and boolean [B,T] episode starts')
        batch, steps, _ = inputs.shape
        if steps < 1:
            raise ValueError('teacher history must contain at least one step')
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
        if cfg.objective_version >= 5:
            state_reference = inputs[..., :cfg.critic_dim]
            physics = inputs[..., cfg.critic_dim:]
            projected = self.input_projection(torch.cat((
                self.state_reference_projection(state_reference),
                self.physics_projection(physics),
            ), -1))
        else:
            projected = self.input_projection(inputs)
        hidden = self.encoder(projected+position, mask=blocked)
        if not self.latent_groups:
            return self.latent_head(hidden)
        return torch.cat(tuple(
            self.latent_head[name](hidden) for name, _ in self.latent_groups), -1)

    def latent_group(self, latent, name):
        """Return one declared semantic group without copying its values."""
        if name not in self.latent_slices:
            raise ValueError(f'unknown or unavailable latent group: {name}')
        return latent[..., self.latent_slices[name]]

    def forward(self, inputs, episode_start, actions, current_core):
        batch, steps = inputs.shape[:2]
        if actions.shape != (batch, steps, self.robot.dof):
            raise ValueError('teacher action shape mismatch')
        if current_core.shape != (batch, steps, 9+2*self.robot.dof):
            raise ValueError('teacher core shape mismatch')
        latent = self.encode(inputs, episode_start)
        return self.decode(latent, actions, current_core)

    def decode(self, latent, actions, current_core):
        """Reuse a causal latent to test the effect of changing executed actions."""
        if self.latent_groups:
            contact_z = self.latent_group(latent, 'contact')
            force_z = self.latent_group(latent, 'force')
            kinematics_z = self.latent_group(latent, 'kinematics')
            dynamics_z = self.latent_group(latent, 'dynamics')
            transition_z = torch.cat((kinematics_z, dynamics_z), -1)
        else:
            transition_z = latent
        conditioned = torch.cat((current_core.detach(), transition_z, actions.detach()), -1)
        output = self.transition(conditioned)
        core_dim = current_core.shape[-1]
        predicted_delta = self.decode_standardized_delta(output[..., :core_dim], horizon=1)
        next_core = current_core.detach()+predicted_delta
        if self.config.objective_version >= 2:
            # Match the deployable Student's physical output manifold, including
            # its finite fallback for a degenerate direction. V1 stays compatible.
            gravity = _safe_unit(next_core[..., 6:9], current_core[..., 6:9].detach())
            next_core = torch.cat((next_core[..., :6], gravity, next_core[..., 9:]), -1)
        delta_core = next_core-current_core.detach()
        if self.latent_groups:
            common = (current_core.detach(), actions.detach())
            contact_logits = self.contact_head(torch.cat((*common, contact_z), -1))
            force = self.force_head(
                torch.cat((*common, force_z), -1)).unflatten(-1, (self.config.contact_slots, 3))
            slip = self.kinematics_head(
                torch.cat((*common, kinematics_z), -1)).unsqueeze(-1)
            contact_physics = torch.cat((force[..., :2], slip, force[..., 2:]), -1)
            contact_mode_logits = self.contact_mode_head(torch.cat(
                (*common, contact_z, kinematics_z), -1)).unflatten(
                    -1, (self.config.contact_slots, 3))
            result = dict(
                latent=latent, next_core=next_core, delta_core=delta_core,
                contact_logits=contact_logits,
                system_id=self.identification(dynamics_z),
                contact_physics=contact_physics,
                contact_mode_logits=contact_mode_logits,
            )
        else:
            result = dict(latent=latent, next_core=next_core, delta_core=delta_core,
                          contact_logits=output[..., core_dim:], system_id=self.identification(latent))
        if self.interaction is not None:
            interaction = self.interaction(conditioned).unflatten(-1, (self.config.contact_slots, 7))
            result['contact_physics'] = interaction[..., :4]
            result['contact_mode_logits'] = interaction[..., 4:]
        return result

    def predict_future_core(self, latent, action_window, current_core, *, horizon):
        """Predict a state delta after an observed sequence of executed actions."""
        if horizon not in self.config.prediction_horizons or horizon <= 1:
            raise ValueError('future Teacher prediction requires a configured horizon > 1')
        if action_window.shape != latent.shape[:-1] + (horizon, self.robot.dof):
            raise ValueError('Teacher future action window has incompatible shape')
        if current_core.shape != latent.shape[:-1] + (9+2*self.robot.dof,):
            raise ValueError('Teacher future current_core has incompatible shape')
        transition_z = (torch.cat((
            self.latent_group(latent, 'kinematics'),
            self.latent_group(latent, 'dynamics')), -1)
            if self.latent_groups else latent)
        standardized_delta = self.future_transitions[str(horizon)](torch.cat(
            (current_core.detach(), transition_z, action_window.detach().flatten(-2)), -1))
        delta = self.decode_standardized_delta(standardized_delta, horizon=horizon)
        result = current_core.detach()+delta
        gravity = _safe_unit(result[..., 6:9], current_core[..., 6:9].detach())
        return torch.cat((result[..., :6], gravity, result[..., 9:]), -1)

    def predict_future_contacts(self, latent, action_window, current_core, *, horizon):
        """Predict future contact state/forces without reading target-frame truth."""
        key = str(horizon)
        if key not in self.future_contact_heads:
            raise ValueError('future contact prediction is not configured for this horizon')
        if action_window.shape != latent.shape[:-1] + (horizon, self.robot.dof):
            raise ValueError('future contact action window has incompatible shape')
        if current_core.shape != latent.shape[:-1] + (9 + 2*self.robot.dof,):
            raise ValueError('future contact current_core has incompatible shape')
        contact_z = self.latent_group(latent, 'contact')
        force_z = self.latent_group(latent, 'force')
        kinematics_z = self.latent_group(latent, 'kinematics')
        actions = action_window.detach().flatten(-2)
        current = current_core.detach()
        heads = self.future_contact_heads[key]
        contact_logits = heads['contact'](torch.cat((current, contact_z, actions), -1))
        force = heads['force'](
            torch.cat((current, force_z, actions), -1)).unflatten(
                -1, (self.config.contact_slots, 3))
        slip = heads['slip'](
            torch.cat((current, kinematics_z, actions), -1)).unsqueeze(-1)
        mode = heads['mode'](torch.cat(
            (current, contact_z, kinematics_z, actions), -1)).unflatten(
                -1, (self.config.contact_slots, 3))
        return {
            'contact_logits': contact_logits,
            'contact_physics': torch.cat((force[..., :2], slip, force[..., 2:]), -1),
            'contact_mode_logits': mode,
        }


def teacher_supervision(output, next_core, core_valid, contacts, contact_mask, parameters, valid,
                        config=None, *, current_core=None, contact_physics=None,
                        contact_modes=None):
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
    total = core+contact
    if config.objective_version >= 4:
        if (contact_physics is None or contact_modes is None
                or contact_physics.shape != output['contact_physics'].shape
                or contact_modes.shape != output['contact_mode_logits'].shape[:-1]):
            raise ValueError('Teacher v4 requires matching contact physics and mode targets')
        interaction_mask = contact_mask.unsqueeze(-1)
        interaction = masked(
            (output['contact_physics']-contact_physics.detach()).square(),
            interaction_mask.expand_as(contact_physics))
        mode_loss = F.cross_entropy(
            output['contact_mode_logits'].flatten(0, -2),
            contact_modes.detach().flatten(), reduction='none').unflatten(0, contact_modes.shape)
        mode = masked(mode_loss, contact_mask)
        total = total + config.interaction_weight*interaction + config.contact_mode_weight*mode
    if config.parameter_reconstruction_weight:
        if output['system_id'].shape != parameters.shape:
            raise ValueError('Teacher parameter reconstruction requires matching system_id targets')
        identification = masked(
            (output['system_id']-parameters.detach()).square().mean(-1), valid)
        total = total + config.parameter_reconstruction_weight*identification
    return total


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

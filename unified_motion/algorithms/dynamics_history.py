"""Replay observed episodes after Dynamics updates without stepping the world."""
from dataclasses import dataclass, fields

import torch

from ..models.physics import DynamicsModel, PhysicsObservation, PhysicsEncoding
from ..physics_data import map_physics, stack_physics


def snapshot_physics(observation: PhysicsObservation) -> PhysicsObservation:
    return PhysicsObservation(**{
        field.name: getattr(observation, field.name).detach().clone()
        for field in fields(observation)
    })


@dataclass(frozen=True)
class DynamicsHistoryStep:
    observation: PhysicsObservation
    episode_start: torch.Tensor
    executed_action: torch.Tensor


class DynamicsHistory:
    """Retain all live episode prefixes, with an explicit memory safety limit.

    Rows may reset asynchronously. Only a common prefix preceding every row's
    latest real reset may be discarded. Replays recompute predictions as well
    as hidden states, so feedback never mixes old and new model weights.
    """

    def __init__(self, max_steps: int = 16384, *, storage_device="cpu", chunk_steps=64,
                 max_bytes: int = 512*1024**2):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("dynamics history max_steps must be positive")
        self.max_steps = max_steps
        if type(chunk_steps) is not int or chunk_steps < 1:
            raise ValueError("history chunk_steps must be positive")
        self.storage_device, self.chunk_steps = torch.device(storage_device), chunk_steps
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("history max_bytes must be positive")
        self.max_bytes, self._frame_bytes = max_bytes, None
        self.steps: list[DynamicsHistoryStep] = []
        self._latest_start = None

    def clear(self):
        self.steps.clear()
        self._latest_start = None
        self._frame_bytes = None

    def snapshot(self):
        result = DynamicsHistory(self.max_steps, storage_device=self.storage_device,
                                 chunk_steps=self.chunk_steps, max_bytes=self.max_bytes)
        result._frame_bytes = self._frame_bytes
        result.steps = list(self.steps)  # Stored tensors are immutable snapshots.
        result._latest_start = None if self._latest_start is None else self._latest_start.clone()
        return result

    def check_capacity(self, starts: torch.Tensor, observation=None):
        starts = starts.to(self.storage_device)
        frame_bytes = self._frame_bytes
        if observation is not None:
            # Includes the command and reset flag, not just state tensors.
            dof = (observation.body_obs.shape[-1]-6)//3
            frame_bytes = sum(getattr(observation, f.name).numel()*getattr(observation, f.name).element_size()
                              for f in fields(observation))
            frame_bytes += starts.numel()*(dof*observation.body_obs.element_size()+starts.element_size())
        if not self.steps:
            if not starts.all().item():
                raise RuntimeError("Dynamics replay requires each row's real episode start")
            retained = 1
        else:
            latest = torch.where(starts, len(self.steps), self._latest_start)
            retained = len(self.steps)+1-int(latest.min().item())
        if retained > self.max_steps:
            raise RuntimeError("Dynamics history exceeded max_steps; increase the explicit limit, "
                               "do not truncate history or reset the environment")
        if frame_bytes is not None and retained*frame_bytes > self.max_bytes:
            raise RuntimeError("Dynamics history exceeded max_bytes before stepping; increase the CPU history budget")
        if frame_bytes is not None:
            self._frame_bytes = frame_bytes

    def append(self, observation, starts, executed_action):
        self.check_capacity(starts, observation)
        starts = starts.to(self.storage_device)
        if self._latest_start is None:
            self._latest_start = torch.zeros_like(starts, dtype=torch.long)
        self._latest_start = torch.where(starts, len(self.steps), self._latest_start)
        self.steps.append(DynamicsHistoryStep(map_physics(observation,
            lambda x: x.detach().to(self.storage_device, copy=True)),
            starts.detach().clone(), executed_action.detach().to(self.storage_device, copy=True)))
        discard = int(self._latest_start.min().item())
        if discard:
            del self.steps[:discard]
            self._latest_start -= discard

    @torch.no_grad()
    def rebuild(self, model: DynamicsModel, env_ids, *, capture_last: int = 0):
        """Return current memory and optional encodings for Actor prefix replay.

        For staggered resets, early outputs of rows awaiting their retained real
        reset are irrelevant: that reset erases them before the returned state.
        Starting those rows from zero is padding, not an environment reset.
        """
        model.eval()
        memory, encodings = None, []
        begin = max(0, len(self.steps) - capture_last)
        device = next(model.parameters()).device
        for offset in range(0, len(self.steps), self.chunk_steps):
            chunk = self.steps[offset:offset+self.chunk_steps]
            frames = map_physics(stack_physics([s.observation for s in chunk]), lambda x: x.to(device))
            # The stateless encoder is batched; recurrence and prediction feedback
            # still follow the exact original time order. No approximate burn-in.
            physics = model.encoder(frames)
            for local, item in enumerate(chunk):
                index = offset+local
                starts = item.episode_start.to(device)
                if index == 0:
                    starts = torch.ones_like(starts)
                frame = frames.at(local)
                encoded = PhysicsEncoding(physics.context[:, local],
                    physics.contact_tokens[:, local], physics.object_tokens[:, local])
                encoding, memory = model.step(frame, episode_start=starts,
                    env_ids=env_ids, memory=memory, physics_encoding=encoded)
                prediction = model.predict_next(frame, item.executed_action.to(device),
                    encoding.latent, physics_encoding=encoding.physics_encoding())
                memory = model.remember_prediction(memory, prediction)
                if index >= begin:
                    encodings.append(encoding)
        return memory, encodings

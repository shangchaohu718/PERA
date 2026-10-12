"""Tensor transforms that preserve every observation field, including availability."""
from dataclasses import fields

import torch

from .models.physics import PhysicsObservation


def map_physics(observation, transform):
    return PhysicsObservation(**{f.name: transform(getattr(observation, f.name))
                                 for f in fields(observation)})


def stack_physics(items):
    return PhysicsObservation(**{f.name: torch.stack([getattr(x, f.name) for x in items], dim=1)
                                 for f in fields(PhysicsObservation)})


def concat_physics(items):
    return PhysicsObservation(**{f.name: torch.cat([getattr(x, f.name) for x in items], dim=1)
                                 for f in fields(PhysicsObservation)})


def slice_physics(observation, begin, end):
    return map_physics(observation, lambda x: x[:, begin:end])

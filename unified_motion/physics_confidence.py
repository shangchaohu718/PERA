"""Held-out error calibration and prior-relative information gating.

This is an empirical gate, not a probability of safe control or an OOD detector.
Deployment only reads the saved state; labels/update() are training-only.
"""
from dataclasses import dataclass, asdict
import math

import torch
from torch import nn


@dataclass(frozen=True)
class PhysicsConfidenceConfig:
    mode: str = "calibrated"
    prior_variance: float = 1/3
    minimum_samples: int = 16
    minimum_skill: float = .05
    feedback_floor: float = .1

    def __post_init__(self):
        if self.mode not in ("bounded", "calibrated", "off", "ungated", "legacy"):
            raise ValueError("unknown Physics confidence mode")
        if not math.isfinite(self.prior_variance) or self.prior_variance <= 0:
            raise ValueError("prior variance must be positive")
        if type(self.minimum_samples) is not int or self.minimum_samples < 1:
            raise ValueError("minimum_samples must be positive")
        if not math.isfinite(self.minimum_skill) or not 0 <= self.minimum_skill < 1:
            raise ValueError("minimum_skill must be in [0,1)")
        if type(self.feedback_floor) not in (float, int) or not math.isfinite(self.feedback_floor) or not 0 < self.feedback_floor <= .25:
            raise ValueError('feedback_floor must be in (0,.25]')


class PhysicsConfidence(nn.Module):
    def __init__(self, dimensions, config=None):
        super().__init__()
        self.config = config or PhysicsConfidenceConfig()
        self.register_buffer("samples", torch.zeros((), dtype=torch.long))
        self.register_buffer("mse", torch.full((dimensions,), self.config.prior_variance))
        self.register_buffer("temperature", torch.ones(dimensions))
        self.register_buffer("coverage", torch.zeros(dimensions))

    @torch.no_grad()
    def update(self, mean, logvar, target, valid):
        """Calibrate the current candidate on labels excluded from its gradient loss.

        Do not pool errors from old models: each candidate must qualify on its
        own held-out window. samples tracks available evidence for this fit.
        """
        errors = (mean-target).square()[valid].reshape(-1, self.mse.numel())
        variance = logvar.exp()[valid].reshape_as(errors)
        if not errors.numel() or not torch.isfinite(errors).all() or not torch.isfinite(variance).all():
            raise ValueError("calibration needs finite held-out samples")
        self.samples.fill_(len(errors))
        self.mse.copy_(errors.mean(0))
        self.temperature.copy_((errors/variance.clamp_min(1e-6)).mean(0).clamp_min(1.))
        self.coverage.copy_((errors <= 3.841459*variance*self.temperature).float().mean(0))

    def forward(self, logvar, age, warmup):
        evidence = (age.to(logvar.dtype)/float(warmup)).clamp(0, 1).unsqueeze(-1)
        if self.config.mode == "legacy":
            evidence = ((age+1).to(logvar.dtype)/float(warmup)).clamp(max=1).unsqueeze(-1)
            return torch.exp(-.5*logvar.exp().mean(-1, keepdim=True))*evidence
        if self.config.mode == "off":
            return logvar[..., :1]*0.
        if self.config.mode == "ungated":
            return torch.ones_like(logvar[..., :1])
        prior = self.config.prior_variance
        if self.config.mode == 'bounded':
            # A bounded feedback budget keeps useful contact/history features trainable.
            # It is not a claim that unidentified parameters are known. Collection,
            # PPO replay and inference use this SAME gate, with no straight-through trick.
            variance = torch.maximum(logvar.exp()*self.temperature, self.mse)
            information = (1-variance/prior).clamp(0, 1).mean(-1, keepdim=True)
            ready = self.samples >= self.config.minimum_samples
            floor = self.config.feedback_floor
            return floor+(1-floor)*information*ready*evidence
        skill = (1-self.mse/prior).clamp(0, 1).amin()
        ready = (self.samples >= self.config.minimum_samples) & (skill >= self.config.minimum_skill)
        # The empirical MSE is a floor: a low variance head cannot certify itself.
        variance = torch.maximum(logvar.exp()*self.temperature, self.mse)
        information = (1-variance/prior).clamp(0, 1).amin(-1, keepdim=True)
        return information*skill*ready*evidence

    def get_extra_state(self):
        return asdict(self.config)

    def set_extra_state(self, value):
        # Older calibrated bundles predate feedback_floor and keep their formula.
        if asdict(PhysicsConfidenceConfig(**value)) != asdict(self.config):
            raise ValueError("Physics confidence configuration differs")

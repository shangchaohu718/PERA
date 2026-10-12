"""Explicit deployment-update schedule, separated from per-episode evidence age."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PhysicsUpdateConfig:
    ema_half_life_steps: float = 1024.
    calibration_fraction: float = .25
    maximum_mean_kl: float = .01
    maximum_token_kl: float = .1
    freeze_base_actor: bool = True

    def __post_init__(self):
        for value in (self.ema_half_life_steps, self.maximum_mean_kl, self.maximum_token_kl):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Physics half-life and KL limits must be positive")
        if not math.isfinite(self.calibration_fraction) or not 0 < self.calibration_fraction < 1:
            raise ValueError("calibration_fraction must be in (0,1)")
        if type(self.freeze_base_actor) is not bool:
            raise ValueError("freeze_base_actor must be bool")

    def ema_decay(self, observed_steps):
        return math.exp(math.log(.5)*observed_steps/self.ema_half_life_steps)

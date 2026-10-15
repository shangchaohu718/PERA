"""Causal reference-plan features for Foot-Ground Teacher V2."""
from __future__ import annotations

import math
import numpy as np


REFERENCE_INTENT_NAMES = (
    "phase_sin", "phase_cos",
    "planned_anchor_left", "planned_anchor_right",
    "planned_slide_left", "planned_slide_right",
)


def reference_intent_from_frames(reference_frame: np.ndarray, role_code: np.ndarray,
                                 *, cycle_start: int, cycle_frames: int) -> np.ndarray:
    """Encode only the plan at t; no realized t+1 contact label is accepted."""
    frame = np.asarray(reference_frame)
    roles = np.asarray(role_code)
    if frame.ndim == 2 and frame.shape[1] == 1:
        frame = frame[:, 0]
    if frame.ndim != 1 or not np.issubdtype(frame.dtype, np.integer):
        raise ValueError("reference_frame must be integer [T] or [T,1]")
    if roles.ndim != 2 or roles.shape[1] != 2 or len(roles) < 1:
        raise ValueError("role_code must have shape [frames,2]")
    if cycle_frames < 1 or cycle_start < 0:
        raise ValueError("invalid reference cycle")
    phase = np.remainder(frame.astype(np.int64) - cycle_start, cycle_frames) / cycle_frames
    current_role = roles[np.remainder(frame.astype(np.int64), len(roles))]
    values = np.column_stack((
        np.sin(2.0 * math.pi * phase), np.cos(2.0 * math.pi * phase),
        current_role[:, 0] == 1, current_role[:, 1] == 1,
        current_role[:, 0] == 2, current_role[:, 1] == 2,
    )).astype(np.float32)
    if values.shape != (len(frame), len(REFERENCE_INTENT_NAMES)):
        raise RuntimeError("reference intent layout mismatch")
    return values

"""Calibrate deterministic Top2 capacities on a fixed, diverse reference bank.

Only expert selection uses the correction. Action mixture probabilities retain
the uncorrected logits. This is outside BPTT/CUDA graph capture. The resulting
bias is frozen during updates and evaluation; live and EMA calibrate separately.
"""

import json
from pathlib import Path
import numpy as np
import torch


def fit_bias(logits, rounds=8):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8 or not np.isfinite(values).all():
        raise ValueError("Expected finite [frames,8] logits")
    bias = -values.mean(0)
    for _ in range(rounds):
        for e in range(8):
            other = np.delete(values + bias, e, axis=1)
            second = np.partition(other, -2, axis=1)[:, -2]
            bias[e] = np.quantile(second - values[:, e], 0.25)
        bias -= bias.mean()
    return bias.astype(np.float32)


class RouterCalibration:
    def __init__(self, path, device, log_dir):
        bank = torch.load(path, map_location="cpu", weights_only=True)["features"]
        self.fit = bank[::16].to(device)
        self.check = bank[1::16].to(device)
        self.path = Path(log_dir) / "router_calibration.jsonl"

    @torch.no_grad()
    def update(self, policy, iteration, name):
        logits = policy.moe.logits(policy.obs_norm.router(self.fit))
        bias = fit_bias(logits.cpu().numpy())
        policy.moe.selection_bias.copy_(torch.from_numpy(bias).to(logits.device))
        held = policy.moe.logits(policy.obs_norm.router(self.check))
        top = policy.moe.select(held)
        usage = torch.nn.functional.one_hot(top, 8).float().mean((0, 1))
        record = {
            "iteration": iteration,
            "policy": name,
            "fit_frames": len(self.fit),
            "other_check_frames": len(self.check),
            "frozen_bias_slot_usage_other_frames": usage.tolist(),
            "bias": bias.tolist(),
            "choice_is_deterministic": True,
            "evaluation_updates_bias": False,
        }
        with self.path.open("a") as out:
            out.write(json.dumps(record) + "\n")
        return record

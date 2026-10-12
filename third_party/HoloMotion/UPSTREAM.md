# HoloMotion upstream lock

This integration was audited against the official repository:

- URL: `https://github.com/HorizonRobotics/HoloMotion.git`
- commit: `cf2018a`
- license: Apache-2.0
- local clean Git object source: `/home/tcl/Desktop/start/HoloMotion`
- local K1 training worktree: `/home/tcl/Desktop/start/holomtion_boosterK1`

The official algorithm is intentionally not copied into `unified_motion` or
modified there. The adjacent K1 worktree remains the source for HoloMotion's
Actor, Critic, MoE Transformer, PPO, training configuration, checkpoint, and
ONNX export. This repository owns only the mjlab boundary and its contract
tests. Keeping that boundary explicit prevents local network implementations
from silently replacing the official baseline.

The integration artifact used for the first adapter gate is the K1
`model_9999.onnx` exported by the completed 10,000-iteration run. Its metadata
declares 22 joint names, PD gains, reset offsets, action scales, and a 506-wide
actor observation.

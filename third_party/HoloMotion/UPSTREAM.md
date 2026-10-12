# HoloMotion upstream lock

This directory vendors the HoloMotion Python package and configuration files
used by the mjlab integration.  They were exported from the official repository:

- URL: `https://github.com/HorizonRobotics/HoloMotion.git`
- commit: `cf2018a282249a04df1db4fa639760884f97b41b`
- license: Apache-2.0

The clean snapshot is under `source/`.  It contains `holomotion/`, the upstream
README and packaging metadata, and the upstream `LICENSE` and `NOTICE`.  It was
created from the Git object at the commit above, not copied from a dirty local
worktree.  Training datasets, checkpoints, exported ONNX files, third-party
robot asset submodules, caches, and deployment trees are intentionally absent.

The local integration still owns only the mjlab boundary and the optional
physics/Teacher extension.  The base Actor, Critic, MoE Transformer, PPO and
HDF5 loader are imported from this vendored upstream package.  Keeping the
boundary explicit prevents local network implementations from silently
replacing the official baseline.

The integration artifact used for the first adapter gate was the K1
`model_9999.onnx` exported by the completed 10,000-iteration run.  The model is
not redistributed in this repository.  Its metadata declares 22 joint names,
PD gains, reset offsets, action scales, and a 506-wide actor observation.

# Runtime environments

For the current HoloMotion scene replay, use [SCENE_REPRODUCTION.md](SCENE_REPRODUCTION.md) and `requirements/scene-replay.txt`. The verified replay environment is Python 3.11.13, Torch 2.9.1+cu128, NumPy 1.26.0, MuJoCo 3.11.0 and Warp 1.14.0. The older environment snapshots below describe historical packaging checks. They are not the current scene replay lock.

> Current release selection: official HoloMotion G1 v1.4 `model_14000` and the matching plugins in `resources/holomotion_weights_manifest.json`. See [HoloMotion release status](HOLOMOTION_RELEASE_STATUS.md). Self-trained MoE-MLP / v2.4 configurations and weights described below are historical artifacts, not the default release.

Use separate compatible interpreters for the CPU checks, native physics and Isaac PPO. No environment was installed or modified while assembling this package.

| Profile checked during packaging | Observed packages | Validation performed |
|---|---|---|
| CPU core | Python 3.11, Torch 2.9.1+cu128, NumPy 1.26.0, MuJoCo 3.5.0 | Routing, gradients, freeze checks and actual-checkpoint output parity |
| Existing native venv | Torch 2.9.1+cu128, NumPy 2.4.6, MuJoCo/MuJoCo-Warp 3.11.0, Warp 1.16.0, mjlab 1.6.0 | Workflow help/import checks; no physical rollout |
| PPO-compatible runtime | The CPU-check interpreter also has Isaac Lab 2.2.0 distribution metadata | PPO parser only; Isaac application not launched |

`provenance/ENVIRONMENTS.json` records read-only metadata from existing interpreters. The separately named `isaaclab` environment currently has different versions; its name alone does not make it the correct PPO runtime. These snapshots are current observations, not a claim that the entire historical paper environment is locked.

The pinned mjlab source is listed in `resources/upstreams.json`. That source requires MuJoCo/MuJoCo-Warp around 3.11 and Warp >=1.14. The `native` extra follows that profile. Do not combine it with the MuJoCo 3.5 CPU CI pins. Install the pinned checkout only into the intended native interpreter after inspecting its dependency declaration. Its dependencies include `tyro`, absent from the CPU-check interpreter.

Isaac PPO needs an existing compatible Isaac Lab installation and the HoloMotion integration dependencies. Use its matching Python; follow that source version's environment procedure. PERA does not install or select a CUDA distribution automatically.

For an existing runtime, use `pip install -e . --no-deps --no-build-isolation`, or set `PERA_PYTHON` when calling `tools/pera.sh`. For a fresh environment, inspect `pyproject.toml`, install the appropriate Torch build explicitly, then the needed extras and pinned upstream source. An editable source checkout is required because configs and reproduction workflows are outside the core Python package. A clean-environment GPU install has not been validated by this packaging task.

Teacher families differ: Instinct uses its pinned InstinctMJ/instinct_rl/mjlab source; MimicX uses the pinned Hugging Face runtime; PMT uses its Isaac Lab interpreter; ATHLETE uses ONNX Runtime and its original tennis simulator; ULTRA uses its original rl_games and MuJoCo runtime. Do not add another environment's entire site-packages to PYTHONPATH. Set family interpreter variables instead.

`pera doctor` reports import availability only. It does not certify ABI compatibility, GPU memory capacity or successful physical execution.

## Current HoloMotion release check

The existing native interpreter loaded the staged official Actor and all 12
plugins on CPU, with frozen hash and finite public-forward checks. This does not
establish a fresh dependency installation. Missing external runtime sources and
public artifacts are listed in `HOLOMOTION_RELEASE_STATUS.md`. No environment or
CUDA package was installed during this audit.

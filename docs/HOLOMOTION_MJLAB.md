# Official HoloMotion K1 → mjlab adapter

The shared K1 task now supports HoloMotion and six custom Actor backends through
`tools/train_k1_mjlab.py --actor ...`, with common Physics residual, checkpoint
resume and frozen evaluation. See [Actor switching](K1_ACTOR_SWITCHING.md).

The 2026-10-12 follow-up adds continuous Dynamics updates and standalone schema 5
Physics checkpoints with calibrated, frozen EMA/residual evaluation. Schema 4 retains
legacy inference semantics. See [algorithm corrections](ALGORITHM_CORRECTIONS_20261012.md) and
[runtime and continuity](PHYSICS_RUNTIME_AND_CONTINUITY.md) for loading, migration,
CPU verification evidence and the remaining real-simulation validation gate.

## Provenance

- Official upstream: `https://github.com/HorizonRobotics/HoloMotion`
- Audited upstream commit: `cf2018a`
- Vendored source: `third_party/HoloMotion/source/holomotion`
- Completed checkpoint used during local validation: `model_9999.pt` (not redistributed)
- Exported actor used during local validation: `exported/model_9999.onnx` (not redistributed)

The clean official source snapshot used by this package is vendored under
`third_party/HoloMotion/source/` at the commit above. The current implementation
adapts the environment-facing contracts and directly loads HoloMotion's
original MoE Transformer Actor and Critic; it does not fork or rewrite those
network implementations. Checkpoints, ONNX exports and datasets remain external.

## Contract implemented

The exported K1 actor has these graph inputs and outputs:

- `obs`: `[1, 506]`
- `past_key_values`: `[1, 2, 1, 32, 2, 64]`
- `step_idx`: `[1]`
- `actions`: `[1, 22]`

The 506 actor values are assembled in the same term order as
`obs_motion_tracking_tf-moe.yaml`: 106 current-state/reference values followed
by ten 40-value future-reference frames. Joint columns are always mapped by the
ONNX `joint_names` metadata, never by an assumed numeric order.

The mjlab action configuration must keep the HoloMotion training semantics:

`target = default_joint_pos + action * action_scale`

The generic K1 mjlab control profile does **not** match the trained policy for
the head and hip-roll joints because the raw MJCF effort limits differ from the
Isaac Lab training limits. `make_holomotion_k1_mjlab_cfg` therefore builds the
PD gains, effort limits, reset offsets, and action scales from the exported
HoloMotion contract. The raw MJCF still supplies rigid-body physics, joint
limits, and armatures.

The HoloMotion-specific config uses the training clock: 200 Hz physics
(`physics_dt=0.005`) with four physics steps per 50 Hz policy action. The older
generic K1 bridge default of 500 Hz/10 steps is intentionally not reused.

`holomotion_mjlab_rewards` implements the complete K1 reward configuration:
the eight positive tracking/alive terms, action-rate and joint-limit penalties,
and the undesired-contact penalty. The contact sensor explicitly covers the 19
K1 bodies selected by the official exclusion rule and retains all four physics
substeps. Termination uses the official gravity-projection distance threshold
of `0.8`; the generic absolute root-height termination is disabled.

The asymmetric critic contract is also available. It assembles the exact
843-value input expected by the existing `model_9999` critic checkpoint (463
current-state values plus ten 38-value future-reference frames).

`holomotion_original_algorithm` reconstructs the original trainable
`PPOTFRefRouterActor` and `PPOCritic` directly from the resolved run config. It
splits the local flat 506/843 tensors back into the original TensorDict leaves
and restores both safetensors files with strict key checking. The original
HoloMotion worktree remains the single owner of the network implementation.
The optional `holomotion` dependency group documents the extra runtime
packages needed in the mjlab environment.

`holomotion_mjlab_task` and `holomotion_mjlab_ppo` provide the first trainable
integration slice. Rollout sampling uses the original Transformer's KV cache;
update replay preserves complete per-environment sequences and uses the
original episode-aware causal mask, GAE, clipped PPO objective, clipped value
loss, entropy bonus, gradient clipping, and AdamW parameter grouping. Updates
also preserve the original environment-wise mini-batching, multi-epoch replay,
analytic KL measurement, windowed KL early stopping, and adaptive learning
rate. The current slice can restore both original AdamW optimizer states. It
also reproduces the active checkpoint losses: auxiliary body-frame linear
velocity prediction, ten-body contact classification, six-body reference and
robot root-relative position prediction, and inactive-expert-to-top-k margin.
The auxiliary contact labels come from a dedicated ordered MJLab sensor rather
than the separate 19-body undesired-contact reward sensor. It intentionally
excludes the inactive configuration branches, curriculum, distributed update,
and full runner state. Those are the remaining parity items before a long
training run.

The exported checkpoint metadata contains the episode's randomized default
joint offsets. A few knee offsets can lie a few milliradians outside the MJCF
hard limit. Physics reset offsets are clipped to the hard limit (and rejected
when the discrepancy exceeds 0.02 rad); actor observation centering continues
to use the exported metadata, matching deployment preprocessing.

## Current gate

CPU contract tests cover metadata validation, 506-D assembly, quaternion/yaw
features, end-of-clip padding, and ONNX-to-mjlab joint permutation. A real
policy rollout additionally requires the local mjlab CUDA stack, K1 assets, a
reference clip with current plus ten future frames, one additional future pose
used only for centered velocity reconstruction, and ONNX Runtime. Reference
root velocities are derived from poses exactly as in official training; motion
file velocity fields are intentionally ignored. Joint velocities are likewise
derived from joint positions with HoloMotion's one-sided boundary and centered
interior differences. This repairs stale first-frame `ref_dof_vel` values found
in official offline-evaluation dumps. It must be
reported as simulation validation, not robot validation.

Run the real evaluator in an environment containing both mjlab and ONNX
Runtime:

```bash
python tools/evaluate_holomotion_k1_mjlab.py \
  --onnx /path/to/exported/model_9999.onnx \
  --asset-root /path/to/booster_k1_assets \
  --motion /path/to/named_k1_reference.npz \
  --device cuda:0 --steps 500 \
  --output runs/holomotion_k1_mjlab/evaluation.json
```

Convert an official HoloMotion offline-evaluation NPZ first. Those files store
`ref_global_rotation_quat` as `xyzw`; the adapter's named reference format is
explicitly `wxyz`:

```bash
python tools/convert_holomotion_eval_reference.py \
  --input /path/to/official_eval_rollout.npz \
  --robot-config /path/to/22dof_training_isaaclab.yaml \
  --source-quaternion-order xyzw \
  --output /tmp/k1_reference_named.npz
```

The evaluator refuses a provider fallback, initializes the robot from the
first reference frame, carries the Transformer KV cache forward, stops at the
official gravity terminal state, and records rewards, contact counts, tracking
errors, target clipping, and actuator saturation for every step.

## Latest CUDA audit

`walk1_physics_audit_v4.json` validates the real mjlab/Warp CUDA path. Correct
joint-velocity reconstruction reduced initial key-body velocity RMSE from
`0.904 m/s` and `4.87 rad/s` to about `2.5e-4` in both channels. Initial reward
is the exact contact-free maximum `13.5`, and no actuator saturation occurs.
The frozen Isaac Lab actor nevertheless reaches the gravity terminal after 65
steps (1.3 seconds), so this is a verified sim-to-sim dynamics gap rather than
a successful policy port. A short in-memory PPO smoke command is available at
`tools/smoke_train_holomotion_k1_mjlab.py`; it performs no checkpoint writes.
Starting a costly PPO run still requires an explicit training decision.

## Bounded training runner

`tools/train_holomotion_k1_mjlab.py` is the resumable runner. Both
`--iterations` and `--output-dir` are mandatory; it never silently inherits the
upstream 10,000-iteration duration. With `--resume-optimizer`, iteration and
AdamW moments continue from the source checkpoint. Each output contains a
copied `config.yaml`, JSON runner provenance, JSONL metrics, the two original
safetensors directories, and the matching optimizer `.pt` file. Existing
checkpoint targets are never overwritten.

The audited run resolves `sampling_strategy: weighted_bin`, but its training
data declares exactly one LaFAN dataset root with ratio `1.0`. Therefore the
effective upstream behavior is uniform sampling within that dataset, not the
score-driven `curriculum` strategy. Repeat `--motion` to load multiple named
NPZ clips into one uniformly sampled library. Per-clip sampled steps, mean
reward, and done count are written into every JSONL iteration record.

The original training manifest is HDF5 v2 and stores only root pose plus joint
pose. `tools/convert_holomotion_hdf5_dataset.py` reuses the original
`Hdf5RootDofDataset` and `TrainingReferenceFK` to recreate body poses, body
velocities, joint velocities, the same 300-frame windows, and the same world
normalization. Its output is loaded with `--motion-dir`. A one-window test from
the local LaFAN manifest produced a finite 300-frame, 23-body clip and completed
a real mjlab PPO iteration.

```bash
python tools/train_holomotion_k1_mjlab.py \
  --run-dir /path/to/source/run \
  --checkpoint-name model_9999 --resume-optimizer \
  --onnx /path/to/source/run/exported/model_9999.onnx \
  --asset-root /path/to/booster_k1 \
  --motion /path/to/named_reference_1.npz \
  --motion /path/to/named_reference_2.npz \
  --output-dir /path/to/new/mjlab_run \
  --iterations 1 --steps-per-env 32 --num-envs 1 --device cuda:0
```

Enable the physics-student / privileged-Teacher extension explicitly:

```bash
python tools/train_holomotion_k1_mjlab.py \
  --run-dir /path/to/source/run --checkpoint-name model_9999 \
  --onnx /path/to/source/run/exported/model_9999.onnx \
  --asset-root /path/to/booster_k1 \
  --motion /path/to/named_reference.npz \
  --output-dir /path/to/new/physics_teacher_run \
  --iterations 1 --steps-per-env 32 --num-envs 1 --device cuda:0 \
  --physics-teacher
```

The student path never reads the mjlab contact sensor. It receives
proprioception plus ordered left/right foot link kinematics. The privileged
Teacher target is deliberately narrow: two normalized, episode-randomized slow
variables (ground friction and trunk-mass scale). Exact left/right contact
point, normal, force, slip, and friction utilization remain privileged labels
for next-state/contact prediction and rewards; they are never Actor inputs.
The deployable latent defaults to 8 values.

The residual Actor input is ordered as global context, student latent,
left-foot token, right-foot token; there is no left/right set pooling at this
boundary. Its residual is gated by both predicted system-identification
uncertainty and evidence age. Dynamics training accumulates across rollout
boundaries, uses 64 burn-in frames followed by 128 gradient-carrying frames,
and recomputes/optimizes that sequence 4 times by default. The Actor consumes
an EMA copy of DynamicsModel to avoid a moving feature target. These values are
configurable through the `--physics-*` options.
Left/right slip and impact weights remain independently configurable. Physics
checkpoints keep the upstream `actor/model.safetensors` layout and store the
residual, online DynamicsModel, EMA DynamicsModel, Teacher, uncertainty heads,
training configuration, and optimizer states in the companion `.pt` file.

The trainable checkpoint loader has already passed a direct parity check: its
zero-observation Actor mean differs from the exported ONNX graph by at most
`1.38e-6`, while the restored Actor and Critic produce finite `[1,22]` and
`[1,1]` outputs respectively.

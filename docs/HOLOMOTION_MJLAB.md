# Official HoloMotion K1 → mjlab adapter

## Provenance

- Official upstream: `https://github.com/HorizonRobotics/HoloMotion`
- Audited upstream commit: `cf2018a`
- Existing K1 training worktree: `/home/tcl/Desktop/start/holomtion_boosterK1`
- Existing completed checkpoint: `model_9999.pt`
- Existing exported actor: `exported/model_9999.onnx`

The official source and the K1 training worktree remain separate from this
package. The current implementation adapts the environment-facing contracts;
it does not yet duplicate HoloMotion's MoE Transformer or PPO implementation.

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
a successful policy port. The next implementation gate is connecting the
official trainable actor/critic and PPO rollout to this mjlab task contract;
starting a costly PPO run still requires an explicit training decision.

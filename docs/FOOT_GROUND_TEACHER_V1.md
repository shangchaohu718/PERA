# K1 Foot-Ground Dynamics Teacher V1

This implementation is offline and intentionally disconnected from PPO and the
Actor. Real Isaac 5.1 collection uses the matching source task; Teacher quality
is decided only after the three-seed, held-out-checkpoint protocol below.

## Isaac Lab / RSL-RL Moonwalk policy compatibility

Native Moonwalk checkpoints are **not** compatible with
`load_k1_checkpoint()`. They use an RSL-RL ELU MLP and a concatenated 125-D
Isaac Lab observation, while this repository's standalone K1 policies use a
different 506-D contract. Use `unified_motion.rsl_k1_policy` instead. The
adapter exactly preserves the checkpoint's empirical normalizer, returns the
deterministic Actor mean before Isaac's action manager, freezes every parameter,
and rejects architectural or observation mismatches.

For a trusted native checkpoint, create a tensor-only collection bundle with:

```bash
python tools/convert_rsl_k1_checkpoint.py SOURCE_MODEL.pt OUTPUT_FROZEN.pt \
  --checkpoint-id globally_unique_run_and_iteration \
  --trusted-source \
  --verify-rollout HISTORICAL_ROLLOUT.npz
```

`--trusted-source` is only needed when newer checkpoints contain NumPy RNG
objects in `infos`; it explicitly permits pickle for that one conversion.
Subsequent loads of `OUTPUT_FROZEN.pt` use `weights_only=True`. The source file
is never modified, and its SHA-256 is stored in the bundle. During collection,
feed the adapter `observation["policy"]` directly from the matching Isaac Lab
task. Do not fabricate it from HoloMotion/MJLab observations; joint ordering,
default-position offsets, reference command, and `last_action` must remain those
of the source task.

The source Isaac task can now be collected directly with the project-local
launcher. For example:

```bash
MOONWALK="/home/tcl/Desktop/start/dance-evaluator and tracker/k1_moonwalk"
"$MOONWALK/scripts/run_isaac51_k1.sh" \
  /home/tcl/Desktop/start/HoloMotion-UM/tools/collect_foot_ground_isaac.py \
  --checkpoint "$MOONWALK/logs/rsl_rl/.../model_999.pt" \
  --policy-bundle \
    /home/tcl/Desktop/start/HoloMotion-UM/data/foot_ground_v1/frozen_policies/v13_direction_model999.pt \
  --checkpoint-id v13_direction_model999 \
  --env-config-class \
    k1_moonwalk.project_task.env_cfg:MoonwalkV13DirectionGateHeelToeEnvCfg \
  --phase-labels "$MOONWALK/motions/phase_labels_61_64_50hz_v2.npz" \
  --output data/foot_ground_v1/v13_direction_seed730201.npz \
  --trajectory-id v13_direction_seed730201 --seed 730201 \
  --headless
```

Collection uses only the tensor-only bundle for inference. The native file is
passed as immutable provenance: before environment allocation, collection
requires both its SHA-256 and the campaign's globally unique checkpoint ID to
match the bundle. Direct native loading is rejected unless the operator uses
the explicit `--trusted-source` escape hatch.

The checked-in campaign config expands six behaviorally distinct checkpoints,
three independent seeds, and three fixed friction conditions into 54 complete
trajectory jobs. Generate its auditable command manifest before execution:

```bash
python tools/run_foot_ground_collection_campaign.py \
  --config configs/teachers/k1_foot_ground_collection_campaign.yaml \
  --job-manifest reports/foot_ground_teacher_v1/collection_jobs.json \
  --headless
```

Audit all six bundle/source bindings independently with:

```bash
python tools/audit_rsl_k1_bundles.py \
  --config configs/teachers/k1_foot_ground_collection_campaign.yaml \
  --output reports/foot_ground_teacher_v1/frozen_policy_bundles.json
```

Add `--execute` only on a host where the Isaac launcher can see a working CUDA
device. Existing trajectory files are skipped rather than overwritten. On this
host, `/dev/nvidia*` was restored with `nvidia-modprobe`; a real 20-step smoke
trajectory passed the raw-buffer/unit audit before the 54-job campaign began.

The collector samples ground-filtered PhysX buffers on every physics substep.
The final substep supplies the real contact point, normal, point velocity and
instantaneous force; all substeps supply the integrated normal impulse. Raw
normal and friction point buffers are retained. Transitions enter the terminal
pre-reset state but no reset-state transition is emitted. If auto-reset makes
the simple inverse-dynamics baseline unavailable, only the residual label is
masked; the otherwise valid terminal dynamics/contact transition is retained.

Feature schemas are fixed in `foot_ground_teacher.features`: Student-visible
input is 94-D (`q/dq/IMU/tau_est/previous action`), current privileged global
input is 12-D, and current per-foot physics is 15-D. Future labels cannot be
passed to these packers.

## Contracts

- Control rate is fixed at 50 Hz (`control_dt_s=0.02`). Solver forces are N.
  The auxiliary control-step impulse integrates substep forces using the physics
  step and is N·s.
- Inputs at index `t` are proprioceptive history, the executed action at `t`, and
  explicitly named current privileged physics. Every target is the state/contact
  at `t+1`. The physical step into a pre-reset terminal state is valid; no row
  whose input is an auto-reset state is emitted. Padding or any transition with
  no physical successor must set `valid_transition=false`.
- Four modes are fixed: separation, stable contact, sliding, impact. Sliding
  intent is supplied from the reference phase; it is not inferred from large
  measured slip, so active moonwalk sliding and accidental slip remain distinct.
- The tangent frame uses projected robot-forward and a deterministic least-aligned
  world-axis fallback. Point velocity is `v_body + omega × (p_contact-p_body)` and
  is relative to an explicit ground velocity.
- Raw fixed-size contact slots are retained. Per-foot point/normal use
  compressive-force weighting; forces sum vectorially. No-contact geometry,
  velocity and force regression is masked.

## Workflow

1. Inventory hashes with `tools/inventory_foot_ground_checkpoints.py`.
2. For the old Moonwalk policies, use `collect_foot_ground_isaac.py` in the source
   Isaac task. `FootGroundMJLabLabeler` remains available only for policies whose
   observation/task contract is genuinely MJLab-compatible.
3. Store one complete episode per NPZ through `write_trajectory()`. Include the
   checkpoint/reference/seed/phase/randomization/termination metadata. Build a
   source-exclusive manifest; this campaign reserves `v13_grounded_model499`
   for validation and `v13_f2r1_model12929_heldout` for test. Smoke data is
   excluded.
4. Audit the complete trajectories and build the manifest before training:

   ```bash
   python tools/build_foot_ground_manifest.py \
     --data-dir data/foot_ground_v1 --exclude-dir smoke \
     --output reports/foot_ground_teacher_v1/dataset_manifest.json \
     --coverage-output reports/foot_ground_teacher_v1/contact_coverage.json \
     --validation-checkpoint v13_grounded_model499 \
     --heldout-checkpoint v13_f2r1_model12929_heldout
   ```

5. Run each contact-weight ablation separately. Do not combine whitening and
   Replay changes in the same comparison:

   ```bash
   python tools/train_foot_ground_teacher.py \
     --manifest data/foot_ground_v1/manifest.json \
     --output-dir runs/foot_ground_v1_seed730201_cw01 \
     --epochs 20 --seed 730201 --contact-weight 0.1
   ```

6. Select the loss weight and contact threshold on validation, then pass the
   frozen numeric threshold to test. `run_foot_ground_teacher_experiments.py`
   runs all `1.0/0.1/0.03 × 3 seeds`, selects the weight by pooled validation
   contact F1, and opens the held-out policy only for the selected weight.

The research quality thresholds are frozen in
`configs/teachers/k1_foot_ground_v1.yaml`. A failed gate blocks Student/Actor
integration; it is reported rather than re-tuned on the held-out checkpoint.

The evaluation report separates feet, contact modes, active/unexpected sliding,
impact and friction bins. It reports persistence and simple-physics residual
baselines. Contact force, point and ground friction may be poorly identifiable
from future Student observations alone; privileged Teacher accuracy is not a
deployment estimate.

## V1 independent result (2026-10-15)

The real campaign produced 54 trajectories and 14,063 valid 50 Hz transitions.
All trajectory/source/force/impulse audits passed. Complete checkpoint sources
were split 36 train / 9 validation / 9 held-out test; no neighboring windows or
checkpoints cross a split.

The first fixed-order baseline collapsed its mode head to separation. A single
controlled change (seeded whole-trajectory shuffle each epoch) fixed contact
calibration but not mode imbalance. Train-only inverse-frequency mode weights
then raised validation mode macro-F1 to about 0.48; increasing from 20 to 100
epochs was required to clear the contact/force/mode validation thresholds.
These controls are retained as separate run directories. Replay remained 1 and
the model, regression standardization, and other loss groups were unchanged.

The final `1.0 / 0.1 / 0.03` contact-weight ablation used three Teacher seeds and
3,600 optimizer updates per run. Weight `1.0` had the best pooled validation
contact F1 and was selected before test access. It passed 17/18 validation checks
but predicted joint-velocity deltas 9.10% worse than Persistence, so the quality
gate already failed. Frozen validation thresholds were nevertheless evaluated
on the reserved unseen policy as a diagnostic. Held-out contact F1 was 0.471
left / 0.441 right, mode macro-F1 0.306 / 0.243, slip-speed RMSE 0.491 / 0.570
m/s, normal-force RMSE 358.6 / 362.2 N, and root angular-delta RMSE 1.549 rad/s.

Therefore Teacher V1 is **not accepted for Student distillation or control
integration**. No Student was trained, no Actor/checkpoint was modified, and no
PPO update was run. The primary next problem is cross-policy domain shift (the
held-out stability policy has far more separation/impact and almost no stable
support), followed by dynamics-head generalization without sacrificing the now
useful contact heads.

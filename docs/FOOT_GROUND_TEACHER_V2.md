# K1 Foot-Ground Dynamics Teacher V2 protocol

V2 tests task allocation and action-conditioned physics before any MoE, Student,
Actor, or PPO integration.  The old V1 held-out checkpoint is development-only;
a new checkpoint-exclusive source is required for final acceptance.

## One-change-at-a-time ablations

| Name | Active supervision | Reference intent |
|---|---|---|
| `v1_reproduction` | all V1 groups | no |
| `v2_a1` | contact/mode, tangent velocity, full force, dynamics | no |
| `v2_a2_core3` | contact/mode, tangent velocity, normal force | no |
| `v2_a1_intent` | same as `v2_a1` | current phase and planned foot roles |

`v2_a1` removes contact geometry, normal velocity, and the redundant scalar
slip-speed loss.  The model may still emit inactive heads for checkpoint and
evaluation-schema stability, but those heads contribute zero training gradient.

Reference intent is six-dimensional: sine/cosine phase, planned anchor flags for
left/right, and planned slide flags for left/right.  It is derived from the
current `reference_frame` only.  Realized next-frame `active_sliding`, contact,
or force labels are never inputs.

## Action counterfactual gate

For a selected state, four Isaac environments are cloned to the same exposed
physical, controller, delayed-actuator, command, and previous-action state.  One
step is executed with the baseline action and with a selected physical joint
target perturbed by `+delta_rad` and `-delta_rad`; the fourth environment repeats
the baseline action to measure the numerical/clone noise floor.  The collector
converts the physical radian perturbation through the resolved action scale and
records both the processed target and the exact delayed target selected by the
official actuator buffer after the final 200 Hz physics substep.  It also
records actuator lag and applied-torque differences; a processed command alone
is not accepted as proof that the perturbation reached the robot.

The evaluation compares physical differences, not just absolute predictions:

- tangent-velocity and normal-force delta MAE/RMSE;
- delta direction cosine and sign accuracy;
- sensitivity gain and action-ignored rate;
- contact and contact-change accuracy;
- symmetric finite-difference Jacobian agreement.

Collection is invalid unless the pre-branch state error is below tolerance,
both processed and delayed actuator targets match the requested physical-radian
perturbations, and the duplicate-baseline branch agrees with the base branch.
Probe informativeness is a separate result: a valid local state may genuinely
have no measurable one-step tangent or force response.  Physical action effects
are reported both relative to three times the measured null-response noise
floor and against the fixed 5 mm/s and 5 N materiality floors.

Tangent-velocity differences are evaluated only where the foot remains in
contact in all three branches.  Normal force includes zero-load branches so
contact creation/loss remains observable.  Contact-transition probes are
reported separately because their dynamics are discontinuous.

## Commands

Create the causal-intent copy of the immutable V1 trajectories:

```bash
python tools/augment_foot_ground_reference_intent.py \
  --manifest reports/foot_ground_teacher_v1/dataset_manifest.json \
  --phase-labels /path/to/phase_labels_61_64_50hz_v2.npz \
  --output-dir data/foot_ground_v2/reference_intent \
  --output-manifest reports/foot_ground_teacher_v2/dataset_manifest_reference_intent.json
```

Create or execute the 4 experiment x 3 seed plan:

```bash
python tools/run_foot_ground_v2_experiments.py \
  --config configs/teachers/k1_foot_ground_v2.yaml \
  --baseline-manifest reports/foot_ground_teacher_v1/dataset_manifest.json \
  --intent-manifest reports/foot_ground_teacher_v2/dataset_manifest_reference_intent.json \
  --output-root runs/foot_ground_teacher_v2/formal \
  --launcher /path/to/python --device cuda:0 [--execute]
```

Isaac counterfactual collection must use the Moonwalk Isaac launcher and a safe
tensor-only policy bundle.  The native RSL-RL checkpoint is provenance only.

No tool in this protocol starts PPO or modifies an Actor or source policy.

The first audited smoke result is
`reports/foot_ground_teacher_v2/counterfactual_action_sensitivity_gate.json`.
It fails both V1 and V2-A1 for control guidance: V1 nearly ignores action
changes, while V2-A1 responds more strongly but reverses tangent direction on
one of two informative foot probes and still captures only about 2.5% of the
material normal-force sensitivity.

Counterfactual supervision is forbidden until
`tools/audit_counterfactual_dataset_coverage.py` passes. Its default gate
requires at least 200 valid probes, three checkpoint sources, three simulator
seeds, three action-delta levels, three friction levels, both action roles and
foot sides, all four contact modes, and a checkpoint-exclusive held-out source.
The current six valid probes fail this gate, so no few-shot fine-tuning result
may be presented as mechanism learning.

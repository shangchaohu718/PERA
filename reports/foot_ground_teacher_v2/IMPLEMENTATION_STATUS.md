# Foot-Ground Teacher V2 — implementation status

## Current decision

V2-A1 and V2-A2 Core-3 have completed 100 epochs x 3 seeds on the unchanged
V1 train/validation split.  Neither result is sufficient to promote a MoE or
to enter Student, Actor, or PPO integration.  The old V1 held-out test was not
opened for V2 model selection.

## Validation-only result

| Metric | V1 | V2-A1 | V2-A2 Core-3 |
|---|---:|---:|---:|
| Contact F1, left/right | .905 / .889 | .909 / .903 | .902 / .900 |
| Contact BCE, left/right | .407 / .451 | .694 / .619 | .777 / .631 |
| Contact Brier, left/right | .123 / .140 | .162 / .154 | .177 / .155 |
| Mode macro-F1, left/right | .481 / .476 | .499 / .537 | .463 / .521 |
| Tangent velocity RMSE m/s, left/right | .152 / .111 | .190 / .181 | .158 / .153 |
| Normal force RMSE N, left/right | 129.1 / 126.7 | 84.7 / 89.5 | 83.0 / 88.2 |
| Delta joint velocity RMSE | 1.409 | 1.438 | not supervised |
| Improvement vs persistence | -9.10% | -11.13% | not supervised |
| Delta root angular velocity RMSE | .308 | .286 | not supervised |

Inactive V2-A2 heads are still emitted for schema compatibility but their
values are not valid evaluation results.

V2-A1 removes the original Geometry dominance and makes all remaining mean
pairwise gradient cosines positive.  It does not create balanced allocation:
mean shared-trunk norms become Contact 2.421, Kinematics 1.389, Force 1.090,
and Dynamics .946.  Contact calibration and tangent velocity become worse.

V2-A2 reduces Force to normal force only.  Mean norms are Contact 2.379,
Kinematics 1.229, and Force .314.  Contact/Force cosine is -.104 with 63.9%
negative audits; Kinematics/Force cosine is -.034 with 64.8% negative audits.
This is evidence for targeted branch separation, not yet for a learned MoE.

## Contact calibration diagnosis

The BCE regression is caused primarily by extreme confident errors, not merely
by probabilities clustering near 0.5.  On the pooled validation predictions:

| Metric left/right | V1 | V2-A1 | V2-A2 Core-3 |
|---|---:|---:|---:|
| ECE | .107 / .098 | .151 / .123 | .165 / .127 |
| MCE | .276 / .447 | .558 / .415 | .615 / .409 |
| Positive probability median | .801 / .794 | .982 / .973 | .984 / .976 |
| Positive probability q01 | .253 / .193 | .00049 / .00133 | .00072 / .00329 |
| Negative probability q90 | .794 / .815 | .984 / .982 | .996 / .993 |

V2 therefore produces a highly polarized distribution: most positive samples
are near one, but a small positive tail is assigned nearly zero probability;
many negative samples are simultaneously assigned nearly one.  A validation
threshold near .05 preserves F1 but does not repair probability quality.

Impact is especially weak.  Left-foot impact BCE rises from 1.03 in V1 to
2.08/2.12 in V2-A1/Core-3.  Separation BCE also worsens despite some improvement
in the mode head.  Contact F1 must not be interpreted as a contact-quality win.

The calibration artifacts contain ten-bin reliability curves, ECE/MCE,
positive/negative probability quantiles, and separate separation/stable/sliding/
impact statistics for both feet.  V2's BCE degradation is therefore identified
as polarized confident error rather than simple uncertainty around 0.5.

Machine-readable pooled metrics and all gradient audits are in
`v2_a_validation_summary.json` and the per-seed run directories.

## Implemented protocol

- Task profiles: `v1`, `v2_a1`, and `v2_a2_core3`.
- Optional six-dimensional causal reference intent at time t.
- Immutable 54-trajectory reference-intent dataset copy and source-exclusive
  manifest.
- Exact same-state four-environment Isaac collector for baseline, +delta,
  -delta, and a duplicate-baseline numerical-noise control.
- Frozen-Teacher counterfactual evaluator for physical deltas, symmetric
  finite differences, sign/direction, sensitivity gain, action-ignored rate,
  contact branch accuracy, and contact-change accuracy.
- Hard collection-quality checks for pre-branch state equality, processed
  physical target differences, duplicate-baseline contact agreement, and
  tangent/force numerical noise.
- V1 checkpoint backward compatibility and explicit active-supervision fields
  so inactive heads cannot be presented as successful predictions.

## Pending runtime work

The unrelated V22 PPO process exited naturally; it was not terminated.  The
counterfactual Isaac smoke has now run with six same-state, four-branch probes.
Every audited probe had exact cloned exposed state, an actual delayed actuator
target of +/-.02 rad after a two-physics-step lag, and a measurable selected-
joint torque change.  Two contact-sensitive probes (frames 15 and 37) failed
the duplicate-baseline numerical-noise gate and were excluded.  This exposes a
remaining hidden-physics/contact-cache cloning limitation around transitions.

The role-paired extension now contains ten attempted probes. Six pass the
collection gate; three have tangent response above the fixed 5 mm/s material
floor. Across those probes and three Teacher seeds, V1 has pooled tangent
cosine .373, sign accuracy .611, sensitivity gain .027, and action-ignored
rate 1.0. V2-A1 increases gain to .212 but has cosine -.315 and sign accuracy
.389: it predicts the wrong direction on two of the three material probes.
The one valid force probe above
5 N showed V2-A1's direction was correct, but its median gain was only .025;
both model families still had force action-ignored rate 1.0.

The Counterfactual Action Sensitivity Gate therefore fails for both V1 and
V2-A1.  Better absolute force RMSE is not evidence of useful action-conditioned
physics.  MoE and Actor integration remain blocked.  The next experiment must
make actuator lag/current delayed target and the intended short future action
plan explicit, and add action-contrastive supervision; simply adding history
or training more epochs is not justified by this result.

Formal V2-A1+Intent has completed 100 epochs x 3 seeds (3,600 updates per
seed) on CPU after unrelated processes occupied the GPU. It used the same
training code, manifest, loss weights, and validation-only selection protocol.
Intent improves tangent RMSE from .190/.181 to .169/.172 m/s and helps
unexpected-slip tangent RMSE (.212 to .187), but contact BCE changes from
.694/.619 to .803/.623 and Brier from .162/.154 to .180/.163. Contact F1 and
mode macro-F1 also decline. Intent is retained only as an optional future
feature; it is not promoted as the V2 winner and does not repair the failed
counterfactual action-sensitivity gate. On the same material probes, Intent has
tangent cosine .085, sign accuracy .611, gain .113, and action-ignored rate
.667; force gain is .027 with ignored rate 1.0.

## Counterfactual training readiness

Action-contrastive fine-tuning has not started. The machine-readable readiness
gate currently finds only 6 valid probes from one checkpoint, one simulator
seed, one delta, and one friction. Material coverage is 2 active-slide and 1
anchor tangent probes, 1 active-slide force probe, no anchor force probe, and
no valid impact sample. This fails every declared scale/generalization gate.

A 576-job source-exclusive campaign is materialized but not launched in full:
288 train, 144 validation, and 144 checkpoint-held-out test jobs; delta and
friction levels are exactly balanced. One compatibility job from each of the
four checkpoint/config families ran successfully. Of those four, two passed
the strict duplicate-baseline noise gate, one was physically valid but had no
contact response, and two noisy contact cases were excluded. This pilot warns
that the full campaign needs quality filtering and likely a more exact hidden
contact-state replay mechanism before its yield is assumed.

A new checkpoint-exclusive final test source must be selected and frozen only
after validation-based architecture selection.  The old V1 test remains a
development diagnostic and cannot be reused as the V2 final claim.

## V3 action-contrastive objective scaffold

An independent action-contrastive objective is now implemented in
`unified_motion/foot_ground_teacher/counterfactual_objective.py`, but it is
deliberately not connected to the trainer. It compares the predicted and Isaac
measured `plus - minus` response from the same pre-branch state. This isolates
action sensitivity from common absolute next-state bias.

The tangent loss requires persistent contact in base/plus/minus and uses a
5 mm/s material-response floor. The normal-force loss retains feet contacting
in any branch and uses a 5 N floor, so loading and unloading transitions are
not discarded. Regressions are normalized in physical units and a separate
tangent-direction cosine loss makes a wrong response direction explicit.
Empty regimes produce differentiable zero losses rather than invalid values.

This scaffold does not authorize training. Wiring it into the formal trainer
remains blocked on `counterfactual_training_readiness.json` reporting
`training_ready: true`; the current report is false. A unit test confirms that
an exact action response has zero loss while an action-ignoring prediction is
penalized. The complete Teacher/RSL/multihorizon regression suite now has 23
passing tests.

The planned `clone` versus `deterministic_replay` Isaac A/B remains a runtime
requirement. On 2026-10-15 the host exposed the loaded NVIDIA 580.65.06 kernel
module but no `/dev/nvidia*` device nodes, and `nvidia-smi` could not communicate
with the driver. No new probe was counted and no external process, driver, or
host service was modified. The collector implementation is syntax-valid, but
the new replay mode must not be described as validated until that A/B runs on
a restored Isaac-capable GPU runtime.

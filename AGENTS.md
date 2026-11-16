# Demonstration policy

User requirement confirmed on 2026-11-13: all subsequent demonstrations must use
the official HoloMotion G1 v1.4 `model_14000` public Actor and experts trained
against that exact frozen Actor. Do not substitute the self-trained Blink / G1
MoE-MLP base, its 30-expert bank, or failed competitive-training checkpoints.

`configs/demo_policy.json` records the required public digest and the matching
demonstration expert sources. The official tennis expert is the accepted 512-env
DAgger `vector_512_dagger_20261104_v2/round_09.domain.pt` (44,000 model updates).
Keep the original observation/action conventions, context normalization, routing
calculation and checkpoint merge bias. Validate base/interface/context hashes.

Use one official Actor for shared-policy demonstrations, with independent robot
temporal caches and at most one enabled task expert per robot/inference call.
Do not label teacher-driven diagnostics or archived self-trained demonstrations
as demonstrations of the official HoloMotion student.

The existing competitive source/checkpoints remain research artifacts; retain
them. The self-trained shared-world showcase is archived for demonstrations. The
official shared-world adapter is implemented and locally checked in
`provenance/OBJECT_OBSTACLE_SHOWCASE_CHECK.json` (seven selected regions),
`provenance/COMPACT_HOLOMOTION_SHOWCASE_CHECK.json` (earlier nine regions) and
`provenance/OFFICIAL_HOLOMOTION_SHOWCASE_CHECK.json` (earlier four regions). Use `python -m pera demo`;
do not bypass the demo policy check. A short demonstration is not a new
statistical task acceptance or robustness evaluation.
No training, downloads, uploads or commits are authorized merely by this switch.

# Release selection

User requirement confirmed on 2026-11-13: the community release must use the
official HoloMotion G1 v1.4 model_14000 base and its matching task experts, not
the self-trained MoE-MLP / suite31 bank. The selected checkpoints retain their
continuous anchor-group routing; do not describe them as global competitive
Top-2. Keep historical experiments, but exclude their weights from this release.
Local staging and audit do not authorize uploads, commits or visibility changes.
Do not claim standalone reproduction until the source/resource distribution and
clean-checkout inference and training checks have actually passed.

User clarification on 2026-11-13: do not redistribute official HoloMotion
Actor/Critic weights. Keep them out of the release folder; direct users to the
pinned official v1.4.0 downloads in resources/holomotion_base.json. Retain hashes
for verification. The original native research checkpoints must remain intact.
Official model-card terms differ from the code license; do not relicense models
or learned plugins under the root MIT notice by assumption.

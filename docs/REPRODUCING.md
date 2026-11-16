# Reproduction protocols and limitations

For the HoloMotion-based open-source release, follow [SCENE_REPRODUCTION.md](SCENE_REPRODUCTION.md). Base pretraining is not included. The expert-training protocols below describe archived research experiments and require their own compatible base and resources.

> Current release selection: official HoloMotion G1 v1.4 `model_14000` and the matching plugins in `resources/holomotion_weights_manifest.json`. See [HoloMotion release status](HOLOMOTION_RELEASE_STATUS.md). Self-trained MoE-MLP / v2.4 configurations and weights described below are historical artifacts, not the default release.

## Scope of v2.4

The main model is the 608-observation / 730-goal / 29-action G1 MoE-MLP with ten observation frames. Its eight routed experts use Top-2 selection, plus one shared FFN. A frozen expansion adds one context-aware FFN and router. Choice uses corrected float32 scores; mixture probabilities use raw selected logits. Initial admission calibration is a probe target, not a forced quota during training. Learned router bias is clamped nonpositive; the separately calibrated fixed selection offset is not a continuous residual gate.

`src/pera/routing.py` is the sole competitive implementation used by retained PPO and DAgger workflows. `src/pera/fusion.py` is the explicitly labeled continuous-fusion comparison. Historical formats are thin wrappers in `reproduction/`; checkpoint field names remain compatible. The shared FFN participates exactly once.

## Expert PPO

512 environments × 32 steps; 3 epochs × 8 minibatches gives **24 Actor Adam updates per iteration**. Only the four new Actor tensors update; a separate MLP Critic learns. The code uses real Isaac Lab physics and the original martial reward implementation under `reproduction/roundhouse_impl.py`. The source normalizes initial XY and yaw, maps joints explicitly and preserves the action/reference residual contract. Evaluation must report physical survival, body/root tracking and actual reference-dependent motion quality separately.

A checkpoint on a training iteration boundary differs from later unsaved optimizer steps. Resume uses the saved optimizer/Critic state, while physics and RNG initialization restart as recorded by the original loop. Do not describe this as bitwise continuation of the entire simulator.

## BC plus online DAgger

Each task starts independently from the same frozen public base. Persistent teacher states, online replay and original reference-state rehearsal contribute to updates. Teacher execution probabilities decrease to zero. Collection queries the teacher on states reached by the mixed/student policy; final evaluation uses zero teacher action execution. The worker checks cached/full Actor parity, finite gradients, actual optimizer steps and frozen public digests.

The original 31-task list used different native physical environment counts. Preserving them reproduces that protocol; changing all tasks to 512 environments changes sampling and resource use and requires new acceptance. Default maximum is 15,000 actual Adam updates per task, with development stopping and independent final seeds. Do not substitute loop calls for optimizer updates.

The inference bank in `pera.switching` installs 29 saved DAgger checkpoints plus one martial PPO expert in a shared frozen Actor. Thirty explicit switches permit at most one enabled task expert; that expert retains its independent competition against the eight public experts. CPU output parity is verified in `provenance/EXPERT_BANK_CHECK.json`. This does not establish joint all-expert routing, continuous physical scene transitions, or acceptance of every saved skill. Context-free old-motion retention also does not certify every external sensor context. See [expert switches](EXPERT_SWITCHES.md).

## Routing A/B and consolidation

The stair A/B is 512 environments and 9,000 actual Adam updates per branch, with matching initial mean contribution. The fusion comparison has no global Top-K admission statistic. Its initial contribution is matched to the competitive condition rather than to a fictitious selection rate.

The separate consolidation stage makes the complete Actor trainable, keeps normalization/selection buffers fixed, uses mixed old/new teacher supervision and writes full combined checkpoints. It uses a new optimizer and a different freeze contract. It must not be reported as frozen-base expert learning.

Use the actual `--help` for these entry points after installing the native runtime:

```bash
python -m pera reproduce collect-old -- --help
python -m pera reproduce consolidate -- --help
python -m pera reproduce retain-ab -- --help
python -m pera reproduce retain-full -- --help
```

The collection and consolidation scripts require the two A/B checkpoints and their replay/selection sources. For a competitive A/B output, the concrete sequence is:

```bash
python -m pera reproduce collect-old -- --domain runs/stairs_competitive/selected.domain.pt --output runs/old_replay_competitive.pt
python -m pera reproduce consolidate -- --domain runs/stairs_competitive/selected.domain.pt --old-bank runs/old_replay_competitive.pt --output runs/consolidated_competitive
```

Repeat with the fusion checkpoint into separate output paths. The exact `--domain`, `--old-bank` and `--output` flags come from the original loop parser. These are optional paper extension experiments, not prerequisites for frozen expert training.

## Transfer and scientific reporting

The transfer preparation/evaluation loops are retained under `reproduction/experiments/martial_transfer_v1_20261106`. They require separately supplied source retarget manifests, CMU descriptions and selected martial checkpoints. Locked protocols identify the original splits and checkpoint hashes. There is no score-based reassignment of clips during packaging.

A short stair reference ends during ascent and does not cover all fifteen steps. Complete reference playback, physical survival, reference-end proximity, top-step arrival and terminal standing stability are distinct claims. MimicX football/tennis evaluate motion tracking; ATHLETE measures physical return cycles; ULTRA requires hand contact, lift and transport.

The historical HoloMotion teacher/student table and the competitive MLP table use different architectures, routing, seeds, sample sizes and budgets. They support descriptive comparisons, not a controlled causal claim that one base is superior. Proving pretrained-base benefit needs a matched random/weak/strong-base study and multiple training seeds. Freezing old tensors alone cannot guarantee old behavior when new experts are enabled.

## Verification performed for this source package

See `provenance/VALIDATION.json`. CPU contract tests and static checks are run in the existing environment. No full GPU training or teacher/physics success sweep was rerun during packaging; matching the previous paper numbers remains dependent on the separately provided resources and original protocol. Missing public artifact URLs and external teacher setup remain explicit release prerequisites, not hidden fallbacks to the development checkout.

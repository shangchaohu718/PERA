# Switched expert bank

> Current release selection: official HoloMotion G1 v1.4 `model_14000` and the matching plugins in `resources/holomotion_weights_manifest.json`. See [HoloMotion release status](HOLOMOTION_RELEASE_STATUS.md). Self-trained MoE-MLP / v2.4 configurations and weights described below are historical artifacts, not the default release.

`checkpoints/composed/pera_30_experts.pt` contains one frozen public G1 Actor and 30 named task experts: the 29 saved DAgger plugins and the martial PPO plugin. All experts are installed once, with at most one enabled at a time. Loading defaults to all off.

## Selection

```python
from pera.switching import SwitchedActor

actor = SwitchedActor.load_bundle("checkpoints/composed/pera_30_experts.pt", device="cpu")
print(actor.expert_names)  # 30 installed slots
print(actor.switches)  # Snapshot of all 30 booleans

actor.set_expert_enabled("parkour_stairs_up", True)
# Enabling a second expert here raises ValueError and leaves the first enabled.
actor.set_expert_enabled("parkour_stairs_up", False)
actor.set_expert_enabled("martial_ppo", True)

actor.select_expert("velocity_walk")  # Atomically closes the previous expert.
actor.set_switches({"martial_ppo": True})  # Unspecified experts are off.
actor.set_switches({})  # All off: original public path.
```

Switch state applies to the entire Actor batch. Different experts per environment in the same batch are not supported. Invalid names, nonboolean switches and multiple enabled names are rejected before changing state.

An enabled expert enters the same competition as its original independently loaded policy: eight frozen public experts plus the selected task expert, with Top-2 admission and the shared FFN once. An enabled expert can lose admission and contribute zero on a state. Closed experts do not execute their routers or FFNs. Their learned biases and calibrated selection offsets are preserved without an additional merge bias or output gate.

## Inputs

The direct inference interface is:

```python
# history: [B,10,608]; goal: [B,730], using the native G1 policy contract.
# task_context: [B, actor.context_dim], from the selected task's original adapter.
actions_sdk = actor(history, goal, context=task_context)  # [B,29]
```

Use the selected expert's saved `context_contract` and matching task observation builder, encoder weights and normalization. The bank does not convert arbitrary camera pixels into task features. Visual encoders, reference motions, simulator scenes and teacher resources remain in the task-specific resource setup. Required context is never silently replaced with zeros. The context-free martial expert needs no context.

Switching preserves the public model and existing observation history. It invalidates cached task context and latent features; supply fresh context and run a fresh forward after switching. `cached_features(goal)` and `predict_cached(cache)` reject caches from a previous switch generation.

For existing native HoloMotion-style task adapters, use the original reference capture with the same bank:

```python
from experiments.g1_competitive_suite31_v1.model import install_capture

install_capture()  # Once per process, before using the existing task observer.
native_actor = actor.reference_actor()  # Shares actor.pi; no second public network.
actor.select_expert("parkour_stairs_up")
native_actor.context = task_context
# After the task observer captures its raw reference/robot state:
actions_holo = native_actor()["actions"]
```

This optional bridge requires `reproduction` and `integrations/HoloMotion` on `PYTHONPATH` plus the task runtime. It retains the original named-joint mapping, action scaling and reference-state construction. Scene placement, reference progression and deciding when to switch remain the caller's responsibility. Switching weights alone does not certify stable physical transitions.

## Composition and verification

```bash
python -m pera compose --config configs/expert_bank.json \
  --output checkpoints/composed/pera_30_experts.pt
python -m pera experts --checkpoint checkpoints/composed/pera_30_experts.pt
python tools/verify_weights.py
python tools/verify_expert_bank.py --report runs/expert_bank_check.json
```

The review bundle already exists. Composition refuses to overwrite it; use a new output path to rebuild. The configuration checks source SHA256 hashes and the loader checks common public weights, base identity, plugin format, shape and finite tensors. The combined file loads without the separate base/plugin files. It contains inference Actor tensors and provenance, not a world model, Critic, optimizer or external visual encoder.

The saved [CPU check](../provenance/EXPERT_BANK_CHECK.json) compares each of the 30 enabled experts against independent loading, checks cached/full output parity, compares all off against the unmodified public Actor, and verifies unchanged public/plugin tensors. Inputs are synthetic; the FP32 and bf16 AMP settings exercise the CPU path, with native CUDA autocast disabled on CPU. This is not GPU or physical skill validation.

Thirty installed checkpoints do not mean thirty accepted skills. Historical acceptance is retained in the [model list](MODELS.md) and [weight manifest](../resources/weights_manifest.json), including the failed ATHLETE tennis checkpoint. ULTRA has no trained student checkpoint in this release and is not a slot. The continuous-fusion ablations and full-Actor consolidation exports are separate experiments and are not installed in this bank.

# Selecting a HoloMotion expert

The current release uses the official HoloMotion G1 v1.4 `model_14000` Actor with one bank containing 25 matching task experts. Each inference call selects zero or one domain. Other domain FFNs do not run.

## Inspect the bank

From the checkout, after obtaining the paired scene resources:

```bash
PYTHONPATH=src python -m pera experts
```

This lists expert IDs and frozen context contracts. It does not start a simulator or evaluate a skill.

## Direct inference

The native HoloMotion observation dictionary and task context must come from the selected scene's observation adapter. Arbitrary camera images or zero-filled context are not compatible inputs.

```python
from pera.holomotion_bank import FrozenHoloMotionBank

bank = FrozenHoloMotionBank.load(
    "checkpoints/holomotion_experts/pera_25skills.experts.pt",
    device="cuda:0",
    base_dir="/path/to/holomotion-v1.4",
)
print(bank.expert_names)
# observations and context are produced by the matching task adapter.
result = bank(observations, expert="mimicx_stairs", context=context)
public_result = bank(observations, expert=None)
```

The bank resets domain selection and context after each call. Unknown expert IDs, multiple IDs, missing context and incompatible context dimensions are rejected. Passing `expert=None` uses the unchanged public path. Native routing remains continuous anchor-group blending with public Top-2; it is not global Top-2 competition among all 25 domains.

For multiple robots, use the existing scene controller in `src/pera/multi_robot.py` and `src/pera/holomotion_showcase.py`: it shares Actor weights and swaps independent observation/Attention state for each robot. A direct call on one Actor does not create independent robot memory automatically. Keep calls sequential; selection is not a thread-local setting.

See [scene replay](SCENE_REPRODUCTION.md) for executable examples and [models](MODELS.md) for local single-checkpoint materialization. The former self-trained 30-expert bank belongs to [historical research](HISTORY.md); its format and switches are incompatible with this release.

# Data and checkpoint resources

> Current release selection: official HoloMotion G1 v1.4 `model_14000` and the matching plugins in `resources/holomotion_weights_manifest.json`. See [HoloMotion release status](../HOLOMOTION_RELEASE_STATUS.md). Self-trained MoE-MLP / v2.4 configurations and weights described below are historical artifacts, not the default release.

This is a source release. Large binary assets are intentionally separate and ignored by Git. `resources/manifest.json` records expected destinations, SHA256 hashes, byte counts and relative source identities. `resources/upstreams.json` pins the Git source repositories. `resources/mimicx_release.json` pins the paired MimicX asset/policy revisions and release indexes.

## Maintainer resource import

If you have the original research artifact tree, preview a verified local import:

```bash
python tools/import_resources.py --source-root /path/to/original/artifacts --group base
# Explicitly copy matching resources; no downloads or environment installation.
python tools/import_resources.py --source-root /path/to/original/artifacts --group base --apply
python tools/import_resources.py --source-root /path/to/original/artifacts --group ppo --apply
```

The source-root path is supplied by the caller. No developer home directory is needed. Different existing destination bytes are refused. Imports go to ignored data/checkpoint/mesh paths, not into tracked source.

## Base policy

- Paper-selected base: `checkpoints/base/best.pt`, original iteration 11,000, SHA256 `fc5df7ad6ec83cce71853a30314e6c4c0109baab144308084279ded0679cb2eb`.
- Optional continuation parent: `checkpoints/base64/best.pt`, original iteration 8,626.
- Motions: `data/base/train/` and `data/base/heldout/`; exact identities and split groups are in `configs/data/train128.json` and `heldout16.json`.
- Calibration bank: `data/base/reference_bank.pt`; rebuild from the 128 training clips with `PYTHONPATH=src python tools/build_reference_bank.py`. It is not fitted on holdout motions.
- Robot: supplied MJCF/URDF text under `assets/g1/`; place matching meshes in `assets/g1/meshes/`. The MuJoCo source model preserves the original actuator settings, disabled self-collision and unenforced joint-speed limits.

The 128-motion data combines LaFAN1, 100STYLE, CMU, AIST++ and BONUS-SEED retargets. The package specifies the selected retargeted segments but does not distribute those arrays or override dataset terms. Some provenance-only source strings use `${SOURCE_ROOT}`; training reads each row's portable `file` field, not those historical provenance strings.

The new PERA base and private retarget exports do not yet have public release URLs in this package. A community numerical release needs these separately hosted permitted artifacts, matching hashes, and dataset acquisition/retarget instructions. Do not present the code-only checkout as already containing them.

## Teachers

| Family | Required source/resources |
|---|---|
| Instinct and parkour | Pinned InstinctMJ, instinct_rl and mjlab; released matching checkpoints, references and scene meshes |
| Velocity walking | Original Unitree velocity ONNX policy and deployment metadata |
| HUSKY | `third_party/HUSKY`; original board scene and released policy; CC BY-NC 4.0 terms |
| MimicX | Paired `2026-10-14-demonstrations` asset/policy snapshots from `resources/mimicx_release.json` |
| PMT | Pinned PMT, optimized reference/terrain, and locally adapted teacher checkpoint for the canonical G1 motors |
| ATHLETE | Original ATHLETE source, tennis assets, ONNX policy and frozen context normalization |
| ULTRA | Original ULTRA runtime, released teacher, physical box meshes and original motion |

The retained adapters preserve task-specific physical success definitions. The original teacher source is external, not replaced by a mock policy. Official HoloMotion v1.4 Actor/Critic weights are not bundled. Download them directly from the official model repository using [the download guide](../HOLOMOTION_DOWNLOAD.md), placing them under `checkpoints/holomotion_v1.4/`. Twelve matching task plugins are staged locally under `checkpoints/holomotion_experts/`; see `resources/holomotion_weights_manifest.json`. No upload was performed. The historical self-trained resource manifest below does not cover this official-base selection.

For Git sources, preview with `python tools/fetch_upstreams.py --plan`; fetch explicitly with `--fetch --only mjlab InstinctMJ instinct_rl PMT`. For the pinned MimicX files, run `PYTHONPATH=src python tools/fetch_mimicx_release.py` only when downloads are intended; this tool performs network acquisition. Nothing fetches weights while importing PERA or printing CLI help.

The PMT teacher was locally adapted; treating it as an untouched official release would be incorrect. Likewise, source manifests do not imply that every listed trained student passed its skill and retention acceptance.

## Optional fixed-checkpoint transfer study

`transfer-prepare -- --help` lists the source-manifest and CMU metadata inputs. This historical extension additionally needs the curriculum retarget shards and the selected initialized/trained martial checkpoints. Those artifacts are not covered by the base/teacher import groups. Supply their permitted exports separately; this package does not claim an automatic download route for them. Shard paths may be absolute caller-supplied paths or paths relative to that manifest. Generated motion arrays are written to ignored `data/transfer/`.

## Locally prepared model files

The local review folder now includes the base and existing task/model checkpoints. See [model contents and historical evaluation status](MODELS_20261113.md) and `resources/weights_manifest.json`. These files are ignored by Git and have not been uploaded. Verify them with `python tools/verify_weights.py`.

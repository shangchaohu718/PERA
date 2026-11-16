# Replaying the HoloMotion scenes

The scene release runs actual student control in MuJoCo. It contains PERA source,
selected upstream runtime source with original licenses, one bank containing 25 matching task experts,
frozen context encoders, robot geometry, initial states and scene references.
The default seven-region demonstration uses box carrying, obstacle vault,
gap crossing, tennis, stair descent, stair ascent and skateboarding.

Official HoloMotion weights are downloaded from the publisher, not bundled.
Use the exact v1.4 `model_14000` Actor, configuration and Critic. A different
checkpoint or the historical self-trained 30-expert bundle is incompatible.

## Installation

Supported replay target: Linux x86_64, Python 3.11, an NVIDIA GPU compatible with
PyTorch CUDA 12.8, an EGL-capable driver, and `ffmpeg` on PATH. The checked machine
uses an RTX 5080. The seven-region replay allocated about 1.6 GiB with the single expert bank (about 3.2 GiB with the legacy Actor/Critic loader) of PyTorch GPU
memory; driver and rendering allocations are additional. Real-time speed is not
required: the output video plays at simulation speed.

From the extracted scene release, in a Python 3.11 environment:

```bash
python -m pip install -r requirements/scene-replay.txt -c requirements/scene-replay.lock.txt
```

The lock pins MuJoCo 3.11.0 and Warp 1.14.0. Teacher action networks and rl_games are not needed for this student replay.
Do not replace it with a historical training-environment requirements file.
The launcher uses checkout-local source directly; editable installation is optional.

## Official base

Choose a model directory outside the PERA checkout. Inspect required files first:

```bash
python tools/fetch_official_base.py --output /path/to/holomotion-v1.4
```

This only verifies existing files and prints missing files (exit code 2).
To obtain missing files directly from the pinned official Hugging Face URLs:

```bash
python tools/fetch_official_base.py --output /path/to/holomotion-v1.4 --download
```

Already verified files are reused. Existing files with unexpected hashes are not
overwritten. See [the upstream download and license guide](HOLOMOTION_DOWNLOAD.md).

## Obtain the private release

Invited users with an authenticated GitHub CLI can download the paired source and resource archive:

```bash
gh release download holomotion-20261116 --repo shangchaohu718/PERA \
  --pattern 'PERA-HoloMotion-25experts-15scenes.tar.gz' --pattern 'SHA256SUMS'
sha256sum --check SHA256SUMS
tar -xzf PERA-HoloMotion-25experts-15scenes.tar.gz
cd PERA
```

The archive contains the runnable source and assets; a Git clone alone excludes the large resource files. Check the archive manifest before running:

```bash
python tools/verify_scene_files.py
python tools/check_holomotion_release.py --base-dir /path/to/holomotion-v1.4
bash scripts/replay_all_scenes.sh --base-dir /path/to/holomotion-v1.4 \
  --device cuda:0 --video --output runs/scenes15
```

The command runs three scene groups sequentially, using the **same expert-bank file**. All 25 FFNs/routers are resident in one Actor in each run. The union contains 15 distinct tasks, listed in [the skill inventory](HOLOMOTION_SKILLS.zh-CN.md). Group durations are 12, 24 and 12 simulated seconds, including a full reference interval for the longer kung-fu routine. Explicit per-task repeats are recorded. This is not a claim of 15 continuous transitions or 15 robots at once.

For just the default seven-region group:

```bash
bash scripts/replay_scenes.sh --bank checkpoints/holomotion_experts/pera_25skills.experts.pt \
  --base-dir /path/to/holomotion-v1.4 --device cuda:0 \
  --seconds 12 --repeat --video --check-isolation --output runs/showcase
```

Use a new output directory for each run. The launcher does not train or contact a
teacher action network. Every robot uses one shared frozen official Actor with
its matching FFN/router and independent temporal memory. The native context
normalization, action mapping and checkpoint merge bias are retained. The
selected checkpoints use continuous anchor-group mixing with native public Top-2.

Outputs are `overview.mp4`, `preview.png`, `world.xml` and `report.json`. The report
records model and plugin identities, source files, dependency versions, physical
control counts, measured task traces, labeled resets and cache-isolation results.
It rejects project modules imported from outside the release checkout. The model
directory may be external; the original research source tree is not required.

## What is verified

The seven-region 12-second replay has been rerun using checkout-local runtime
source and external official models. All seven recorded task traces exactly
matched the previous demonstration on the same machine and environment. This
checks the scene port and packaging, rather than a new statistical skill claim.

The full 15-task suite also passed in a fresh isolated environment, using an archive extracted into an independent directory. The seven original traces matched exactly. The additional tasks completed their checked intervals without tracking loss; two noisy-contact task traces differ slightly from the earlier local probe, as recorded in the verification report. Package binaries were reused locally rather than downloaded again; see [the installation record](../provenance/ISOLATED_ENVIRONMENT.json).

Artifact delivery is recorded separately in [the canonical repository](https://github.com/shangchaohu718/PERA/blob/main/provenance/RELEASE_DELIVERY.json). Training reproduction remains unverified.

The gallery videos are earlier single-task recordings; the current shared-world
release has explicit scene-port physics and labeled episode loops. It does not
claim identical camera pixels or identical native single-task simulator physics.
It does not reproduce training from scratch, paper-wide success rates, continuous
transitions between regions, or acceptance of every historical expert.

## Distribution

The local scene archive includes the required non-base assets even though large
files are excluded from ordinary Git tracking. A source-only checkout is
insufficient until its paired scene assets have been distributed. Do not substitute
videos for those files. The intended release archive can be built with:

```bash
python tools/build_scene_release.py --output /path/to/PERA-scenes.tar.gz
```

The builder verifies resource hashes, includes a complete file manifest and
excludes official HoloMotion base weights and historical base checkpoints.
Models, motions and third-party assets retain their upstream terms.

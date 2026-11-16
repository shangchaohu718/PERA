# Current HoloMotion scene release

Use [SCENE_REPRODUCTION.md](SCENE_REPRODUCTION.md) for the current standalone source/asset archive and external official models. The original native-source workflow below is historical.

# Shared-world demonstration

Seven robots share one MuJoCo world and one official HoloMotion G1 v1.4
`model_14000` Actor. The compact layout includes physical box carrying, wall vaulting, gap crossing,
stair descent, stair ascent, skateboarding and tennis returns. Each robot enables one matching trained FFN/router
expert. All weights remain frozen; teacher action heads are not queried.

## Run

Use an existing compatible native interpreter. Local validation used Torch
2.9.1, MuJoCo 3.11 and the pinned upstream task sources.

```bash
python -m pera demo --native-source-root /path/to/native/artifacts \
  --device cuda:0 --repeat --seconds 12 --video --check-isolation \
  --output runs/showcase
```

The native root contains `checkpoints/holomotion_v1.4`, the expert files pinned
in `configs/demo_policy.json`, and the pinned upstream sources. Prepare additional resources from existing local artifacts:

```bash
python tools/prepare_compact_showcase.py --source-root /path/to/native/artifacts --apply
python tools/prepare_object_showcase.py --source-root /path/to/native/artifacts --stage box --apply
python tools/prepare_object_showcase.py --source-root /path/to/native/artifacts --stage vault_wall --device cpu --apply
python tools/prepare_object_showcase.py --source-root /path/to/native/artifacts --stage gaps --device cpu --apply
```

The exporters use the existing native environment and assets. CPU export is
supported. Resource hashes are recorded in the manifests listed by
`configs/showcase.json`. The default recording uses a closer overhead overview without
project/method banners or a status footer. Scene names remain visible, and
brief Reset labels identify episode loops. Nothing is downloaded or trained. Use `--viewer` for an interactive window.
Recording requires local `ffmpeg`; choose a new output directory. CPU inference
is supported with `--device cpu`; a compatible local OpenGL backend is still
required for rendering.

Outputs include `overview.mp4`, `preview.png`, a world XML inspection snapshot
and `report.json`. Recreate the world from its config/resources; the XML alone
is not a standalone asset pack. The overview and three rotating closeups show
the same live physical world. The depth-information panel is hidden in this layout. Experts that require
depth still receive their original visual inputs. Camera position is configurable in
`configs/showcase.json`. The previous nine-region layout is `configs/showcase_nine.json`; the four-region layout is preserved in
`configs/showcase_four.json`.

## Models and inputs

`configs/demo_policy.json` pins the official Actor/config hashes, public and
interface digests, expert file hashes and frozen context contracts. Original
checkpoint routing and merge biases are preserved.

| Region | Expert | Additional input |
|---|---|---|
| Box carrying | ULTRA largebox | Native normalized 4052D observation; no camera pixels |
| Wall vault | Instinct roll-vault-wall | Frozen 32D encoding of live local depth |
| Gap crossing | Instinct Parkour gaps | Frozen 128D depth encoding plus 5D command/clock |
| Stair descent | MimicX descent | Original descent reference |
| Stair ascent | MimicX stairs | Original ascent reference |
| Skateboard | HUSKY skateboard | Frozen 128D proprioceptive encoding |
| Tennis returns | ATHLETE tennis | Native normalized 631D observable ball/robot history |

Vaulting and gap crossing use camera pixels. Display cameras are
separate from Actor inputs. Encoder and normalization hashes are checked against
the saved expert context contracts. Parkour preserves native depth crop, blur,
normalization, history and command conventions. It receives no teacher action or
future trajectory.

One official Actor holds all the selected experts. Dispatch selects one expert and
restores that robot's attention K/V memory, lengths, write indices, rotary
positions and previous routing state. Other experts are disabled for that call.
`--check-isolation` compares eight input steps in single-robot, interleaved and
reverse-order dispatch using the actual weights. It does not advance physics.
The self-trained competitive showcase remains archived and is refused by the
demo launcher.

## Physics and resets

There is one physics `MjData`, advanced by `mj_step`. Seven native auxiliary
models/data objects mirror its state for FK, sensors and tennis planner
bookkeeping; they never advance robot physics. The common physical step is the native ULTRA rate of 1/1020 seconds, with its
Euler/PGS solver and 50 solver iterations. Box control runs exactly every 17
physical steps (60Hz). The other robots request 50Hz control, quantized to the
shared physical clock with less than 1ms timing error. Per-robot control counts
and actual physics steps are recorded separately. Native masses, inertias, drive gains, armature and
joint conventions are retained. Most infinite planes become finite floor patches. Box carrying retains its
native physical plane; a noncolliding finite patch supplies its display.
Separate collision masks preserve each region's internal contacts and prevent
unintended contacts between regions. This demonstration does not test
cross-region physical interactions.

The gap corridor retains native terrain elevations; only unused surrounding
tiles are removed. Each visual sensor casts rays into its own native geometry
mirror at the actual shared-world robot pose. This prevents another region's
infinite plane from filling the gaps in its depth input. Geometry mirrors do
not advance physics. Display culling uses compiled mesh centers, retaining the
actual vault obstacle while hiding tall background walls. Tennis court decoration is cropped; the net collider and
ball rules remain unchanged. Ball relaunch changes only object state.

With `--repeat`, reference-based robots reset at their own reference boundaries.
The compact skateboard track loops every 3.6 seconds; gap crossing has a
10-second maximum interval. Gap crossing repeats after reaching within 0.4m, moving at least
2m, remaining upright above 0.65m pelvis height and dwelling for 1 second.
The earlier unlooped replay lost stability after reaching the goal; native
10-second survival is not established by this exhibition. Each reset clears only
that robot's temporal state and is labeled
and counted. Tennis continues while relaunching balls. This is a multi-robot
skill exhibition with episode loops.

## Local evidence

`provenance/OBJECT_OBSTACLE_SHOWCASE_CHECK.json` records the object/obstacle replay,
weight identities, isolation checks, native input/terrain parity and actual
measurements. The earlier nine-region replay is in
`provenance/COMPACT_HOLOMOTION_SHOWCASE_CHECK.json`; the four-region replay is in
`provenance/OFFICIAL_HOLOMOTION_SHOWCASE_CHECK.json`. Height/tilt warnings and
reference endpoints are diagnostics, not native acceptance gates or a
statistical success-rate estimate. Historical failed runs are retained.

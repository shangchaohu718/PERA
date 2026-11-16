# Resources

The canonical source and resource release is [shangchaohu718/PERA](https://github.com/shangchaohu718/PERA). It remains private; invited users need authenticated access.

## Scene release

The paired `PERA-HoloMotion-25experts-15scenes.tar.gz` archive supplies:

- PERA inference and scene-replay source;
- the single 25-expert bank;
- references, initial states, matching frozen context encoders and normalization;
- robot, terrain, object and court geometry for the 15-task suite;
- pinned, licensed runtime subsets of ATHLETE, ULTRA, InstinctMJ, instinct_rl and mjlab;
- a per-file SHA256 manifest.

Follow [SCENE_REPRODUCTION.md](SCENE_REPRODUCTION.md) to obtain and verify the archive. Large resources are excluded from ordinary Git tracking. A clone alone is insufficient for simulation.

The archive does **not** contain official HoloMotion Actor/Critic weights. Obtain the fixed version from the [publisher](HOLOMOTION_DOWNLOAD.md). It also excludes full teacher-policy bundles, the historical 128-motion training dataset, environments, CUDA installers and private training logs.

## Identities

| Manifest | Purpose |
|---|---|
| `resources/holomotion_base.json` | Pinned upstream base URLs, revision and hashes |
| `resources/holomotion_bank25.json` | Combined expert-bank identity |
| `configs/holomotion_experts25.json` | Per-expert source, contracts, hashes and historical evaluation notes |
| `configs/reproduction_suite15.json` | Three replay groups and durations |
| `resources/*showcase_manifest.json` | Exact selected scene files |
| `resources/runtime_sources.json` | Vendored runtime files, upstream revisions and licenses |
| `SCENE_RELEASE_MANIFEST.json` | Generated archive-wide file hashes |

The 15 scenes run as three independent groups. All 25 expert networks are installed in each run, but this release does not claim that all 25 historical native tasks have been packaged or statistically re-evaluated.

## Terms

PERA source contributions use the root MIT license. Third-party code, geometry, motions, frozen encoders and learned models retain their applicable upstream terms; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). HUSKY resources include noncommercial restrictions. This package is not a blanket MIT license for all assets.

[Archived resource notes](history/RESOURCES_20261113.md) describe prior self-trained and training-data experiments; they are not required for the current student replay.

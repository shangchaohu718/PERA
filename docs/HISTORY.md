# Historical research

The release now exposes only the official HoloMotion scene path and expert inspection through the main CLI. Self-trained suite31 training, martial PPO, stair routing ablations, full-Actor fine-tuning, transfer scripts and their resource manifests are not current scene dependencies.

Their original source and dated documentation remain available in [the preserved pre-cleanup Git revision](https://github.com/shangchaohu718/PERA/tree/b7bdc1ab98a9af433a554140ea10124a19b5a27a). A separate local backup was retained during cleanup. Original research checkpoints and experiment results were not changed.

The `docs/history/` and `provenance/history/` directories contain dated snapshots, not the latest release state. They are excluded from the inference archive. Current state is in [the scene release status](HOLOMOTION_RELEASE_STATUS.md) and [the reproduction record](../provenance/SCENE_REPRODUCTION_CHECK.json).

Small legacy model/observation compatibility modules remain where retained adapters or CPU tests import them. This does not enable the self-trained base in a demonstration; the official model policy guard remains mandatory.

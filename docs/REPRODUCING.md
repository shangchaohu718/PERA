# Reproduction scope

The supported path is [the fifteen-scene HoloMotion student replay](SCENE_REPRODUCTION.md). It uses the official v1.4 `model_14000` Actor, the released 25-expert bank and the paired scene resources. Official base weights are downloaded separately from the publisher.

## Checks

```bash
python tools/verify_scene_files.py
python tools/check_holomotion_release.py --base-dir /path/to/holomotion-v1.4
bash scripts/replay_all_scenes.sh --base-dir /path/to/holomotion-v1.4 \
  --device cuda:0 --video --output runs/scenes15
```

The first command applies to an extracted scene archive; it verifies the generated archive manifest. The second checks resource identities. The final command performs real student control and produces three videos and measured reports, covering 15 distinct tasks. It uses no teacher action network.

## Boundaries

The recorded clean-directory and isolated-environment replay is described in [the verification record](../provenance/SCENE_REPRODUCTION_CHECK.json). It establishes the packaged scene path, not training from scratch, paper-wide success rates, uninterrupted transitions between tasks or physical acceptance of every historical expert.

Teacher/student training adapters require separate full upstream sources, teacher models, data and simulator environments. They are not complete turnkey training recipes; see [training prerequisites](TRAINING.md). Historical self-trained PPO, suite31 and full-Actor fine-tuning commands have been removed from the public CLI. Their source and dated results are preserved in [the historical revision](HISTORY.md).

The CPU tests check routing, gradients, switches, input validation and freezing. Some are explicitly legacy-format compatibility tests. Test coverage is not a substitute for a closed-loop physical rollout or a task success study.

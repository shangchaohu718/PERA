# Scene demonstration

Use [the scene reproduction instructions](SCENE_REPRODUCTION.md) for installation, scene assets and the external official HoloMotion base.

## Record all fifteen scenes

```bash
bash scripts/replay_all_scenes.sh --base-dir /path/to/holomotion-v1.4 \
  --device cuda:0 --video --output runs/scenes15
```

The three groups run sequentially. `--video` records offscreen; it does not open a window. Each group writes `overview.mp4`, `preview.png`, `report.json` and `world.xml`. The parent directory has group logs and an aggregate report. Choose a new output directory each time.

## Interactive seven-scene window

On a graphical Linux desktop:

```bash
MUJOCO_GL=glfw bash scripts/replay_scenes.sh \
  --base-dir /path/to/holomotion-v1.4 --device cuda:0 \
  --viewer --repeat --seconds 30 --output runs/viewer
```

Do not start another replay merely because the recording command does not show a window. Check its process and log first.

Each group shares one frozen Actor with all 25 experts installed, independent robot temporal caches and at most one task domain selected per call. The demonstrations use student actions. Labeled task loops may reset individual episodes; this is not a continuous fifteen-task route.

The native-source preparation tools are migration utilities for maintainers with the original artifacts. They are not required to run the paired scene archive.

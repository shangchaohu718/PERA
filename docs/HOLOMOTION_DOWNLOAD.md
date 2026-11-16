# Official HoloMotion base download

Official HoloMotion Actor/Critic weights are **not included or mirrored** in
this release. Obtain them directly from
[HorizonRobotics/HoloMotion_models](https://huggingface.co/HorizonRobotics/HoloMotion_models).

Use **HoloMotion_motion_tracking_model_v1.4.0**, checkpoint **model_14000**.
The validated model repository revision is
`e3c90dd1f7483d02d3b4d2e5fb3a359c6ff9c67a`.

## Required files

Download these three files from the pinned official version:

- [config.yaml](https://huggingface.co/HorizonRobotics/HoloMotion_models/resolve/e3c90dd1f7483d02d3b4d2e5fb3a359c6ff9c67a/HoloMotion_motion_tracking_model_v1.4.0/config.yaml)
- [model_14000/actor/model.safetensors](https://huggingface.co/HorizonRobotics/HoloMotion_models/resolve/e3c90dd1f7483d02d3b4d2e5fb3a359c6ff9c67a/HoloMotion_motion_tracking_model_v1.4.0/model_14000/actor/model.safetensors)
- [model_14000/critic/model.safetensors](https://huggingface.co/HorizonRobotics/HoloMotion_models/resolve/e3c90dd1f7483d02d3b4d2e5fb3a359c6ff9c67a/HoloMotion_motion_tracking_model_v1.4.0/model_14000/critic/model.safetensors)

Place them in a model directory **outside** the PERA source/release folder:

```text
/path/to/holomotion-v1.4/
├── config.yaml
└── model_14000/
    ├── actor/model.safetensors
    └── critic/model.safetensors
```

Then verify the base and locally supplied plugins:

```bash
python tools/check_holomotion_release.py --base-dir /path/to/holomotion-v1.4
```

The Actor file is 1,636,251,760 bytes and the Critic is 59,001,428 bytes.
Exact SHA256 values, destinations and immutable upstream URLs are also in
[holomotion_base.json](../resources/holomotion_base.json). Keep this version:
a newer checkpoint or an ONNX export does not replace these files for the
current strict loader. Verification reports missing files until the user has
installed the official models and the required task plugins.

## License and attribution

The [source repository](https://github.com/HorizonRobotics/HoloMotion/blob/master/LICENSE)
uses Apache-2.0. The
[official model card](https://huggingface.co/HorizonRobotics/HoloMotion_models)
labels the model CC BY-NC-SA 4.0. Source-code permission and model permission
must be considered separately. Our packaging choice is to reference official
model downloads, not redistribute those weights. The root MIT notice does not
relicense pretrained models, learned plugins, frozen encoders or motion data;
retain their applicable source and training-data terms.

The separately supplied task scenes/runtime and clean-checkout training status
remain as documented in [the release audit](HOLOMOTION_RELEASE_STATUS.md).

The scene launcher accepts this external directory using `--base-dir`. See [scene replay](SCENE_REPRODUCTION.md) for a verify-first downloader and the complete run command.

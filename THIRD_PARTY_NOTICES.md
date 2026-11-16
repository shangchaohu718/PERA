# Third-party notices

The adapted runtime in `src/blinkmimic/` is based on [BlinkMimic](https://github.com/fuyichen0725/blinkmimic).

Portions of `src/blinkmimic/` and `tools/` come from [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)
(Chong Zhang) and [InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack) (Huy-Quang-Dao), used under their MIT
Licenses, reproduced below.

## InstantTrack

```
MIT License

Copyright (c) 2026 Huy-Quang-Dao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## supermimic_vibed

```
MIT License

Copyright (c) 2026 ETH Zurich, Chong Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Dependencies and data

The Python dependencies (mjlab, MuJoCo, MuJoCo Warp, NVIDIA Warp, PyTorch, matplotlib, Pillow) are installed from
their own distributions under their own licenses (Apache-2.0, BSD-3-Clause and others) and are not redistributed here.
The historical LAFAN1 motion-array dataset (Ubisoft La Forge, CC BY-NC-ND 4.0) is not bundled. Files derived from LAFAN1 (videos, GIFs, checkpoints) are not covered by this repository's MIT License. The selected Unitree G1 model descriptions and ATHLETE deployment meshes are included for scene replay; retain their BSD-3-Clause notice in `licenses/Unitree.txt` and the accompanying source notices. Other upstream assets remain separately supplied as described by the scene resource manifests.

## HoloMotion and robot integration

`integrations/HoloMotion/` contains only the selected HoloMotion runtime sources needed by the retained workflows. Original source version: HorizonRobotics/HoloMotion commit `cf2018a282249a04df1db4fa639760884f97b41b`; local modifications and selected-file hashes are tracked in `provenance/`. Preserve `licenses/HoloMotion.txt`. Robot model text retains `licenses/Unitree.txt`. Large upstream assets are supplied separately.

## Optional HUSKY bridge

`src/modumotion_g1/husky_skateboarding.py` derives observation/control conventions and scene adaptation from TeleHuman/humanoid_skateboarding. This portion retains CC BY-NC 4.0 terms; see `licenses/HUSKY-CC-BY-NC-4.0.md`. It is not covered by an unrestricted commercial-use grant from the root MIT notice.

## External teachers

InstinctMJ, instinct_rl, mjlab, PMT, MimicX, ATHLETE and ULTRA identify the upstream source/data/model projects. Their names identify provenance and compatibility, not an endorsement or transfer of their licenses. Full teacher-policy bundles and original training datasets are not included. The selected runtime code, frozen context encoders and scene references included for inference are described below and in the scene manifests. Pinned Git revisions and paired MimicX release revisions are in `resources/`.

## Official HoloMotion model downloads

The HoloMotion model card labels its pretrained models CC BY-NC-SA 4.0,
separately from the Apache-2.0 source license. This release does not bundle or
mirror official Actor/Critic weights. Users obtain the pinned v1.4.0 models
from the original publisher using `docs/HOLOMOTION_DOWNLOAD.md`. Task plugins,
encoders and motion assets retain applicable source and data terms; the root
MIT license does not grant additional rights over them.

## Bundled scene runtime subset

`vendor/` supplies only the runtime source and ATHLETE deployment geometry used
by the HoloMotion scene replay. Original notices are retained beside each subset:

| Source | Snapshot and license |
|---|---|
| ATHLETE | [MIT notice](vendor/ATHLETE/LICENSE.md) |
| ULTRA | [Apache-2.0](vendor/ULTRA/LICENSE), [InterMimic notice](vendor/ULTRA/LICENSE-InterMimic) |
| InstinctMJ | [CC BY-NC 4.0](vendor/InstinctMJ/LICENSE) |
| mjlab | [license](vendor/mjlab/LICENSE) |
| instinct_rl | [CC BY-NC 4.0](vendor/instinct_rl/LICENSE) |

[Runtime source identities](resources/runtime_sources.json) pin the exact selected
file bytes, upstream URLs and revisions. These are runtime subsets, not complete
upstream repositories or training distributions. Scene archives also contain
matching plugins, frozen encoders and scene references; they retain their upstream
model and data terms. The official HoloMotion Actor/Critic weights remain external.
The earlier external-teacher note describes the original source-only package.

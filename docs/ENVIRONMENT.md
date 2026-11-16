# Environment

The HoloMotion scene replay targets Linux x86_64, Python 3.11, an NVIDIA GPU compatible with CUDA 12.8 and an EGL-capable driver. Install `ffmpeg` separately.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/scene-replay.txt -c requirements/scene-replay.lock.txt
```

`scene-replay.txt` declares the replay dependencies and official PyTorch CUDA wheel index. `scene-replay.lock.txt` pins the transitive versions used for isolated validation. Run from the source checkout; the launcher sets its local module path and does not require editable installation.

Core versions: Torch 2.9.1+cu128, torchvision 0.24.1+cu128, NumPy 1.26.0, MuJoCo/MuJoCo-Warp 3.11.0, Warp 1.14.0 and ONNX Runtime GPU 1.23.2. ONNX feature sessions explicitly use the CPU provider. Teacher action heads are not required for student replay.

The release validation uses a fresh isolated environment with dependency resolution and compatibility checks. Large CUDA/Torch binaries are repacked from locally installed distributions after verifying their recorded file hashes; small missing dependencies are obtained separately. This avoids repeated CUDA downloads. It verifies isolated installation and runtime closure, not a fresh download of every upstream wheel. Versions and actual replay results are recorded in [SCENE_REPRODUCTION_CHECK.json](../provenance/SCENE_REPRODUCTION_CHECK.json).

Use `PERA_PYTHON=/path/to/python` before either shell launcher to select an existing interpreter. Do not add another environment's `site-packages` to `PYTHONPATH`.

The single-bank scene runs allocate about 1.6 GiB of PyTorch GPU memory on the checked RTX 5080; driver, rendering and other-process allocations are additional. Simulation can run slower than real time while producing original-speed video.

See [the complete reproduction instructions](SCENE_REPRODUCTION.md). The [archived environment notes](history/ENVIRONMENT_20261113.md) concern earlier training profiles; they are not interchangeable with this replay environment. From-scratch teacher/student training reproduction is outside the current acceptance scope.

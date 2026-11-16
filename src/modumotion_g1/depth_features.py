import hashlib
from pathlib import Path

"""Only the released frozen depth feature bridge; no world-model runner."""
from torch import nn
from instinct_rl.utils.utils import get_obs_slice


class FrozenDepthFeatures(nn.Module):
    """Released depth encoder and its exact frozen input normalization."""

    def __init__(self, native_actor, normalizer, obs_format):
        super().__init__()
        section, self.depth_shape = get_obs_slice(obs_format, "depth_image")
        self.encoder = native_actor.encoders._parallel_blocks["depth_image"]
        self.register_buffer("mean", normalizer._mean[:, section].clone())
        self.register_buffer("std", normalizer._std[:, section].clone())
        self.eps = normalizer.eps
        self.requires_grad_(False)
        self.eval()

    def forward(self, depth):
        flat = depth.reshape(depth.shape[0], -1)
        normalized = (flat - self.mean) / (self.std + self.eps)
        return self.encoder(normalized.reshape(-1, *self.depth_shape))


def tensor_module_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optimizer_step(optimizer):
    steps = {int(v["step"].item()) for v in optimizer.state.values() if "step" in v}
    if len(steps) > 1:
        raise RuntimeError("Optimizer parameters have inconsistent step counts")
    return next(iter(steps), 0)

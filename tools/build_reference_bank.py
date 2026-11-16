import json
from types import SimpleNamespace
import numpy as np, torch
from blinkmimic.moe_policy import reference_router_features
from blinkmimic.rotations import quat_to_mat, so3_log
from pera.paths import ROOT as R

E = R / "resources/base"
(R / "data/base").mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
mapping = torch.tensor(
    json.loads((R / "configs/data/train128.json").read_text())["holomotion_from_sdk"]
)
banks = []
labels = []
with torch.no_grad():
    for path in sorted((R / "data/base/train").glob("*.npz")):
        with np.load(path) as d:
            root, quat, q = [
                torch.tensor(d[n]).float() for n in ["root_pos", "root_quat_wxyz", "joint_pos"]
            ]
        mat = quat_to_mat(quat)
        v = torch.gradient(root, spacing=0.02, dim=0)[0]
        w = torch.zeros_like(root)
        w[1:-1] = so3_log(mat[2:] @ mat[:-2].transpose(-1, -2)) / 0.04
        w[0] = w[1]
        w[-1] = w[-2]
        motion = SimpleNamespace(
            root_p=root,
            root_R=mat,
            q=q,
            root_v_local=(mat.transpose(-1, -2) @ v[..., None])[..., 0],
            root_w_local=(mat.transpose(-1, -2) @ w[..., None])[..., 0],
            future_frames=lambda f, k: (f[:, None] + torch.arange(1, k + 1)).clamp(max=len(q) - 1),
        )
        f = torch.linspace(0, len(q) - 1, 256).long()
        state = (root[f], mat[f], v[f], w[f], q[f], q[f] * 0)
        banks.append(reference_router_features(motion, state, f, mapping))
        labels.extend([path.stem] * len(f))
    bank = torch.cat(banks)
    assert bank.shape == (32768, 509) and bank.isfinite().all()
    torch.save(dict(features=bank, clip_names=labels), R / "data/base/reference_bank.pt")
    print(bank.shape)

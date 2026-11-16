"""Original released MimicX environments, existing local runtime only."""

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from pera.paths import ROOT

BASE = ROOT / "third_party/mimicx_release"
ASSETS = BASE / "MimicX-Assets"
POLICIES = BASE / "MimicX-Policies"
RELEASE = "releases/2026-10-14-demonstrations"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".temporary.json")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def release_runner():
    path = ASSETS / RELEASE / "run_policy.py"
    spec = importlib.util.spec_from_file_location("released_mimicx_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(task):
    return release_runner().prepare(POLICIES, ASSETS, task)


def setup(task, seed, num_envs=512):
    # These are the same task registration and reset-cache refresh as the
    # publisher's run_policy.py, with only the requested parallelism changed.
    sys.path.insert(0, str(ASSETS / RELEASE / "runtime"))
    import mjlab.tasks
    from mimicx.parkour.environment import install_initial_reference_refresh

    released = release_runner()
    row, paths = released.prepare(POLICIES, ASSETS, task)
    registered = released.register(row, paths, ASSETS)
    install_initial_reference_refresh()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    cfg = load_env_cfg(registered, play=True)
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.commands["motion"].motion_file = str(paths["motion"])
    cfg.commands["motion"].debug_vis = False
    agent = load_rl_cfg(registered)
    agent.seed = seed
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = (load_runner_cls(registered) or MjlabOnPolicyRunner)(
        wrapped, asdict(agent), device="cuda:0"
    )
    runner.load(
        str(paths["checkpoint"]), load_cfg={"actor": True}, strict=True, map_location="cuda:0"
    )
    teacher = runner.get_inference_policy(device="cuda:0")
    teacher.eval().requires_grad_(False)
    return wrapped, teacher, row, paths


def module_digest(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

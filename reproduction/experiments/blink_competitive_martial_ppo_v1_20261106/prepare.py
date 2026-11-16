"""Create a private, resolved all-seven reference/Isaac configuration."""

import json
import shutil
from datetime import datetime
from experiments.blink_competitive_martial_ppo_v1_20261106.common import (
    BASE,
    BASE_SHA,
    REPORT,
    ROOT,
    CODE,
    SOURCE,
    OLD_CONFIG,
    bootstrap,
    sha,
    write_json,
)

bootstrap()
from omegaconf import OmegaConf
from holomotion.src.utils.config import setup_hydra_resolvers


def prepare():
    assert sha(BASE) == BASE_SHA, "Frozen base checkpoint changed"
    REPORT.mkdir(parents=True, exist_ok=True)
    config_file = REPORT / "config.yaml"
    if config_file.exists():
        return config_file
    source = json.loads((SOURCE / "manifest.json").read_text())
    reference = REPORT / "references_h5"
    reference.mkdir(exist_ok=True)
    (reference / "shards").mkdir(exist_ok=True)
    for item in source["hdf5_shards"]:
        link = reference / item["file"]
        if not link.exists():
            link.symlink_to(SOURCE / item["file"])
    source["root"] = str(reference)
    write_json(reference / "manifest.json", source)
    setup_hydra_resolvers()
    OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)
    c = OmegaConf.load(OLD_CONFIG)
    OmegaConf.set_struct(c, False)
    c.motion_h5_path = [str(reference)]
    c.num_envs = 512
    c.headless = True
    c.experiment_name = "blink_martial_seven_competitive_ppo"
    c.robot.motion.train_hdf5_roots = [str(reference)]
    c.robot.motion.val_hdf5_roots = [str(reference)]
    c.robot.motion.max_frame_length = max(v["length"] for v in source["clips"].values())
    c.robot.motion.min_frame_length = 1
    c.robot.motion.world_frame_normalization = True
    c.robot.motion.cache.max_num_clips = 7
    c.robot.motion.cache_max_num_clips = 7
    c.robot.motion.dataloader.num_workers = 0
    c.robot.motion.dataloader.timeout = 0
    c.robot.motion.dataloader.persistent_workers = False
    c.robot.motion.cache.swap_interval_steps = 2**31 - 1
    # Never write into the robot USD cache shared with other experiments.
    private_usd = REPORT / "usd"
    if not private_usd.exists():
        shutil.copytree(
            ROOT / "integrations/HoloMotion/assets/robots/unitree/G1/29dof/usd", private_usd
        )
    c.robot.asset.usd_cache_root = str(private_usd)
    c.robot.asset.force_usd_conversion = False
    e = c.env.config
    e.disable_ref_viz = True
    e.simulation.episode_length_s = (c.robot.motion.max_frame_length + 1) / 50.0
    e.commands.ref_motion.params.resample_time_interval_s = 1e9
    e.commands.ref_motion.params.debug_vis = False
    e.save_rendering_dir = str(REPORT / "renderings")
    e.eval_log_dir = str(REPORT / "eval_logs")
    c.algo._target_ = "private_manual_ppo"
    c.algo.config.public_checkpoint = str(BASE)
    c.algo.config.fixed_merge_bias = 0.0
    c.algo.config.num_learning_iterations = 1000
    # Materialize the exact old reward config before saving anything.
    old = OmegaConf.load(OLD_CONFIG)
    assert OmegaConf.to_container(c.rewards, resolve=True) == OmegaConf.to_container(
        old.rewards, resolve=True
    )
    resolved = OmegaConf.to_container(c, resolve=True)
    OmegaConf.save(OmegaConf.create(resolved), config_file)
    write_json(
        REPORT / "plan.json",
        dict(
            robot="unitree_g1_29dof",
            method="PPO",
            world_model=False,
            frozen_base=str(BASE),
            frozen_base_sha256=BASE_SHA,
            base_iteration=11000,
            old_experts=8,
            new_experts=1,
            top_k=2,
            shared_expert_frozen=True,
            actor_trainable_tensors=["gate_up", "down", "router.weight", "router.bias"],
            initialization="copy existing second-slot FFN, fit router raw score, calibrate new selection offset",
            teacher_queries=0,
            reference_names=list(source["clips"]),
            nominal_seconds=sum(v["length"] for v in source["clips"].values()) / 50,
            num_envs=512,
            steps_per_round=32,
            ppo_epochs=3,
            minibatches=8,
            actor_lr=3e-5,
            critic_lr=1e-4,
            fixed_gaussian_std=0.1,
            target_rounds=1000,
            max_actor_updates=24000,
            reward_source=str(OLD_CONFIG),
            reward_weights_unchanged=True,
            reference_shard_sha256={
                x["file"]: sha(SOURCE / x["file"]) for x in source["hdf5_shards"]
            },
            original_reward_sha256=sha(ROOT / "reproduction/roundhouse_impl.py"),
            source_sha256={p.name: sha(p) for p in CODE.glob("*.py")},
            unrelated_stairs_experiment_untouched=True,
            acceptance="per-clip phase-zero physical evaluation of base / initialized / trained, plus old-skill regression check",
        ),
    )
    return config_file


if __name__ == "__main__":
    print(prepare())

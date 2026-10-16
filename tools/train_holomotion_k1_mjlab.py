"""Train HoloMotion or a custom Actor on the shared K1 mjlab task.

The number of iterations and output directory are mandatory so an accidental
invocation cannot silently start the upstream 10,000-iteration schedule.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

# Support the documented ``python tools/train_holomotion_k1_mjlab.py`` form
# without requiring an editable install of this repository.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import yaml

from unified_motion.data.motion_sampler import MotionLibrary, MotionSampler
from unified_motion.envs.mjlab_k1 import K1MJLabBridge
from unified_motion.holomotion_mjlab import (
    HoloMotionK1MJLabAdapter,
    HoloMotionOnnxContract,
    make_holomotion_k1_mjlab_cfg,
)
from unified_motion.holomotion_mjlab_runner import (
    HoloMotionMJLabRunner,
    HoloMotionRunnerConfig,
)
from unified_motion.holomotion_mjlab_task import (
    HoloMotionK1MJLabTask,
    HoloMotionPhysicsRandomizationConfig,
)
from unified_motion.holomotion_original_algorithm import (
    initialize_original_holomotion_k1_algorithm,
    load_original_holomotion_k1_algorithm,
)
from unified_motion.physics_confidence import PhysicsConfidenceConfig
from unified_motion.physics_update import PhysicsUpdateConfig
from unified_motion.k1_policy_backends import ACTOR_BACKENDS, K1ActorConfig
from unified_motion.k1_training import create_custom_k1_algorithm, create_k1_trainer
from unified_motion.k1_policy_runtime import resume_k1_trainer, runtime_configuration
from unified_motion.k1_policy_runtime import load_k1_checkpoint

DEFAULT_HOLOMOTION_ROOT = (
    Path(__file__).resolve().parents[1] / "third_party" / "HoloMotion" / "source"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holomotion-root",
        type=Path,
        default=DEFAULT_HOLOMOTION_ROOT,
        help="HoloMotion source root (default: vendored audited snapshot)",
    )
    parser.add_argument("--actor", choices=(*ACTOR_BACKENDS, "dense"), help="default: holomotion; dense aliases transformer; resume infers the backend")
    parser.add_argument("--actor-config", type=Path, help="custom Actor YAML mapping (K1ActorConfig fields)")
    parser.add_argument("--resume", type=Path, help="standalone complete K1 checkpoint; restores its configuration")
    parser.add_argument("--teacher-pretrain-from", type=Path,
        help="freeze this non-Physics policy checkpoint and train only a new privileged Teacher")
    parser.add_argument("--teacher-pretrain-run-dir", type=Path,
        help="freeze an original HoloMotion run checkpoint and train a new privileged Teacher")
    parser.add_argument("--frozen-teacher-from", type=Path,
        help=("load only a trained privileged Teacher into a fresh Actor experiment; "
              "freeze the Teacher and train a randomly initialized Actor using either "
              "the legacy residual or --teacher-latent-observation"))
    parser.add_argument("--frozen-teacher-residual-limit", type=float, default=0.1,
        help="maximum absolute Teacher action residual for a fresh-Actor experiment (default: 0.1)")
    parser.add_argument("--teacher-latent-observation", action="store_true",
        help=("with --frozen-teacher-from, remove the action residual and feed the "
              "frozen Teacher latent z directly into a custom PPO Actor"))
    parser.add_argument("--teacher-controls-residual", action="store_true",
        help=("start a privileged control experiment from a trained Teacher checkpoint: "
              "freeze base Actor/Teacher, discard the old residual, and train a new residual "
              "from Teacher next-state/contact predictions only"))
    holomotion_source = parser.add_mutually_exclusive_group()
    holomotion_source.add_argument("--run-dir", type=Path, help="original HoloMotion weights/config directory")
    holomotion_source.add_argument(
        "--holomotion-config",
        type=Path,
        help="compact HoloMotion architecture/PPO YAML; randomly initializes all weights",
    )
    parser.add_argument("--checkpoint-name", default="model_9999")
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="restore optimizer moments and continue after the checkpoint iteration",
    )
    contract_source = parser.add_mutually_exclusive_group()
    contract_source.add_argument("--onnx", type=Path, help="joint/PD/observation metadata, not the inference engine")
    contract_source.add_argument("--task-contract", type=Path, help="JSON HoloMotionOnnxContract for custom training without ONNX")
    parser.add_argument("--asset-root", type=Path, required=True)
    motion_source = parser.add_mutually_exclusive_group(required=True)
    motion_source.add_argument(
        "--motion",
        type=Path,
        action="append",
        help="named motion NPZ; repeat this option to build a multi-motion library",
    )
    motion_source.add_argument(
        "--motion-dir",
        type=Path,
        help="folder of project NPZ windows produced by the HDF5 converter",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--steps-per-env", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--progress-interval", type=int, default=10,
                        help="print reward, speed and ETA every N iterations")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--advisor-config', type=Path, help='optional asynchronous DeepSeek/GLM proposal-only adviser')
    parser.add_argument('--trial-adjustment', type=Path, help='explicit experimental coefficient proposal applied before collection')
    parser.add_argument('--observation-normalization', choices=('auto', 'update', 'frozen'), default='auto',
        help='auto: fresh weights update stats; imported/legacy weights preserve stats; resume keeps the saved mode')
    parser.add_argument("--physics-teacher", action="store_true")
    parser.add_argument('--physics-teacher-architecture', choices=('transformer', 'mlp'), default='transformer')
    parser.add_argument('--physics-teacher-config', type=Path, help='training-only Transformer Teacher YAML')
    parser.add_argument('--teacher-delta-whitening-stats', type=Path,
        help='fixed train-split delta statistics JSON for an explicitly whitened Teacher config')
    parser.add_argument("--physics", action="store_true", help="direct-ID Physics residual without Teacher")
    parser.add_argument("--finetune-base-actor", action="store_true", help="unfreeze base Actor in Physics runs")
    parser.add_argument("--physics-gate", choices=("bounded", "calibrated", "off", "ungated"), default="bounded")
    parser.add_argument('--physics-feedback-floor', type=float, default=.1)
    parser.add_argument("--physics-ema-half-life-steps", type=float, default=1024.)
    parser.add_argument("--physics-max-update-kl", type=float, default=.01)
    parser.add_argument("--physics-friction-range", type=float, nargs=2, default=(.2, 1.2),
                        metavar=("MIN", "MAX"))
    parser.add_argument("--physics-mass-scale-range", type=float, nargs=2, default=(.8, 1.2),
                        metavar=("MIN", "MAX"))
    parser.add_argument("--physics-history-mib", type=int, default=512)
    parser.add_argument("--physics-scene-encoder", choices=("set", "gnn"), default="set")
    parser.add_argument("--physics-graph-layers", type=int, default=2)
    parser.add_argument(
        "--matched-physics-task",
        action="store_true",
        help=("enable the same physics randomization as the Physics run, "
              "without adding the Physics network"),
    )
    parser.add_argument("--physics-width", type=int, default=64)
    parser.add_argument("--physics-memory-width", type=int, default=64)
    parser.add_argument("--physics-latent-width", type=int, default=8)
    parser.add_argument("--physics-hidden-dim", type=int, default=128)
    parser.add_argument("--physics-burn-in-steps", type=int, default=64)
    parser.add_argument("--physics-train-steps", type=int, default=128)
    parser.add_argument("--physics-confidence-warmup-steps", type=int, default=128)
    parser.add_argument("--physics-ema-decay", type=float, default=None,
                        help="explicit legacy per-update decay; overrides the step half-life")
    parser.add_argument("--physics-dynamics-epochs", type=int, default=4)
    args = parser.parse_args()
    if args.actor == "dense":
        args.actor = "transformer"
    teacher_sources = sum(value is not None for value in (
        args.teacher_pretrain_from, args.teacher_pretrain_run_dir))
    if (teacher_sources > 1 or args.resume is not None and teacher_sources
            or args.frozen_teacher_from is not None and (args.resume is not None or teacher_sources)):
        parser.error("--resume, Teacher pretraining sources and --frozen-teacher-from are mutually exclusive")
    if args.teacher_controls_residual and args.resume is None:
        parser.error("--teacher-controls-residual requires --resume with a trained Teacher checkpoint")
    if args.teacher_latent_observation and args.frozen_teacher_from is None:
        parser.error("--teacher-latent-observation requires --frozen-teacher-from")
    if args.teacher_latent_observation and args.teacher_controls_residual:
        parser.error("Teacher latent observation and Teacher residual are mutually exclusive")
    if teacher_sources:
        if args.run_dir is not None or args.holomotion_config is not None or args.actor_config is not None:
            parser.error("Teacher pretraining restores its Actor from its selected Teacher source")
        if args.physics or args.resume_optimizer or args.finetune_base_actor:
            parser.error("Teacher-only pretraining cannot enable joint Physics/PPO options")
        args.physics_teacher = True
        args.physics_teacher_architecture = "transformer"
    if args.frozen_teacher_from is not None:
        if args.teacher_controls_residual:
            parser.error("--frozen-teacher-from already selects Teacher-controlled residual training")
        if args.physics or args.physics_teacher or args.physics_teacher_config is not None:
            parser.error("--frozen-teacher-from restores its saved Teacher; remove Physics/Teacher creation flags")
        if args.teacher_delta_whitening_stats is not None:
            parser.error("--frozen-teacher-from restores saved whitening statistics")
        if args.run_dir is not None or args.resume_optimizer:
            parser.error("--frozen-teacher-from requires a randomly initialized Actor, not imported Actor weights")
        args.physics_teacher = True
        args.physics_teacher_architecture = "transformer"
        if args.teacher_latent_observation and args.actor in (None, "holomotion"):
            parser.error("--teacher-latent-observation currently requires a custom Actor such as --actor mlp")
    if args.resume is not None:
        if (args.resume_optimizer or args.run_dir is not None
                or args.holomotion_config is not None or args.actor_config is not None):
            parser.error("--resume restores its bundle; do not combine with run-dir, holomotion-config, resume-optimizer or actor-config")
        restore_options = {"--steps-per-env", "--finetune-base-actor", "--matched-physics-task", "--physics", "--physics-teacher"}
        explicit = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
        conflicting = {flag for flag in explicit if (flag in restore_options
            or flag.startswith("--physics-") and flag != "--physics-history-mib"
            )}
        if conflicting:
            parser.error("--resume restores saved training settings; remove overrides: "+", ".join(sorted(conflicting)))
    elif not teacher_sources:
        args.actor = args.actor or "holomotion"
        if (args.onnx is None and args.task_contract is None
                and args.frozen_teacher_from is None):
            parser.error("provide --onnx or --task-contract for a new run")
        if (args.actor == "holomotion"
                and args.run_dir is None and args.holomotion_config is None):
            parser.error("--actor holomotion requires --run-dir or --holomotion-config")
        if args.actor != "holomotion" and (
            args.run_dir is not None or args.holomotion_config is not None
            or args.resume_optimizer
        ):
            parser.error("custom Actors start fresh; use --resume for their complete bundles")
        if args.resume_optimizer and args.run_dir is None:
            parser.error("--resume-optimizer requires --run-dir and an original checkpoint")
        if args.actor == "holomotion" and args.actor_config is not None:
            parser.error("HoloMotion architecture comes from its source run, not --actor-config")
        if args.physics_scene_encoder == "gnn" and not (args.physics or args.physics_teacher):
            parser.error("--physics-scene-encoder gnn requires --physics or --physics-teacher")
    if args.physics_teacher_config is not None and (
            not args.physics_teacher or args.physics_teacher_architecture != 'transformer'):
        parser.error('--physics-teacher-config requires the Transformer Teacher')
    if args.teacher_delta_whitening_stats is not None and args.physics_teacher_config is None:
        parser.error('--teacher-delta-whitening-stats requires --physics-teacher-config')
    if args.resume is not None and args.teacher_delta_whitening_stats is not None:
        parser.error('resume restores saved Teacher whitening statistics; remove the override')
    positive = (
        "iterations",
        "steps_per_env",
        "num_envs",
        "save_interval",
        "log_interval",
        "progress_interval",
        "physics_burn_in_steps",
        "physics_train_steps",
        "physics_confidence_warmup_steps",
        "physics_dynamics_epochs",
        "physics_history_mib",
        "physics_graph_layers",
    )
    if any(getattr(args, name) < 1 for name in positive):
        parser.error("iteration, step, environment, save, and log counts must be positive")
    if any(getattr(args, name) < 2 for name in (
        "physics_width", "physics_memory_width", "physics_latent_width",
        "physics_hidden_dim",
    )):
        parser.error("physics model widths must be at least 2")
    if not args.device.startswith("cuda"):
        parser.error("mjlab training requires a CUDA device")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if (not np.isfinite(args.frozen_teacher_residual_limit)
            or not 0.0 < args.frozen_teacher_residual_limit <= 1.0):
        parser.error("--frozen-teacher-residual-limit must be in (0,1]")
    if args.physics_ema_decay is not None and not 0.0 < args.physics_ema_decay < 1.0:
        parser.error("physics EMA decay must be in (0,1)")
    for name in ("physics_friction_range", "physics_mass_scale_range"):
        bounds = tuple(getattr(args, name))
        if any(not np.isfinite(v) or v <= 0 for v in bounds) or bounds[0] >= bounds[1]:
            parser.error(name.replace('_', '-')+" must be an increasing positive range")
    try:
        PhysicsUpdateConfig(ema_half_life_steps=args.physics_ema_half_life_steps,
                            maximum_mean_kl=args.physics_max_update_kl)
        PhysicsConfidenceConfig(mode=args.physics_gate, feedback_floor=args.physics_feedback_floor)
    except ValueError as error:
        parser.error(str(error))
    return args


def main() -> None:
    args = parse_args()
    teacher_sources = sum(value is not None for value in (
        args.teacher_pretrain_from, args.teacher_pretrain_run_dir))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no mjlab training was started")
    from mjlab.envs import ManagerBasedRlEnv

    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    resume_payload = None
    teacher_source_payload = None
    frozen_teacher_payload = None
    if args.teacher_pretrain_from is not None:
        teacher_source_payload = torch.load(
            args.teacher_pretrain_from, map_location="cpu", weights_only=True)
        if "physics_schema" in teacher_source_payload:
            raise ValueError("Teacher-only must start from a non-Physics policy checkpoint")
        saved_runtime = runtime_configuration(teacher_source_payload)
        saved_network = saved_runtime.get("network_config", teacher_source_payload.get("network_config"))
        args.actor = saved_network.get("backend", "holomotion")
        physics_enabled = True
        saved_contract = saved_runtime.get("task_contract")
        contract = HoloMotionOnnxContract(**saved_contract) if saved_contract is not None else None
    elif args.teacher_pretrain_run_dir is not None:
        args.actor = "holomotion"
        physics_enabled = True
        contract = None
    elif args.resume is not None:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        saved_runtime = runtime_configuration(resume_payload)
        saved_network = saved_runtime.get("network_config", resume_payload.get("network_config"))
        saved_actor = saved_network.get("backend", "holomotion")
        if args.actor is not None and args.actor != saved_actor:
            raise ValueError("--actor disagrees with checkpoint; changing architecture starts a new experiment")
        args.actor = saved_actor
        physics_enabled = "physics_schema" in resume_payload
        if (args.physics or args.physics_teacher) and not physics_enabled:
            raise ValueError("cannot add Physics through resume; start a new experiment")
        args.physics_teacher = (saved_runtime.get("dynamics_config", {}).get("teacher_student", False)
                                or saved_runtime.get('privileged_teacher_config') is not None)
        args.steps_per_env = saved_runtime["rollout_steps"]
        saved_contract = saved_runtime.get("task_contract")
        contract = HoloMotionOnnxContract(**saved_contract) if saved_contract is not None else None
    elif args.frozen_teacher_from is not None:
        frozen_teacher_payload = torch.load(
            args.frozen_teacher_from, map_location="cpu", weights_only=True)
        if not isinstance(frozen_teacher_payload, dict):
            raise ValueError("frozen Teacher checkpoint must be a dictionary")
        saved_teacher = frozen_teacher_payload.get("privileged_teacher")
        saved_runtime = runtime_configuration(frozen_teacher_payload)
        if (frozen_teacher_payload.get("physics_schema") != 5 or saved_teacher is None
                or saved_teacher.get("updates", 0) < 1):
            raise ValueError("--frozen-teacher-from requires a trained schema-5 privileged Teacher checkpoint")
        from unified_motion.privileged_teacher import PrivilegedTeacherConfig
        frozen_teacher_config = PrivilegedTeacherConfig(**saved_teacher.get("config", {}))
        if args.teacher_latent_observation and frozen_teacher_config.objective_version < 5:
            raise ValueError(
                "--teacher-latent-observation requires a shared Critic/Teacher v5 checkpoint"
            )
        if not args.teacher_latent_observation and frozen_teacher_config.objective_version >= 4:
            raise ValueError("Teacher v4/v5 cannot use the legacy action residual")
        physics_enabled = True
        saved_contract = saved_runtime.get("task_contract")
        contract = HoloMotionOnnxContract(**saved_contract) if saved_contract is not None else None
    else:
        contract = None
        physics_enabled = args.physics or args.physics_teacher
    supplied = (HoloMotionOnnxContract.from_onnx(args.onnx) if args.onnx is not None else
        HoloMotionOnnxContract.from_json(args.task_contract)
        if args.task_contract is not None else None)
    if supplied is not None:
        if contract is not None and asdict(contract) != asdict(supplied):
            raise ValueError("supplied task contract disagrees with checkpoint")
        contract = supplied
    if contract is None:
        raise ValueError("checkpoint has no task contract; provide --onnx or --task-contract")
    if args.motion_dir is not None:
        motion_paths = sorted(args.motion_dir.expanduser().rglob("*.npz"))
        if not motion_paths:
            raise FileNotFoundError(f"{args.motion_dir} contains no named motion NPZ files")
        # ``--motion-dir`` is documented as the output of
        # convert_holomotion_hdf5_dataset.py.  Those files use the explicit
        # named MJLab contract, not the unrelated project-NPZ schema.
        library = MotionLibrary.from_named_mjlab_files(
            motion_paths, robot_id="k1_22dof", device=device
        )
    else:
        motion_paths = args.motion
        library = MotionLibrary.from_named_mjlab_files(
            args.motion, robot_id="k1_22dof", device=device
        )
    offsets = tuple(index * 0.02 for index in range(contract.future_frames + 2))
    sampler = MotionSampler(
        library,
        num_envs=args.num_envs,
        step_dt=0.02,
        future_offsets_s=offsets,
    )
    cfg = make_holomotion_k1_mjlab_cfg(
        args.asset_root, contract, num_envs=args.num_envs
    )
    cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg, device=str(device))
    matched_physics_task = physics_enabled or args.matched_physics_task
    task = HoloMotionK1MJLabTask(
        HoloMotionK1MJLabAdapter(
            K1MJLabBridge(env, reference_steps=contract.future_frames + 2), contract
        ),
        sampler,
        physics_randomization=((None if saved_runtime.get("physics_randomization") is None else
            HoloMotionPhysicsRandomizationConfig(**saved_runtime["physics_randomization"]))
            if resume_payload is not None or frozen_teacher_payload is not None else
            HoloMotionPhysicsRandomizationConfig(
                friction_range=tuple(args.physics_friction_range),
                trunk_mass_scale_range=tuple(args.physics_mass_scale_range))
            if matched_physics_task else None
        ),
    )
    advisor = None
    try:
        if args.resume is not None:
            trainer = resume_k1_trainer(args.resume, task, holomotion_root=args.holomotion_root,
                dynamics_history_bytes=args.physics_history_mib*1024**2)
            if args.teacher_controls_residual:
                if getattr(trainer, '_physics_control_mode', 'student') == 'teacher':
                    raise ValueError("checkpoint already uses Teacher-controlled residual; resume it without --teacher-controls-residual")
                trainer.prepare_teacher_control()
        else:
            source_ppo_config = None
            if args.teacher_pretrain_from is not None:
                from unified_motion.holomotion_original_algorithm import OriginalHoloMotionK1Algorithm
                from unified_motion.holomotion_mjlab_ppo import HoloMotionPPOConfig
                from unified_motion.holomotion_physics_runtime import validate_task_contract
                loaded_source = load_k1_checkpoint(args.teacher_pretrain_from, device=device,
                                                   holomotion_root=args.holomotion_root)
                validate_task_contract(runtime_configuration(loaded_source.payload), task)
                algorithm = OriginalHoloMotionK1Algorithm(
                    loaded_source.actor, loaded_source.critic, None,
                    args.teacher_pretrain_from.parent, loaded_source.payload["network_config"])
                source_ppo_config = HoloMotionPPOConfig(**loaded_source.payload["ppo_config"])
            elif args.teacher_pretrain_run_dir is not None:
                algorithm = load_original_holomotion_k1_algorithm(
                    holomotion_root=args.holomotion_root,
                    run_dir=args.teacher_pretrain_run_dir,
                    checkpoint_name=args.checkpoint_name,
                    device=device,
                )
            elif args.actor == "holomotion":
                algorithm = (
                    initialize_original_holomotion_k1_algorithm(
                        holomotion_root=args.holomotion_root,
                        config_path=args.holomotion_config,
                        device=device,
                    )
                    if args.holomotion_config is not None
                    else load_original_holomotion_k1_algorithm(
                        holomotion_root=args.holomotion_root,
                        run_dir=args.run_dir,
                        checkpoint_name=args.checkpoint_name,
                        device=device,
                    )
                )
            else:
                teacher_latent_dim = None
                if frozen_teacher_payload is not None and args.teacher_latent_observation:
                    teacher_latent_dim = int(
                        saved_runtime["dynamics_config"]["latent_width"])
                critic_privileged_dim = None
                if frozen_teacher_payload is not None and args.teacher_latent_observation:
                    from unified_motion.privileged_teacher import PrivilegedTeacherConfig
                    teacher_contract = PrivilegedTeacherConfig(
                        **frozen_teacher_payload["privileged_teacher"]["config"])
                    critic_privileged_dim = teacher_contract.critic_privileged_dim(
                        int(saved_runtime["dynamics_config"]["system_id_dim"]))
                algorithm = create_custom_k1_algorithm(
                    custom_actor_config(
                        args.actor, args.actor_config,
                        teacher_latent_dim=teacher_latent_dim,
                        critic_privileged_dim=critic_privileged_dim),
                    device=device)
            dynamics_options = dict(
                    physics_width=args.physics_width,
                    memory_width=args.physics_memory_width,
                    latent_width=args.physics_latent_width,
                    hidden_dim=args.physics_hidden_dim,
                    scene_encoder=args.physics_scene_encoder,
                    graph_layers=args.physics_graph_layers,
                    contact_object_indices=[-1,-1] if args.physics_scene_encoder == "gnn" else None)
            if frozen_teacher_payload is not None:
                dynamics_options = dict(saved_runtime["dynamics_config"])
                for key in ("robot", "system_id_dim", "teacher_student", "uncertainty"):
                    dynamics_options.pop(key, None)
            physics_options = {} if not physics_enabled else dict(
                dynamics_config=dynamics_options,
                dynamics_burn_in_steps=args.physics_burn_in_steps,
                dynamics_train_steps=args.physics_train_steps,
                confidence_warmup_steps=args.physics_confidence_warmup_steps,
                dynamics_ema_decay=args.physics_ema_decay,
                dynamics_epochs=args.physics_dynamics_epochs,
                dynamics_history_bytes=args.physics_history_mib*1024**2,
                confidence_config=PhysicsConfidenceConfig(mode=args.physics_gate, feedback_floor=args.physics_feedback_floor),
                update_config=PhysicsUpdateConfig(ema_half_life_steps=args.physics_ema_half_life_steps,
                    maximum_mean_kl=args.physics_max_update_kl,
                    freeze_base_actor=(teacher_sources > 0 or (args.actor == "holomotion"
                        and args.run_dir is not None
                        and not args.finetune_base_actor))),
            )
            frozen_teacher_config = (None if frozen_teacher_payload is None else
                                     frozen_teacher_payload["privileged_teacher"]["config"])
            # A frozen-Teacher branch adds randomly initialized scaffolding
            # after the fresh Actor/Critic have been built.  Fork the RNG so
            # that scaffolding does not change the baseline's first rollout
            # exploration stream; the zero-init residual then starts from the
            # exact same base action distribution.
            rng_scope = (torch.random.fork_rng(devices=[device])
                         if frozen_teacher_payload is not None else nullcontext())
            with rng_scope:
                trainer = create_k1_trainer(algorithm, task, physics=physics_enabled,
                    teacher=args.physics_teacher,
                    ppo_config=source_ppo_config,
                    privileged_teacher_config=(
                        ({} if args.physics_teacher_config is None else
                         load_teacher_config(args.physics_teacher_config,
                                             args.teacher_delta_whitening_stats))
                        if args.physics_teacher and args.physics_teacher_architecture == 'transformer'
                        and frozen_teacher_config is None else frozen_teacher_config),
                    **physics_options)
            if teacher_sources:
                trainer.prepare_teacher_only()
            if frozen_teacher_payload is not None:
                from unified_motion.holomotion_physics_runtime import validate_task_contract
                validate_task_contract(runtime_configuration(frozen_teacher_payload), task)
                trainer.restore_privileged_teacher(frozen_teacher_payload)
                if args.teacher_latent_observation:
                    trainer.prepare_teacher_observation()
                else:
                    trainer.prepare_teacher_control(
                        freeze_base_actor=False,
                        residual_limit=args.frozen_teacher_residual_limit,
                    )
        from unified_motion.llm.advisor import apply_trial_adjustment, tuning_snapshot, start_training_trial
        start_iteration = 0 if resume_payload is None else resume_payload["iter"]+1
        if args.resume_optimizer:
            resumed = trainer.load_optimizer_checkpoint(
                args.run_dir / f"{args.checkpoint_name}.pt"
            )
            start_iteration = resumed.iteration + 1
        if args.observation_normalization != 'auto':
            trainer.configure_observation_normalization(enabled=args.observation_normalization == 'update')
        start_training_trial(trainer)
        if args.trial_adjustment is not None:
            apply_trial_adjustment(trainer, json.loads(args.trial_adjustment.read_text(encoding='utf-8')))
        if args.resume is not None:
            from unified_motion.llm.validation import file_sha256
            trainer.training_source_checkpoint_sha256 = file_sha256(args.resume)
        elif args.frozen_teacher_from is not None:
            from unified_motion.llm.validation import file_sha256
            trainer.training_source_checkpoint_sha256 = file_sha256(args.frozen_teacher_from)
        trainer.training_seed = args.seed
        from unified_motion.llm.validation import file_sha256
        trainer.training_protocol = dict(schema=1, seed=args.seed, num_envs=args.num_envs,
            observation_normalization=dict(trainer._observation_normalization),
            steps_per_env=args.steps_per_env, motion_sha256=[file_sha256(path) for path in motion_paths],
            asset_root=str(args.asset_root.resolve()), task_contract=asdict(contract),
            physics_randomization=(None if task.physics_randomization is None else asdict(task.physics_randomization)))
        runner_config = HoloMotionRunnerConfig(
            num_iterations=args.iterations,
            steps_per_env=args.steps_per_env,
            start_iteration=start_iteration,
            save_interval=args.save_interval,
            log_interval=args.log_interval,
            progress_interval=args.progress_interval,
            teacher_only=bool(teacher_sources),
        )
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        if args.run_dir is not None:
            source_config = args.run_dir.expanduser().resolve() / "config.yaml"
            destination_config = output / "config.yaml"
            if not destination_config.exists():
                shutil.copy2(source_config, destination_config)
        if args.holomotion_config is not None:
            source_config = args.holomotion_config.expanduser().resolve()
            destination_config = output / "holomotion_config.yaml"
            if not destination_config.exists():
                shutil.copy2(source_config, destination_config)
        metadata = {
            "actor_backend": args.actor,
            "network_config": trainer._network_config,
            "source_run": None if args.run_dir is None else str(args.run_dir.expanduser().resolve()),
            "source_checkpoint": str(args.resume) if args.resume else (args.checkpoint_name if args.run_dir else None),
            "teacher_pretrain_source": (None if args.teacher_pretrain_from is None else
                                         str(args.teacher_pretrain_from.expanduser().resolve())),
            "teacher_pretrain_run_dir": (None if args.teacher_pretrain_run_dir is None else
                                           str(args.teacher_pretrain_run_dir.expanduser().resolve())),
            "frozen_teacher_source": (None if args.frozen_teacher_from is None else
                                        str(args.frozen_teacher_from.expanduser().resolve())),
            "initialization_config": (str(args.holomotion_config.expanduser().resolve())
                if args.holomotion_config is not None else None),
            "motion_files": (
                [str(path.expanduser().resolve()) for path in args.motion]
                if args.motion is not None
                else None
            ),
            "motion_dir": (
                str(args.motion_dir.expanduser().resolve())
                if args.motion_dir is not None
                else None
            ),
            "resume_optimizer": args.resume_optimizer or args.resume is not None,
            "seed": args.seed,
            "training_tuning": tuning_snapshot(trainer),
            "trial_adjustment": getattr(trainer, 'training_adjustment', None),
            "physics_teacher": args.physics_teacher,
            "physics_control_mode": getattr(trainer, '_physics_control_mode', None),
            "privileged_teacher_config": (None if getattr(trainer, 'privileged_teacher', None) is None
                                          else asdict(trainer.privileged_teacher.config)),
            "physics_enabled": physics_enabled,
            "matched_physics_task": matched_physics_task,
            "physics_model": ({
                "physics_width": trainer.dynamics.physics_width,
                "memory_width": trainer.dynamics.memory_width,
                "latent_width": trainer.dynamics.latent_width,
                "hidden_dim": trainer.dynamics.hidden_dim,
                "scene_encoder": trainer.dynamics.encoder.scene_encoder,
                "teacher_dim": task.physics_teacher_dim,
                "uncertainty": True,
                "burn_in_steps": trainer.dynamics_burn_in_steps,
                "train_steps": trainer.dynamics_train_steps,
                "confidence_warmup_steps": trainer.confidence_warmup_steps,
                "ema_decay": trainer.dynamics_ema_decay,
                "dynamics_epochs": trainer.dynamics_epochs,
                "history_mib": args.physics_history_mib,
                "update_config": asdict(trainer.update_config),
                "confidence_config": asdict(trainer.confidence.config),
            } if physics_enabled else None),
            "runner": asdict(runner_config),
        }
        metadata_path = output / "mjlab_runner.json"
        if metadata_path.exists():
            raise FileExistsError(f"runner metadata already exists: {metadata_path}")
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if args.advisor_config is not None:
            from unified_motion.llm.advisor import PeriodicTrainingAdvisor
            advisor = PeriodicTrainingAdvisor(
                yaml.safe_load(args.advisor_config.read_text(encoding='utf-8')), output/'advisor')
        records = HoloMotionMJLabRunner(trainer, output, runner_config, advisor=advisor).run()
        print(json.dumps(records[-1], ensure_ascii=False, indent=2))
    finally:
        task.close()
        if advisor is not None:
            advisor.close()


def custom_actor_config(backend, path=None, *, teacher_latent_dim=None,
                        critic_privileged_dim=None):
    values = {} if path is None else yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Actor config must be a YAML mapping")
    values = dict(values)
    selected = values.pop("backbone", backend)
    if ("transformer" if selected == "dense" else selected) != backend:
        raise ValueError("Actor config backbone disagrees with --actor")
    if teacher_latent_dim is not None:
        configured = values.get("teacher_latent_dim", teacher_latent_dim)
        if configured != teacher_latent_dim:
            raise ValueError("Actor config teacher_latent_dim disagrees with frozen Teacher")
        values["teacher_latent_dim"] = teacher_latent_dim
    if critic_privileged_dim is not None:
        configured = values.get("critic_privileged_dim", critic_privileged_dim)
        if configured != critic_privileged_dim:
            raise ValueError("Actor config critic_privileged_dim disagrees with frozen Teacher")
        values["critic_privileged_dim"] = critic_privileged_dim
    return K1ActorConfig(backbone=backend, **values)


def load_teacher_config(path, whitening_stats_path=None):
    from unified_motion.privileged_teacher import PrivilegedTeacherConfig
    values = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(values, dict):
        raise ValueError('Transformer Teacher config must be a YAML mapping')
    if whitening_stats_path is not None:
        document = json.loads(Path(whitening_stats_path).read_text(encoding='utf-8'))
        if (document.get('schema_version') != 1
                or document.get('kind') != 'teacher_delta_whitening_statistics'
                or not isinstance(document.get('statistics'), dict)):
            raise ValueError('invalid Teacher delta whitening statistics document')
        values = dict(values)
        if values.get('delta_whitening') is not True:
            raise ValueError('Teacher config must explicitly enable delta_whitening')
        if 'delta_whitening_statistics' in values:
            raise ValueError('Teacher config must not embed statistics when a stats file is supplied')
        values['delta_whitening_statistics'] = document['statistics']
    return asdict(PrivilegedTeacherConfig(**values))


if __name__ == "__main__":
    main()

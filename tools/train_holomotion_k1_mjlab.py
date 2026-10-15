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
    HoloMotionFootRewardConfig,
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--advisor-config', type=Path, help='optional asynchronous DeepSeek/GLM proposal-only adviser')
    parser.add_argument('--trial-adjustment', type=Path, help='explicit experimental coefficient proposal applied before collection')
    parser.add_argument('--observation-normalization', choices=('auto', 'update', 'frozen'), default='auto',
        help='auto: fresh weights update stats; imported/legacy weights preserve stats; resume keeps the saved mode')
    parser.add_argument("--physics-teacher", action="store_true")
    parser.add_argument('--physics-teacher-architecture', choices=('transformer', 'mlp'), default='transformer')
    parser.add_argument('--physics-teacher-config', type=Path, help='training-only Transformer Teacher YAML')
    parser.add_argument("--physics", action="store_true", help="direct-ID Physics residual without Teacher")
    parser.add_argument("--finetune-base-actor", action="store_true", help="unfreeze base Actor in Physics runs")
    parser.add_argument("--physics-gate", choices=("bounded", "calibrated", "off", "ungated"), default="bounded")
    parser.add_argument('--physics-feedback-floor', type=float, default=.1)
    parser.add_argument("--physics-ema-half-life-steps", type=float, default=1024.)
    parser.add_argument("--physics-max-update-kl", type=float, default=.01)
    parser.add_argument("--physics-history-mib", type=int, default=512)
    parser.add_argument("--physics-scene-encoder", choices=("set", "gnn"), default="set")
    parser.add_argument("--physics-graph-layers", type=int, default=2)
    parser.add_argument(
        "--matched-physics-task",
        action="store_true",
        help=("enable the same foot rewards and physics randomization as the "
              "physics-teacher run, without adding the physics network"),
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
    parser.add_argument("--left-foot-slip-weight", type=float, default=0.02)
    parser.add_argument("--right-foot-slip-weight", type=float, default=0.02)
    parser.add_argument("--left-foot-impact-weight", type=float, default=0.01)
    parser.add_argument("--right-foot-impact-weight", type=float, default=0.01)
    parser.add_argument("--foot-impact-threshold-n", type=float, default=100.0)
    args = parser.parse_args()
    if args.actor == "dense":
        args.actor = "transformer"
    if args.resume is not None:
        if (args.resume_optimizer or args.run_dir is not None
                or args.holomotion_config is not None or args.actor_config is not None):
            parser.error("--resume restores its bundle; do not combine with run-dir, holomotion-config, resume-optimizer or actor-config")
        restore_options = {"--steps-per-env", "--finetune-base-actor", "--matched-physics-task", "--physics", "--physics-teacher"}
        explicit = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
        conflicting = {flag for flag in explicit if (flag in restore_options
            or flag.startswith("--physics-") and flag != "--physics-history-mib"
            or flag.startswith("--left-foot-") or flag.startswith("--right-foot-")
            or flag == "--foot-impact-threshold-n")}
        if conflicting:
            parser.error("--resume restores saved training settings; remove overrides: "+", ".join(sorted(conflicting)))
    else:
        args.actor = args.actor or "holomotion"
        if args.onnx is None and args.task_contract is None:
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
    positive = (
        "iterations",
        "steps_per_env",
        "num_envs",
        "save_interval",
        "log_interval",
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
    if args.physics_ema_decay is not None and not 0.0 < args.physics_ema_decay < 1.0:
        parser.error("physics EMA decay must be in (0,1)")
    try:
        PhysicsUpdateConfig(ema_half_life_steps=args.physics_ema_half_life_steps,
                            maximum_mean_kl=args.physics_max_update_kl)
        PhysicsConfidenceConfig(mode=args.physics_gate, feedback_floor=args.physics_feedback_floor)
    except ValueError as error:
        parser.error(str(error))
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no mjlab training was started")
    from mjlab.envs import ManagerBasedRlEnv

    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    resume_payload = None
    if args.resume is not None:
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
        library = MotionLibrary.from_project_folder(
            args.motion_dir, robot_id="k1_22dof", device=device
        )
    else:
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
        foot_reward_config=(HoloMotionFootRewardConfig(**(saved_runtime.get("foot_reward") or {}))
            if resume_payload is not None else
            HoloMotionFootRewardConfig(
                left_slip_weight=args.left_foot_slip_weight,
                right_slip_weight=args.right_foot_slip_weight,
                left_impact_weight=args.left_foot_impact_weight,
                right_impact_weight=args.right_foot_impact_weight,
                impact_threshold_n=args.foot_impact_threshold_n,
            )
            if matched_physics_task else None
        ),
        physics_randomization=((None if saved_runtime.get("physics_randomization") is None else
            HoloMotionPhysicsRandomizationConfig(**saved_runtime["physics_randomization"]))
            if resume_payload is not None else
            HoloMotionPhysicsRandomizationConfig()
            if matched_physics_task else None
        ),
    )
    advisor = None
    try:
        if args.resume is not None:
            trainer = resume_k1_trainer(args.resume, task, holomotion_root=args.holomotion_root,
                dynamics_history_bytes=args.physics_history_mib*1024**2)
        else:
            if args.actor == "holomotion":
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
                algorithm = create_custom_k1_algorithm(
                    custom_actor_config(args.actor, args.actor_config), device=device)
            physics_options = {} if not physics_enabled else dict(
                dynamics_config=dict(
                    physics_width=args.physics_width,
                    memory_width=args.physics_memory_width,
                    latent_width=args.physics_latent_width,
                    hidden_dim=args.physics_hidden_dim,
                    scene_encoder=args.physics_scene_encoder,
                    graph_layers=args.physics_graph_layers,
                    contact_object_indices=[-1,-1] if args.physics_scene_encoder == "gnn" else None),
                dynamics_burn_in_steps=args.physics_burn_in_steps,
                dynamics_train_steps=args.physics_train_steps,
                confidence_warmup_steps=args.physics_confidence_warmup_steps,
                dynamics_ema_decay=args.physics_ema_decay,
                dynamics_epochs=args.physics_dynamics_epochs,
                dynamics_history_bytes=args.physics_history_mib*1024**2,
                confidence_config=PhysicsConfidenceConfig(mode=args.physics_gate, feedback_floor=args.physics_feedback_floor),
                update_config=PhysicsUpdateConfig(ema_half_life_steps=args.physics_ema_half_life_steps,
                    maximum_mean_kl=args.physics_max_update_kl,
                    freeze_base_actor=(args.actor == "holomotion"
                        and args.run_dir is not None
                        and not args.finetune_base_actor)),
            )
            trainer = create_k1_trainer(algorithm, task, physics=physics_enabled,
                teacher=args.physics_teacher,
                privileged_teacher_config=(
                    ({} if args.physics_teacher_config is None else
                     load_teacher_config(args.physics_teacher_config))
                    if args.physics_teacher and args.physics_teacher_architecture == 'transformer' else None),
                **physics_options)
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
        trainer.training_seed = args.seed
        from unified_motion.llm.validation import file_sha256
        motion_paths = (sorted(args.motion_dir.expanduser().rglob('*.npz'))
                        if args.motion_dir is not None else args.motion)
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
            "foot_reward": asdict(task.foot_reward_config),
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


def custom_actor_config(backend, path=None):
    values = {} if path is None else yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Actor config must be a YAML mapping")
    values = dict(values)
    selected = values.pop("backbone", backend)
    if ("transformer" if selected == "dense" else selected) != backend:
        raise ValueError("Actor config backbone disagrees with --actor")
    return K1ActorConfig(backbone=backend, **values)


def load_teacher_config(path):
    from unified_motion.privileged_teacher import PrivilegedTeacherConfig
    values = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(values, dict):
        raise ValueError('Transformer Teacher config must be a YAML mapping')
    return asdict(PrivilegedTeacherConfig(**values))


if __name__ == "__main__":
    main()

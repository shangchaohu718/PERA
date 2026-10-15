"""Finite training runner for the original HoloMotion algorithm on mjlab."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .holomotion_mjlab_ppo import HoloMotionMJLabPPO
from .llm.advisor import tuning_snapshot, start_training_trial


@dataclass(frozen=True)
class HoloMotionRunnerConfig:
    num_iterations: int
    steps_per_env: int = 32
    start_iteration: int = 0
    save_interval: int = 1000
    log_interval: int = 1
    progress_interval: int = 10
    random_start: bool = True
    teacher_only: bool = False

    def __post_init__(self) -> None:
        for name in (
            "num_iterations",
            "steps_per_env",
            "save_interval",
            "log_interval",
            "progress_interval",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.start_iteration < 0:
            raise ValueError("start_iteration must be non-negative")
        if type(self.teacher_only) is not bool:
            raise TypeError("teacher_only must be boolean")


class HoloMotionMJLabRunner:
    """Run a bounded number of PPO iterations and save compatible checkpoints."""

    def __init__(
        self,
        trainer: HoloMotionMJLabPPO,
        output_dir: str | Path,
        config: HoloMotionRunnerConfig,
        advisor=None,
    ) -> None:
        self.trainer = trainer
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.config = config
        self.advisor = advisor
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "metrics.jsonl"

    def _log(self, record: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            stream.write("\n")

    def run(self) -> list[dict[str, Any]]:
        cfg = self.config
        if not hasattr(self.trainer, 'training_trial'):
            start_training_trial(self.trainer)
        if self.trainer.training_trial['initial_settings'] != tuning_snapshot(self.trainer):
            raise ValueError('training settings changed after trial initialization')
        minimum_remaining = (
            cfg.steps_per_env * self.trainer.task.sampler.step_dt
            + float(self.trainer.task.sampler.offsets_s[-1])
        )
        self.trainer.reset(
            random_start=cfg.random_start,
            min_remaining_s=minimum_remaining,
        )
        records = []
        self.trainer.training_compute_seconds = 0.
        self.trainer.training_transitions = 0
        last_checkpoint_iteration = None
        stop = cfg.start_iteration + cfg.num_iterations
        run_started = time.perf_counter()
        wall_started = time.time()
        for iteration in range(cfg.start_iteration, stop):
            collection_start = time.perf_counter()
            rollout = self.trainer.collect(cfg.steps_per_env)
            collection_seconds = time.perf_counter() - collection_start
            update_start = time.perf_counter()
            metrics = None if cfg.teacher_only else self.trainer.update(rollout)
            ppo_update_seconds = 0.0 if cfg.teacher_only else time.perf_counter() - update_start
            physics_metrics = None
            physics_update_seconds = 0.0
            if rollout.physics_observation is not None:
                physics_start = time.perf_counter()
                physics_metrics = (self.trainer.update_teacher_only(rollout)
                                   if cfg.teacher_only else self.trainer.update_dynamics(rollout))
                physics_update_seconds = time.perf_counter() - physics_start
            update_seconds = ppo_update_seconds + physics_update_seconds
            self.trainer.training_compute_seconds += collection_seconds+update_seconds
            self.trainer.training_transitions += rollout.actions.shape[0]*rollout.actions.shape[1]
            clip_statistics = {}
            for clip_id, clip_name in enumerate(
                self.trainer.task.sampler.library.clip_names
            ):
                selected = rollout.clip_ids == clip_id
                count = int(selected.sum())
                if count == 0:
                    continue
                clip_statistics[clip_name] = {
                    "clip_id": clip_id,
                    "sampled_steps": count,
                    "reward_mean": float(rollout.rewards.squeeze(-1)[selected].mean()),
                    "done_count": int(rollout.dones.squeeze(-1)[selected].sum()),
                }
                if rollout.reward_terms is not None:
                    clip_statistics[clip_name]['tracking'] = {
                        name: float(value[selected].float().mean())
                        for name, value in rollout.reward_terms.items()
                        if name.endswith(('_error_m', '_error_rad', '_error_mps', '_error_radps',
                                          '_rmse_mps', '_rmse_radps')) or name.startswith(('left_foot_', 'right_foot_'))}
                    for term, label in [('physical_failure', 'physical_failures'),
                                        ('reference_completion', 'reference_completions')]:
                        if term in rollout.reward_terms:
                            clip_statistics[clip_name][label] = int(rollout.reward_terms[term][selected].sum())
            record = {
                "iteration": iteration,
                "reward_mean": float(rollout.rewards.mean()),
                "done_count": int(rollout.dones.sum()),
                "collection_seconds": collection_seconds,
                "update_seconds": update_seconds,
                "ppo_update_seconds": ppo_update_seconds,
                "physics_update_seconds": physics_update_seconds,
                "clips": clip_statistics,
                "training_tuning": tuning_snapshot(self.trainer),
                **({} if metrics is None else asdict(metrics)),
            }
            if rollout.physics_confidence is not None:
                record["physics_confidence_mean"] = float(
                    rollout.physics_confidence.mean()
                )
            if rollout.reward_terms is not None:
                for term, label in [('physical_failure', 'physical_failures'),
                                    ('reference_completion', 'reference_completions')]:
                    if term in rollout.reward_terms:
                        record[label] = int(rollout.reward_terms[term].sum())
                record["reward_terms"] = {
                    name: float(value.float().mean())
                    for name, value in rollout.reward_terms.items()
                }
            if physics_metrics is not None:
                record.update({
                    f"physics_{key}": value
                    for key, value in (physics_metrics.items() if isinstance(physics_metrics, dict)
                                       else asdict(physics_metrics).items())
                })
            records.append(record)
            if iteration % cfg.log_interval == 0:
                self._log(record)
            completed = iteration - cfg.start_iteration + 1
            if completed % cfg.progress_interval == 0 or iteration == stop - 1:
                elapsed = time.perf_counter() - run_started
                seconds_per_iteration = elapsed / completed
                remaining_seconds = seconds_per_iteration * (stop - iteration - 1)
                finish = time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(time.time() + remaining_seconds),
                )
                print(
                    f"[progress] {completed}/{cfg.num_iterations} "
                    f"iter={iteration} reward={record['reward_mean']:.4f} "
                    f"sec/iter={seconds_per_iteration:.2f} "
                    f"ETA={remaining_seconds / 60:.1f}min finish={finish}",
                    flush=True,
                )
            if self.advisor is not None:
                self.advisor.on_iteration(record, self.trainer)
            if iteration % cfg.save_interval == 0:
                self.trainer.save_checkpoint(
                    self.output_dir / f"model_{iteration}.pt",
                    iteration=iteration,
                    infos=record,
                )
                last_checkpoint_iteration = iteration

        final_iteration = stop - 1
        if final_iteration != last_checkpoint_iteration:
            self.trainer.save_checkpoint(
                self.output_dir / f"model_{final_iteration}.pt",
                iteration=final_iteration,
                infos=records[-1],
            )
        wall_finished = time.time()
        wall_seconds = wall_finished - wall_started
        summary = {
            "schema": 1,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(wall_started)),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(wall_finished)),
            "wall_seconds": wall_seconds,
            "compute_seconds": self.trainer.training_compute_seconds,
            "iterations": cfg.num_iterations,
            "start_iteration": cfg.start_iteration,
            "final_iteration": final_iteration,
            "transitions": self.trainer.training_transitions,
            "transitions_per_second": self.trainer.training_transitions / max(wall_seconds, 1e-12),
            "seconds_per_iteration": wall_seconds / cfg.num_iterations,
            "final_reward_mean": records[-1]["reward_mean"],
        }
        (self.output_dir / "training_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[complete] wall={wall_seconds / 60:.1f}min "
            f"throughput={summary['transitions_per_second']:.0f} transitions/s "
            f"final_reward={summary['final_reward_mean']:.4f}",
            flush=True,
        )
        return records

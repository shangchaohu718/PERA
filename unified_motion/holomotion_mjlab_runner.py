"""Finite training runner for the original HoloMotion algorithm on mjlab."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .holomotion_mjlab_ppo import HoloMotionMJLabPPO


@dataclass(frozen=True)
class HoloMotionRunnerConfig:
    num_iterations: int
    steps_per_env: int = 32
    start_iteration: int = 0
    save_interval: int = 1000
    log_interval: int = 1
    random_start: bool = True

    def __post_init__(self) -> None:
        for name in (
            "num_iterations",
            "steps_per_env",
            "save_interval",
            "log_interval",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.start_iteration < 0:
            raise ValueError("start_iteration must be non-negative")


class HoloMotionMJLabRunner:
    """Run a bounded number of PPO iterations and save compatible checkpoints."""

    def __init__(
        self,
        trainer: HoloMotionMJLabPPO,
        output_dir: str | Path,
        config: HoloMotionRunnerConfig,
    ) -> None:
        self.trainer = trainer
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.config = config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "metrics.jsonl"

    def _log(self, record: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            stream.write("\n")

    def run(self) -> list[dict[str, Any]]:
        cfg = self.config
        minimum_remaining = (
            cfg.steps_per_env * self.trainer.task.sampler.step_dt
            + float(self.trainer.task.sampler.offsets_s[-1])
        )
        self.trainer.reset(
            random_start=cfg.random_start,
            min_remaining_s=minimum_remaining,
        )
        records = []
        last_checkpoint_iteration = None
        stop = cfg.start_iteration + cfg.num_iterations
        for iteration in range(cfg.start_iteration, stop):
            collection_start = time.perf_counter()
            rollout = self.trainer.collect(cfg.steps_per_env)
            collection_seconds = time.perf_counter() - collection_start
            update_start = time.perf_counter()
            metrics = self.trainer.update(rollout)
            ppo_update_seconds = time.perf_counter() - update_start
            physics_metrics = None
            physics_update_seconds = 0.0
            if rollout.physics_observation is not None:
                physics_start = time.perf_counter()
                physics_metrics = self.trainer.update_dynamics(rollout)
                physics_update_seconds = time.perf_counter() - physics_start
            update_seconds = ppo_update_seconds + physics_update_seconds
            clip_statistics = {}
            for clip_id, clip_name in enumerate(
                self.trainer.task.sampler.library.clip_names
            ):
                selected = rollout.clip_ids == clip_id
                count = int(selected.sum())
                if count == 0:
                    continue
                clip_statistics[clip_name] = {
                    "sampled_steps": count,
                    "reward_mean": float(rollout.rewards.squeeze(-1)[selected].mean()),
                    "done_count": int(rollout.dones.squeeze(-1)[selected].sum()),
                }
            record = {
                "iteration": iteration,
                "reward_mean": float(rollout.rewards.mean()),
                "done_count": int(rollout.dones.sum()),
                "collection_seconds": collection_seconds,
                "update_seconds": update_seconds,
                "ppo_update_seconds": ppo_update_seconds,
                "physics_update_seconds": physics_update_seconds,
                "clips": clip_statistics,
                **asdict(metrics),
            }
            if rollout.physics_confidence is not None:
                record["physics_confidence_mean"] = float(
                    rollout.physics_confidence.mean()
                )
            if rollout.reward_terms is not None:
                record["reward_terms"] = {
                    name: float(value.float().mean())
                    for name, value in rollout.reward_terms.items()
                }
            if physics_metrics is not None:
                record.update({
                    f"physics_{key}": value
                    for key, value in asdict(physics_metrics).items()
                })
            records.append(record)
            if iteration % cfg.log_interval == 0:
                self._log(record)
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
        return records

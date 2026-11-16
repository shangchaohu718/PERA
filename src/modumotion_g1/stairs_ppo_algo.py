"""Pure PPO with an immutable public Actor and explicit update accounting."""

import json
import statistics
import time
from pathlib import Path

import torch
from holomotion.src.algo.ppo_tf import PPOTF

from modumotion_g1.domain_moe import (
    domain_parameters,
    public_digest,
    set_active_domains,
)
from modumotion_g1.frozen_feature_ppo import (
    FEATURE_KEY,
    FrozenFeaturePPOCache,
)
from modumotion_g1.train_algo import DomainTrainPPO


class StairsTrainPPO(DomainTrainPPO):
    def load(self, ckpt_path):
        result = super().load(ckpt_path)
        actor = self.accelerator.unwrap_model(self.actor)
        bias = float(self.config.get("fixed_merge_bias", -0.5))
        set_active_domains(actor, [self.domain_name], {self.domain_name: bias})
        self.feature_cache = None
        if self.config.get("frozen_feature_ppo", True):
            if self.rollout_forward_chunk_size:
                raise ValueError("Disable rollout chunking for feature PPO")
            self.feature_cache = FrozenFeaturePPOCache(self.domain_name)
            self.feature_cache.attach_actor(actor)
        self.sequence_parity_verified = False
        self.initial_steps = self.optimizer_steps()
        self.initial_iteration = int(self.current_learning_iteration)
        self.run_started = time.monotonic()
        self.last_public_check = self.initial_iteration - 1
        self.initial_domain = [p.detach().clone() for p in domain_parameters(actor)]
        self.initial_critic_parameter = (
            next(self.accelerator.unwrap_model(self.critic).parameters()).detach().clone()
        )
        return result

    def _build_storage(self, obs_td):
        if self.feature_cache is not None:
            obs_td = self.feature_cache.augment_observation(obs_td, template=True)
        return super()._build_storage(obs_td)

    def _build_transition(self, obs_td, actor_out, critic_out):
        if self.feature_cache is not None:
            obs_td = self.feature_cache.augment_observation(obs_td)
        return super()._build_transition(obs_td, actor_out, critic_out)

    def update(self):
        if self.feature_cache is not None and not self.sequence_parity_verified:
            actor = self.accelerator.unwrap_model(self.actor)
            batch = next(self._sequence_batches(self.num_mini_batches, 1))
            obs, actions = batch[:2]
            mask = batch[10]
            parameters = domain_parameters(actor)
            outputs = []
            gradients = []
            for observation in (obs.exclude(FEATURE_KEY), obs):
                output = actor(
                    observation,
                    actions=actions,
                    mode="sequence_logp",
                    attn_mask=mask,
                    update_obs_norm=False,
                )
                ratio = (output["actions_log_prob"] - batch[6]).exp()
                advantage = batch[3]
                loss = torch.maximum(
                    -advantage * ratio,
                    -advantage * ratio.clamp(1 - self.clip_param, 1 + self.clip_param),
                ).mean()
                gradients.append(torch.autograd.grad(loss, parameters))
                outputs.append(
                    {k: output[k].detach() for k in ("mu", "actions_log_prob", "entropy")}
                )
            for key in outputs[0]:
                torch.testing.assert_close(
                    outputs[0][key],
                    outputs[1][key],
                    atol=2e-4,
                    rtol=2e-4,
                )
            for full, cached in zip(*gradients, strict=True):
                torch.testing.assert_close(full, cached, atol=2e-5, rtol=2e-3)
            self.sequence_parity_verified = True
            (Path(self.log_dir) / "gpu_feature_parity.json").write_text(
                json.dumps(
                    {
                        "full_cached_mu_logp_entropy_match": True,
                        "ppo_objective_four_parameter_gradients_match": True,
                        "max_mu_error": float((outputs[0]["mu"] - outputs[1]["mu"]).abs().max()),
                    },
                    indent=2,
                )
                + "\n"
            )
        return super().update()

    def optimizer_steps(self):
        values = [
            int(state["step"].item())
            for state in self.actor_optimizer.state.values()
            if "step" in state
        ]
        if values and (len(values) != 4 or len(set(values)) != 1):
            raise RuntimeError(f"Inconsistent domain Adam steps: {values}")
        return values[0] if values else 0

    def _post_update_hook(self, loss_dict):
        # The native hook is empty. Check the optimizer/gradient boundary on
        # every iteration; copy/hash the 423M frozen weights periodically.
        PPOTF._post_update_hook(self, loss_dict)
        actor = self.accelerator.unwrap_model(self.actor)
        parameters = domain_parameters(actor)
        actual = {id(p) for group in self.actor_optimizer.param_groups for p in group["params"]}
        if actual != {id(p) for p in parameters}:
            raise RuntimeError("Unexpected Actor optimizer parameters")
        if any(
            p.grad is not None
            for name, p in actor.named_parameters()
            if ".domain_plugins." not in name
        ):
            raise RuntimeError("Public Actor received gradients")
        if not all(torch.isfinite(p).all() for p in parameters):
            raise RuntimeError("Nonfinite domain parameter")
        iteration = int(self.current_learning_iteration)
        interval = int(self.config.get("public_hash_interval", 50))
        checked = iteration == self.initial_iteration or (iteration + 1) % interval == 0
        if checked:
            if public_digest(actor) != actor.public_sha256:
                raise RuntimeError("Frozen public Actor changed")
            self.last_public_check = iteration
        metrics = self._last_update_metrics
        row = {
            "iteration": iteration,
            "iterations_this_run": iteration - self.initial_iteration + 1,
            "optimizer_steps_total": self.optimizer_steps(),
            "optimizer_steps_added": self.optimizer_steps() - self.initial_steps,
            "updates_this_iteration": int(metrics["0-Train/num_updates_executed"]),
            "environment_steps_this_run": (
                (iteration - self.initial_iteration + 1)
                * int(self.env.num_envs)
                * int(self.num_steps_per_env)
            ),
            "losses": {k: float(v) for k, v in loss_dict.items()},
            "kl_windowed": float(metrics["0-Train/kl_windowed"]),
            "kl_stopped": bool(metrics["0-Train/kl_stop_triggered"]),
            "mean_episode_return": statistics.fmean(self.rewbuffer) if self.rewbuffer else None,
            "mean_episode_length": statistics.fmean(self.lenbuffer) if self.lenbuffer else None,
            "actor_learning_rate": self.actor_optimizer.param_groups[0]["lr"],
            "fixed_merge_bias": actor.actor_module.layers[0].merge_bias[self.domain_name],
            "public_hash_checked": checked,
            "last_public_hash_check_iteration": self.last_public_check,
            "elapsed_s": time.monotonic() - self.run_started,
            "cuda_peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
            "cached_sequence_updates": (
                self.feature_cache.cached_sequence_count if self.feature_cache is not None else 0
            ),
        }
        target = Path(self.log_dir) / "progress.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(row, indent=2) + "\n")
        temporary.replace(target)
        with (Path(self.log_dir) / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if iteration % int(self.log_interval) == 0:
            print(json.dumps({"stairs_ppo": row}), flush=True)

    def save(self, path, infos=None):
        super().save(path, infos)
        state = torch.load(path, map_location="cpu", weights_only=True)
        state["actual_actor_optimizer_steps"] = self.optimizer_steps()
        state["fixed_merge_bias"] = float(self.config.get("fixed_merge_bias", -0.5))
        torch.save(state, path)

    def _post_training_hook(self):
        actor = self.accelerator.unwrap_model(self.actor)
        critic = self.accelerator.unwrap_model(self.critic)
        changed = [
            not torch.equal(old, new)
            for old, new in zip(self.initial_domain, domain_parameters(actor), strict=True)
        ]
        result = {
            "method": "PPO",
            "teacher_action_queries": 0,
            "public_sha256": actor.public_sha256,
            "public_unchanged": public_digest(actor) == actor.public_sha256,
            "domain_tensors_changed": changed,
            "critic_changed": not torch.equal(
                self.initial_critic_parameter, next(critic.parameters())
            ),
            "iterations_this_run": int(self.current_learning_iteration)
            - self.initial_iteration
            + 1,
            "optimizer_steps_total": self.optimizer_steps(),
            "optimizer_steps_added": self.optimizer_steps() - self.initial_steps,
            "final_checkpoint": f"model_{self.current_learning_iteration}.pt",
            "elapsed_s": time.monotonic() - self.run_started,
        }
        if not result["public_unchanged"] or not all(changed) or not result["critic_changed"]:
            raise RuntimeError(f"Training boundary validation failed: {result}")
        (Path(self.log_dir) / "result.json").write_text(json.dumps(result, indent=2) + "\n")

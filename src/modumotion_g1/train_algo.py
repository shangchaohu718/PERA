"""Single-process PPO for one independently trained G1 domain plugin."""

from pathlib import Path

import torch
from holomotion.src.algo.ppo_tf import PPOTF
from safetensors.torch import load_file

from modumotion_g1.domain_moe import (
    add_domain,
    domain_parameters,
    install_domains,
    load_domain,
    public_digest,
    save_domain,
    set_trainable_domain,
)


class DomainTrainPPO(PPOTF):
    def _setup_models_and_optimizer(self):
        if self.accelerator.num_processes != 1:
            raise NotImplementedError("Domain training is currently validated on one process/GPU")
        if self.config.get("checkpoint_load_mode") != "finetune":
            raise ValueError("Public checkpoint must be loaded in finetune mode")
        if self.config.schedule != "fixed":
            raise ValueError("Use a fixed learning rate for initial domain experiments")
        if self.config.get("export_policy", False):
            raise ValueError("Domain ONNX export has not been validated")
        super()._setup_models_and_optimizer()
        self.domain_name = str(self.config.domain_name)
        self._domain_ready = False

    def load(self, ckpt_path):
        if self._domain_ready:
            raise ValueError("Create a fresh trainer to load a different domain checkpoint")
        native_path = Path(self.config.public_checkpoint).resolve()
        resume = None
        if Path(ckpt_path).resolve() != native_path:
            resume = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if (
                resume.get("format") != "modumotion_g1_domain_ppo_v1"
                or resume["name"] != self.domain_name
            ):
                raise ValueError("Not a compatible domain PPO training checkpoint")
        result = super().load(str(native_path))
        actor = self.accelerator.unwrap_model(self.actor)
        install_domains(actor)
        if resume is None:
            add_domain(
                actor,
                self.domain_name,
                source_expert=int(self.config.get("source_expert", 0)),
                initial_bias=float(self.config.get("initial_domain_bias", -2.0)),
            )
        else:
            load_domain(actor, Path(ckpt_path).with_suffix(".domain.pt"))
            set_trainable_domain(actor, self.domain_name)
        old_optimizer = self.actor_optimizer
        optimizer_cls = getattr(torch.optim, self.optimizer_type)
        kwargs = self._build_optimizer_kwargs(optimizer_cls)
        params = domain_parameters(actor)
        optimizer = optimizer_cls(
            params,
            lr=self.actor_learning_rate,
            betas=(self.actor_beta1, self.actor_beta2),
            **kwargs,
        )
        self.actor_optimizer = self.accelerator.prepare_optimizer(optimizer)
        # The native optimizer was created before the public Actor was frozen.
        # Remove it from Accelerate's registered optimizer list as well.
        self.accelerator._optimizers = [
            o for o in self.accelerator._optimizers if o is not old_optimizer
        ]
        if resume is not None:
            if resume["public_sha256"] != actor.public_sha256:
                raise ValueError("PPO state belongs to another public base")
            critic = self.accelerator.unwrap_model(self.critic)
            critic.load_state_dict(
                load_file(
                    str(Path(ckpt_path).with_suffix(".critic.safetensors")),
                    device=str(self.device),
                ),
                strict=True,
            )
            self.actor_optimizer.load_state_dict(resume["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(resume["critic_optimizer_state_dict"])
            self.current_learning_iteration = int(resume["iter"]) + 1
        self._domain_ready = True
        return result

    def _post_update_hook(self, loss_dict):
        super()._post_update_hook(loss_dict)
        actor = self.accelerator.unwrap_model(self.actor)
        if public_digest(actor) != actor.public_sha256:
            raise RuntimeError("PPO update changed the frozen public Actor")
        actual = {id(p) for group in self.actor_optimizer.param_groups for p in group["params"]}
        if actual != {id(p) for p in domain_parameters(actor)}:
            raise RuntimeError("Actor optimizer contains unexpected parameters")
        if not all(torch.isfinite(p).all() for p in domain_parameters(actor)):
            raise RuntimeError("Domain parameters became nonfinite")

    def save(self, path, infos=None):
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        actor = self.accelerator.unwrap_model(self.actor)
        save_domain(
            actor,
            self.domain_name,
            path.with_suffix(".domain.pt"),
            {
                "training": "PPO",
                "iteration": int(self.current_learning_iteration),
            },
        )
        critic = self.accelerator.unwrap_model(self.critic)
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in critic.state_dict().items()},
            str(path.with_suffix(".critic.safetensors")),
        )
        torch.save(
            {
                "format": "modumotion_g1_domain_ppo_v1",
                "name": self.domain_name,
                "public_sha256": actor.public_sha256,
                "iter": int(self.current_learning_iteration),
                "actor_optimizer_state_dict": (self.actor_optimizer.state_dict()),
                "critic_optimizer_state_dict": (self.critic_optimizer.state_dict()),
                "infos": infos,
            },
            path,
        )

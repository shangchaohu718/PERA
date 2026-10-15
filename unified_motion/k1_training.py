"""Construct every Actor on the shared K1 task, PPO and optional Physics path."""
from pathlib import Path

from .k1_policy_backends import K1ActorConfig, custom_network_config, build_k1_networks
from .holomotion_original_algorithm import OriginalHoloMotionK1Algorithm
from .holomotion_mjlab_ppo import HoloMotionMJLabPPO, HoloMotionPPOConfig
from .holomotion_physics_training import HoloMotionPhysicsMJLabPPO
from .models.physics import DynamicsModel
from .physics_update import PhysicsUpdateConfig


def create_custom_k1_algorithm(config: K1ActorConfig, *, device="cpu"):
    network = custom_network_config(config)
    actor, critic = build_k1_networks(network, device=device)
    return OriginalHoloMotionK1Algorithm(actor, critic, None, Path("."), network)


def create_k1_trainer(algorithm, task, *, physics=False, teacher=False,
                      ppo_config=None, dynamics_config=None, update_config=None,
                      privileged_teacher_config=None, **physics_options):
    custom = (algorithm.network_config or {}).get("backend", "holomotion") != "holomotion"
    if custom and ppo_config is None:
        ppo_config = HoloMotionPPOConfig(aux_state_pred_enabled=False,
            inactive_expert_margin_enabled=False)
    if not physics:
        if teacher or physics_options or dynamics_config or update_config or privileged_teacher_config is not None:
            raise ValueError("Physics options require physics=True")
        return HoloMotionMJLabPPO(algorithm, task, config=ppo_config)
    config = dict(dynamics_config or {})
    config.setdefault('prediction_horizons', (1, 4))
    if privileged_teacher_config is not None and not teacher:
        raise ValueError('privileged Transformer config requires teacher=True')
    config.update(system_id_dim=task.physics_teacher_dim,
                  teacher_student=teacher and privileged_teacher_config is None, uncertainty=True)
    dynamics = DynamicsModel(task.bridge.robot, **config).to(task.device)
    # Missing provenance must never freeze a freshly initialized policy. The
    # safetensors loader marks restored weights; callers may override explicitly.
    update_config = update_config or PhysicsUpdateConfig(
        freeze_base_actor=not custom and getattr(algorithm, 'pretrained', False) is True)
    return HoloMotionPhysicsMJLabPPO(algorithm, task, dynamics, config=ppo_config,
                                   update_config=update_config,
                                   privileged_teacher_config=privileged_teacher_config, **physics_options)

import copy

import torch

from unified_motion.holomotion_mjlab_ppo import HoloMotionPPORollout
from unified_motion.holomotion_physics_training import HoloMotionPhysicsMJLabPPO
from unified_motion.holomotion_mjlab_task import HoloMotionFootRewardConfig
from unified_motion.models.physics import DynamicsModel, PhysicsObservation
from unified_motion.robot_contract import get_robot_contract


def _physics(batch=2, steps=3, *, exact_next=False):
    robot=get_robot_contract("k1_22dof")
    body=torch.zeros(batch,steps,6+3*robot.dof)
    body[...,5]=-1
    contact=torch.zeros(batch,steps,2,12)
    if exact_next:
        contact[...,0,10]=1
        contact[...,1,10]=torch.tensor([0,1,0],dtype=body.dtype)
        contact[...,0,9]=30
        contact[...,1,9]=10
    return PhysicsObservation(
        body,torch.zeros(batch,steps,3),contact,
        torch.ones(batch,steps,2,dtype=torch.bool),
        torch.zeros(batch,steps,0,16),torch.zeros(batch,steps,0,dtype=torch.bool),
    )


def _rollout(model):
    b,t,n=2,3,22
    starts=torch.zeros(b,t,dtype=torch.bool);starts[:,0]=True
    previous_valid=torch.zeros(b,t,dtype=torch.bool);previous_valid[:,1:]=True
    z=lambda *shape:torch.zeros(*shape)
    return HoloMotionPPORollout(
        z(b,t,506),z(b,t,843),torch.randn(b,t,n)*.01,z(b,t,1),z(b,t,n),
        torch.ones(b,t,n)*.2,z(b,t,1),z(b,t,1),z(b,t,1),
        torch.zeros(b,t,1,dtype=torch.bool),z(b,t,1),z(b,t,3),z(b,t,10),
        z(b,t,6,3),z(b,t,6,3),torch.zeros(b,t,dtype=torch.long),
        physics_features=z(b,t,3*model.physics_width+model.latent_width),
        physics_observation=_physics(b,t),next_physics_observation=_physics(b,t,exact_next=True),
        physics_episode_start=starts,physics_initial_hidden=z(b,model.memory_width),
        previous_predicted_core=z(b,t,9+2*n),previous_prediction_valid=previous_valid,
        physics_teacher_target=torch.randn(b,t,model.system_id_dim)*.1,
        dynamics_stamp=tuple(number for p in model.parameters() for number in (id(p),p._version)),
    )


def test_physics_update_trains_prediction_and_teacher_with_separate_feet():
    model=DynamicsModel(
        get_robot_contract("k1_22dof"),physics_width=8,memory_width=8,
        latent_width=4,hidden_dim=16,system_id_dim=2,teacher_student=True,
        uncertainty=True,
    )
    trainer=HoloMotionPhysicsMJLabPPO.__new__(HoloMotionPhysicsMJLabPPO)
    trainer.dynamics=model
    trainer.actor_dynamics=copy.deepcopy(model)
    trainer.dynamics_optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    trainer.system_identification_weight=.1
    trainer.teacher_reconstruction_weight=.1
    trainer.teacher_distillation_weight=.05
    trainer.dynamics_burn_in_steps=2
    trainer.dynamics_train_steps=4
    trainer.dynamics_ema_decay=.99
    trainer.dynamics_epochs=2
    trainer._dynamics_rollouts=[]
    trainer._observation=object();trainer._physics_memory=object();trainer._physics_episode_start=object()
    trainer._physics_age=object()
    # A 6-step physics window is deliberately assembled from two independent
    # 3-step PPO rollouts, while Dynamics weights remain frozen between them.
    assert trainer.update_dynamics(_rollout(model)) is None
    assert len(trainer._dynamics_rollouts)==1
    metrics=trainer.update_dynamics(_rollout(model))
    assert torch.isfinite(torch.tensor(metrics.total_loss))
    assert metrics.left_contact_loss>0 and metrics.right_contact_loss>0
    assert metrics.teacher_reconstruction_loss>0
    assert metrics.mean_system_id_std>0 and metrics.mean_state_std>0
    assert metrics.optimizer_steps==2
    assert model.teacher_encoder[0].weight.grad is not None
    assert trainer._observation is None and trainer._physics_memory is None
    assert trainer._physics_age is None


def test_left_and_right_foot_reward_weights_are_independent():
    cfg=HoloMotionFootRewardConfig(
        left_slip_weight=.1,right_slip_weight=.2,
        left_impact_weight=.3,right_impact_weight=.4,
    )
    assert cfg.left_slip_weight!=cfg.right_slip_weight
    assert cfg.left_impact_weight!=cfg.right_impact_weight

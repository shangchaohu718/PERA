"""CPU 数学/接口测评专用夹具：不是 mjlab，不是 K1 动力学的替代品。

人为线性转移用来抓取时间、重置、快照、PPO 接线错误；不能证明机器人会走。
"""
import torch
from unified_motion.robot_contract import get_robot_contract
from unified_motion.reference_observations import ReferenceObservation
from unified_motion.models.physics_dynamics import PhysicsObservation, ControlGoal, DynamicsModel, PhysicsMLPActorCritic
from unified_motion.models.physics_policy_v2 import PhysicsConditionedActorCritic, PhysicsPolicyV2Config
from unified_motion.algorithms.sequence_training import TaskTransition


class TensorTask:
    """提供确定性的测试转移，故意暴露可被原地修改的 observation。"""
    auto_reset = False
    def __init__(self, robot_id='k1_22dof', batch=2, contacts=2, objects=1, multimodal=False, endings=True):
        self.robot_id=robot_id; self.robot=get_robot_contract(robot_id)
        self.num_envs=batch; self.device=torch.device('cpu'); self.env_ids=tuple(range(batch)); self.step_dt=.02
        self.c,self.m=contacts,objects; self.multimodal=multimodal; self.endings=endings
        self.body=torch.zeros(batch,6+3*self.robot.dof)
        self.vel=torch.zeros(batch,3)
        self.contact=torch.zeros(batch,contacts,12)
        self.object=torch.zeros(batch,objects,16)
        self.contact_mask=torch.ones(batch,contacts,dtype=torch.bool)
        self.object_mask=torch.ones(batch,objects,dtype=torch.bool)
        self.ticks=torch.zeros(batch,dtype=torch.long)
        self._starts=torch.ones(batch,dtype=torch.bool)
        self.pending=torch.zeros(batch,dtype=torch.bool)
        self.reset_calls=0
        self.reset()
    @property
    def episode_start(self): return self._starts.clone()
    def reset(self, env_ids=None):
        ids=torch.arange(self.num_envs) if env_ids is None else env_ids
        self.body[ids]=0; self.body[ids,5]=-1
        self.vel[ids]=0; self.contact[ids]=0
        if self.c:
            self.contact[ids,:,8]=1; self.contact[ids,:,9]=1
            self.contact[ids,:,10:]=1
        self.object[ids]=0
        if self.m:
            self.object[ids,:,3]=1; self.object[ids,:,13:]=1
        self.ticks[ids]=0; self._starts[ids]=True; self.pending[ids]=False
        self.reset_calls+=1
    def read(self):
        return PhysicsObservation(self.body,self.vel,self.contact,self.contact_mask,self.object,self.object_mask)
    def goal(self):
        n=self.robot.dof; b=self.num_envs
        ref=torch.zeros(b,3,16+2*n); ref[:,:,0]=torch.tensor([0,.04,.1])
        ref[:,:,4]=1; ref[:,:,8]=1; ref[:,:,16]=.15
        mask=torch.ones(b,3,dtype=torch.bool); mask[:,-1]=False
        goal=ControlGoal(motion=ReferenceObservation(ref,mask))
        if self.multimodal:
            if self.c:
                goal.contact_features=torch.zeros(b,self.c,7)
                goal.contact_features[:,:,6]=1
                goal.contact_mask=torch.ones(b,self.c,dtype=torch.bool)
                goal.contact_mask[:,0]=False
            if self.m:
                goal.object_features=torch.zeros(b,self.m,6)
                goal.object_features[:,:,0]=.2
                goal.object_mask=torch.ones(b,self.m,dtype=torch.bool)
        return goal
    def step(self, action):
        if self.pending.any(): raise RuntimeError('reset required')
        n=self.robot.dof
        vel=self.body[:,6+n:6+2*n]
        vel.mul_(.93).add_(action*.02)
        self.body[:,6:6+n].add_(vel*.02)
        self.body[:,-n:]=action
        self.vel[:,0]=vel.mean(-1)
        self.ticks+=1; self._starts[:]=False
        if self.c:
            self.contact[:,:,3]=self.vel[:,0:1]
            self.contact[:,:,10]=(self.ticks[:,None]%3!=0).float()
            self.contact[:,:,9]=self.contact[:,:,10]
        if self.m: self.object[:,:,0]+=.001
        terminated=(self.ticks==5)&(torch.arange(self.num_envs)%2==0) if self.endings else torch.zeros(self.num_envs,dtype=torch.bool)
        truncated=(self.ticks==7)&~terminated if self.endings else torch.zeros_like(terminated)
        self.pending=terminated|truncated
        reward=1-(self.body[:,6:6+n]-.1).square().mean(-1)-.001*action.square().mean(-1)
        return TaskTransition(reward,terminated,truncated,{'tracking':reward.clone()})


def make_models(kind='dense', robot_id='k1_22dof', **task_kwargs):
    torch.manual_seed(71)
    task=TensorTask(robot_id,**task_kwargs)
    dyn=DynamicsModel(task.robot, physics_width=8,memory_width=8,latent_width=4,hidden_dim=16)
    if kind=='mlp':
        pol=PhysicsMLPActorCritic(task.robot,physics_width=8,latent_width=4,hidden_dim=16)
    else:
        cfg=PhysicsPolicyV2Config(backbone=kind,d_model=16,input_hidden_dim=24,
              n_layers=2,n_heads=4,n_kv_heads=2,context_len=3,dense_hidden_dim=24,
              num_fine_experts=4,top_k=2,expert_hidden_dim=16,router_hidden_dim=16,
              physics_width=8,latent_width=4,critic_hidden_dim=16)
        pol=PhysicsConditionedActorCritic({robot_id:task.robot},cfg)
    return task,dyn,pol

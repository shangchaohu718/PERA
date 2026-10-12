"""CPU解析诊断场景，不是K1/G1动力学、不替代MuJoCo，不用于报告机器人成功率。

一个受控质点与水平移动平面接触，具有质量、粘性阻力、外力和单侧地面约束。
其它关节列仅用于检查网络输入/输出尺寸。文件用途是无需CUDA验证完整训练链路。
"""
from __future__ import annotations
import torch
from ..robot_contract import get_robot_contract
from ..models.physics import PhysicsObservation
from ..algorithms.sequence_training import TaskTransition
from ..tasks.interaction import ContactPair, RelationState, PhysicalRewardState


class DiagnosticBackend:
    """输入机器人接口ID和环境数；输出供PlanTask使用的解析诊断后端。

    robot_id只控制张量宽度，绝不表示这里模拟了该机器人。
    """
    backend_kind = "analytic_diagnostic_not_humanoid"

    def __init__(self, robot_id="k1_22dof", num_envs=2, step_dt=.02, device="cpu"):
        self.robot_id = robot_id; self.robot = get_robot_contract(robot_id)
        self.num_envs = num_envs; self.step_dt = step_dt; self.device = torch.device(device)
        self.env_ids = tuple(range(num_envs))
        self.pairs = (ContactPair("probe_surface", "diagnostic_probe", "surface", "surface"),)
        # 与read()的object_features槽位严格对应，供可选GNN核对关系而不是猜索引。
        self.object_names = ("surface",)
        self.pos = torch.zeros(num_envs, 3, device=self.device)
        self.vel = torch.zeros_like(self.pos)
        self.surface = torch.zeros_like(self.pos)
        self.force = torch.zeros_like(self.pos)
        self.q = torch.zeros(num_envs, self.robot.dof, device=self.device)
        self.qd = torch.zeros_like(self.q); self.action = torch.zeros_like(self.q)
        self.identity = torch.zeros(num_envs, 4, device=self.device); self.identity[:, 0] = 1
        self.params = {"mass_scale": torch.ones(num_envs,device=self.device),
                       "friction": torch.ones(num_envs,device=self.device)*.5,
                       "support_speed_m_s": torch.zeros(num_envs,device=self.device)}
        self.reset()

    def reset(self, env_ids=None):
        """只重置指定诊断环境的状态；下一步参数由课程重新写入。"""
        i = slice(None) if env_ids is None else env_ids
        self.pos[i]=0; self.pos[i,2]=.01
        self.vel[i]=0; self.surface[i]=0; self.force[i]=0
        self.q[i]=0; self.qd[i]=0; self.action[i]=0

    def set_parameters(self, env_ids, values):
        """真正写进解析运动方程的参数，并读回；不认识的参数直接报错。"""
        if set(values)-set(self.params): raise ValueError("诊断后端不支持这些物理参数")
        for k,v in values.items(): self.params[k][env_ids] = v
        return {k: self.params[k][env_ids].clone() for k in values}

    def apply_force(self, force_w):
        """输入[B,3]外力（牛顿），下一控制步进入F=ma；每次调用覆盖上次外力。"""
        self.force.copy_(force_w)

    def relations(self):
        """读取实际诊断质点相对平面的运动，不根据目标伪造接触。"""
        rel = self.vel.clone(); rel[:,0] -= self.params["support_speed_m_s"]
        found=(self.pos[:,2:3] <= .010001)
        normal_force=found.to(self.pos.dtype)*self.params["mass_scale"][:,None]*9.81
        return RelationState(torch.zeros_like(self.pos), self.identity,
                             self.pos[:,None].clone(), rel[:,None],
                             self.surface[:,None].clone(), self.identity[:,None],
                             found,normal_force)

    def read(self):
        """打包网络需要的尺寸；这是数学测试信号，不是机器人真实传感器数据。"""
        b,n = self.num_envs,self.robot.dof
        obs = self.q.new_zeros(b,6+3*n); obs[:,5]=-1
        obs[:,6:6+n]=self.q; obs[:,6+n:6+2*n]=self.qd; obs[:,-n:]=self.action
        r=self.relations()
        c=self.q.new_zeros(b,1,12); c[...,:3]=r.point_pos; c[...,3:6]=r.relative_velocity
        c[...,8]=1; c[...,10]=r.found; c[...,11]=1
        c[...,9]=r.found.to(c.dtype)*self.params["mass_scale"][:,None]*9.81
        obj=self.q.new_zeros(b,1,16); obj[...,:3]=self.surface[:,None]; obj[...,3]=1
        obj[...,7]=self.params["support_speed_m_s"][:,None]; obj[...,13:]=1
        mask=torch.ones(b,1,dtype=torch.bool,device=self.device)
        return PhysicsObservation(obs,torch.zeros_like(self.pos),c,mask,obj,mask.clone())

    def reward_state(self):
        """诊断后端只提供形状正确的解析量，不伪装成人形机器人碰撞。"""
        limits=torch.stack((torch.full_like(self.q,-1.),torch.full_like(self.q,1.)),-1)
        return PhysicalRewardState(torch.zeros_like(self.pos),self.pos.clone(),self.q.clone(),
                                   limits,torch.zeros(self.num_envs,dtype=torch.bool,device=self.device))

    def step(self, action):
        """执行半隐式质点积分和单侧平面约束；返回诊断状态，计划负责目标奖励。"""
        self.action.copy_(action)
        mass=self.params["mass_scale"][:,None]
        mu=self.params["friction"][:,None]
        for _ in range(4):
            dt=self.step_dt/4
            rel=self.vel.clone(); rel[:,0]-=self.params["support_speed_m_s"]
            acceleration=(action[:,:3]*8+self.force-mu*rel*4)/mass
            acceleration[:,2]-=9.81
            self.vel.add_(acceleration*dt); self.pos.add_(self.vel*dt)
            hit=self.pos[:,2]<.01
            self.pos[hit,2]=.01; self.vel[hit,2]=self.vel[hit,2].clamp_min(0)
            self.surface[:,0]+=self.params["support_speed_m_s"]*dt
            self.qd.add_((action-2*self.qd-self.q)*dt); self.q.add_(self.qd*dt)
        failed=(self.pos.abs()>5).any(-1)
        return TaskTransition(self.q.new_zeros(self.num_envs),failed,torch.zeros_like(failed))

    def close(self):
        """解析后端没有外部资源；与真实后端保持统一关闭接口。"""
        return None

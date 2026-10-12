"""真实mjlab接触关系适配，不依赖滑步标签，不用诊断场景替代真实后端。

复用R18 K1的执行器与关节合同。新增任意已注册body—对象body接触关系。
当前工厂只加载官方K1及地面；附加物体必须由调用者在cfg.scene.entities显式注册。
包含源码级接线，但本交付环境没有mjlab/CUDA，实际物理兼容性必须另行验收。
"""
from __future__ import annotations
import re
import torch
from ..algorithms.sequence_training import TaskTransition
from ..models.physics import PhysicsObservation
from ..tasks.interaction import (ContactPair, RelationState, PhysicalRewardState, rotate,
                                 relative_point_velocity)
from .mjlab_physics import _contact_primary_names, _quat_relative


def add_relation_sensors(cfg, pairs):
    """输入环境配置及ContactPair列表；输出原配置，追加每条关系的独立真实传感器。

    一个传感器只匹配一个robot body和一个对象body。maxforce只是一处代表接触力，非总力。
    必须在创建ManagerBasedRlEnv之前调用。不是按脚高度猜测接触。
    """
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    sensors=list(cfg.scene.sensors)
    for i,p in enumerate(pairs):
        secondary=(ContactMatch(mode="body",pattern="terrain") if p.object=="terrain"
                   else ContactMatch(mode="body",pattern=re.escape(p.object_body),entity=p.object))
        sensors.append(ContactSensorCfg(name=f"interaction_pair_{i}",
            primary=ContactMatch(mode="body",pattern=re.escape(p.body),entity="robot"),
            secondary=secondary, secondary_policy="error",
            fields=("found","force","pos","normal","tangent"),
            reduce="maxforce",global_frame=True,num_slots=1))
        # maxforce用于接触几何；承重必须使用整条接触关系的净力，不能拿单点最大力代替。
        sensors.append(ContactSensorCfg(name=f"interaction_pair_load_{i}",
            primary=ContactMatch(mode="body",pattern=re.escape(p.body),entity="robot"),
            secondary=secondary, secondary_policy="error", fields=("found","force"),
            reduce="netforce",num_slots=1))
    foot_bodies=tuple(dict.fromkeys(p.body for p in pairs if p.object=="terrain"))
    if foot_bodies:
        sensors.append(ContactSensorCfg(name="interaction_undesired_ground",
            primary=ContactMatch(mode="body",pattern=r".*",entity="robot",exclude=foot_bodies),
            secondary=ContactMatch(mode="body",pattern="terrain"),secondary_policy="error",
            fields=("found",),reduce="maxforce",num_slots=1))
    if len({s.name for s in sensors})!=len(sensors): raise ValueError("接触传感器重名")
    cfg.scene.sensors=tuple(sensors)
    return cfg


class MJLabInteractionBackend:
    """输入已创建的K1MJLabBridge、固定关系和显式对象尺寸，输出通用任务后端。

    质量随机化当前缩放机器人根部件质量和惯量；不是随机缩放全部关节或假定真实负载位置。
    力矩/速度曲线仍沿用R18名义执行器。参数写入后读取实际tensor核对。
    """
    backend_kind="mjlab_k1_real"

    def __init__(self, bridge, pairs, *, object_sizes=None):
        self.bridge=bridge; self.env=bridge.env; self.entity=bridge.entity
        self.pairs=tuple(pairs); self.object_sizes=dict(object_sizes or {})
        self.object_names=tuple(self.object_sizes)  # 顺序就是read()输出的物体槽位顺序。
        for k in ("robot_id","robot","num_envs","device","env_ids"):
            setattr(self,k,getattr(bridge,k))
        self.step_dt=self.env.step_dt
        self.sensors=[]; self.load_sensors=[]; self.body_ids=[]; self.counterparts=[]
        for i,p in enumerate(self.pairs):
            if p.body not in self.entity.body_names: raise ValueError(f"未知机器人body:{p.body}")
            self.body_ids.append(self.entity.body_names.index(p.body))
            sensor=self.env.scene[f"interaction_pair_{i}"]
            if len(_contact_primary_names(sensor))!=1 or sensor.cfg.reduce!="maxforce" or not sensor.cfg.global_frame:
                raise ValueError("每条关系需要唯一primary的世界系maxforce传感器")
            self.sensors.append(sensor)
            load_sensor=self.env.scene[f"interaction_pair_load_{i}"]
            if len(_contact_primary_names(load_sensor))!=1 or load_sensor.cfg.reduce!="netforce":
                raise ValueError("每条关系需要唯一primary的netforce承重传感器")
            self.load_sensors.append(load_sensor)
            if p.object=="terrain": self.counterparts.append(None)
            else:
                e=self.env.scene[p.object]
                self.counterparts.append((e,e.body_names.index(p.object_body)))
                if p.object not in self.object_sizes:
                    raise ValueError("外部对象必须显式配置size_m，不能推测模型尺寸")
        self._prepared=False
        self.undesired_sensor=(self.env.scene["interaction_undesired_ground"]
                               if any(p.object=="terrain" for p in self.pairs) else None)

    def _body_state(self):
        """读取两侧刚体运动和body固定测量点；输出供关系观测/目标转换共用的快照。"""
        d=self.entity.data
        pose=d.body_link_pose_w[:,self.body_ids]; vel=d.body_link_vel_w[:,self.body_ids]
        offsets=pose.new_tensor([p.point_local_m for p in self.pairs])
        point=pose[...,:3]+rotate(pose[...,3:7],offsets.expand(self.num_envs,-1,-1))
        cp=torch.zeros_like(pose); cp[...,3]=1
        cv=torch.zeros_like(vel)
        for j,entry in enumerate(self.counterparts):
            if entry is None:
                # mjlab 会把并行环境平铺到不同世界坐标原点。terrain 本身没有
                # Entity data，但任务中的地面局部目标必须以各环境原点为基准，
                # 不能让所有机器人都追世界原点附近的同一双脚目标。
                origins=self.env.scene.env_origins
                if origins.shape != (self.num_envs,3) or origins.device != pose.device:
                    raise ValueError("mjlab scene.env_origins 与并行环境不匹配")
                cp[:,j,:3]=origins.to(dtype=pose.dtype)
            else:
                e,k=entry; cp[:,j]=e.data.body_link_pose_w[:,k]; cv[:,j]=e.data.body_link_vel_w[:,k]
        rel=relative_point_velocity(point,pose[...,:3],vel[...,:3],vel[...,3:],
                                    cp[...,:3],cv[...,:3],cv[...,3:])
        found=torch.stack([s.data.found[:,0]>0 for s in self.sensors],-1)
        loads=[]
        for sensor,load_sensor in zip(self.sensors,self.load_sensors):
            data=sensor.data; load=load_sensor.data
            if data.normal is None or load.force is None:
                raise ValueError("接触传感器缺少真实净力/法线")
            normal=data.normal[:,0]
            normal=normal/normal.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            loads.append((load.force[:,0]*normal).sum(-1).abs())
        normal_force=torch.stack(loads,-1)
        normal_force=torch.where(found,normal_force,0.)
        return RelationState(d.root_link_pos_w.clone(),d.root_link_quat_w.clone(),point,rel,
                             cp[...,:3],cp[...,3:],found,normal_force)

    def relations(self):
        """输出世界系关系测量；不改变仿真时钟。"""
        return self._body_state()

    def read(self):
        """读取[B,...]真实身体、关系和外部物体；受力值来自真实传感器，非参考数据。"""
        r=self.relations(); d=self.entity.data; b=self.num_envs
        obs=self.bridge.observation().clone()
        c=obs.new_zeros(b,len(self.pairs),12)
        c[...,:3]=rotate(r.root_quat,r.point_pos-r.root_pos[:,None],True)
        c[...,3:6]=rotate(r.root_quat,r.relative_velocity,True)
        for j,sensor in enumerate(self.sensors):
            data=sensor.data
            if data.normal is None or data.force is None: raise ValueError("接触传感器缺少真实力/法线")
            normal=data.normal[:,0]
            normal=normal/normal.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            c[:,j,6:9]=torch.where(r.found[:,j,None],rotate(r.root_quat,normal,True),0.)
            c[:,j,9]=torch.where(r.found[:,j],(data.force[:,0]*normal).sum(-1).abs(),0.)
            c[:,j,10]=r.found[:,j]; c[:,j,11]=1
        objects=[]
        for name,size in self.object_sizes.items():
            other=self.env.scene[name].data
            objects.append(torch.cat((rotate(r.root_quat,other.root_link_pos_w-r.root_pos,True),
                _quat_relative(r.root_quat,other.root_link_quat_w),
                rotate(r.root_quat,other.root_link_vel_w[:,:3]-d.root_link_vel_w[:,:3],True),
                rotate(r.root_quat,other.root_link_vel_w[:,3:]-d.root_link_vel_w[:,3:],True),
                obs.new_tensor(size).expand(b,3)),-1))
        of=torch.stack(objects,1) if objects else obs.new_zeros(b,0,16)
        return PhysicsObservation(obs,d.root_link_lin_vel_b.clone(),c,
            torch.ones(b,len(self.pairs),dtype=torch.bool,device=self.device),of,
            torch.ones(b,len(objects),dtype=torch.bool,device=self.device))

    def reward_state(self):
        """读取奖励专用真值；这些量不会经过PhysicsObservation进入可部署Actor。"""
        d=self.entity.data
        root_id=int(self.entity.indexing.root_body_id)
        com=d.data.subtree_com[:,root_id].clone()
        undesired=(torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
                   if self.undesired_sensor is None else
                   (self.undesired_sensor.data.found>0).any(-1))
        return PhysicalRewardState(d.root_link_ang_vel_b.clone(),com,d.joint_pos.clone(),
                                   d.soft_joint_pos_limits.clone(),undesired)

    def reset(self, env_ids=None):
        """显式重置选中的环境；不吞掉最终观测。"""
        self.bridge.reset(env_ids)

    def step(self, action):
        """策略动作按已有合同映射到PD目标；本函数不修改原始action概率。"""
        tr=self.bridge.step(action)
        return TaskTransition(torch.zeros(self.num_envs,device=self.device),tr.terminated,tr.truncated)

    def _prepare_parameters(self):
        """为各环境展开摩擦/质量数组和派生常数，只做一次；不同环境不能共享随机参数。"""
        if self._prepared: return
        from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel
        def marker(): pass
        marked=requires_model_fields("geom_friction","body_mass","body_inertia",
                                      recompute=RecomputeLevel.set_const)(marker)
        self.env.sim.expand_model_fields(tuple(marked.model_fields))
        self._base_mass=self.env.sim.get_default_field("body_mass").clone()
        self._base_inertia=self.env.sim.get_default_field("body_inertia").clone()
        self._root_id=int(self.entity.indexing.root_body_id)
        self._prepared=True

    def set_parameters(self, env_ids, values):
        """把摩擦、根部件质量真正写入mjlab模型并读回。

        非零support_speed当前需另接带驱动的移动平台，默认K1平面后端明确拒绝，不伪装支持。
        参数变化后重算质量相关常数，再forward刷新可观察状态。
        """
        if set(values)-{"friction","mass_scale","support_speed_m_s"}:
            raise ValueError("mjlab后端收到不支持的参数")
        if "support_speed_m_s" in values and values["support_speed_m_s"].abs().max().item()>0:
            raise NotImplementedError("默认K1平面场景没有移动平台驱动；不能把非零速度写进观测冒充物理运动")
        self._prepare_parameters(); sim=self.env.sim; result={}
        if "friction" in values:
            # 对场景全部geom写相同滑动摩擦，避免接触双方混合规则掩盖变化。
            f=sim.model.geom_friction
            f[env_ids,:,0]=values["friction"][:,None]
            result["friction"]=f[env_ids,0,0].clone()
        if "mass_scale" in values:
            from mjlab.managers.event_manager import RecomputeLevel
            k=self._root_id
            sim.model.body_mass[env_ids,k]=self._base_mass[k]*values["mass_scale"]
            sim.model.body_inertia[env_ids,k]=self._base_inertia[k]*values["mass_scale"][:,None]
            sim.recompute_constants(RecomputeLevel.set_const)
            result["mass_scale"]=(sim.model.body_mass[env_ids,k]/self._base_mass[k]).clone()
        if "support_speed_m_s" in values:
            result["support_speed_m_s"]=torch.zeros_like(values["support_speed_m_s"])
        sim.forward()
        return result

    def apply_force(self, force_w):
        """输入根部件世界外力[B,3]；零力会清除上一次推力，避免意外持续施力。"""
        self.entity.data.write_external_wrench(force=force_w[:,None,:],
                                              torque=torch.zeros_like(force_w[:,None,:]),body_ids=[0])

    def close(self):
        """释放实际mjlab资源。"""
        self.env.close()

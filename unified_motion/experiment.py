"""YAML研究入口：配置→课程任务→独立动力学→MLP/Dense/MoE→训练或冻结评测。

运行示例：
    python -m unified_motion.experiment --config configs/diagnostic.yaml --mode train
    python -m unified_motion.experiment --config configs/diagnostic.yaml --mode evaluate \
        --checkpoint runs/diagnostic/checkpoint.pt

默认diagnostic是解析数学场景，不是机器人；真实mjlab必须显式选择，不会自动降级。
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from .api import (DynamicsModel, build_policy, PPOConfig, PhysicsSequenceCollector,
                  PhysicsSequencePPO, DynamicsUpdater)
from .models.physics import PhysicsMLPActorCritic
from .tasks.interaction import (ContactPair, PlanTarget, PlanStage, ContactPlanTask,
                                ContactRewardConfig, DomainScheduler, RandomizedTask)


class _UniqueLoader(yaml.SafeLoader):
    """只读取普通YAML数据，同时拒绝重复键，避免前面的值被静默覆盖。"""


def _mapping(loader, node, deep=False):
    """输入YAML映射节点；输出唯一键字典，不执行文件中的Python代码。"""
    result={}
    for k,v in node.value:
        key=loader.construct_object(k,deep=deep)
        if key in result: raise ValueError(f"YAML存在重复键：{key}")
        result[key]=loader.construct_object(v,deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,_mapping)


def load_config(path):
    """读取并检查YAML，拼错参数报错而不是悄悄使用默认值；返回普通字典。"""
    config=yaml.load(Path(path).read_text(encoding="utf-8"),Loader=_UniqueLoader)
    if not isinstance(config,dict): raise ValueError("YAML顶层必须是字典")
    allowed={"schema","seed","backend","output","device","robot_id","num_envs","step_dt",
             "model","dynamics","ppo","run","domains","train_domains","evaluation_domains",
             "curriculum","push_interval_steps","plan","evaluation_plan","contacts","asset_root",
             "object_sizes","adaptation","task_reward"}
    unknown=set(config)-allowed
    if unknown: raise ValueError(f"未知配置项：{sorted(unknown)}")
    if config.get("schema")!=1: raise ValueError("YAML schema必须是1")
    if config.get("backend") not in ("diagnostic","mjlab_contact"):
        raise ValueError("backend必须显式写diagnostic或mjlab_contact")
    for key in ("num_envs","seed"):
        if type(config.get(key)) is not int or config[key] < (1 if key=="num_envs" else 0):
            raise ValueError(f"{key}必须为合法整数")
    if config["backend"] == "mjlab_contact" and config.get("robot_id") != "k1_22dof":
        raise ValueError("当前真实场景工厂仅适配K1，不能把ID改成G1就冒充G1环境")
    dt=config.get("step_dt")
    if isinstance(dt,bool) or not isinstance(dt,(float,int)) or not 0<dt<=.1:
        raise ValueError("step_dt必须在(0,0.1]秒")
    checks={
        "model": {"backbone","d_model","hidden_dim","n_layers","context_len","num_experts","init_std",
                  "recurrent_hidden_dim","max_replay_steps",
                  "lru_state_dim","lru_r_min","lru_r_max","lru_max_phase","lru_scan"},
        "dynamics": {"physics_width","memory_width","latent_width","hidden_dim","interaction_feedback",
                     "scene_encoder","graph_layers","contact_object_indices"},
        "run": {"updates","horizon","dynamics_every","dynamics_learning_rate","evaluation_episodes",
                "evaluation_max_steps"},
        "curriculum": {"level","promote_threshold","window"},
    }
    for key,names in checks.items():
        values=config.get(key,{})
        if not isinstance(values,dict) or set(values)-names:
            raise ValueError(f"{key}存在未知字段")
    run=config["run"]
    for key in ("updates","horizon","dynamics_every","evaluation_episodes","evaluation_max_steps"):
        if type(run.get(key)) is not int or run[key]<1: raise ValueError(f"run.{key}必须正整数")
    model = config.get("model", {})
    kind = model.get("backbone", "mlp")
    if kind not in ("mlp", "dense", "transformer", "moe", "gru", "lstm", "lru"):
        raise ValueError("model.backbone必须是mlp/transformer/dense/moe/gru/lstm/lru")
    memory_width = model.get("recurrent_hidden_dim")
    if memory_width is not None and (kind not in ("gru", "lstm") or
                                     type(memory_width) is not int or memory_width < 1):
        raise ValueError("recurrent_hidden_dim只用于GRU/LSTM，且必须是正整数")
    if "max_replay_steps" in model and (
            type(model["max_replay_steps"]) is not int or model["max_replay_steps"] < 1):
        raise ValueError("max_replay_steps必须是正整数")
    from .models.lru import validate_lru_settings
    lru_keys = {"lru_state_dim", "lru_r_min", "lru_r_max", "lru_max_phase", "lru_scan"}
    if kind != "lru" and lru_keys & model.keys():
        raise ValueError("lru_*字段只用于LRU，其他主干不接受无效配置")
    if kind == "lru":
        validate_lru_settings(model.get("lru_state_dim"), model.get("lru_r_min", .9),
            model.get("lru_r_max", .999), model.get("lru_max_phase", 2*math.pi),
            model.get("lru_scan", "parallel"))
    dyn = config.get("dynamics", {})
    scene_encoder = dyn.get("scene_encoder", "set")
    if scene_encoder not in ("set", "gnn"):
        raise ValueError("dynamics.scene_encoder必须是set或gnn")
    if type(dyn.get("graph_layers", 2)) is not int or dyn.get("graph_layers", 2) < 1:
        raise ValueError("graph_layers必须是正整数")
    mapping = dyn.get("contact_object_indices")
    if scene_encoder == "gnn":
        if not isinstance(mapping, list) or any(type(i) is not int or i < -1 for i in mapping):
            raise ValueError("GNN必须显式给出contact_object_indices整数列表，-1表示静态世界")
    elif mapping is not None:
        raise ValueError("set编码器不会使用contact_object_indices，请删除这项无效配置")
    PPOConfig(**config.get("ppo",{}))
    reward_cfg=config.get("task_reward",{})
    if not isinstance(reward_cfg,dict):
        raise ValueError("task_reward必须是字典")
    ContactRewardConfig(**reward_cfg)
    adaptation=config.get("adaptation",{})
    allowed_adaptation={"privileged_critic","system_identification_weight","actor_observation",
                        "actor_dynamics",
                        "teacher_distillation_weight","teacher_reconstruction_weight"}
    if not isinstance(adaptation,dict) or set(adaptation)-allowed_adaptation:
        raise ValueError("adaptation存在未知字段")
    privileged_critic=adaptation.get("privileged_critic",False)
    if type(privileged_critic) is not bool:
        raise ValueError("adaptation.privileged_critic必须是bool")
    actor_observation=adaptation.get("actor_observation","full")
    if actor_observation not in ("full","deployable"):
        raise ValueError("adaptation.actor_observation必须是full或deployable")
    actor_dynamics=adaptation.get("actor_dynamics","online")
    if actor_dynamics not in ("none","online","oracle"):
        raise ValueError("adaptation.actor_dynamics必须是none、online或oracle")
    if actor_dynamics=="oracle" and not privileged_critic:
        raise ValueError("oracle Actor需要同时开启privileged_critic")
    system_id_weight=adaptation.get("system_identification_weight",0.0)
    if (isinstance(system_id_weight,bool) or not isinstance(system_id_weight,(float,int))
            or not math.isfinite(system_id_weight) or system_id_weight<0):
        raise ValueError("adaptation.system_identification_weight必须是非负有限数")
    if system_id_weight and not privileged_critic:
        raise ValueError("系统辨识当前要求同时开启privileged_critic，以共用严格隔离的真值通道")
    teacher_distill=adaptation.get("teacher_distillation_weight",0.0)
    teacher_reconstruct=adaptation.get("teacher_reconstruction_weight",0.0)
    for name,value in (("teacher_distillation_weight",teacher_distill),
                       ("teacher_reconstruction_weight",teacher_reconstruct)):
        if (isinstance(value,bool) or not isinstance(value,(float,int))
                or not math.isfinite(value) or value<0):
            raise ValueError(f"adaptation.{name}必须是非负有限数")
    if bool(teacher_distill) != bool(teacher_reconstruct):
        raise ValueError("Teacher蒸馏与重建权重必须同时开启或关闭")
    if teacher_distill and not system_id_weight:
        raise ValueError("Teacher–Student需要先开启system_identification_weight")
    _scheduler(config)
    for key in ("plan","evaluation_plan"):
        if key in config: _stages(config[key])
    return config


def _scheduler(config):
    """创建可复现课程；只读取配置，不修改环境。"""
    return DomainScheduler(config["domains"],config["train_domains"],config["evaluation_domains"],
                           seed=config["seed"],**config.get("curriculum",{}))


def _stages(raw):
    """把YAML中的阶段和目标构造成有校验的数据类，不自动生成新接触计划。"""
    if not isinstance(raw,list) or not raw: raise ValueError("plan必须为非空阶段列表")
    stages=[]
    for row in raw:
        row=dict(row)
        targets=tuple(PlanTarget(**t) for t in row.pop("targets"))
        stages.append(PlanStage(targets=targets,**row))
    return tuple(stages)


def build_task(config, *, evaluation=False):
    """创建指定后端的任务。真实后端缺CUDA/资产/依赖时直接失败，绝不换诊断场景。"""
    backend_name=config["backend"]
    if backend_name=="diagnostic":
        from .envs.diagnostic_interaction import DiagnosticBackend
        backend=DiagnosticBackend(config["robot_id"],config["num_envs"],config["step_dt"],config["device"])
    else:
        if not str(config["device"]).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("真实mjlab需要CUDA；未启动物理环境，也没有降级为诊断后端")
        from mjlab.envs import ManagerBasedRlEnv
        from .envs.mjlab_k1 import K1MJLabBridge, make_k1_mjlab_cfg
        from .envs.mjlab_interaction import add_relation_sensors, MJLabInteractionBackend
        pairs=tuple(ContactPair(**p) for p in config["contacts"])
        cfg=make_k1_mjlab_cfg(str(Path(config["asset_root"]).expanduser()),
                              num_envs=config["num_envs"],physics_dt=config["step_dt"]/10,decimation=10)
        cfg.seed=config["seed"]
        cfg=add_relation_sensors(cfg,pairs)
        env=ManagerBasedRlEnv(cfg,device=config["device"])
        try:
            backend=MJLabInteractionBackend(K1MJLabBridge(env),pairs,
                                            object_sizes=config.get("object_sizes"))
        except Exception:
            env.close();raise
    try:
        plan=config.get("evaluation_plan",config["plan"]) if evaluation else config["plan"]
        task=ContactPlanTask(backend,_stages(plan),ContactRewardConfig(**config.get("task_reward",{})))
        adaptation=config.get("adaptation",{})
        return RandomizedTask(task,_scheduler(config),evaluation=evaluation,
                               push_interval_steps=config.get("push_interval_steps",0),
                               privileged_learning=adaptation.get("privileged_critic",False),
                               actor_observation=adaptation.get("actor_observation","full"),
                               actor_dynamics=adaptation.get("actor_dynamics","online"))
    except Exception:
        backend.close();raise


def model_digest(*models):
    """计算参数和buffer内容指纹；GRU/KV运行时记忆不在state_dict中，允许它们随观测变化。"""
    h=hashlib.sha256()
    for index,model in enumerate(models):
        for key,value in sorted(model.state_dict().items()):
            h.update(f"{index}/{key}".encode())
            if isinstance(value,torch.Tensor):
                t=value.detach().cpu().contiguous()
                h.update(str((t.dtype,tuple(t.shape))).encode())
                h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
            else: h.update(json.dumps(value,sort_keys=True,default=str).encode())
    return h.hexdigest()


def build_models(config, task):
    """同一组任务和物理接口创建Dynamics与三种策略之一，宽度由YAML统一传递。"""
    dyn_cfg = dict(config.get("dynamics", {}))
    if dyn_cfg.get("scene_encoder", "set") == "gnn":
        # 验证YAML里的编号真的对应环境提供的对象；不能只验证编号没越界。
        names = tuple(getattr(task.backend, "object_names", ()))
        expected = []
        for pair in task.backend.pairs:
            if pair.object in names:
                expected.append(names.index(pair.object))
            elif pair.object == "terrain":
                expected.append(-1)
            else:
                raise ValueError(f"GNN关系对象{pair.object!r}没有对应的可观察物体槽位")
        if tuple(dyn_cfg["contact_object_indices"]) != tuple(expected):
            raise ValueError(f"GNN关系映射与后端不符，实际应为{expected}，对象顺序为{names}")
    adaptation=config.get("adaptation",{})
    privileged_dim=task.privileged_dim if adaptation.get("privileged_critic",False) else 0
    if adaptation.get("system_identification_weight",0):
        dyn_cfg["system_id_dim"]=privileged_dim
    if adaptation.get("teacher_distillation_weight",0):
        dyn_cfg["teacher_student"]=True
    # 两套网络使用隔离的CPU随机流。否则Teacher额外层会消耗随机数，导致同一种子
    # 消融中的Actor初始权重不同，结构效果与初始化噪声混在一起。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"]+1001)
        dyn=DynamicsModel(task.robot,**dyn_cfg).to(task.device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"]+2001)
        policy=build_policy({task.robot_id:task.robot},physics_width=dyn.physics_width,
                            latent_width=dyn.latent_width,
                            privileged_dims=({task.robot_id:privileged_dim} if privileged_dim else None),
                            **config.get("model",{})).to(task.device)
    return dyn,policy


def _seed(seed):
    """固定Python/NumPy/Torch随机数，不宣称不同GPU内核逐位完全一致。"""
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def run_training(config, checkpoint=None):
    """完整训练入口：采样→PPO→独立动力学监督→按需重建记忆→保存日志和权重。

    输入：已校验YAML字典、可选本程序checkpoint。
    输出：checkpoint.pt、metrics.jsonl、resolved_config.yaml和split_report.json。
    resume恢复模型/优化器/课程，但重新初始化物理状态，不声称逐步无缝续跑。
    """
    _seed(config["seed"])
    task=build_task(config)
    try:
        dyn,policy=build_models(config,task)
        ppo=PhysicsSequencePPO(policy,PPOConfig(**config.get("ppo",{})))
        updater=DynamicsUpdater(dyn,learning_rate=config["run"].get("dynamics_learning_rate",1e-4),
            system_id_weight=config.get("adaptation",{}).get("system_identification_weight",0.0),
            teacher_distillation_weight=config.get("adaptation",{}).get("teacher_distillation_weight",0.0),
            teacher_reconstruction_weight=config.get("adaptation",{}).get("teacher_reconstruction_weight",0.0))
        collector=PhysicsSequenceCollector(task,dyn,policy)
        output=Path(config["output"]);output.mkdir(parents=True,exist_ok=True)
        start=0
        if checkpoint:
            saved=torch.load(checkpoint,map_location=task.device,weights_only=True)
            if saved.get("experiment_schema")!=1: raise ValueError("不是本入口的checkpoint")
            if saved["config"]!=config: raise ValueError("resume的配置与checkpoint不同，拒绝静默改任务")
            ppo.load_state_dict(saved["ppo"]);updater.load_state_dict(saved["dynamics"])
            start=saved["iteration"]
            task.scheduler.level=saved["course_level"]
            task.scheduler.generator.set_state(saved["course_rng"].cpu())
            task.scheduler.history=list(saved["course_history"])
            torch.set_rng_state(saved["torch_rng"].cpu())
        (output/"resolved_config.yaml").write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding="utf-8")
        (output/"split_report.json").write_text(json.dumps(task.scheduler.split_report(),indent=2),encoding="utf-8")
        run=config["run"]
        with (output/"metrics.jsonl").open("a",encoding="utf-8") as log:
            for i in range(start+1,start+run["updates"]+1):
                rollout=collector.collect(run["horizon"])
                stats=ppo.update(rollout)
                ds=None
                if i%run["dynamics_every"]==0:
                    ds=updater.update(rollout);collector.on_dynamics_updated()
                rollout_terms={key:value.float().mean().item()
                               for key,value in rollout.reward_terms.items()}
                rollout_terms.update(
                    termination_rate=rollout.terminated.float().mean().item(),
                    truncation_rate=rollout.truncated.float().mean().item())
                row={"iteration":i,"backend":task.backend_kind,
                     "reward_mean":rollout.reward.mean().item(),"policy_std":rollout.old_std.mean().item(),
                     "course_level":task.scheduler.level,"ppo":asdict(stats),
                     "dynamics":None if ds is None else asdict(ds),
                     "rollout":rollout_terms}
                encoded=json.dumps(row,ensure_ascii=False,allow_nan=False)
                print(encoded,flush=True);log.write(encoded+"\n");log.flush()
        saved={"experiment_schema":1,"config":config,"iteration":i,
               "ppo":ppo.state_dict(),"dynamics":updater.state_dict(),
               "course_level":task.scheduler.level,"course_rng":task.scheduler.generator.get_state(),
               "course_history":task.scheduler.history,"torch_rng":torch.get_rng_state()}
        tmp=output/"checkpoint.tmp";torch.save(saved,tmp);tmp.replace(output/"checkpoint.pt")
        return output/"checkpoint.pt"
    finally: task.backend.close()


@torch.no_grad()
def evaluate_frozen(task, dynamics, policy, *, episodes=2, max_steps=1000):
    """冻结权重评测：每一步检查参数/buffer指纹，允许GRU/KV按新观测改变。

    输入：评测域的任务、已经训练/加载的两个模型、每环境回合数、步数预算。
    输出：成功率、位置/相对速度误差、回合完成情况和权重指纹。
    没在预算内跑完指定回合时，completed=False；不把未结束环境算成功。
    同一个有限测试集的通过，不等于通用零样本技能证明。
    """
    if not getattr(task,"evaluation",False): raise ValueError("冻结评测必须使用evaluation域任务")
    if type(episodes) is not int or episodes<1 or type(max_steps) is not int or max_steps<1:
        raise ValueError("评测回合数和步数必须正整数")
    policy.eval();dynamics.eval();before=model_digest(policy,dynamics)
    task.reset();memory=None;cache=None
    count=torch.zeros(task.num_envs,dtype=torch.long,device=task.device)
    successes=0;pe_sum=0.;ve_sum=0.;samples=0
    stage_sum=0.;stage_samples=0;maximum_stage_index_observed=0
    push_interval=int(getattr(task,"push_interval_steps",0))
    recovering=torch.zeros(task.num_envs,dtype=torch.bool,device=task.device)
    recovery_age=torch.zeros(task.num_envs,dtype=torch.long,device=task.device)
    pushes_observed=0;pushes_recovered=0;recovery_steps_sum=0
    for tick in range(max_steps):
        frame=task.read();starts=task.episode_start.clone()
        encoded,memory=dynamics.step(frame,episode_start=starts,env_ids=task.env_ids,memory=memory)
        goal=task.goal()
        privileged=task.privileged() if getattr(task,"privileged_dim",0) else None
        actor_encoded=(task.policy_dynamics(encoded,privileged)
                       if hasattr(task,"policy_dynamics") else encoded)
        if isinstance(policy,PhysicsMLPActorCritic):
            action=policy(frame,actor_encoded,goal,privileged=privileged).mean
        else:
            _,action,_,cache=policy.step(task.robot_id,frame,actor_encoded,goal,
                    env_ids=task.env_ids,episode_start=starts,cache=cache,
                    privileged=privileged,deterministic=True)
        pred=dynamics.predict_next(frame,action,encoded.latent)
        memory=dynamics.remember_prediction(memory,pred)
        active=count<episodes
        push_due=(torch.zeros_like(active) if not push_interval else
                  ((task.clock+1) % push_interval == 0) & active)
        tr=task.step(action)
        metrics=task.last_metrics
        if active.any().item():
            pe_sum+=metrics["position_error_m"][active].sum().item()
            ve_sum+=metrics["relative_speed_error_m_s"][active].sum().item()
            samples+=int(active.sum())
            if "stage" in metrics:
                active_stage=metrics["stage"][active]
                stage_sum+=active_stage.sum().item();stage_samples+=int(active_stage.numel())
                if active_stage.numel():
                    maximum_stage_index_observed=max(maximum_stage_index_observed,
                                                     int(active_stage.max().item()))
        if push_interval and "within_tolerance" in metrics:
            pushes_observed+=int(push_due.sum())
            recovering=(recovering|push_due)&active
            recovery_age+=recovering.long()
            restored=recovering&metrics["within_tolerance"]&active
            pushes_recovered+=int(restored.sum())
            recovery_steps_sum+=int(recovery_age[restored].sum())
            recovering[restored]=False;recovery_age[restored]=0
        done=tr.terminated|tr.truncated
        counted=done&active
        successes+=int(metrics["success"][counted].sum())
        count+=counted.long()
        recovering[done]=False;recovery_age[done]=0
        if model_digest(policy,dynamics)!=before:
            raise RuntimeError("冻结评测中参数或buffer发生变化，结果无效")
        if (count>=episodes).all().item(): break
        if done.any().item(): task.reset(done.nonzero(as_tuple=False).flatten())
    completed=int(count.sum())
    result={"backend":task.backend_kind,"completed":bool((count>=episodes).all()),
            "episodes":completed,"successes":successes,
            "success_rate":successes/max(completed,1),"mean_position_error_m":pe_sum/max(samples,1),
            "mean_relative_speed_error_m_s":ve_sum/max(samples,1),"control_steps":tick+1,
            "mean_stage_index":stage_sum/max(stage_samples,1),
            "maximum_stage_index_observed":maximum_stage_index_observed,
            "weights_unchanged":True,"weight_sha256":before,
            "holdout":task.scheduler.split_report(),
            "robot_skill_verified":False}
    if push_interval:
        result.update(push_interval_steps=push_interval,pushes_observed=pushes_observed,
                      pushes_recovered=pushes_recovered,
                      push_recovery_rate=pushes_recovered/max(pushes_observed,1),
                      mean_push_recovery_steps=recovery_steps_sum/max(pushes_recovered,1))
    return result


def run_evaluation(config, checkpoint):
    """从checkpoint加载模型，只在独立评测域运行；保存报告，不创建任何优化器。"""
    _seed(config["seed"])
    task=build_task(config,evaluation=True)
    try:
        dyn,policy=build_models(config,task)
        saved=torch.load(checkpoint,map_location=task.device,weights_only=True)
        if saved.get("experiment_schema")!=1: raise ValueError("checkpoint schema不匹配")
        # 不允许为了通过测试更换本应锁定的模型或物理域。
        if saved["config"]!=config: raise ValueError("评测配置必须与训练时预先登记的配置一致")
        policy.load_state_dict(saved["ppo"]["model"],strict=True)
        dyn.load_state_dict(saved["dynamics"]["model"],strict=True)
        result=evaluate_frozen(task,dyn,policy,episodes=config["run"]["evaluation_episodes"],
                                max_steps=config["run"]["evaluation_max_steps"])
        output=Path(config["output"])/"frozen_evaluation.json"
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        if not result["completed"]: raise RuntimeError("评测预算不足，指定回合没有全部跑完")
        return result
    finally: task.backend.close()


def describe_models(config):
    """在CPU创建配置中的网络，统计真实参数数量，不启动物理环境。

    输入：已校验的YAML字典。输出：可序列化的参数报告。
    数量不等于速度：没有GPU计时，不能据此宣称哪种结构一定训练更快。
    """
    from .api import get_robot_contract
    _seed(config["seed"])
    robot = get_robot_contract(config["robot_id"])
    dyn = DynamicsModel(robot, **config.get("dynamics", {}))
    policy = build_policy({config["robot_id"]: robot}, physics_width=dyn.physics_width,
                         latent_width=dyn.latent_width, **config.get("model", {}))
    count = lambda module: sum(p.numel() for p in module.parameters())
    graph = dyn.encoder.graph
    result = dict(backbone=config.get("model", {}).get("backbone", "mlp"),
                  scene_encoder=dyn.encoder.scene_encoder, policy_parameters=count(policy),
                  dynamics_parameters=count(dyn), graph_parameters=0 if graph is None else count(graph),
                  total_parameters=count(policy)+count(dyn),
                  note="参数统计，不是速度/显存/动作质量评测")
    from .models.moe import ReferenceRoutedMoE
    result["canonical_backbone"] = "dense" if result["backbone"] == "transformer" else result["backbone"]
    result["moe_layer_count"] = sum(isinstance(m, ReferenceRoutedMoE) for m in policy.modules())
    if result["backbone"] == "lru":
        cfg = policy.cfg
        result["actor_memory_floats_per_env"] = 2*cfg.n_layers*(cfg.lru_state_dim or cfg.d_model)
        result["lru_scan"] = cfg.lru_scan
        result["replay_policy"] = "真实回合历史重放；并行扫描不等于可以截断历史"
    elif result["backbone"] in ("gru", "lstm"):
        cfg = policy.cfg
        result["actor_memory_floats_per_env"] = cfg.n_layers*(cfg.recurrent_hidden_dim or cfg.d_model)*(2 if cfg.backbone=="lstm" else 1)
        result["replay_policy"] = "真实回合历史重放；上限不是隐式截断窗口"
    return result


def main():
    """终端入口；导入模块不会自动训练。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--mode",choices=("train","evaluate","validate","inspect"),default="train")
    parser.add_argument("--checkpoint",type=Path)
    args=parser.parse_args();config=load_config(args.config)
    if args.mode=="validate":
        print(json.dumps({"configuration":"PASS","split":_scheduler(config).split_report()},indent=2))
    elif args.mode=="inspect": print(json.dumps(describe_models(config), ensure_ascii=False, indent=2))
    elif args.mode=="train": run_training(config,args.checkpoint)
    elif args.checkpoint is None: parser.error("evaluate必须给--checkpoint，不用随机网络冒充训练模型")
    else: run_evaluation(config,args.checkpoint)


if __name__=="__main__": main()

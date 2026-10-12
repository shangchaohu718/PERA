"""唯一正式训练命令：公开动作文件 → K1 mjlab → 物理策略 PPO + 独立动力学。

使用 python -m unified_motion.train --help 查看参数。
不会偷偷在 CPU 上用假物理替代 mjlab；CPU 集成测评独立放在 tests/。
此入口目前只创建 K1 环境。G1 网络接口通过 CPU 测试，不代表 G1 实体环境已适配。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import random

import numpy as np
import torch

from . import __version__
from .api import (
    build_policy, get_robot_contract, MotionLibrary, MotionSampler,
    MotionTrackingReward, MotionTrackingRewardConfig, DynamicsModel, PPOConfig, MotionTrackingTask,
    PhysicsSequenceCollector, PhysicsSequencePPO, DynamicsUpdater,
)
from .algorithms.common import learning_rate_scale


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """读取终端设置；输入 argv 可用于统一命令行测试，None 表示系统参数。

    输出：Namespace，包含资产/动作路径、策略种类、环境数、训练预算和保存位置。
    这里只解析与验证参数，不启动 GPU，不导入 mjlab。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', type=Path, required=True, help='booster_assets 仓库根目录')
    parser.add_argument('--motions', type=Path, required=True,
                        help='项目标准NPZ目录，或单个带名字的HoloMotion/MJLab NPZ')
    parser.add_argument('--backbone', choices=('mlp','transformer','dense','moe','gru','lstm','lru'), default='mlp')
    parser.add_argument('--device', default='cuda:0', help='mjlab 使用的 NVIDIA GPU')
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--horizon', type=int, default=32, help='每轮采样控制步数')
    parser.add_argument('--updates', type=int, default=100, help='本次追加多少轮采样/优化')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--entropy-coef', type=float, default=0.005)
    parser.add_argument('--minibatch-envs', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--lr-schedule', choices=('constant','linear','cosine'), default='cosine')
    parser.add_argument('--lr-min-ratio', type=float, default=0.1)
    parser.add_argument('--balance-coef', type=float, default=0.01,
                        help='MoE专家负载均衡权重；非MoE主干该项恒为0')
    parser.add_argument('--dynamics-lr', type=float, default=1e-4)
    parser.add_argument('--dynamics-every', type=int, default=1, help='每多少轮更新一次动力学；更新后重新开始环境和记忆')
    parser.add_argument('--dynamics-epochs', type=int, default=4,
                        help='每批真实动力学序列重复优化次数')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--latent-width', type=int, default=64,
                        help='Dynamics在线隐变量宽度；旧checkpoint默认64，v2建议先测试16')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--context-len', type=int, default=32, help='仅Transformer的注意力窗口')
    parser.add_argument('--recurrent-hidden-dim', type=int, default=None, help='GRU/LSTM Actor记忆宽度')
    parser.add_argument('--max-replay-steps', type=int, default=4096, help='循环Actor完整历史重放的保护上限')
    parser.add_argument('--experts', type=int, default=4)
    parser.add_argument('--init-std', type=float, default=1.)
    parser.add_argument('--adaptation', choices=('full','none'), default='full',
                        help='full启用可部署Actor、特权Critic、系统辨识和Teacher–Student')
    parser.add_argument('--reference-state-init', action=argparse.BooleanOptionalAction, default=True,
                        help='reset时从参考当前帧初始化根/关节状态；正式动作模仿默认开启')
    parser.add_argument('--random-reference-start', action=argparse.BooleanOptionalAction, default=True,
                        help='训练reset时随机采样参考阶段；冻结评测应关闭')
    parser.add_argument('--system-id-weight', type=float, default=.1)
    parser.add_argument('--teacher-distillation-weight', type=float, default=.05)
    parser.add_argument('--teacher-reconstruction-weight', type=float, default=.1)
    parser.add_argument('--prediction-horizons', type=int, nargs='+', default=[1],
                        help='Dynamics直接预测跨度，必须以1开头递增；例如 1 4 8')
    parser.add_argument('--multi-step-weight', type=float, default=0.0)
    parser.add_argument('--latent-slow-weight', type=float, default=0.0)
    parser.add_argument('--extended-contact-supervision', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Teacher/System-ID额外监督切向力、滑移速度和摩擦利用率')
    parser.add_argument('--body-pos-weight', type=float, default=1.0)
    parser.add_argument('--body-rot-weight', type=float, default=.5)
    parser.add_argument('--body-lin-vel-weight', type=float, default=.2)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--output', type=Path, default=Path('runs/k1_physics'))
    parser.add_argument('--resume', type=Path, help='本项目 checkpoint；恢复权重/优化器，不恢复物理场景')
    parser.add_argument('--allow-local-mjlab-stack', action='store_true',
                        help='允许使用版本不在审计白名单、但所有模块可导入的本机MJLab栈；仍会记录实际版本')
    parser.add_argument('--model-check-values', action=argparse.BooleanOptionalAction, default=True,
                        help='模型内部逐层NaN/Inf检查；关闭可减少GPU同步，训练器关键检查仍保留')
    args = parser.parse_args(argv)
    for name in ('num_envs','horizon','updates','epochs','minibatch_envs','dynamics_every','dynamics_epochs',
                 'd_model','hidden_dim','latent_width','layers','context_len','experts','save_every'):
        if getattr(args,name) <= 0:
            parser.error(f'--{name.replace("_","-")} 必须大于0')
    for name in ('system_id_weight','teacher_distillation_weight','teacher_reconstruction_weight',
                 'multi_step_weight','latent_slow_weight','entropy_coef','balance_coef',
                 'body_pos_weight','body_rot_weight','body_lin_vel_weight'):
        value=getattr(args,name)
        if not np.isfinite(value) or value < 0:
            parser.error(f'--{name.replace("_","-")} 必须是非负有限数')
    if not np.isfinite(args.lr_min_ratio) or not 0 < args.lr_min_ratio <= 1:
        parser.error('--lr-min-ratio必须在(0,1]内')
    if args.adaptation=='full' and (not args.system_id_weight or
            bool(args.teacher_distillation_weight) != bool(args.teacher_reconstruction_weight)):
        parser.error('full adaptation要求非零system-id，且Teacher蒸馏/重建须同时开启')
    horizons=tuple(args.prediction_horizons)
    if (not horizons or horizons[0]!=1 or tuple(sorted(set(horizons)))!=horizons
            or any(value<1 for value in horizons)):
        parser.error('--prediction-horizons必须是以1开头的严格递增正整数')
    if max(horizons)>=args.horizon:
        parser.error('最大prediction horizon必须小于rollout horizon')
    if bool(args.multi_step_weight) != (len(horizons)>1):
        parser.error('多步prediction horizons和非零multi-step-weight必须同时设置')
    if args.extended_contact_supervision and args.adaptation!='full':
        parser.error('扩展接触监督要求full adaptation')
    if not args.device.startswith('cuda'):
        parser.error('此命令是实际 mjlab 入口，不用假 CPU 环境替代 GPU 物理仿真')
    return args


def _save_checkpoint(path: Path, ppo: PhysicsSequencePPO,
                     dynamics: DynamicsUpdater, iteration: int, args: argparse.Namespace) -> None:
    """先写临时文件再更名，减少中断时损坏已有 checkpoint 的风险。

    输入：输出路径、两个优化器、当前轮次和配置；输出：磁盘 checkpoint。
    只保存权重、优化器、计数和随机数，不保存 mjlab 场景、接触状态或运行时缓存。
    """
    payload = dict(schema=1, project_version=__version__, iteration=iteration,
        ppo=ppo.state_dict(), dynamics=dynamics.state_dict(),
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    """构造真实环境并顺序运行采样、策略优化、动力学优化。

    输入：parse_args 输出。输出：JSONL 训练日志和 checkpoint；同时输出进度统计。
    有任何概率复算或数值错误立即报错，不通过跳过坏数据来掩盖问题。
    """
    if not torch.cuda.is_available():
        raise RuntimeError('未检测到可用 CUDA；CPU 回归使用 python tools/evaluate.py。未启动物理训练。')
    # 启动之前核对真实依赖与资产。这里只审计，不下载、不替换本地软件。
    # 即使本检查通过，也必须另外用 verify_physics.py 验证编译和物理步进。
    from .envs.physics_validation import dependency_report, audit_k1_assets
    dependencies = dependency_report()
    if dependencies["status"] != "PASS":
        packages=dependencies.get("packages",{})
        importable=packages and all(x.get("installed") and x.get("importable") for x in packages.values())
        if not args.allow_local_mjlab_stack or not importable:
            raise RuntimeError(f"仿真依赖预检未通过：{dependencies['reason']}")
        print(f"警告：使用白名单外但可导入的本机MJLab栈：{dependencies['reason']}",flush=True)
    assets = audit_k1_assets(args.asset_root)
    if assets["status"] != "PASS":
        raise RuntimeError(f"真实资产不完整：{assets.get('missing_files', [])}")
    # 依赖缺失要直说，不能换成一个玩具仿真器让训练看起来成功。
    from mjlab.envs import ManagerBasedRlEnv
    from .envs.mjlab_k1 import K1MJLabBridge
    from .envs.mjlab_physics import K1PhysicsBridge, make_k1_physics_cfg
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device=torch.device(args.device)
    robot=get_robot_contract('k1_22dof')
    library=(MotionLibrary.from_named_mjlab_file(args.motions,robot_id='k1_22dof',device=device)
             if args.motions.is_file() else
             MotionLibrary.from_project_folder(args.motions,robot_id='k1_22dof',device=device))
    cfg=make_k1_physics_cfg(str(args.asset_root.expanduser()),num_envs=args.num_envs)
    cfg.seed=args.seed
    cfg.episode_length_s=max(float(cfg.episode_length_s),
                             float(library.clip_durations_s.max().item())+.02)
    args.output.mkdir(parents=True,exist_ok=True)
    versions={name:importlib.metadata.version(name) for name in
              ('torch','numpy','mjlab','mujoco','mujoco-warp','warp-lang')}
    # 把当次的 XML、mesh、Kp/Kd 与零动作裁剪写进运行目录，防止同名资产悄悄变化。
    (args.output/'asset_manifest.json').write_text(
        json.dumps(assets,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if assets['zero_action_changes']:
        print('注意：默认姿态与零动作执行目标不完全相同，详见 asset_manifest.json',flush=True)
    (args.output/'environment.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    (args.output/'dependency_report.json').write_text(
        json.dumps(dependencies,ensure_ascii=False,indent=2),encoding='utf-8')
    env=ManagerBasedRlEnv(cfg,device=str(device))
    try:
        bridge=K1MJLabBridge(env)
        physics_bridge=K1PhysicsBridge(bridge)
        sampler=MotionSampler(library,num_envs=args.num_envs,step_dt=env.step_dt,
                              future_offsets_s=(0.,.04,.10,.20))
        reward_cfg=MotionTrackingRewardConfig(body_pos_weight=args.body_pos_weight,
            body_rot_weight=args.body_rot_weight,body_lin_vel_weight=args.body_lin_vel_weight)
        full_adaptation=args.adaptation=='full'
        task=MotionTrackingTask(physics_bridge,sampler,MotionTrackingReward(robot,reward_cfg),
            actor_observation='deployable' if full_adaptation else 'full',
            privileged_learning=full_adaptation,
            extended_contact_supervision=args.extended_contact_supervision,
            reference_state_init=args.reference_state_init,
            random_reference_start=args.random_reference_start,
            minimum_reference_remaining_s=min(1.28,
                float(library.clip_durations_s.min().item())))
        dynamics=DynamicsModel(robot,latent_width=args.latent_width,
                               system_id_dim=task.privileged_dim if full_adaptation else 0,
                               teacher_student=full_adaptation,
                               prediction_horizons=tuple(args.prediction_horizons),
                               check_values=args.model_check_values).to(device)
        policy=build_policy({'k1_22dof':robot},backbone=args.backbone,d_model=args.d_model,
            hidden_dim=args.hidden_dim,n_layers=args.layers,context_len=args.context_len,
            num_experts=args.experts,init_std=args.init_std,
            latent_width=args.latent_width,
            recurrent_hidden_dim=args.recurrent_hidden_dim,max_replay_steps=args.max_replay_steps,
            privileged_dims=({'k1_22dof':task.privileged_dim} if full_adaptation else None),
            check_values=args.model_check_values).to(device)
        ppo=PhysicsSequencePPO(policy,PPOConfig(epochs=args.epochs,minibatch_envs=args.minibatch_envs,
                                              learning_rate=args.learning_rate,
                                              entropy_coef=args.entropy_coef,
                                              balance_coef=args.balance_coef))
        dyn_optimizer=DynamicsUpdater(dynamics,learning_rate=args.dynamics_lr,
            system_id_weight=args.system_id_weight if full_adaptation else 0.,
            teacher_distillation_weight=args.teacher_distillation_weight if full_adaptation else 0.,
            teacher_reconstruction_weight=args.teacher_reconstruction_weight if full_adaptation else 0.,
            multi_step_weight=args.multi_step_weight,
            latent_slow_weight=args.latent_slow_weight,
            epochs=args.dynamics_epochs)
        collector=PhysicsSequenceCollector(task,dynamics,policy)
        start=0
        if args.resume:
            ckpt=torch.load(args.resume,map_location=device,weights_only=True)
            if ckpt.get('schema')!=1:
                raise ValueError('不是此训练入口的 checkpoint')
            ppo.load_state_dict(ckpt['ppo']);dyn_optimizer.load_state_dict(ckpt['dynamics'])
            start=int(ckpt['iteration'])
            torch.set_rng_state(ckpt['torch_rng'].cpu())
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt['cuda_rng']])
        # 每次启动都建立新的物理场景；resume 不声称逐步复现被中断的环境。
        with (args.output/'metrics.jsonl').open('a',encoding='utf-8') as log:
            for iteration in range(start+1,start+args.updates+1):
                progress=(iteration-start-1)/max(args.updates-1,1)
                lr_scale=learning_rate_scale(progress,schedule=args.lr_schedule,
                                             min_ratio=args.lr_min_ratio)
                policy_lr=ppo.set_learning_rate_scale(lr_scale)
                dynamics_lr=dyn_optimizer.set_learning_rate_scale(lr_scale)
                rollout=collector.collect(args.horizon)
                stats=ppo.update(rollout)
                dyn_stats=None
                if iteration % args.dynamics_every==0:
                    dyn_stats=dyn_optimizer.update(rollout)
                    collector.on_dynamics_updated()
                reward_terms={k:v.float().mean().item() for k,v in rollout.reward_terms.items()}
                row=dict(iteration=iteration,reward_mean=rollout.reward.mean().item(),
                         policy_std=rollout.old_std.mean().item(),ppo=asdict(stats),
                         dynamics=None if dyn_stats is None else asdict(dyn_stats),
                         rollout=reward_terms,policy_learning_rate=policy_lr,
                         dynamics_learning_rate=dynamics_lr,
                         learning_rate_scale=lr_scale)
                encoded=json.dumps(row,ensure_ascii=False,allow_nan=False)
                log.write(encoded+'\n');log.flush();print(encoded,flush=True)
                if iteration % args.save_every==0 or iteration==start+args.updates:
                    _save_checkpoint(args.output/f'checkpoint_{iteration:06d}.pt',ppo,dyn_optimizer,iteration,args)
    finally:
        env.close()


def main() -> None:
    """终端入口；读取参数并启动 run。导入本文件不会自动训练。"""
    run(parse_args())


if __name__=='__main__':
    main()

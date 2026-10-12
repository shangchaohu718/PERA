"""R19集中回归：真实接触后端另行验收；这里只测数学、学习链路和防泄漏约定。"""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
import pytest
import yaml

from unified_motion.experiment import (load_config, build_task, build_models,
    evaluate_frozen, model_digest)
from unified_motion.tasks.interaction import (ContactPair, PlanTarget, PlanStage,
    ContactPlanTask, DomainScheduler, RandomizedTask, relative_point_velocity,
    unloading_score, ContactRewardConfig)
from unified_motion.envs.diagnostic_interaction import DiagnosticBackend
from unified_motion.envs.mjlab_interaction import MJLabInteractionBackend
from unified_motion.algorithms.sequence_training import (PhysicsSequenceCollector,
    PhysicsSequencePPO, DynamicsUpdater, stack_tree, clone_tree)
from unified_motion.algorithms.common import PPOConfig
from unified_motion.models.physics import DynamicsEncoding, DynamicsModel

ROOT=Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT/'configs/diagnostic.yaml')


def task_at_origin(hold=2, timeout=5):
    b=DiagnosticBackend(num_envs=2)
    plan=PlanStage('hold',(PlanTarget('probe_surface',(0.,0.,.01)),),
                   hold_steps=hold,max_steps=timeout,position_tolerance_m=.01)
    return ContactPlanTask(b,(plan,))


def test_unloading_reward_is_continuous_and_monotonic():
    force=torch.tensor([[100.,20.,0.]])
    desired=torch.tensor([[0.,0.,1.]])
    mask=torch.ones(1,3,dtype=torch.bool)
    score=unloading_score(force,desired,mask,25.)
    expected=(torch.exp(torch.tensor(-4.))+torch.exp(torch.tensor(-.8)))/2
    torch.testing.assert_close(score,expected[None])
    lower=unloading_score(force*.5,desired,mask,25.)
    assert (lower>score).all()


def test_dense_contact_rewards_are_logged_and_action_history_resets():
    backend=DiagnosticBackend(num_envs=2)
    stage=PlanStage('hold',(PlanTarget('probe_surface',(0.,0.,.01)),),hold_steps=3)
    cfg=ContactRewardConfig(support_weight=.3,upright_weight=.2,com_transfer_weight=.2,
        action_rate_weight=.02,joint_limit_weight=.1,undesired_contact_weight=.1,
        root_ang_vel_weight=.05,foot_slide_weight=.05)
    task=ContactPlanTask(backend,(stage,),cfg);task.reset()
    action=torch.ones(2,task.robot.dof)*.2
    first=task.step(action)
    assert first.terms['action_rate_penalty'].eq(0).all()
    second=task.step(-action)
    assert second.terms['action_rate_penalty'].gt(0).all()
    assert {'support','upright','com_transfer','joint_limit_penalty',
            'undesired_contact_penalty','root_ang_vel_penalty','foot_slide_penalty'} <= second.terms.keys()
    task.reset(torch.tensor([0]))
    third=task.step(-action)
    assert third.terms['action_rate_penalty'][0]==0


@pytest.mark.parametrize('kind',['mlp','dense','moe'])
def test_closed_loop_with_interaction_feedback(kind):
    c=config();c['model']['backbone']=kind
    task=build_task(c);dyn,pol=build_models(c,task)
    collector=PhysicsSequenceCollector(task,dyn,pol)
    ppo=PhysicsSequencePPO(pol,PPOConfig(epochs=1,minibatch_envs=2))
    update=DynamicsUpdater(dyn)
    rollout=collector.collect(6)
    assert rollout.previous_interaction_prediction is not None
    first=model_digest(pol)
    ppo.update(rollout)
    assert model_digest(pol)!=first
    # 上一轮PPO改变策略后，仍能重建时序缓存；有前缀不能打乱时间。
    rollout=collector.collect(6)
    assert rollout.prefix is not None if kind!='mlp' else rollout.prefix is None
    ppo.update(rollout)
    stat=update.update(rollout)
    assert stat.loss>=0
    assert dyn.feedback_net[0].weight.grad is not None
    assert torch.isfinite(dyn.feedback_net[0].weight.grad).all()
    collector.on_dynamics_updated()
    assert collector.collect(2).episode_start[:,0].all()


def test_sequence_replay_matches_interaction_memory():
    torch.manual_seed(9)
    b=DiagnosticBackend(num_envs=2)
    dyn=DynamicsModel(b.robot,physics_width=8,memory_width=8,latent_width=4,
                      hidden_dim=16,interaction_feedback=True).eval()
    memory=None;obs=[];preds=[];valid=[];zs=[];contexts=[];starts=[]
    for t in range(5):
        o=b.read();start=torch.full((2,),t==0)
        old=None if memory is None else memory.interaction_prediction
        d,memory=dyn.step(o,episode_start=start,env_ids=b.env_ids,memory=memory)
        action=torch.full((2,b.robot.dof),.1)
        prediction=dyn.predict_next(o,action,d.latent)
        if old is None:
            old=clone_tree(prediction);old.contact_mask.zero_();old.object_mask.zero_()
        obs.append(clone_tree(o));preds.append(clone_tree(old));valid.append(~start)
        starts.append(start);zs.append(d.latent);contexts.append(d.context)
        memory=dyn.remember_prediction(memory,prediction);b.step(action)
    seq=dyn.forward_sequence(stack_tree(obs),episode_start=torch.stack(starts,1),
        previous_predicted_core=torch.stack([p.next_core for p in preds],1),
        previous_prediction_valid=torch.stack(valid,1),
        previous_interaction_prediction=stack_tree(preds))
    torch.testing.assert_close(seq.latent,torch.stack(zs,1))
    torch.testing.assert_close(seq.context,torch.stack(contexts,1))


def test_object_and_contact_errors_actually_change_context():
    b=DiagnosticBackend(num_envs=2)
    dyn=DynamicsModel(b.robot,physics_width=8,memory_width=8,latent_width=4,
                      hidden_dim=16,interaction_feedback=True).eval()
    o=b.read();enc,mem=dyn.step(o,episode_start=torch.ones(2,dtype=torch.bool),env_ids=b.env_ids)
    pred=dyn.predict_next(o,torch.zeros(2,b.robot.dof),enc.latent)
    mem=dyn.remember_prediction(mem,pred)
    altered=clone_tree(mem);altered.owner=mem.owner
    altered.interaction_prediction.next_object_pos+=.2
    a,_=dyn.step(o,episode_start=torch.zeros(2,dtype=torch.bool),env_ids=b.env_ids,memory=mem)
    z,_=dyn.step(o,episode_start=torch.zeros(2,dtype=torch.bool),env_ids=b.env_ids,memory=altered)
    assert not torch.allclose(a.context,z.context)
    # reset以后不允许使用旧预测，无论旧预测多大。
    a,_=dyn.step(o,episode_start=torch.ones(2,dtype=torch.bool),env_ids=b.env_ids,memory=mem)
    z,_=dyn.step(o,episode_start=torch.ones(2,dtype=torch.bool),env_ids=b.env_ids,memory=altered)
    torch.testing.assert_close(a.context,z.context)


def test_quaternion_sign_does_not_create_feedback_error():
    b=DiagnosticBackend();o=b.read()
    dyn=DynamicsModel(b.robot,physics_width=8,memory_width=8,latent_width=4,hidden_dim=16)
    pred=dyn.predict_next(o,torch.zeros(2,b.robot.dof),torch.zeros(2,4))
    a=dyn.interaction_error(o,pred,torch.ones(2,dtype=torch.bool))
    pred.next_object_quat.neg_()
    torch.testing.assert_close(a,dyn.interaction_error(o,pred,torch.ones(2,dtype=torch.bool)))


def test_masked_object_does_not_create_feedback():
    b=DiagnosticBackend();o=b.read()
    dyn=DynamicsModel(b.robot,physics_width=8,memory_width=8,latent_width=4,hidden_dim=16)
    pred=dyn.predict_next(o,torch.zeros(2,b.robot.dof),torch.zeros(2,4))
    pred.object_mask.zero_();pred.next_object_pos+=1e3
    error=dyn.interaction_error(o,pred,torch.ones(2,dtype=torch.bool))
    assert (error[:,3:]==0).all()


def test_rotating_counterparty_velocity():
    point=torch.tensor([[1.,0.,0.]])
    zero=torch.zeros_like(point);omega=torch.tensor([[0.,0.,2.]])
    value=relative_point_velocity(point,zero,zero,zero,zero,zero,omega)
    torch.testing.assert_close(value,torch.tensor([[0.,-2.,0.]]))


def test_contact_hold_requires_contiguous_frames():
    task=task_at_origin(hold=2)
    task.reset();a=torch.zeros(2,task.robot.dof)
    tr=task.step(a);assert not tr.terminated.any()
    tr=task.step(a);assert tr.terminated.all() and task.last_success.all()
    with pytest.raises(RuntimeError): task.step(a)


def test_stage_transition_and_failure_semantics():
    task=task_at_origin();task.stages=(task.stages[0],PlanStage('far',
      (PlanTarget('probe_surface',(3.,0.,.01)),),max_steps=2,hold_steps=1))
    task.reset();a=torch.zeros(2,task.robot.dof)
    task.step(a);task.step(a)
    assert (task.phase==1).all()
    task.step(a);end=task.step(a)
    assert end.terminated.all() and not end.truncated.any() and not task.last_success.any()


def test_object_local_goal_moves_with_object():
    task=task_at_origin();task.reset()
    first=task.goal().contact_features.clone()
    task.backend.surface[:,0]+=.25
    second=task.goal().contact_features
    torch.testing.assert_close(second[...,0],first[...,0]+.25)


def test_mjlab_terrain_relation_uses_each_environment_origin():
    """并行K1的地面局部目标不能全部落在世界原点。"""
    b=2
    data=SimpleNamespace(
        root_link_pos_w=torch.tensor([[1.,-1.,.57],[-1.,1.,.57]]),
        root_link_quat_w=torch.tensor([[1.,0.,0.,0.]]).repeat(b,1),
        body_link_pose_w=torch.tensor([[[1.,-1.,.02,1.,0.,0.,0.]],
                                       [[-1.,1.,.02,1.,0.,0.,0.]]]),
        body_link_vel_w=torch.zeros(b,1,6),
    )
    backend=object.__new__(MJLabInteractionBackend)
    backend.num_envs=b
    backend.entity=SimpleNamespace(data=data)
    backend.pairs=(ContactPair('foot_ground','foot',point_local_m=(0.,0.,0.)),)
    backend.body_ids=[0]
    backend.counterparts=[None]
    backend.sensors=[SimpleNamespace(data=SimpleNamespace(
        found=torch.ones(b,1),normal=torch.tensor([[[0.,0.,1.]]]).repeat(b,1,1),
        force=torch.tensor([[[0.,0.,10.]]]).repeat(b,1,1)))]
    backend.load_sensors=[SimpleNamespace(data=SimpleNamespace(
        force=torch.tensor([[[0.,0.,10.]]]).repeat(b,1,1)))]
    origins=torch.tensor([[1.,-1.,0.],[-1.,1.,0.]])
    backend.env=SimpleNamespace(scene=SimpleNamespace(env_origins=origins))
    relation=backend._body_state()
    torch.testing.assert_close(relation.object_pos[:,0],origins)


def test_partial_reset_does_not_reset_other_phase():
    task=task_at_origin(hold=4);task.reset();task.step(torch.zeros(2,task.robot.dof))
    task.reset(torch.tensor([0]))
    assert task.age.tolist()==[0,1]


def test_registry_mutation_is_rejected():
    task=task_at_origin();task.reset();task.backend.pairs=(ContactPair('other','body'),)
    with pytest.raises(ValueError): task.read()


def test_curriculum_changes_only_from_training_episode_results():
    c=config();task=build_task(c)
    task.scheduler.record(torch.ones(20,dtype=torch.bool))
    assert abs(task.scheduler.level-.4)<1e-6
    evaluation=build_task(c,evaluation=True)
    level=evaluation.scheduler.level;evaluation.reset()
    for _ in range(40):
        tr=evaluation.step(torch.zeros(2,evaluation.robot.dof))
        if (tr.terminated|tr.truncated).any(): evaluation.reset()
    assert evaluation.scheduler.level==level


def test_readback_mismatch_is_rejected():
    task=build_task(config())
    original=task.backend.set_parameters
    def wrong(ids,values):
        result=original(ids,values);result['friction']+=1;return result
    task.backend.set_parameters=wrong
    with pytest.raises(AssertionError):task.reset()


def test_delay_queue_preserves_raw_action_in_observation():
    c=config()
    for d in c['domains'].values(): d['delay_steps']=[1,1];d['push_force_n']=[0,0]
    task=build_task(c);task.reset()
    a=torch.ones(2,task.robot.dof)*.2
    task.step(a)
    assert (task.backend.action==0).all()
    torch.testing.assert_close(task.read().body_obs[:,-task.robot.dof:],a)
    task.step(a*2)
    torch.testing.assert_close(task.backend.action,a)
    task.reset(torch.tensor([0]));assert (task.queue[0]==0).all()


def test_asymmetric_online_identification_has_no_actor_privileged_path():
    c=config()
    c['adaptation']={
        'actor_observation':'deployable',
        'privileged_critic':True,
        'system_identification_weight':.1,
    }
    task=build_task(c);task.reset()
    frame=task.read();privileged=task.privileged()
    assert privileged.shape==(task.num_envs,task.privileged_dim)
    assert task.privileged_dim==len(task.privileged_names)>0
    # 精确根线速度、接触法线和力不进入Actor观测。
    assert (frame.base_lin_vel_b==0).all()
    assert (frame.contact_features[...,3:10]==0).all()
    assert (frame.contact_features[...,11]==0).all()
    assert not frame.object_mask.any() and (frame.object_features==0).all()
    dyn,policy=build_models(c,task);dyn.eval()
    encoding,_=dyn.step(frame,episode_start=task.episode_start,env_ids=task.env_ids)
    a=policy(frame,encoding,task.goal(),privileged=privileged)
    b=policy(frame,encoding,task.goal(),privileged=privileged+1)
    torch.testing.assert_close(a.mean,b.mean,rtol=0,atol=0)
    torch.testing.assert_close(a.std,b.std,rtol=0,atol=0)
    assert not torch.allclose(a.value,b.value)

    collector=PhysicsSequenceCollector(task,dyn,policy)
    rollout=collector.collect(6)
    updater=DynamicsUpdater(dyn,system_id_weight=.1)
    stats=updater.update(rollout)
    assert stats.terms['system_identification']>=0
    assert dyn.system_id_head[0].weight.grad is not None
    assert torch.isfinite(dyn.system_id_head[0].weight.grad).all()


def test_system_identification_requires_privileged_critic_channel():
    c=config();c['adaptation']={'system_identification_weight':.1}
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'config.yaml'
        path.write_text(yaml.safe_dump(c),encoding='utf-8')
        with pytest.raises(ValueError,match='privileged_critic'):load_config(path)


def test_actor_dynamics_ablation_keeps_privileged_channels_explicit():
    c=config()
    c['adaptation']={'actor_observation':'deployable','actor_dynamics':'none',
                     'privileged_critic':True}
    task=build_task(c);task.reset()
    sample=DynamicsEncoding(torch.randn(2,8),torch.randn(2,16),torch.randn(2,16),
                            torch.randn(2,53))
    none=task.policy_dynamics(sample,task.privileged())
    assert (none.latent==0).all() and (none.prediction_error==0).all()
    torch.testing.assert_close(none.context,sample.context)

    c['adaptation']['actor_dynamics']='online'
    online_task=build_task(c);online_task.reset()
    assert online_task.policy_dynamics(sample,online_task.privileged()) is sample

    c['adaptation']['actor_dynamics']='oracle'
    oracle_task=build_task(c);oracle_task.reset()
    privileged=oracle_task.privileged()
    oracle=oracle_task.policy_dynamics(sample,privileged)
    torch.testing.assert_close(oracle.latent[:,:oracle_task.privileged_dim],privileged)
    assert (oracle.latent[:,oracle_task.privileged_dim:]==0).all()
    assert (oracle.prediction_error==0).all()


def test_oracle_actor_is_rejected_without_privileged_channel():
    c=config();c['adaptation']={'actor_dynamics':'oracle'}
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'config.yaml'
        path.write_text(yaml.safe_dump(c),encoding='utf-8')
        with pytest.raises(ValueError,match='privileged_critic'):load_config(path)


def test_teacher_extra_layers_do_not_change_actor_initialization():
    direct=config();direct['adaptation']={'privileged_critic':True,
        'system_identification_weight':.1}
    teacher=deepcopy(direct);teacher['adaptation'].update(
        teacher_distillation_weight=.05,teacher_reconstruction_weight=.1)
    direct_task=build_task(direct);teacher_task=build_task(teacher)
    _,direct_policy=build_models(direct,direct_task)
    _,teacher_policy=build_models(teacher,teacher_task)
    assert direct_policy.state_dict().keys()==teacher_policy.state_dict().keys()
    for name,value in direct_policy.state_dict().items():
        other=teacher_policy.state_dict()[name]
        if isinstance(value,torch.Tensor):
            torch.testing.assert_close(value,other,rtol=0,atol=0,
                                       msg=f'Actor初始参数不一致：{name}')
        else:
            assert value==other


def test_teacher_student_stop_gradient_and_reconstruction_contract():
    backend=DiagnosticBackend()
    model=DynamicsModel(backend.robot,physics_width=8,memory_width=8,latent_width=4,
                        hidden_dim=16,system_id_dim=7,teacher_student=True)
    privileged=torch.randn(2,3,7)
    student=torch.randn(2,3,4,requires_grad=True)
    teacher=model.encode_privileged_teacher(privileged)
    (student-teacher.detach()).square().mean().backward()
    assert student.grad is not None and student.grad.abs().sum()>0
    assert all(p.grad is None for p in model.teacher_encoder.parameters())
    model.zero_grad(set_to_none=True)
    teacher=model.encode_privileged_teacher(privileged)
    (model.reconstruct_privileged(teacher)-privileged).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0
               for p in model.teacher_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0
               for p in model.teacher_decoder.parameters())


def test_teacher_student_full_training_path_reports_both_losses():
    c=config()
    c['adaptation']={'actor_observation':'deployable','privileged_critic':True,
        'system_identification_weight':.1,'teacher_distillation_weight':.05,
        'teacher_reconstruction_weight':.1}
    task=build_task(c);dyn,policy=build_models(c,task)
    rollout=PhysicsSequenceCollector(task,dyn,policy).collect(6)
    stats=DynamicsUpdater(dyn,system_id_weight=.1,teacher_distillation_weight=.05,
                          teacher_reconstruction_weight=.1).update(rollout)
    assert stats.terms['teacher_distillation']>=0
    assert stats.terms['teacher_reconstruction']>=0


def test_split_reports_overlap_honestly():
    c=config();s=DomainScheduler(c['domains'],c['train_domains'],c['evaluation_domains'])
    assert s.split_report()['all_parameter_supports_disjoint']
    c['domains']['heldout_low_friction']=deepcopy(c['domains']['familiar'])
    s=DomainScheduler(c['domains'],c['train_domains'],c['evaluation_domains'])
    assert not s.split_report()['all_parameter_supports_disjoint']


def test_randomization_seed_is_reproducible():
    c=config()
    a=DomainScheduler(c['domains'],c['train_domains'],c['evaluation_domains'],seed=8)
    b=DomainScheduler(c['domains'],c['train_domains'],c['evaluation_domains'],seed=8)
    av,an=a.sample(10);bv,bn=b.sample(10)
    assert an==bn
    for k in av: torch.testing.assert_close(av[k],bv[k])


def test_yaml_duplicate_and_unknown_rejected():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'x.yaml';p.write_text('schema: 1\nschema: 1\n')
        with pytest.raises(ValueError):load_config(p)
        p.write_text((ROOT/'configs/diagnostic.yaml').read_text()+'\naccidental_typo: 42\n')
        with pytest.raises(ValueError):load_config(p)


@pytest.mark.parametrize('kind',['mlp','dense','moe'])
def test_frozen_evaluation_preserves_weights_and_counts_endings(kind):
    c=config();c['model']['backbone']=kind
    task=build_task(c,evaluation=True);dyn,pol=build_models(c,task)
    before=model_digest(dyn,pol)
    result=evaluate_frozen(task,dyn,pol,episodes=1,max_steps=50)
    assert result['completed'] and result['episodes']==2
    assert result['weights_unchanged'] and before==model_digest(dyn,pol)
    assert result['robot_skill_verified'] is False


def test_frozen_evaluation_detects_buffer_mutation():
    c=config();task=build_task(c,evaluation=True);dyn,pol=build_models(c,task)
    pol.register_buffer('tamper_marker',torch.tensor(0.))
    original=task.step
    def step(a):
        tr=original(a);pol.tamper_marker.add_(1);return tr
    task.step=step
    with pytest.raises(RuntimeError,match='参数或buffer'):evaluate_frozen(task,dyn,pol,episodes=1,max_steps=2)


def test_frozen_budget_does_not_claim_completion():
    task=build_task(config(),evaluation=True);dyn,pol=build_models(config(),task)
    report=evaluate_frozen(task,dyn,pol,episodes=1,max_steps=1)
    assert not report['completed'] and report['episodes']==0


def test_real_backend_does_not_fallback_to_diagnostic():
    c=load_config(ROOT/'configs/k1_contact.yaml')
    if torch.cuda.is_available():pytest.skip('这里只检查无CUDA环境的拒绝路径')
    with pytest.raises(RuntimeError,match='没有降级'):build_task(c)

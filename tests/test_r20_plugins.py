"""R20集中回归：测试图/循环插口和真实参数更新；解析任务不是机器人仿真。"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import io

import pytest
import torch
import yaml

from unified_motion.api import build_policy, get_robot_contract
from unified_motion.models.recurrent import RecurrentCore, RecurrentState, RecurrentCache
from unified_motion.models.graph_encoder import GraphBatch, MessagePassingEncoder, PhysicsRelationGraph
from unified_motion.models.physics import DynamicsModel, PhysicsEncoder
from unified_motion.experiment import load_config, build_task, build_models, evaluate_frozen, model_digest
from unified_motion.algorithms.sequence_training import (
    PhysicsSequenceCollector, PhysicsSequencePPO, DynamicsUpdater, clone_tree,
    at_time, stack_tree, _history_size,
)
from unified_motion.algorithms.common import PPOConfig
from unified_motion.envs.diagnostic_interaction import DiagnosticBackend
from physics_fixtures import TensorTask

ROOT = Path(__file__).resolve().parents[1]


def make_core(kind='gru', dtype=torch.float32):
    torch.manual_seed(101)
    return RecurrentCore(8, 6, 8, kind=kind, n_layers=2).to(dtype)


def core_inputs(dtype=torch.float32):
    x = torch.randn(3, 9, 8, dtype=dtype)
    starts = torch.zeros(3, 9, dtype=torch.bool)
    starts[:, 0] = True
    starts[0, 4] = True
    valid = torch.ones(3, 9, dtype=torch.bool)
    valid[2, 6:] = False
    return x, starts, valid


@pytest.mark.parametrize('kind', ['gru', 'lstm'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_recurrent_whole_sequence_matches_steps(kind, dtype):
    model = make_core(kind, dtype)
    x, starts, valid = core_inputs(dtype)
    whole, state = model.forward_sequence(x, episode_start=starts, valid_mask=valid)
    h = None; frames = []
    for t in range(x.shape[1]):
        y, h = model.step(x[:, t], h, episode_start=starts[:, t], valid_mask=valid[:, t])
        frames.append(y)
    torch.testing.assert_close(whole, torch.stack(frames, 1))
    torch.testing.assert_close(state.hidden, h.hidden)
    if kind == 'lstm': torch.testing.assert_close(state.cell, h.cell)
    assert (whole[~valid] == 0).all()


@pytest.mark.parametrize('kind', ['gru', 'lstm'])
def test_future_does_not_affect_past_and_reset_clears_both_states(kind):
    model = make_core(kind)
    x, starts, valid = core_inputs()
    a, _ = model.forward_sequence(x, episode_start=starts, valid_mask=valid)
    dirty = x.clone(); dirty[:, 5:] += 100
    z, _ = model.forward_sequence(dirty, episode_start=starts, valid_mask=valid)
    torch.testing.assert_close(a[:, :5], z[:, :5])
    _, previous = model.forward_sequence(x[:, :4], episode_start=starts[:, :4])
    mixed, mixed_state = model.step(x[:, 4], previous, episode_start=starts[:, 4])
    fresh, fresh_state = model.step(x[:, 4], None, episode_start=torch.ones(3, dtype=torch.bool))
    torch.testing.assert_close(mixed[0], fresh[0])
    torch.testing.assert_close(mixed_state.hidden[:, 0], fresh_state.hidden[:, 0])
    assert not torch.allclose(mixed[1], fresh[1])
    if kind == 'lstm': torch.testing.assert_close(mixed_state.cell[:, 0], fresh_state.cell[:, 0])


@pytest.mark.parametrize('kind', ['gru', 'lstm'])
def test_masked_recurrent_input_zero_gradient_and_state_preserved(kind):
    model = make_core(kind)
    x, starts, valid = core_inputs()
    x.requires_grad_()
    y, state = model.forward_sequence(x, episode_start=starts, valid_mask=valid)
    y.square().sum().backward()
    assert (x.grad[~valid] == 0).all()
    _, before = model.forward_sequence(x.detach()[:, :6], episode_start=starts[:, :6])
    torch.testing.assert_close(state.hidden[:, 2], before.hidden[:, 2])
    if kind == 'lstm': torch.testing.assert_close(state.cell[:, 2], before.cell[:, 2])
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize('kind', ['gru', 'lstm'])
def test_recurrent_serialization_and_input_validation(kind):
    model = make_core(kind)
    copied = make_core(kind); copied.load_state_dict(deepcopy(model.state_dict()))
    x, s, v = core_inputs()
    torch.testing.assert_close(model.forward_sequence(x, episode_start=s)[0],
                               copied.forward_sequence(x, episode_start=s)[0])
    wrong = model.initial_state(3)
    wrong.cell = None if kind == 'lstm' else torch.zeros_like(wrong.hidden)
    with pytest.raises(ValueError):model.step(x[:,0], wrong, episode_start=s[:,0])
    v[0,2] = False
    with pytest.raises(ValueError):model.forward_sequence(x, episode_start=s, valid_mask=v)
    x[0,0,0] = float('nan')
    with pytest.raises(ValueError):model.forward_sequence(x, episode_start=s)


@pytest.mark.parametrize('kind', ['gru', 'lstm'])
def test_recurrent_autocast_is_finite(kind):
    model=make_core(kind)
    x,s,v=core_inputs()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y,h=model.forward_sequence(x,episode_start=s,valid_mask=v)
    assert y.dtype==torch.float32 and h.hidden.dtype==torch.float32
    y.square().sum().backward()
    assert torch.isfinite(y).all()


def graph_inputs(dtype=torch.float32):
    nodes = torch.randn(2, 4, 8, dtype=dtype)
    masks = torch.tensor([[True,True,True,False],[True,True,True,True]])
    edges = torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]])
    kinds = torch.tensor([0,0,1,1,2,2])
    return GraphBatch(nodes,masks,edges,kinds)


def test_graph_message_aggregation_matches_independent_dense_reference():
    torch.manual_seed(7)
    model=MessagePassingEncoder(8,n_layers=1).double()
    g=graph_inputs(torch.float64)
    actual=model(g)
    x=torch.where(g.node_mask[...,None],g.nodes,0.)
    layer=model.layers[0]; total=torch.zeros_like(x);count=torch.zeros_like(g.node_mask,dtype=x.dtype)
    for e in range(g.edge_index.shape[1]):
        src,dst=g.edge_index[:,e].tolist()
        valid=g.node_mask[:,src]&g.node_mask[:,dst]
        m=torch.cat((x[:,src],x[:,dst],layer.edge_embedding(g.edge_type[e]).expand(2,-1)),dim=-1)
        m=layer.message(torch.where(valid[:,None],m,0.))
        total[:,dst]+=torch.where(valid[:,None],m,0.)
        count[:,dst]+=valid.to(x.dtype)
    expected=layer.norm(x+layer.update(torch.cat((x,total/count.clamp_min(1)[...,None]),-1)))
    expected=torch.where(g.node_mask[...,None],expected,0.)
    torch.testing.assert_close(actual,expected)


def test_graph_node_permutation_equivariance_and_edge_order_invariance():
    model=MessagePassingEncoder(8)
    g=graph_inputs();a=model(g)
    perm=torch.tensor([2,0,3,1]);inv=torch.argsort(perm)
    altered=GraphBatch(g.nodes[:,perm],g.node_mask[:,perm],inv[g.edge_index],g.edge_type)
    torch.testing.assert_close(model(altered),a[:,perm],atol=2e-6,rtol=2e-5)
    e=torch.randperm(g.edge_type.numel())
    torch.testing.assert_close(model(replace(g,edge_index=g.edge_index[:,e],edge_type=g.edge_type[e])),a)


def test_graph_masks_block_garbage_and_gradients():
    model=MessagePassingEncoder(8)
    g=graph_inputs();a=model(g)
    dirty=torch.where(g.node_mask[...,None],g.nodes,torch.finfo(torch.float32).max/2).requires_grad_()
    y=model(replace(g,nodes=dirty))
    torch.testing.assert_close(a,y)
    y.square().sum().backward()
    assert (dirty.grad[~g.node_mask]==0).all()
    assert torch.isfinite(dirty.grad).all()


def test_graph_edges_really_affect_output():
    model=MessagePassingEncoder(8)
    g=graph_inputs()
    a=model(g)
    empty=replace(g,edge_index=torch.empty(2,0,dtype=torch.long),edge_type=torch.empty(0,dtype=torch.long))
    assert not torch.allclose(a,model(empty))
    blocked=replace(g,edge_mask=torch.zeros(2,g.edge_type.numel(),dtype=torch.bool))
    torch.testing.assert_close(model(empty),model(blocked))


def test_graph_does_not_mix_environments_or_times():
    model=MessagePassingEncoder(8)
    g=graph_inputs()
    seq=replace(g,nodes=g.nodes[:,None].repeat(1,3,1,1),node_mask=g.node_mask[:,None].repeat(1,3,1))
    out=model(seq)
    dirty=seq.nodes.clone();dirty[1,1:]+=10
    other=model(replace(seq,nodes=dirty))
    torch.testing.assert_close(out[0],other[0]);torch.testing.assert_close(out[:,0],other[:,0])


@pytest.mark.parametrize('problem',['negative','range','self','duplicate','type','mask','nan'])
def test_graph_rejects_bad_data(problem):
    model=MessagePassingEncoder(8);g=graph_inputs()
    if problem=='negative':g.edge_index[0,0]=-1
    elif problem=='range':g.edge_index[0,0]=4
    elif problem=='self':g.edge_index[0,0]=g.edge_index[1,0]
    elif problem=='duplicate':g.edge_index[:,1]=g.edge_index[:,0]
    elif problem=='type':g.edge_type[0]=3
    elif problem=='mask':g.node_mask=g.node_mask.float()
    elif problem=='nan':g.nodes[0,0,0]=float('nan')
    with pytest.raises((ValueError,TypeError)):model(g)


def test_graph_numerical_gradient():
    model=MessagePassingEncoder(2,n_layers=1).double()
    x=torch.randn(1,2,2,dtype=torch.float64,requires_grad=True)
    def fn(v):return model(GraphBatch(v,torch.ones(1,2,dtype=torch.bool),
                                     torch.tensor([[0,1],[1,0]]),torch.tensor([0,0])))
    assert torch.autograd.gradcheck(fn,(x,),eps=1e-6,atol=1e-4)


def test_graph_no_contacts_no_objects_is_valid():
    layer=PhysicsRelationGraph(8,[])
    body=torch.randn(2,8)
    out=layer(body,torch.zeros(2,0,8),torch.zeros(2,0,8),
               torch.zeros(2,0,dtype=torch.bool),torch.zeros(2,0,dtype=torch.bool))
    assert out[0].shape==(2,8) and torch.isfinite(out[0]).all()
    assert out[1].shape==(2,0,8)


def test_graph_dynamics_has_supervised_gradients_and_quaternion_invariance():
    b=DiagnosticBackend();o=b.read()
    model=DynamicsModel(b.robot,physics_width=8,memory_width=8,latent_width=4,hidden_dim=16,
                        scene_encoder='gnn',contact_object_indices=[0],interaction_feedback=True).eval()
    enc,mem=model.step(o,episode_start=torch.ones(2,dtype=torch.bool),env_ids=b.env_ids)
    pred=model.predict_next(o,torch.zeros(2,b.robot.dof),enc.latent)
    model.prediction_loss(pred,o,valid_transition=torch.ones(2,dtype=torch.bool)).total.backward()
    grad=model.encoder.graph.gnn.layers[0].message[0].weight.grad
    assert grad is not None and grad.abs().sum()>0 and torch.isfinite(grad).all()
    dirty=clone_tree(o);dirty.object_features[...,3:7].neg_()
    torch.testing.assert_close(model.encoder(o).context,model.encoder(dirty).context)


def test_graph_bad_mapping_and_checkpoint_semantics():
    b=DiagnosticBackend();o=b.read()
    model=PhysicsEncoder(b.robot,width=8,hidden_dim=16,scene_encoder='gnn',contact_object_indices=[0])
    wrong=PhysicsEncoder(b.robot,width=8,hidden_dim=16,scene_encoder='gnn',contact_object_indices=[-1])
    with pytest.raises(ValueError):wrong.load_state_dict(model.state_dict())
    wrong=PhysicsEncoder(b.robot,width=8,hidden_dim=16,scene_encoder='gnn',contact_object_indices=[1])
    with pytest.raises(ValueError):wrong(o)
    with pytest.raises(ValueError):PhysicsEncoder(b.robot,scene_encoder='gnn')


def setup_rnn(kind, graph=False, robot='k1_22dof', max_replay=512):
    task=TensorTask(robot,batch=2,contacts=1,objects=1,endings=False)
    dyn=DynamicsModel(task.robot,physics_width=8,memory_width=8,latent_width=4,hidden_dim=16,
        interaction_feedback=True,scene_encoder='gnn' if graph else 'set',
        contact_object_indices=[0] if graph else None)
    policy=build_policy({robot:task.robot},backbone=kind,d_model=16,hidden_dim=16,
        n_layers=2,context_len=2,physics_width=8,latent_width=4,
        recurrent_hidden_dim=12,max_replay_steps=max_replay,init_std=.15)
    return task,dyn,policy


@pytest.mark.parametrize('kind',['gru','lstm'])
@pytest.mark.parametrize('graph',[False,True])
def test_recurrent_collector_retains_history_and_updates_policy_and_dynamics(kind,graph):
    task,dyn,pol=setup_rnn(kind,graph)
    collector=PhysicsSequenceCollector(task,dyn,pol)
    ppo=PhysicsSequencePPO(pol,PPOConfig(epochs=1,minibatch_envs=1))
    assert _history_size(pol) is None
    for i in range(3):
        roll=collector.collect(6)
        assert (0 if roll.prefix is None else roll.prefix.starts.shape[1])==i*6
        # n_layers*(context_len-1)=2，真正保留的循环历史远大于它。
        before=model_digest(pol)
        stat=ppo.update(roll)
        assert stat.optimizer_steps>0 and stat.old_policy_error<1e-4
        assert model_digest(pol)!=before
    loss=DynamicsUpdater(dyn,epochs=1).update(roll)
    assert loss.loss>=0
    if graph:assert dyn.encoder.graph.gnn.layers[0].message[0].weight.grad.abs().sum()>0
    collector.on_dynamics_updated();assert collector.collect(2).episode_start[:,0].all()


@pytest.mark.parametrize('kind',['gru','lstm'])
@pytest.mark.parametrize('robot',['k1_22dof','g1_23dof','g1_29dof'])
def test_recurrent_all_robot_interfaces_and_checkpoint(kind,robot):
    task,dyn,pol=setup_rnn(kind,robot=robot)
    c=PhysicsSequenceCollector(task,dyn,pol);roll=c.collect(5)
    assert roll.action.shape==(2,5,task.robot.dof)
    _,_,copy=setup_rnn(kind,robot=robot)
    copy.load_state_dict(deepcopy(pol.state_dict()))
    # 独立构造的同权重模型复算旧动作，不复用另一模型的缓存。
    ppo=PhysicsSequencePPO(copy,PPOConfig(epochs=1,minibatch_envs=2))
    assert ppo.verify_old_policy(roll)<1e-4


@pytest.mark.parametrize('kind',['gru','lstm'])
def test_recurrent_missing_history_and_cap_are_not_silently_ignored(kind):
    task,dyn,p=setup_rnn(kind,max_replay=8);c=PhysicsSequenceCollector(task,dyn,p)
    c.collect(4);r=c.collect(4)
    r.prefix=None
    with pytest.raises(ValueError):PhysicsSequencePPO(p).verify_old_policy(r)
    with pytest.raises(RuntimeError,match='max_replay_steps'):c.collect(1)


@pytest.mark.parametrize('kind',['gru','lstm'])
def test_recurrent_cache_rejects_environment_and_weight_changes(kind):
    task,dyn,p=setup_rnn(kind);p.eval();dyn.eval();task.reset()
    frame=task.read();starts=task.episode_start
    d,_=dyn.step(frame,episode_start=starts,env_ids=task.env_ids)
    _,_,_,cache=p.step(task.robot_id,frame,d,task.goal(),env_ids=task.env_ids,episode_start=starts)
    assert isinstance(cache,RecurrentCache)
    with pytest.raises(ValueError):
        p.step(task.robot_id,frame,d,task.goal(),env_ids=tuple(reversed(task.env_ids)),
                episode_start=starts,cache=cache)
    with torch.no_grad():next(p.encoder.parameters()).add_(.001)
    with pytest.raises(ValueError,match='参数已改变'):
        p.step(task.robot_id,frame,d,task.goal(),env_ids=task.env_ids,episode_start=starts,cache=cache)


@pytest.mark.parametrize('kind',['gru','lstm'])
def test_recurrent_partial_resets_replay_exactly(kind):
    task,dyn,p=setup_rnn(kind)
    task.endings=True
    c=PhysicsSequenceCollector(task,dyn,p);u=PhysicsSequencePPO(p,PPOConfig(epochs=1,minibatch_envs=2))
    for _ in range(4):
        r=c.collect(7);s=u.update(r)
        assert s.old_policy_error<1e-4


@pytest.mark.parametrize('kind',['mlp','dense','moe','gru','lstm'])
def test_yaml_task_graph_and_frozen_eval_all_choices(kind):
    cfg=load_config(ROOT/'configs/diagnostic.yaml')
    cfg['model']['backbone']=kind
    cfg['dynamics'].update(scene_encoder='gnn',graph_layers=1,contact_object_indices=[0])
    task=build_task(cfg,evaluation=True);d,p=build_models(cfg,task)
    result=evaluate_frozen(task,d,p,episodes=1,max_steps=50)
    assert result['completed'] and result['weights_unchanged']
    assert result['backend']=='analytic_diagnostic_not_humanoid'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('kind',['gru','lstm'])
def test_cuda_recurrent_core(kind):
    model=make_core(kind).cuda();x,s,v=core_inputs()
    y,_=model.forward_sequence(x.cuda(),episode_start=s.cuda(),valid_mask=v.cuda())
    y.square().mean().backward();assert torch.isfinite(y).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_message_passing():
    model=MessagePassingEncoder(8).cuda();g=graph_inputs()
    g=GraphBatch(g.nodes.cuda(),g.node_mask.cuda(),g.edge_index.cuda(),g.edge_type.cuda())
    y=model(g);y.square().mean().backward();assert torch.isfinite(y).all()


@pytest.mark.parametrize('field,value',[
    ('scene_encoder','typo'),('graph_layers',0),('contact_object_indices',[-2]),
])
def test_invalid_graph_yaml_rejected(field,value,tmp_path):
    cfg=load_config(ROOT/'configs/diagnostic_gnn_gru.yaml')
    cfg['dynamics'][field]=value
    path=tmp_path/'invalid.yaml';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):load_config(path)


def test_graph_mapping_checked_against_named_environment_slots():
    cfg=load_config(ROOT/'configs/diagnostic_gnn_gru.yaml')
    cfg['dynamics']['contact_object_indices']=[-1]  # 合法整数，但实际对象是surface而不是世界。
    task=build_task(cfg)
    with pytest.raises(ValueError,match='映射与后端不符'):build_models(cfg,task)


def test_all_yaml_configs_validate_and_inspect_without_simulator():
    from unified_motion.experiment import describe_models
    for path in (ROOT/'configs').glob('*.yaml'):
        cfg=load_config(path)
        report=describe_models(cfg)
        assert report['total_parameters']==report['policy_parameters']+report['dynamics_parameters']
        assert report['total_parameters']>0


@pytest.mark.parametrize('kind',['gru','lstm'])
def test_recurrent_padding_in_full_actor(kind):
    task,dyn,p=setup_rnn(kind)
    r=PhysicsSequenceCollector(task,dyn,p).collect(5)
    valid=torch.ones(2,5,dtype=torch.bool);valid[1,3:]=False
    out=p.forward_sequence(task.robot_id,r.inputs.physics,r.inputs.dynamics(),r.inputs.goal,
             valid_mask=valid,episode_start=r.episode_start)
    assert (out.mean[~valid]==0).all() and (out.values[~valid]==0).all()
    assert (out.std[~valid]==1).all()
    out.mean.square().mean().backward()
    assert any(x.grad is not None and x.grad.abs().sum()>0 for x in p.backbone.core.parameters())


def test_graph_parameters_change_only_in_dynamics_update():
    task,dyn,p=setup_rnn('gru',graph=True)
    r=PhysicsSequenceCollector(task,dyn,p).collect(6)
    before=model_digest(dyn)
    PhysicsSequencePPO(p,PPOConfig(epochs=1,minibatch_envs=2)).update(r)
    assert model_digest(dyn)==before
    DynamicsUpdater(dyn).update(r)
    assert model_digest(dyn)!=before


def test_full_rnn_checkpoint_safe_roundtrip():
    task,dyn,p=setup_rnn('lstm',graph=True)
    c=PhysicsSequenceCollector(task,dyn,p)
    ppo=PhysicsSequencePPO(p,PPOConfig(epochs=1,minibatch_envs=2));up=DynamicsUpdater(dyn)
    r=c.collect(5);ppo.update(r);up.update(r)
    buffer=io.BytesIO();torch.save({'ppo':ppo.state_dict(),'dynamics':up.state_dict()},buffer)
    buffer.seek(0);saved=torch.load(buffer,weights_only=True)
    ppo.load_state_dict(saved['ppo']);up.load_state_dict(saved['dynamics'])
    c.on_dynamics_updated();r=c.collect(3)
    assert ppo.verify_old_policy(r)<1e-4


@pytest.mark.parametrize('kind',['gru','lstm'])
def test_one_recurrent_policy_instance_serves_k1_and_g1_without_cache_mix(kind):
    names=['k1_22dof','g1_23dof','g1_29dof']
    robots={name:get_robot_contract(name) for name in names}
    p=build_policy(robots,backbone=kind,d_model=16,hidden_dim=16,n_layers=1,
                   physics_width=8,latent_width=4,recurrent_hidden_dim=12).eval()
    previous_cache=None
    for name in names:
        task=TensorTask(name,batch=2,contacts=1,objects=1,endings=False)
        d=DynamicsModel(task.robot,physics_width=8,memory_width=8,latent_width=4,hidden_dim=16).eval()
        obs=task.read();dyn,_=d.step(obs,episode_start=task.episode_start,env_ids=task.env_ids)
        if previous_cache is not None:
            with pytest.raises(ValueError):
                p.step(name,obs,dyn,task.goal(),env_ids=task.env_ids,
                       episode_start=task.episode_start,cache=previous_cache)
        result,action,lp,previous_cache=p.step(name,obs,dyn,task.goal(),env_ids=task.env_ids,
                                              episode_start=task.episode_start)
        assert action.shape==(2,robots[name].dof) and torch.isfinite(lp).all()

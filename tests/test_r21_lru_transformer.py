"""R21：独立公式/梯度、线性扫描、边界、PPO接线和无MoE主干的集中回归。

TensorTask与diagnostic只验证算法，不是K1/G1真实物理；不以成功率替代代码检查。
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import io
import math

import pytest
import torch
import yaml

from unified_motion.api import build_policy, get_robot_contract
from unified_motion.models.lru import LRULayer, LRUCore, LRUState, _parallel_prefix
from unified_motion.models.moe import ReferenceRoutedMoE
from unified_motion.models.attention import CausalSelfAttention
from unified_motion.models.recurrent import RecurrentCache, RecurrentCore, RecurrentState
from unified_motion.models.physics_policy import PhysicsPolicyConfig, PhysicsConditionedActorCritic, DenseReferenceTransformer
from unified_motion.models.physics import DynamicsModel
from unified_motion.algorithms.sequence_training import (
    PhysicsSequenceCollector, PhysicsSequencePPO, DynamicsUpdater, clone_tree, _history_size,
)
from unified_motion.algorithms.common import PPOConfig
from unified_motion.experiment import load_config, build_task, build_models, evaluate_frozen, describe_models, model_digest
from physics_fixtures import TensorTask

ROOT = Path(__file__).resolve().parents[1]


def inputs(t=11, dtype=torch.float64):
    torch.manual_seed(211)
    x = torch.randn(3, t, 5, dtype=dtype)
    starts = torch.zeros(3, t, dtype=torch.bool)
    starts[:, 0] = True
    if t > 4:
        starts[0, 4] = True
    valid = torch.ones(3, t, dtype=torch.bool)
    if t > 6:
        valid[2, -3:] = False
    return x, starts, valid


def independent_complex(layer, x, initial, starts, valid):
    # 独立地使用真正复数算式作数值基准，不复用被测扫描或实部/虚部乘法函数。
    lam = torch.exp(-torch.exp(layer.nu_log) + 1j*torch.exp(layer.theta_log))
    bb = torch.complex(layer.b_real, layer.b_imag)*layer.gamma_log.exp()[:, None]
    cc = torch.complex(layer.c_real, layer.c_imag)
    h = torch.complex(initial[..., 0], initial[..., 1])
    ys = []
    for t in range(x.shape[1]):
        u = torch.where(valid[:, t, None], x[:, t], 0.)
        h0 = torch.where(starts[:, t, None], 0., h)
        proposal = lam*h0 + u.to(bb.dtype)@bb.T
        h = torch.where(valid[:, t, None], proposal, h)
        y = (h@cc.T).real + layer.skip*u
        ys.append(torch.where(valid[:, t, None], y, 0.))
    return torch.stack(ys, 1), torch.stack((h.real, h.imag), -1)


@pytest.mark.parametrize('t', [1, 2, 3, 5, 8, 11, 32, 65])
@pytest.mark.parametrize('mode', ['parallel', 'sequential'])
def test_layer_matches_independent_complex_formula(t, mode):
    x, starts, valid = inputs(t)
    layer = LRULayer(5, 7).double()
    h0 = torch.randn(3, 7, 2, dtype=x.dtype)
    a, h = layer(x, initial_state=h0, episode_start=starts, valid_mask=valid, scan_mode=mode)
    b, j = independent_complex(layer, x, h0, starts, valid)
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(h, j, rtol=1e-10, atol=1e-10)


def test_layer_is_linear_even_though_full_network_has_nonlinearity():
    torch.manual_seed(5)
    layer = LRULayer(3, 4).double()
    x, y = torch.randn(2, 9, 3, dtype=torch.float64), torch.randn(2, 9, 3, dtype=torch.float64)
    starts = torch.zeros(2, 9, dtype=torch.bool)
    f = lambda u: layer(u, episode_start=starts)[0]
    torch.testing.assert_close(f(2*x-3*y), 2*f(x)-3*f(y), atol=1e-10, rtol=1e-10)


def test_hand_computed_impulse_response():
    layer = LRULayer(1, 1).double()
    with torch.no_grad():
        layer.nu_log.fill_(math.log(-math.log(.5)))
        layer.theta_log.fill_(math.log(2*math.pi))
        layer.gamma_log.fill_(0.)
        layer.b_real.fill_(1.); layer.b_imag.zero_()
        layer.c_real.fill_(1.); layer.c_imag.zero_(); layer.skip.zero_()
    x = torch.zeros(1, 5, 1, dtype=torch.float64); x[:, 0] = 1
    y, _ = layer(x, episode_start=torch.zeros(1, 5, dtype=torch.bool))
    torch.testing.assert_close(y.flatten(), torch.tensor([1., .5, .25, .125, .0625], dtype=x.dtype))


def test_parallel_gradients_match_independent_serial_complex():
    x, starts, valid = inputs(7)
    layer = LRULayer(5, 4).double()
    reference = deepcopy(layer)
    x = x.requires_grad_(); xx = x.detach().clone().requires_grad_()
    h0 = torch.randn(3, 4, 2, dtype=x.dtype, requires_grad=True)
    hh0 = h0.detach().clone().requires_grad_()
    # 第一环境给非零真实初态，用于验证初态梯度，不全部被首帧reset清空。
    starts[0, 0] = False
    a, h = layer(x, initial_state=h0, episode_start=starts, valid_mask=valid)
    b, j = independent_complex(reference, xx, hh0, starts, valid)
    weight = torch.randn_like(a)
    (a.mul(weight).sum()+h.square().sum()).backward()
    (b.mul(weight).sum()+j.square().sum()).backward()
    torch.testing.assert_close(x.grad, xx.grad, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(h0.grad, hh0.grad, rtol=1e-8, atol=1e-9)
    for p, q in zip(layer.parameters(), reference.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-8, atol=1e-9)


def test_finite_difference_gradcheck_inputs_initial_state_and_parameters():
    layer = LRULayer(2, 2).double()
    x = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)
    h0 = torch.randn(1, 2, 2, dtype=x.dtype, requires_grad=True)
    starts = torch.tensor([[False, False, True]])
    names, params = zip(*layer.named_parameters())
    def call(u, h, *parameters):
        y, end = torch.func.functional_call(layer, dict(zip(names, parameters)), (u,),
                             {'initial_state': h, 'episode_start': starts})
        return y, end
    assert torch.autograd.gradcheck(call, (x, h0, *params), fast_mode=True)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('mode', ['parallel', 'sequential'])
def test_full_core_whole_sequence_steps_and_input_projection(dtype, mode):
    x, s, v = inputs(17, dtype)
    model = LRUCore(5, 7, 8, n_layers=2, scan_mode=mode).to(dtype)
    full, state = model(x, episode_start=s, valid_mask=v)
    memory, frames = None, []
    for t in range(x.shape[1]):
        y, memory = model.step(x[:, t], memory, episode_start=s[:, t], valid_mask=v[:, t])
        frames.append(y)
    tol = 1e-5 if dtype==torch.float32 else 1e-10
    torch.testing.assert_close(full, torch.stack(frames, 1), rtol=tol, atol=tol)
    torch.testing.assert_close(state.hidden, memory.hidden, rtol=tol, atol=tol)
    assert state.hidden.shape == (2, 3, 7, 2)


def test_no_future_or_cross_environment_or_cross_episode_information():
    x, s, v = inputs(13)
    model = LRUCore(5, 6, 8, n_layers=2).double()
    y, _ = model(x, episode_start=s, valid_mask=v)
    changed = x.clone(); changed[:, 8:] += 100.
    yy, _ = model(changed, episode_start=s, valid_mask=v)
    torch.testing.assert_close(yy[:, :8], y[:, :8], rtol=1e-10, atol=1e-10)
    changed = x.clone(); changed[0, :4] -= 100.
    yy, _ = model(changed, episode_start=s, valid_mask=v)
    torch.testing.assert_close(yy[0, 4:], y[0, 4:], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(yy[1:], y[1:], rtol=1e-10, atol=1e-10)


def test_padding_large_finite_numbers_no_output_or_gradient_pollution():
    x, s, v = inputs(11)
    x[~v] = 1e200
    x.requires_grad_()
    model = LRUCore(5, 6, 8, n_layers=2).double()
    out, _ = model(x, episode_start=s, valid_mask=v)
    out.square().sum().backward()
    assert (out[~v] == 0).all() and (x.grad[~v] == 0).all()
    assert torch.isfinite(x.grad).all()


def test_all_padding_holds_nonzero_state_and_zero_output():
    model = LRUCore(5, 7, 8, n_layers=2)
    x = torch.zeros(2, 7, 5)
    h = model.initial_state(2); h.hidden.normal_()
    old = h.hidden.clone()
    y, end = model(x, episode_start=torch.zeros(2, 7, dtype=torch.bool),
                   valid_mask=torch.zeros(2, 7, dtype=torch.bool), initial_state=h)
    assert (y==0).all()
    torch.testing.assert_close(old, end.hidden, rtol=0, atol=0)
    torch.testing.assert_close(h.hidden, old, rtol=0, atol=0)


@pytest.mark.parametrize('kwargs', [
    {'state_dim': 0}, {'state_dim': True}, {'n_layers': 0}, {'r_min': -.1},
    {'r_min': .999, 'r_max': .9}, {'r_max': 1.}, {'max_phase': 0.},
    {'max_phase': 7.}, {'scan_mode': 'typo'}, {'r_min': float('nan')},
])
def test_lru_bad_parameters_rejected(kwargs):
    args=dict(input_dim=5, state_dim=7, output_dim=8); args.update(kwargs)
    with pytest.raises(ValueError): LRUCore(**args)


def test_bad_state_masks_and_precision_rejected():
    model=LRUCore(5, 7, 8);x,s,v=inputs(7, torch.float32)
    with pytest.raises(TypeError): model(x, episode_start=s, initial_state=RecurrentState(torch.zeros(2,3,7)))
    with pytest.raises(ValueError): model(x, episode_start=s, initial_state=LRUState(torch.zeros(1,3,7)))
    bad=s.clone();bad[2,-1]=True
    with pytest.raises(ValueError): model(x,episode_start=bad,valid_mask=v)
    bad=v.clone();bad[0,1]=False
    with pytest.raises(ValueError): model(x,episode_start=s,valid_mask=bad)
    with pytest.raises(ValueError): model(x.half(),episode_start=s)
    x[0,0,0]=float('nan')
    with pytest.raises(ValueError): model(x,episode_start=s)


def test_coefficient_spectrum_initialization_and_invalid_parameters():
    torch.manual_seed(88)
    model=LRULayer(5, 100, r_min=.8, r_max=.95)
    lam,gamma=model.coefficients()
    radius=lam.norm(dim=-1)
    assert (radius>=.8).all() and (radius<.95).all()
    torch.testing.assert_close(gamma.square(),1-radius.square(),rtol=1e-5,atol=1e-6)
    for value in (float('nan'),float('inf'),-1000.):
        with torch.no_grad(): model.nu_log[0]=value
        with pytest.raises(ValueError):model.coefficients()


def test_long_sequence_stable_and_parallel_matches_serial():
    torch.manual_seed(17)
    model=LRUCore(8,12,8,n_layers=2)
    x=torch.randn(2,513,8)
    starts=torch.zeros(2,513,dtype=torch.bool);starts[:,0]=True;starts[0,199]=True
    with torch.no_grad():
        a,h=model(x,episode_start=starts)
        b,j=model(x,episode_start=starts,scan_mode='sequential')
    torch.testing.assert_close(a,b,rtol=3e-5,atol=1e-5)
    torch.testing.assert_close(h.hidden,j.hidden,rtol=3e-5,atol=1e-5)


def test_outer_cpu_autocast_keeps_lru_memory_float32_and_gradients_finite():
    x,s,v=inputs(9,torch.float32)
    model=LRUCore(5,7,8,n_layers=2)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        y,h=model(x,episode_start=s,valid_mask=v)
        y.square().sum().backward()
    assert y.dtype==h.hidden.dtype==torch.float32
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def setup_lru(graph=False, robot='k1_22dof', cap=512):
    torch.manual_seed(42)
    task=TensorTask(robot,batch=2,contacts=1,objects=1,endings=False)
    dyn=DynamicsModel(task.robot,physics_width=8,memory_width=8,latent_width=4,hidden_dim=16,
        interaction_feedback=True,scene_encoder='gnn' if graph else 'set',
        contact_object_indices=[0] if graph else None)
    pol=build_policy({robot:task.robot},backbone='lru',d_model=16,hidden_dim=16,n_layers=2,
        context_len=2,physics_width=8,latent_width=4,lru_state_dim=10,init_std=.15,max_replay_steps=cap)
    return task,dyn,pol


@pytest.mark.parametrize('graph',[False,True])
def test_lru_exact_ppo_replay_across_updates_and_independent_dynamics(graph):
    task,dyn,p=setup_lru(graph)
    c=PhysicsSequenceCollector(task,dyn,p);ppo=PhysicsSequencePPO(p,PPOConfig(epochs=1,minibatch_envs=1))
    assert _history_size(p) is None
    for i in range(3):
        r=c.collect(7)
        assert (0 if r.prefix is None else r.prefix.starts.shape[1])==i*7
        before=model_digest(dyn);old_policy=model_digest(p)
        stats=ppo.update(r)
        assert stats.old_policy_error<1e-4 and stats.optimizer_steps>0
        assert model_digest(dyn)==before and model_digest(p)!=old_policy
    DynamicsUpdater(dyn,epochs=1).update(r)
    assert model_digest(dyn)!=before
    c.on_dynamics_updated()
    again=c.collect(3)
    assert not again.episode_start[:,0].any()
    assert ppo.verify_old_policy(again)<1e-4


@pytest.mark.parametrize('robot',['k1_22dof','g1_23dof','g1_29dof'])
def test_lru_multi_robot_shapes_restore_and_partial_resets(robot):
    task,dyn,p=setup_lru(robot=robot);task.endings=True
    c=PhysicsSequenceCollector(task,dyn,p);u=PhysicsSequencePPO(p,PPOConfig(epochs=1,minibatch_envs=2))
    for _ in range(3):
        r=c.collect(7)
        assert r.action.shape==(2,7,task.robot.dof)
        assert u.update(r).old_policy_error<1e-4
    _,_,copy=setup_lru(robot=robot)
    copy.load_state_dict(p.state_dict(),strict=True)
    r=c.collect(4)
    assert PhysicsSequencePPO(copy).verify_old_policy(r)<1e-4


def test_lru_prefix_must_not_be_dropped_and_cap_is_not_truncation():
    t,d,p=setup_lru(cap=8);c=PhysicsSequenceCollector(t,d,p)
    c.collect(4);r=c.collect(4)
    r.prefix=None
    with pytest.raises(ValueError):PhysicsSequencePPO(p).verify_old_policy(r)
    with pytest.raises(RuntimeError,match='max_replay_steps'):c.collect(1)


def test_lru_cache_rejects_wrong_env_robot_weights_and_state_kind():
    t,d,p=setup_lru();p.eval();d.eval();t.reset()
    o=t.read();starts=t.episode_start;enc,_=d.step(o,episode_start=starts,env_ids=t.env_ids)
    _,_,_,cache=p.step(t.robot_id,o,enc,t.goal(),env_ids=t.env_ids,episode_start=starts)
    assert isinstance(cache,RecurrentCache) and isinstance(cache.state,LRUState)
    with pytest.raises(ValueError): p.step(t.robot_id,o,enc,t.goal(),env_ids=tuple(reversed(t.env_ids)),episode_start=starts,cache=cache)
    wrong=deepcopy(cache);wrong.owner=cache.owner;wrong.stamp=cache.stamp;wrong.policy_stamp=cache.policy_stamp
    wrong.state=RecurrentState(torch.zeros(2,2,10))
    with pytest.raises(TypeError):p.step(t.robot_id,o,enc,t.goal(),env_ids=t.env_ids,episode_start=starts,cache=wrong)
    with torch.no_grad():next(p.encoder.parameters()).add_(.001)
    with pytest.raises(ValueError,match='参数已改变'):p.step(t.robot_id,o,enc,t.goal(),env_ids=t.env_ids,episode_start=starts,cache=cache)


def test_lru_full_optimizer_checkpoint_roundtrip():
    t,d,p=setup_lru(True);c=PhysicsSequenceCollector(t,d,p)
    u=PhysicsSequencePPO(p,PPOConfig(epochs=1,minibatch_envs=2));v=DynamicsUpdater(d)
    r=c.collect(6);u.update(r);v.update(r)
    stream=io.BytesIO();torch.save(dict(ppo=u.state_dict(),dynamics=v.state_dict()),stream)
    stream.seek(0);saved=torch.load(stream,weights_only=True)
    u.load_state_dict(saved['ppo']);v.load_state_dict(saved['dynamics'])
    c.on_dynamics_updated();r=c.collect(5)
    assert u.verify_old_policy(r)<1e-4


def test_transformer_alias_is_same_weights_and_contains_no_moe_or_recurrent_actor():
    robot={'k1_22dof':get_robot_contract('k1_22dof')}
    args=dict(d_model=16,hidden_dim=16,n_layers=2,physics_width=8,latent_width=4)
    torch.manual_seed(17);a=build_policy(robot,backbone='dense',**args)
    torch.manual_seed(17);b=build_policy(robot,backbone='transformer',**args)
    assert type(a.backbone) is type(b.backbone) is DenseReferenceTransformer
    assert a.get_extra_state()==b.get_extra_state()
    assert model_digest(a)==model_digest(b)
    assert not any(isinstance(m,(ReferenceRoutedMoE,RecurrentCore,LRUCore)) for m in b.modules())
    assert sum(isinstance(m,CausalSelfAttention) for m in b.modules())==2
    assert not any('router' in name for name,_ in b.named_parameters())
    direct=PhysicsPolicyConfig(backbone='transformer')
    assert direct.backbone=='dense'
    b.load_state_dict(a.state_dict(),strict=True)


@pytest.mark.parametrize('kind',['transformer','lru'])
@pytest.mark.parametrize('graph',[False,True])
def test_new_yaml_full_frozen_path_and_structure_counts(kind,graph):
    suffix=('gnn_' if graph else '')+kind
    cfg=load_config(ROOT/f'configs/diagnostic_{suffix}.yaml')
    report=describe_models(cfg)
    assert report['moe_layer_count']==0
    if kind=='lru':assert report['actor_memory_floats_per_env']==128
    task=build_task(cfg,evaluation=True);d,p=build_models(cfg,task)
    out=evaluate_frozen(task,d,p,episodes=1,max_steps=80)
    assert out['completed'] and out['weights_unchanged']
    assert not out['robot_skill_verified']
    assert out['backend']=='analytic_diagnostic_not_humanoid'


@pytest.mark.parametrize('field,value',[
    ('lru_state_dim',0),('lru_r_min',-.1),('lru_r_max',1.),('lru_max_phase',0.),('lru_scan','bad'),
])
def test_bad_lru_yaml_rejected(field,value,tmp_path):
    cfg=load_config(ROOT/'configs/diagnostic_lru.yaml');cfg['model'][field]=value
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):load_config(path)


def test_other_models_reject_misplaced_lru_options(tmp_path):
    with pytest.raises(ValueError):build_policy({'k1':get_robot_contract('k1_22dof')},backbone='transformer',lru_r_min=.9)
    cfg=load_config(ROOT/'configs/diagnostic_lru.yaml');cfg['model']['backbone']='transformer'
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):load_config(path)


def test_checkpoint_semantics_cannot_swap_lru_and_gru():
    m=LRUCore(8,8,8)
    with pytest.raises(ValueError):m.set_extra_state({**m.get_extra_state(),'kind':'gru'})
    _,_,p=setup_lru()
    new=deepcopy(p.get_extra_state());new['config']['lru_scan']='sequential'
    with pytest.raises(ValueError):p.set_extra_state(new)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('scan',['parallel','sequential'])
def test_cuda_lru_scan_and_gradients(scan):
    x,s,v=inputs(33,torch.float32)
    p=LRUCore(5,8,8,n_layers=2,scan_mode=scan).cuda()
    y,_=p(x.cuda(),episode_start=s.cuda(),valid_mask=v.cuda())
    y.square().sum().backward()
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in p.parameters())

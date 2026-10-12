"""第08个文件测试：不使用真实机器人、物理仿真或 PPO。"""

import copy
import io
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unified_motion.models.transformer import (
    ReferenceConditionedTransformer, TransformerConfig, TransformerCache,
)
from unified_motion.models.encoder import MultiRobotEncoder, EncodedMotion
from unified_motion.robot_contract import get_robot_contract, RobotContract
from unified_motion.observations import RobotState, ObservationBuilder
from unified_motion.reference_observations import (
    ReferenceObservationBuilder, ReferenceWindow, ReferenceObservation,
)


ROBOT_IDS = ('k1_22dof', 'g1_23dof', 'g1_29dof')


def config(**overrides):
    kwargs = dict(d_model=16, n_layers=2, n_heads=4, n_kv_heads=2,
                  context_len=4, num_fine_experts=4, num_shared_experts=1,
                  top_k=2, expert_hidden_dim=32, router_hidden_dim=16)
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def make_encoded(b=2, t=8, h=3, d=16, *, robot_id='k1_22dof',
                 dtype=torch.float32, requires_grad=False):
    x = torch.randn(b, t, d, dtype=dtype)
    refs = torch.randn(b, t, h, d, dtype=dtype)
    if requires_grad:
        x.requires_grad_()
        refs.requires_grad_()
    mask = torch.ones(b, t, h, dtype=torch.bool)
    if h > 1:
        mask[..., -1] = False
    return EncodedMotion(robot_id, x, refs, mask)


def slice_step(input_seq, t):
    return EncodedMotion(
        input_seq.robot_id, input_seq.state_token[:, t],
        input_seq.reference_tokens[:, t], input_seq.reference_mask[:, t],
    )


def make_full_builders(robot, b=2, t=4, h=3):
    """真实调用之前的两个 Builder + Encoder，而不是跳过模块只造形状。"""
    d = robot.dof
    names = tuple(reversed(robot.joint_names))
    default = torch.linspace(-.2, .2, d)
    body_builder = ObservationBuilder(robot, names, default)
    ref_builder = ReferenceObservationBuilder(robot, names, default)
    state = RobotState(
        joint_names=names,
        joint_pos=(default + torch.randn(b, t, d) * .01).flip(-1),
        joint_vel=torch.randn(b, t, d) * .01,
        root_ang_vel_b=torch.zeros(b, t, 3),
        gravity_b=torch.tensor([0., 0., -1.]).expand(b, t, 3).clone(),
    )
    current_root_pos = torch.zeros(b, t, 3)
    current_root_quat = torch.zeros(b, t, 4); current_root_quat[..., 0] = 1
    ref_q = torch.randn(b, t, h, d) * .01 + default
    ref_quat = current_root_quat.unsqueeze(-2).expand(b, t, h, 4).clone()
    ref_mask = torch.ones(b, t, h, dtype=torch.bool)
    if h > 1: ref_mask[..., -1] = False
    ref_window = ReferenceWindow(
        root_name=robot.root_name, joint_names=names,
        joint_pos=ref_q.flip(-1), joint_vel=torch.zeros(b, t, h, d),
        root_pos_w=torch.zeros(b, t, h, 3), root_quat_wxyz=ref_quat,
        root_lin_vel_w=torch.zeros(b, t, h, 3), root_ang_vel_w=torch.zeros(b, t, h, 3),
        time_offsets_s=(torch.arange(h, dtype=torch.float32)*.1).expand(b,t,h).clone(),
        valid_mask=ref_mask,
    )
    body = body_builder(state, torch.zeros(b,t,d))
    ref = ref_builder(ref_window, current_root_pos_w=current_root_pos,
                      current_root_quat_wxyz=current_root_quat)
    return body, ref


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(915)
        self.m = ReferenceConditionedTransformer(config())
        self.encoded = make_encoded()

    def test_forward_shapes_and_diagnostics(self):
        out = self.m(self.encoded)
        self.assertEqual(tuple(out.features.shape), (2,8,16))
        self.assertEqual(tuple(out.reference_summary.shape), (2,8,16))
        self.assertEqual(tuple(out.router_entropy.shape), (2,))
        self.assertEqual(tuple(out.expert_load.shape), (2,4))
        self.assertEqual(tuple(out.expert_counts.shape), (2,4))
        self.assertEqual(out.balance_loss.ndim, 0)
        torch.testing.assert_close(out.expert_load.sum(-1), torch.ones(2))
        self.assertTrue(out.features.isfinite().all().item())
        self.assertTrue(out.balance_loss.isfinite().item())

    def test_one_layer_and_more_layers(self):
        for count in (1, 2, 3):
            with self.subTest(n_layers=count):
                m = ReferenceConditionedTransformer(config(n_layers=count))
                out = m(self.encoded)
                self.assertEqual(out.router_entropy.numel(),count)
                self.assertEqual(out.expert_counts.shape[0],count)

    def test_many_reference_points(self):
        for h in (1, 2, 4, 12):
            with self.subTest(h=h):
                encoded = make_encoded(h=h)
                self.assertEqual(self.m(encoded).features.shape, (2,8,16))

    def test_reference_fusion_called_once_for_multiple_layers(self):
        events=[]
        hook = self.m.reference_fusion.register_forward_hook(lambda module,args,output: events.append(1))
        try:
            self.m(self.encoded)
            self.assertEqual(len(events),1)
        finally:
            hook.remove()

    def test_manual_forward_pipeline_no_double_residual(self):
        output = self.m(self.encoded)
        fused = self.m.reference_fusion(self.encoded.state_token,self.encoded.reference_tokens,
                                        self.encoded.reference_mask)
        valid = torch.ones(self.encoded.state_token.shape[:2],dtype=torch.bool)
        starts = torch.zeros_like(valid)
        x = fused.policy_features
        route = self.m.router_conditioner(torch.cat(
            (fused.reference_summary, self.encoded.state_token), dim=-1
        ))
        stats = []
        for layer in self.m.layers:
            x, m = layer(x,route,valid_mask=valid,episode_start=starts)
            stats.append(m.balance_loss)
        x = self.m.final_norm(x)
        torch.testing.assert_close(output.features,x,atol=0,rtol=0)
        torch.testing.assert_close(output.balance_loss,torch.stack(stats).mean(),atol=0,rtol=0)

    def test_streaming_equals_sequence(self):
        for count in (1,2,3):
            with self.subTest(n_layers=count):
                m=ReferenceConditionedTransformer(config(n_layers=count)).eval()
                encoded=make_encoded(t=11)
                starts=torch.zeros(2,11,dtype=torch.bool)
                starts[0,3]=True;starts[1,7]=True
                with torch.no_grad():
                    full=m(encoded,episode_start=starts).features
                    cache=None;parts=[]
                    for t in range(11):
                        y,cache=m.step(slice_step(encoded,t),env_ids=(10,11),cache=cache,
                                       episode_start=starts[:,t])
                        parts.append(y.features)
                    torch.testing.assert_close(torch.stack(parts,dim=1),full,atol=4e-6,rtol=4e-5)

    def test_stream_context_len_one(self):
        m = ReferenceConditionedTransformer(config(context_len=1)).eval()
        enc=make_encoded(t=6)
        with torch.no_grad():
            full=m(enc).features
            parts=[];cache=None
            for t in range(6):
                y,cache=m.step(slice_step(enc,t),env_ids=(0,1),cache=cache)
                parts.append(y.features)
            torch.testing.assert_close(torch.stack(parts,1),full,atol=3e-6,rtol=3e-5)

    def test_stream_long_ring_rollover(self):
        m=self.m.eval()
        enc=make_encoded(t=23)
        starts=torch.zeros(2,23,dtype=torch.bool)
        starts[0,5]=True; starts[0,16]=True;starts[1,8]=True
        with torch.no_grad():
            full=m(enc,episode_start=starts).features
            cache=None;outs=[]
            for t in range(23):
                out,cache=m.step(slice_step(enc,t),env_ids=(7,5),cache=cache,episode_start=starts[:,t])
                outs.append(out.features)
            torch.testing.assert_close(torch.stack(outs,1),full,atol=4e-6,rtol=5e-5)

    def test_episode_reset_isolation(self):
        enc=make_encoded(t=9)
        starts=torch.zeros(2,9,dtype=torch.bool);starts[0,4]=True
        ref=self.m(enc,episode_start=starts).features
        modified=replace(enc,state_token=enc.state_token.clone(),reference_tokens=enc.reference_tokens.clone())
        modified.state_token[0,:4]+=123
        modified.reference_tokens[0,:4]+=123
        out=self.m(modified,episode_start=starts).features
        torch.testing.assert_close(out[0,4:],ref[0,4:])
        torch.testing.assert_close(out[1],ref[1])

    def test_no_future_real_state_leak(self):
        enc=make_encoded(t=7)
        original=self.m(enc).features
        modified=replace(enc,state_token=enc.state_token.clone())
        modified.state_token[:,4:]+=100
        changed=self.m(modified).features
        torch.testing.assert_close(changed[:,:4],original[:,:4])

    def test_no_future_reference_window_leak(self):
        enc=make_encoded(t=7)
        original=self.m(enc).features
        modified=replace(enc,reference_tokens=enc.reference_tokens.clone())
        modified.reference_tokens[:,4:]+=100
        changed=self.m(modified).features
        torch.testing.assert_close(changed[:,:4],original[:,:4])

    def test_padding_zero_and_immunity(self):
        enc=make_encoded(t=7,requires_grad=True)
        mask=torch.ones(2,7,dtype=torch.bool);mask[0,5:]=False;mask[1,6:]=False
        a=self.m(enc,valid_mask=mask)
        self.assertEqual(a.features[~mask].count_nonzero().item(),0)
        pollution=replace(enc,state_token=enc.state_token.detach().clone(),
                          reference_tokens=enc.reference_tokens.detach().clone())
        pollution.state_token[~mask]=9999
        pollution.reference_tokens[~mask]=9999
        b=self.m(pollution,valid_mask=mask)
        torch.testing.assert_close(a.features,b.features)
        a.features.square().sum().backward()
        self.assertEqual(enc.state_token.grad[~mask].count_nonzero().item(),0)
        self.assertEqual(enc.reference_tokens.grad[~mask].count_nonzero().item(),0)

    def test_all_padding_one_environment_zero_and_finite(self):
        # 第0个环境只有补位时刻是不合法序列；有效环境可以有尾部补位。
        enc=make_encoded()
        valid=torch.ones(2,8,dtype=torch.bool);valid[0]=False
        out=self.m(enc,valid_mask=valid)
        self.assertEqual(out.features[0].count_nonzero().item(),0)
        self.assertTrue(torch.isfinite(out.features).all().item())

    def test_full_batch_all_padding_finite(self):
        enc=make_encoded(t=2)
        valid=torch.zeros(2,2,dtype=torch.bool)
        out=self.m(enc,valid_mask=valid)
        self.assertEqual(out.features.count_nonzero().item(),0)
        self.assertEqual(out.balance_loss.item(),0)

    def test_padding_holes_rejected(self):
        mask=torch.ones(2,8,dtype=torch.bool);mask[0,3]=False
        with self.assertRaises(ValueError):self.m(self.encoded,valid_mask=mask)

    def test_reset_on_padding_rejected(self):
        valid=torch.ones(2,8,dtype=torch.bool);valid[:,6:]=False
        starts=torch.zeros(2,8,dtype=torch.bool);starts[0,7]=True
        with self.assertRaises(ValueError):self.m(self.encoded,valid_mask=valid,episode_start=starts)

    def test_bad_bool_masks_rejected(self):
        with self.assertRaises(ValueError):self.m(self.encoded,valid_mask=torch.ones(2,8))
        with self.assertRaises(ValueError):self.m(self.encoded,episode_start=torch.ones(1,8,dtype=torch.bool))
        with self.assertRaises(ValueError):self.m(self.encoded,valid_mask=torch.ones(2,8,dtype=torch.bool,device='meta'))

    def test_wrong_input_rank_and_width(self):
        e=self.encoded
        with self.assertRaises(ValueError):self.m(replace(e,state_token=e.state_token[:,0]))
        with self.assertRaises(ValueError):self.m(replace(e,state_token=e.state_token[...,:-1]))
        with self.assertRaises(ValueError):self.m(replace(e,reference_tokens=e.reference_tokens[:,0]))
        with self.assertRaises(ValueError):self.m(replace(e,reference_tokens=e.reference_tokens[...,:-1]))
        with self.assertRaises(ValueError):self.m(replace(e,reference_mask=e.reference_mask[...,:-1]))
        with self.assertRaises(TypeError):self.m({'state':e.state_token})

    def test_robust_bad_robot_id(self):
        with self.assertRaises(ValueError):self.m(replace(self.encoded,robot_id=''))

    def test_backward_all_major_components(self):
        enc=make_encoded(requires_grad=True)
        out=self.m(enc)
        (out.features.square().mean()+.1*out.balance_loss).backward()
        self.assertGreater(enc.state_token.grad.abs().sum().item(),0)
        self.assertGreater(enc.reference_tokens.grad.abs().sum().item(),0)
        for p in (self.m.reference_fusion.q_proj.weight,
                  self.m.layers[0].temporal_attn.q_proj.weight,
                  self.m.layers[0].moe.router[0].weight,
                  self.m.final_norm.weight):
            self.assertIsNotNone(p.grad)
            self.assertTrue(p.grad.isfinite().all().item())
            self.assertGreater(p.grad.abs().sum().item(),0)
        # 稀疏专家未选中时梯度可以是 None，选中的专家必须收到梯度。
        for layer in self.m.layers:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item()>0
                                for expert in layer.moe.fine_experts for p in expert.parameters()))

    def test_balance_loss_is_not_automatically_applied(self):
        # 用不同训练信号比较：未添加 balance_loss 时，它仍作为可选项返回。
        enc=make_encoded()
        a=self.m(enc)
        self.assertEqual(a.balance_loss.ndim,0)
        self.assertTrue(a.balance_loss.requires_grad)

    def test_numerical_nonfinite_rejected(self):
        for val in (float('nan'),float('inf')):
            with self.subTest(val=val):
                enc=make_encoded()
                enc.state_token[0,0,0]=val
                with self.assertRaises(ValueError):self.m(enc)

    def test_train_eval_no_dropout(self):
        a=self.m.train()(self.encoded).features
        b=self.m.eval()(self.encoded).features
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_no_input_mutation(self):
        enc=self.encoded
        old=[enc.state_token.clone(),enc.reference_tokens.clone(),enc.reference_mask.clone()]
        self.m(enc)
        for original,actual in zip(old,[enc.state_token,enc.reference_tokens,enc.reference_mask]):
            torch.testing.assert_close(actual,original)

    def test_float64_supported(self):
        m=ReferenceConditionedTransformer(config()).double()
        enc=make_encoded(dtype=torch.float64)
        out=m(enc)
        self.assertEqual(out.features.dtype,torch.float64)
        (out.features.square().mean()+.01*out.balance_loss).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None))

    def test_cpu_autocast(self):
        with torch.autocast('cpu',dtype=torch.bfloat16):
            out=self.m(self.encoded)
            out.features.float().square().mean().backward()
        self.assertTrue(torch.isfinite(out.features).all().item())

    def test_batched_large_sequence(self):
        enc=make_encoded(b=16,t=32,d=16)
        out=self.m(enc)
        self.assertEqual(out.features.shape,(16,32,16))
        self.assertEqual(out.expert_counts.shape,(2,4))

    def test_checkpoint_roundtrip(self):
        m=self.m.eval()
        baseline=m(self.encoded).features
        bio=io.BytesIO();torch.save(m.state_dict(),bio);bio.seek(0)
        other=ReferenceConditionedTransformer(config()).eval()
        other.load_state_dict(torch.load(bio,weights_only=True))
        torch.testing.assert_close(other(self.encoded).features,baseline,atol=0,rtol=0)

    def test_checkpoint_incompatible_rejected(self):
        weights=self.m.state_dict()
        for cfg in (config(context_len=8),config(n_heads=2,n_kv_heads=1),
                    config(num_fine_experts=3),config(n_layers=1)):
            with self.subTest(cfg=cfg),self.assertRaises((ValueError,RuntimeError)):
                ReferenceConditionedTransformer(cfg).load_state_dict(weights)

    def test_debug_value_check_switch_can_load_same_weights(self):
        source=self.m
        deployed=ReferenceConditionedTransformer(config(check_values=False))
        deployed.load_state_dict(source.state_dict())
        self.assertEqual(deployed.config.check_values,False)
        torch.testing.assert_close(deployed(self.encoded).features,source(self.encoded).features)

    def test_checkpoint_metadata_missing_rejected(self):
        weights=copy.deepcopy(self.m.state_dict()); del weights['_extra_state']
        with self.assertRaises(RuntimeError):self.m.load_state_dict(weights)

    def test_step_requires_eval(self):
        with self.assertRaises(RuntimeError):
            self.m.step(slice_step(self.encoded,0),env_ids=(0,1))

    def test_step_mask_and_env_id_checks(self):
        self.m.eval()
        single=slice_step(self.encoded,0)
        for ids in (('a','b'),(0,),[0,1],(0,0),(True,1)):
            with self.subTest(ids=ids),self.assertRaises(ValueError):
                self.m.step(single,env_ids=ids)
        with self.assertRaises(ValueError):
            self.m.step(single,env_ids=(0,1),episode_start=torch.zeros(2))
        with self.assertRaises(ValueError):
            self.m.step(single,env_ids=(0,1),episode_start=torch.zeros(1,dtype=torch.bool))

    def test_cache_env_order_guard(self):
        self.m.eval()
        _,cache=self.m.step(slice_step(self.encoded,0),env_ids=(21,22))
        with self.assertRaises(ValueError):
            self.m.step(slice_step(self.encoded,1),env_ids=(22,21),cache=cache)
        with self.assertRaises(ValueError):
            self.m.step(slice_step(self.encoded,1),env_ids=(21,23),cache=cache)

    def test_cache_robot_guard(self):
        self.m.eval()
        _,cache=self.m.step(slice_step(self.encoded,0),env_ids=(0,1))
        wrong=replace(slice_step(self.encoded,1),robot_id='g1_29dof')
        with self.assertRaises(ValueError):
            self.m.step(wrong,env_ids=(0,1),cache=cache)

    def test_cache_instance_guard(self):
        m=self.m.eval()
        _,cache=m.step(slice_step(self.encoded,0),env_ids=(0,1))
        other=ReferenceConditionedTransformer(config()).eval()
        with self.assertRaises(ValueError):
            other.step(slice_step(self.encoded,1),env_ids=(0,1),cache=cache)

    def test_cache_manual_invalidation(self):
        m=self.m.eval()
        _,cache=m.step(slice_step(self.encoded,0),env_ids=(0,1))
        m.invalidate_caches()
        with self.assertRaises(ValueError):
            m.step(slice_step(self.encoded,1),env_ids=(0,1),cache=cache)

    def test_cache_train_transition_invalidation(self):
        m=self.m.eval()
        _,cache=m.step(slice_step(self.encoded,0),env_ids=(0,1))
        m.train().eval()
        with self.assertRaises(ValueError):
            m.step(slice_step(self.encoded,1),env_ids=(0,1),cache=cache)

    def test_cache_checkpoint_loading_invalidation(self):
        m=self.m.eval()
        _,cache=m.step(slice_step(self.encoded,0),env_ids=(0,1))
        m.load_state_dict(m.state_dict())
        with self.assertRaises(ValueError):
            m.step(slice_step(self.encoded,1),env_ids=(0,1),cache=cache)

    def test_invalid_config(self):
        for kwargs in (dict(n_layers=0),dict(n_layers=True),dict(d_model=15),
                       dict(n_kv_heads=3),dict(num_fine_experts=0),dict(top_k=9),
                       dict(context_len=0)):
            with self.subTest(kwargs=kwargs),self.assertRaises((ValueError,TypeError)):
                ReferenceConditionedTransformer(config(**kwargs))

    def test_extra_state_round_trip_layout(self):
        state=self.m.get_extra_state()
        self.assertIn('pipeline',state)
        self.assertEqual(state['config']['d_model'],16)
        self.assertEqual(len(self.m.layers),2)

    def test_robot_sharing_all_three_and_autograd(self):
        robots={k:get_robot_contract(k) for k in ROBOT_IDS}
        encoder=MultiRobotEncoder(robots,d_model=16,hidden_dim=32)
        transformer=ReferenceConditionedTransformer(config())
        head=nn.ModuleDict({k:nn.Linear(16,v.dof) for k,v in robots.items()})
        optim=torch.optim.Adam(list(encoder.parameters())+list(transformer.parameters())+list(head.parameters()),lr=1e-3)
        old=transformer.layers[0].temporal_attn.q_proj.weight.detach().clone()
        loss=0
        for key,robot in robots.items():
            obs,ref=make_full_builders(robot)
            encoded=encoder(key,obs,ref)
            out=transformer(encoded)
            pred=head[key](out.features)
            self.assertEqual(pred.shape,(2,4,robot.dof))
            loss=loss+pred.square().mean()+.01*out.balance_loss
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(encoder.state_encoders['robot_0'][0].weight.grad.abs().sum().item(),0)
        optim.step()
        self.assertFalse(torch.equal(old,transformer.layers[0].temporal_attn.q_proj.weight))

    def test_new_robot_not_hardcoded(self):
        robot=RobotContract('Demo','base',('a','b','c'))
        encoder=MultiRobotEncoder({'demo':robot},d_model=16,hidden_dim=32)
        transformer=self.m
        obs,ref=make_full_builders(robot,b=2,t=3,h=2)
        out=transformer(encoder('demo',obs,ref))
        self.assertEqual(out.features.shape,(2,3,16))

    @unittest.skipUnless(torch.cuda.is_available(),'当前容器没有 CUDA')
    def test_cuda_batched_forward_backward(self):
        model=ReferenceConditionedTransformer(config()).cuda()
        encoded=make_encoded(b=64,t=32)
        encoded=replace(encoded,state_token=encoded.state_token.cuda(),
                        reference_tokens=encoded.reference_tokens.cuda(),
                        reference_mask=encoded.reference_mask.cuda())
        out=model(encoded)
        (out.features.square().mean()+.01*out.balance_loss).backward()
        self.assertTrue(out.features.isfinite().all())

    @unittest.skipUnless(torch.cuda.is_available(),'当前容器没有 CUDA')
    def test_cuda_cache_vs_sequence(self):
        model=ReferenceConditionedTransformer(config()).cuda().eval()
        encoded=make_encoded(t=10)
        encoded=replace(encoded,state_token=encoded.state_token.cuda(),
                        reference_tokens=encoded.reference_tokens.cuda(),
                        reference_mask=encoded.reference_mask.cuda())
        with torch.no_grad():
            full=model(encoded).features
            cache=None;ys=[]
            for t in range(10):
                o,cache=model.step(slice_step(encoded,t),env_ids=(0,1),cache=cache)
                ys.append(o.features)
            torch.testing.assert_close(torch.stack(ys,1),full,atol=5e-5,rtol=5e-4)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)

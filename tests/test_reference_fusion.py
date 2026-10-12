"""第06模块回归测试：独立数学、梯度、补位、时间隔离与多机器人集成。"""
import copy
import io
import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from unified_motion.models.reference_fusion import ReferenceFusion
from unified_motion.models.attention import CausalSelfAttention
from unified_motion.models.encoder import MultiRobotEncoder
from unified_motion.robot_contract import RobotContract, get_robot_contract
from unified_motion.reference_observations import ReferenceObservation
from test_encoder import build_inputs


def manual_fusion(model, x, refs, mask, times):
    """按环境/时间/头显式循环；不用 SDPA 或模块 forward 验证数学含义。"""
    leading, h, d = x.shape[:-1], refs.shape[-2], x.shape[-1]
    xx = x.reshape(-1, d)
    rr = refs.reshape(-1, h, d)
    mm = mask.reshape(-1, h)
    tt = times.reshape(-1)
    rows, summaries = [], []
    groups = model.n_heads // model.n_kv_heads

    def norm(value, layer):
        centered = value - value.mean(-1, keepdim=True)
        variance = centered.square().mean(-1, keepdim=True)
        return centered / torch.sqrt(variance + layer.eps) * layer.weight + layer.bias

    for i in range(xx.shape[0]):
        if not tt[i]:
            rows.append(xx[i] * 0.0)
            summaries.append(rr[i].sum(0) * 0.0)
            continue
        available = rr[i][mm[i]]
        q = (model.q_proj.weight @ norm(xx[i], model.query_norm)).reshape(model.n_heads, model.head_dim)
        normalized = norm(available, model.reference_norm)
        keys = (normalized @ model.k_proj.weight.T).reshape(-1, model.n_kv_heads, model.head_dim)
        values = (normalized @ model.v_proj.weight.T).reshape(-1, model.n_kv_heads, model.head_dim)
        heads = []
        for head in range(model.n_heads):
            scores = keys[:, head // groups] @ q[head] / math.sqrt(model.head_dim)
            heads.append(scores.softmax(-1) @ values[:, head // groups])
        rows.append(xx[i] + model.out_proj.weight @ torch.cat(heads))
        summaries.append(available.mean(0))
    return (torch.stack(rows).reshape(*leading, d),
            torch.stack(summaries).reshape(*leading, d))


class ReferenceFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)

    def fixture(self, leading=(2, 4), h=3, dtype=torch.float32, **kw):
        cfg = dict(d_model=16, n_heads=4, n_kv_heads=2)
        cfg.update(kw)
        model = ReferenceFusion(**cfg).to(dtype=dtype)
        d = cfg['d_model']
        x = torch.randn(*leading, d, dtype=dtype)
        refs = torch.randn(*leading, h, d, dtype=dtype)
        valid = torch.ones(*leading, h, dtype=torch.bool)
        if h > 1:
            valid[..., -1] = False
        times = torch.ones(*leading, dtype=torch.bool)
        return model, x, refs, valid, times

    def test_shapes_mha_gqa_mqa(self):
        for kv in (1, 2, 4):
            for leading in ((3,), (2, 5)):
                for h in (1, 2, 5):
                    with self.subTest(kv=kv, leading=leading, h=h):
                        m,x,r,v,t = self.fixture(leading,h,n_kv_heads=kv)
                        out=m(x,r,v,time_mask=t)
                        self.assertEqual(out.policy_features.shape,x.shape)
                        self.assertEqual(out.reference_summary.shape,x.shape)
                        self.assertTrue(torch.equal(out.time_mask,t))

    def test_independent_math_forward_float64(self):
        for kv in (1,2,4):
            m,x,r,v,t=self.fixture(dtype=torch.float64,n_kv_heads=kv)
            t[0,-1]=False; v[0,-1]=False
            out=m(x,r,v,time_mask=t)
            expected, summary=manual_fusion(m,x,r,v,t)
            torch.testing.assert_close(out.policy_features,expected,atol=1e-12,rtol=1e-10)
            torch.testing.assert_close(out.reference_summary,summary,atol=1e-12,rtol=1e-10)

    def test_independent_math_gradients(self):
        m,x,r,v,t=self.fixture(leading=(2,2),dtype=torch.float64)
        other=copy.deepcopy(m)
        x.requires_grad_(); r.requires_grad_()
        xx=x.detach().clone().requires_grad_(); rr=r.detach().clone().requires_grad_()
        a=m(x,r,v,time_mask=t)
        b,s=manual_fusion(other,xx,rr,v,t)
        weight=torch.randn_like(x); sw=torch.randn_like(x)
        ((a.policy_features*weight).sum()+(a.reference_summary*sw).sum()).backward()
        ((b*weight).sum()+(s*sw).sum()).backward()
        torch.testing.assert_close(x.grad,xx.grad,atol=1e-11,rtol=1e-9)
        torch.testing.assert_close(r.grad,rr.grad,atol=1e-11,rtol=1e-9)
        for (name,p),(other_name,q) in zip(m.named_parameters(),other.named_parameters()):
            self.assertEqual(name,other_name)
            torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-8)

    def test_numerical_autograd_gradcheck(self):
        m,x,r,v,t=self.fixture(leading=(1,),h=2,dtype=torch.float64,d_model=4,n_heads=2,n_kv_heads=1)
        v[:]=True
        x.requires_grad_();r.requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            lambda a,b:m(a,b,v).policy_features,(x,r),fast_mode=True))

    def test_masked_reference_garbage_cannot_change_output(self):
        m,x,r,v,t=self.fixture()
        initial=m(x,r,v)
        for value in (0.0,1e30,-1e30):
            changed=torch.where(v[...,None],r,value)
            out=m(x,changed,v)
            torch.testing.assert_close(out.policy_features,initial.policy_features,atol=0,rtol=0)
            torch.testing.assert_close(out.reference_summary,initial.reference_summary,atol=0,rtol=0)

    def test_extra_padding_not_change_result(self):
        m,x,r,v,t=self.fixture()
        out=m(x,r,v)
        r2=torch.cat((r,torch.randn(*r.shape[:-2],6,r.shape[-1])*100),dim=-2)
        v2=torch.cat((v,torch.zeros(*v.shape[:-1],6,dtype=torch.bool)),dim=-1)
        other=m(x,r2,v2)
        torch.testing.assert_close(out.policy_features,other.policy_features,atol=3e-7,rtol=1e-5)
        torch.testing.assert_close(out.reference_summary,other.reference_summary)

    def test_masked_reference_gradient_is_zero(self):
        m,x,r,v,t=self.fixture()
        x.requires_grad_();r.requires_grad_()
        out=m(x,r,v)
        (out.policy_features.square().sum()+out.reference_summary.square().sum()).backward()
        self.assertEqual(r.grad[~v].count_nonzero().item(),0)
        self.assertGreater(r.grad[v].abs().sum().item(),0)

    def test_valid_reference_changes_output(self):
        m,x,r,v,t=self.fixture()
        out=m(x,r,v).policy_features
        altered=r.clone();altered[...,0,::2]+=4
        self.assertGreater((m(x,altered,v).policy_features-out).abs().max().item(),1e-3)

    def test_last_future_target_is_visible(self):
        # H不是T，不允许三角屏蔽让唯一query只能读取第0个参考。
        m,x,r,v,t=self.fixture(h=4)
        v[:]=True
        out=m(x,r,v).policy_features
        changed=r.clone();changed[...,-1,::2]+=5
        self.assertGreater((m(x,changed,v).policy_features-out).abs().max().item(),1e-3)

    def test_fusion_does_not_mix_timesteps_or_environments(self):
        m,x,r,v,t=self.fixture()
        before=m(x,r,v)
        xx=x.clone();rr=r.clone();xx[1,2]+=torch.arange(16);rr[1,2,:,::2]+=5
        after=m(xx,rr,v)
        keep=torch.ones_like(t);keep[1,2]=False
        torch.testing.assert_close(before.policy_features[keep],after.policy_features[keep],atol=0,rtol=0)
        torch.testing.assert_close(before.reference_summary[keep],after.reference_summary[keep],atol=0,rtol=0)

    def test_reference_summary_only_depends_on_reference(self):
        m,x,r,v,t=self.fixture()
        out=m(x,r,v)
        other=m(torch.randn_like(x)*10,r,v)
        torch.testing.assert_close(out.reference_summary,other.reference_summary,atol=0,rtol=0)
        torch.testing.assert_close(out.reference_summary,r[...,:2,:].mean(-2))

    def test_single_reference_residual_equation(self):
        m,x,r,v,t=self.fixture(h=1,n_kv_heads=4)
        expected=x+m.out_proj(m.v_proj(m.reference_norm(r)).squeeze(-2))
        torch.testing.assert_close(m(x,r,v).policy_features,expected)

    def test_reference_permutation_preserving_time_features(self):
        # 交换完整的带时间token，只改变存放顺序，不改变目标含义。
        m,x,r,v,t=self.fixture(h=4)
        v[:]=True
        permutation=torch.tensor([2,0,3,1])
        out=m(x,r,v);other=m(x,r[...,permutation,:],v[...,permutation])
        torch.testing.assert_close(out.policy_features,other.policy_features)
        torch.testing.assert_close(out.reference_summary,other.reference_summary)

    def test_time_padding_zero_outputs_and_gradients(self):
        m,x,r,v,t=self.fixture()
        t[0,2:]=False;t[1,3:]=False;v[~t]=False
        x.requires_grad_();r.requires_grad_()
        out=m(x,r,v,time_mask=t)
        self.assertEqual(out.policy_features[~t].count_nonzero().item(),0)
        self.assertEqual(out.reference_summary[~t].count_nonzero().item(),0)
        (out.policy_features.square().sum()+out.reference_summary.square().sum()).backward()
        self.assertEqual(x.grad[~t].count_nonzero().item(),0)
        self.assertEqual(r.grad[~t].count_nonzero().item(),0)
        xx=torch.where(t[...,None],x.detach(),1e30)
        rr=torch.where(t[...,None,None],r.detach(),-1e30)
        torch.testing.assert_close(m(xx,rr,v,time_mask=t).policy_features,out.policy_features)

    def test_all_padding_batch_finite_zero(self):
        m,x,r,v,t=self.fixture()
        t[:]=False;v[:]=False
        out=m(x,r,v,time_mask=t)
        self.assertEqual(out.policy_features.count_nonzero().item(),0)
        self.assertEqual(out.reference_summary.count_nonzero().item(),0)
        out.policy_features.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_single_step_and_sequence_equivalence(self):
        m,x,r,v,t=self.fixture()
        seq=m(x,r,v)
        steps=[m(x[:,i],r[:,i],v[:,i]) for i in range(x.shape[1])]
        torch.testing.assert_close(seq.policy_features,torch.stack([s.policy_features for s in steps],1))
        torch.testing.assert_close(seq.reference_summary,torch.stack([s.reference_summary for s in steps],1))

    def test_noncontiguous_inputs(self):
        m,x,r,v,t=self.fixture()
        expected=m(x,r,v)
        out=m(x.transpose(0,1),r.transpose(0,1),v.transpose(0,1))
        torch.testing.assert_close(out.policy_features,expected.policy_features.transpose(0,1))

    def test_no_input_mutation_and_mask_ownership(self):
        m,x,r,v,t=self.fixture()
        originals=[z.clone() for z in (x,r,v,t)]
        out=m(x,r,v,time_mask=t)
        for value,initial in zip((x,r,v,t),originals):
            self.assertTrue(torch.equal(value,initial))
        t.zero_()
        self.assertTrue(out.time_mask.all())

    def test_train_eval_deterministic_equal(self):
        m,x,r,v,t=self.fixture()
        a=m.train()(x,r,v).policy_features
        b=m.eval()(x,r,v).policy_features
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_weight_serialization_roundtrip(self):
        m,x,r,v,t=self.fixture()
        buffer=io.BytesIO();torch.save(m.state_dict(),buffer);buffer.seek(0)
        loaded=ReferenceFusion(d_model=16,n_heads=4,n_kv_heads=2)
        loaded.load_state_dict(torch.load(buffer,weights_only=True))
        torch.testing.assert_close(m(x,r,v).policy_features,loaded(x,r,v).policy_features,atol=0,rtol=0)

    def test_same_shapes_different_heads_checkpoint_rejected(self):
        # 两者KV输出宽度相同，不能只靠参数shape判断兼容。
        m,*_=self.fixture()
        other=ReferenceFusion(d_model=16,n_heads=2,n_kv_heads=1)
        with self.assertRaises(ValueError):other.load_state_dict(m.state_dict())
        state=copy.deepcopy(m.state_dict());state['_extra_state']['schema']=99
        with self.assertRaises(ValueError):m.load_state_dict(state)

    def test_missing_metadata_checkpoint_rejected(self):
        m,*_=self.fixture();state=m.state_dict();del state['_extra_state']
        with self.assertRaises(RuntimeError):m.load_state_dict(state)

    def test_config_errors(self):
        for kw in (dict(d_model=0),dict(d_model=True),dict(n_heads=0),dict(n_kv_heads=5),
                   dict(d_model=15),dict(d_model=1,n_heads=1,n_kv_heads=1),dict(check_values=1)):
            with self.subTest(kw=kw),self.assertRaises((TypeError,ValueError)):
                cfg=dict(d_model=16,n_heads=4,n_kv_heads=2);cfg.update(kw);ReferenceFusion(**cfg)

    def test_odd_head_dimension_allowed_without_rope(self):
        m,x,r,v,t=self.fixture(d_model=12,n_heads=4,n_kv_heads=2)
        self.assertEqual(m(x,r,v).policy_features.shape,x.shape)

    def test_invalid_tensor_types(self):
        m,x,r,v,t=self.fixture()
        for args in ((None,r,v),(x,[],v),(x,r,[])):
            with self.assertRaises(TypeError):m(*args)
        with self.assertRaises(TypeError):m(x,r,v,time_mask=[])

    def test_mismatched_dimensions(self):
        m,x,r,v,t=self.fixture()
        for args in ((x[...,0],r,v),(x[None],r,v),(x[...,:-1],r,v),
                     (x,r[...,:-1],v),(x,r[:1],v[:1]),(x,r,v[:1]),
                     (x,r[:,0],v[:,0])):
            with self.subTest(shapes=[z.shape for z in args]),self.assertRaises(ValueError):m(*args)
        with self.assertRaises(ValueError):m(x,r,v,time_mask=t[:1])

    def test_empty_dimensions(self):
        m,x,r,v,t=self.fixture()
        for args in ((x[:0],r[:0],v[:0]),(x[:,:0],r[:,:0],v[:,:0]),(x,r[...,:0,:],v[...,:0])):
            with self.assertRaises(ValueError):m(*args)

    def test_dtype_and_device_errors(self):
        m,x,r,v,t=self.fixture()
        for args in ((x.double(),r,v),(x,r.long(),v),(x,r,v.float())):
            with self.assertRaises(TypeError):m(*args)
        with self.assertRaises(TypeError):m(x,r,v,time_mask=t.long())
        for args in ((x.to('meta'),r,v),(x,r.to('meta'),v),(x,r,v.to('meta'))):
            with self.assertRaises(ValueError):m(*args)
        with self.assertRaises(ValueError):m(x,r,v,time_mask=t.to('meta'))

    def test_nan_and_inf_rejected_even_in_padding(self):
        m,x,r,v,t=self.fixture()
        for value in (float('nan'),float('inf'),-float('inf')):
            for which in ('x','r'):
                xx=x.clone();rr=r.clone()
                if which=='x':xx[0,0,0]=value
                else:rr[0,0,-1,0]=value
                with self.subTest(which=which,value=value),self.assertRaises(ValueError):m(xx,rr,v)

    def test_no_reference_real_frame_rejected(self):
        m,x,r,v,t=self.fixture();v[0,0]=False
        with self.assertRaises(ValueError):m(x,r,v)

    def test_holes_in_reference_or_time_rejected(self):
        m,x,r,v,t=self.fixture();v[0,0]=torch.tensor([True,False,True])
        with self.assertRaises(ValueError):m(x,r,v)
        v[0,0]=True;t[0]=torch.tensor([True,False,True,True])
        with self.assertRaises(ValueError):m(x,r,v,time_mask=t)

    def test_fast_check_mode_equivalence(self):
        m,x,r,v,t=self.fixture();expected=m(x,r,v)
        m.check_values=False
        torch.testing.assert_close(expected.policy_features,m(x,r,v).policy_features)
        with self.assertRaises(ValueError):m(x[:1],r,v)

    def test_nonfinite_output_rejected(self):
        m,x,r,v,t=self.fixture()
        with torch.no_grad():m.out_proj.weight.fill_(float('inf'))
        with self.assertRaises(ValueError):m(x,r,v)

    def test_half_model_rejected(self):
        m,x,r,v,t=self.fixture();m.half()
        with self.assertRaises(TypeError):m(x.half(),r.half(),v)

    def test_cpu_autocast_float_and_encoded_inputs(self):
        for both_low in (False,True):
            m,x,r,v,t=self.fixture()
            baseline=m(x,r,v).policy_features
            with torch.autocast('cpu',dtype=torch.bfloat16):
                out=m(x.to(torch.bfloat16) if both_low else x,r.to(torch.bfloat16),v)
                out.policy_features.float().square().mean().backward()
            self.assertTrue(torch.isfinite(out.policy_features).all())
            torch.testing.assert_close(out.policy_features.float(),baseline,atol=.035,rtol=.05)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_parameter_count_default(self):
        self.assertEqual(sum(p.numel() for p in ReferenceFusion().parameters()),197632)

    def test_all_existing_modules_three_robots_trainable(self):
        robots={k:get_robot_contract(k) for k in ('k1_22dof','g1_23dof','g1_29dof')}
        encoder=MultiRobotEncoder(robots,d_model=16,hidden_dim=32)
        attention=CausalSelfAttention(d_model=16,n_heads=4,n_kv_heads=2,context_len=3)
        fusion=ReferenceFusion(d_model=16,n_heads=4,n_kv_heads=2)
        heads=nn.ModuleDict({k:nn.Linear(16,robot.dof) for k,robot in robots.items()})
        all_parameters=list(encoder.parameters())+list(attention.parameters())+list(fusion.parameters())+list(heads.parameters())
        optimizer=torch.optim.Adam(all_parameters,lr=1e-3)
        old=encoder.reference_encoders['robot_0'][0].weight.detach().clone()
        loss=0.0
        for key,robot in robots.items():
            obs,ref,*_=build_inputs(robot,leading=(2,4),h=3)
            encoded=encoder(key,obs,ref)
            # 临时组装仅用来测试梯度，并非正式Transformer/Actor。
            history=encoded.state_token+attention(encoded.state_token)
            out=fusion(history,encoded.reference_tokens,encoded.reference_mask)
            prediction=heads[key](out.policy_features)
            self.assertEqual(prediction.shape,(2,4,robot.dof))
            loss=loss+prediction.square().mean()+.01*out.reference_summary.square().mean()
        loss.backward()
        for p in all_parameters:
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(old,encoder.reference_encoders['robot_0'][0].weight))

    def test_new_robot_contract_no_fusion_changes(self):
        robot=RobotContract('test_robot','base',('j1','j2','j3'))
        encoder=MultiRobotEncoder({'test':robot},d_model=16,hidden_dim=32)
        obs,ref,*_=build_inputs(robot,leading=(2,),h=2)
        encoded=encoder('test',obs,ref)
        fusion=ReferenceFusion(d_model=16,n_heads=4,n_kv_heads=2)
        self.assertEqual(fusion(encoded.state_token,encoded.reference_tokens,encoded.reference_mask).policy_features.shape,(2,16))

    def test_temporal_attention_fusion_streaming_reset_and_future_isolation(self):
        attention=CausalSelfAttention(d_model=16,n_heads=4,n_kv_heads=2,context_len=4).eval()
        fusion,x,refs,mask,times=self.fixture(leading=(2,11))
        fusion.eval()
        starts=torch.zeros(2,11,dtype=torch.bool);starts[0,5]=True;starts[1,8]=True
        with torch.no_grad():
            def run(xx,rr):
                hist=xx+attention(xx,episode_start=starts)
                return fusion(hist,rr,mask).policy_features
            full=run(x,refs)
            cache=None;steps=[]
            for i in range(x.shape[1]):
                y,cache=attention.step(x[:,i],cache,episode_start=starts[:,i])
                steps.append(fusion(x[:,i]+y,refs[:,i],mask[:,i]).policy_features)
            torch.testing.assert_close(full,torch.stack(steps,1),atol=2e-6,rtol=2e-5)
            changed=x.clone();changed[:,7:]+=torch.randn_like(changed[:,7:])*10
            rr=refs.clone();rr[:,7:]+=torch.randn_like(rr[:,7:])*10
            torch.testing.assert_close(full[:,:7],run(changed,rr)[:,:7],atol=0,rtol=0)
            before_reset=x.clone();before_reset[0,:5]+=torch.randn_like(before_reset[0,:5])*10
            torch.testing.assert_close(full[0,5:],run(before_reset,refs)[0,5:],atol=0,rtol=0)

    def test_large_batched_input_without_full_robot_training(self):
        m,x,r,v,t=self.fixture(leading=(1024,50),h=3)
        with torch.no_grad():
            out=m(x,r,v)
        self.assertEqual(out.policy_features.shape,(1024,50,16))
        self.assertTrue(torch.isfinite(out.policy_features).all())

    @unittest.skipUnless(torch.cuda.is_available(),'当前没有CUDA设备')
    def test_cuda_forward_backward_and_cpu_parity(self):
        m,x,r,v,t=self.fixture()
        expected=m(x,r,v).policy_features
        m.cuda();out=m(x.cuda(),r.cuda(),v.cuda()).policy_features
        torch.testing.assert_close(out.cpu(),expected,atol=2e-5,rtol=2e-4)
        out.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(),'当前没有CUDA设备')
    def test_cuda_autocast(self):
        m,x,r,v,t=self.fixture();m.cuda()
        with torch.autocast('cuda',dtype=torch.float16):
            out=m(x.cuda(),r.cuda(),v.cuda()).policy_features
        out.float().square().mean().backward()
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)

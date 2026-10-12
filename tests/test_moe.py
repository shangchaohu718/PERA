"""第 07 模块测试：稀疏路由梯度、有效性掩码、跨机器人集成、序列接口。"""
import copy
import io
import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_motion.robot_contract import get_robot_contract
from unified_motion.models.encoder import MultiRobotEncoder
from unified_motion.models.attention import CausalSelfAttention
from unified_motion.models.reference_fusion import ReferenceFusion
from unified_motion.models.moe import ReferenceRoutedMoE, SwiGLUExpert
from test_encoder import build_inputs

IDS = ('k1_22dof', 'g1_23dof', 'g1_29dof')


def make_moe(*, dtype=torch.float32, top_k=2, shared=1, fine=4,
             check_values=True):
    return ReferenceRoutedMoE(
        d_model=16, num_fine_experts=fine, num_shared_experts=shared,
        top_k=top_k, expert_hidden_dim=32, router_hidden_dim=16,
        check_values=check_values,
    ).to(dtype=dtype)


class MoETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(429)

    def fixture(self, *, leading=(2, 5), dtype=torch.float32, **kw):
        m = make_moe(dtype=dtype, **kw)
        x = torch.randn(*leading, 16, dtype=dtype)
        ref = torch.randn_like(x)
        mask = torch.ones(*leading, dtype=torch.bool)
        if len(leading) == 2:
            mask[0, -2:] = False
        return m, x, ref, mask

    def test_swiglu_expert_manual_formula(self):
        expert = SwiGLUExpert(16, 32)
        x = torch.randn(3, 16)
        gate, val = expert.up_gate(x).chunk(2, dim=-1)
        expected = expert.down(torch.nn.functional.silu(gate) * val)
        torch.testing.assert_close(expert(x), expected)

    def test_default_parameter_count(self):
        m = ReferenceRoutedMoE()
        # 9 个 SwiGLU，每个 3*256*512 个参数；Router: 256*256+256 +256*8+8。
        self.assertEqual(sum(p.numel() for p in m.parameters()), 9*3*256*512 + (256*256+256+256*8+8))

    def test_output_shapes_and_probs(self):
        m, x, r, mask = self.fixture()
        out = m(x, r, valid_mask=mask)
        self.assertEqual(out.delta.shape, x.shape)
        self.assertEqual(out.topk_indices.shape, (2,5,2))
        self.assertEqual(out.topk_weights.shape, (2,5,2))
        self.assertEqual(out.router_probs.shape, (2,5,4))
        self.assertEqual(out.expert_counts.shape, (4,))
        self.assertEqual(out.expert_load.shape, (4,))
        self.assertEqual(out.balance_loss.shape, ())
        self.assertEqual(out.router_entropy.shape, ())
        torch.testing.assert_close(out.topk_weights[mask].sum(-1), torch.ones(mask.sum()))
        torch.testing.assert_close(out.router_probs[mask].sum(-1), torch.ones(mask.sum()))
        self.assertTrue((out.topk_indices[mask] >= 0).all().item())
        self.assertTrue((out.topk_indices[mask] < 4).all().item())
        self.assertTrue((out.topk_indices[mask][:,0] != out.topk_indices[mask][:,1]).all().item())
        self.assertEqual(out.expert_counts.sum().item(), int(mask.sum())*2)
        self.assertAlmostEqual(out.expert_load.sum().item(), 1, places=6)
        self.assertGreaterEqual(out.balance_loss.item(), 0)

    def test_mask_padding_has_zero_outputs(self):
        m, x, r, mask = self.fixture()
        out = m(x, r, valid_mask=mask)
        self.assertEqual(out.delta[~mask].abs().sum().item(), 0)
        self.assertEqual(out.router_probs[~mask].abs().sum().item(), 0)
        self.assertEqual(out.topk_weights[~mask].abs().sum().item(), 0)
        self.assertTrue((out.topk_indices[~mask] == -1).all().item())

    def test_invalid_values_are_ignored_if_finite(self):
        m, x, r, mask = self.fixture()
        out = m(x,r,valid_mask=mask)
        xx, rr = x.clone(), r.clone()
        xx[~mask] = 1e5
        rr[~mask] = -1e5
        later = m(xx,rr,valid_mask=mask)
        for k in ('delta','router_probs','topk_weights','expert_load','balance_loss'):
            torch.testing.assert_close(getattr(out,k),getattr(later,k))

    def test_all_padding(self):
        m,x,r,mask = self.fixture()
        mask[:] = False
        out = m(x,r,valid_mask=mask)
        self.assertTrue((out.delta == 0).all().item())
        self.assertTrue((out.router_probs == 0).all().item())
        self.assertTrue((out.topk_weights == 0).all().item())
        self.assertTrue((out.topk_indices == -1).all().item())
        self.assertEqual(out.expert_counts.sum().item(),0)
        self.assertEqual(out.expert_load.sum().item(),0)
        self.assertEqual(out.balance_loss.item(),0)
        self.assertEqual(out.router_entropy.item(),0)

    def test_forward_backward_finite(self):
        m,x,r,mask = self.fixture()
        x.requires_grad_();r.requires_grad_()
        out = m(x,r,valid_mask=mask)
        (out.delta.square().mean() + .01*out.balance_loss).backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertTrue(torch.isfinite(r.grad).all().item())
        self.assertGreater(r.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                            for p in m.router.parameters()))
        self.assertEqual(x.grad[~mask].abs().sum().item(),0)
        self.assertEqual(r.grad[~mask].abs().sum().item(),0)

    def test_sparse_matches_dense_baseline_gradients(self):
        m,x,r,mask = self.fixture(leading=(2,4), dtype=torch.float64)
        x.requires_grad_();r.requires_grad_()
        sparse = m(x,r,valid_mask=mask)
        loss = sparse.delta.square().sum() + .07*sparse.balance_loss
        loss.backward()
        sparse_dx=x.grad.detach().clone();sparse_dr=r.grad.detach().clone()
        sparse_grads={name:(None if p.grad is None else p.grad.detach().clone())
                      for name,p in m.named_parameters()}
        m.zero_grad(set_to_none=True);x.grad=None;r.grad=None
        # 独立计算一次 Router 的数学过程，构建全专家密集基线，不能复用旧计算图。
        logits = m.router(r)
        indices = sparse.topk_indices.detach().clamp_min(0)
        weights = torch.softmax(torch.gather(logits, -1, indices), dim=-1)
        weights = torch.where(mask.unsqueeze(-1), weights, 0.0)
        all_experts = torch.stack([expert(x) for expert in m.fine_experts],dim=-2)
        selected = torch.gather(
            all_experts, -2, indices.unsqueeze(-1).expand(*indices.shape,16),
        )
        dense_fine = (selected*weights.unsqueeze(-1)).sum(dim=-2)
        dense_shared = sum(expert(x) for expert in m.shared_experts)/len(m.shared_experts)
        target = torch.where(mask.unsqueeze(-1), dense_fine+dense_shared,0.0)
        torch.testing.assert_close(sparse.delta, target, atol=1e-12,rtol=1e-12)
        all_probs = torch.softmax(logits, dim=-1)
        mean_probs = (all_probs * mask.unsqueeze(-1)).sum(dim=(0,1)) / mask.sum()
        balance = len(m.fine_experts) * (sparse.expert_load * mean_probs).sum()
        (target.square().sum()+.07*balance).backward()
        torch.testing.assert_close(x.grad, sparse_dx,atol=1e-10,rtol=1e-9)
        torch.testing.assert_close(r.grad, sparse_dr,atol=1e-10,rtol=1e-9)
        for name,p in m.named_parameters():
            # 路由未触发的专家，稀疏实现没有梯度；密集实现则是全零梯度。
            actual = torch.zeros_like(p) if p.grad is None else p.grad
            expected = torch.zeros_like(p) if sparse_grads[name] is None else sparse_grads[name]
            torch.testing.assert_close(actual, expected,atol=1e-10,rtol=1e-9)

    def test_router_probability_changes_with_reference(self):
        m, x, r, mask = self.fixture()
        a = m(x, r, valid_mask=mask)
        b = m(x, -r, valid_mask=mask)
        self.assertFalse(torch.allclose(a.router_probs[mask], b.router_probs[mask]))

    def test_topk1_and_no_shared(self):
        m,x,r,mask=self.fixture(top_k=1, shared=0)
        out=m(x,r,valid_mask=mask)
        self.assertEqual(out.topk_indices.shape[-1],1)
        torch.testing.assert_close(out.topk_weights[mask],torch.ones_like(out.topk_weights[mask]))
        self.assertTrue(torch.isfinite(out.delta).all().item())

    def test_topk_equals_number_of_experts(self):
        m,x,r,mask=self.fixture(top_k=4)
        out=m(x,r,valid_mask=mask)
        self.assertEqual(out.expert_counts.sum().item(),int(mask.sum())*4)
        self.assertTrue((out.expert_counts == mask.sum()).all().item())
        torch.testing.assert_close(out.expert_load, torch.full((4,),.25))

    def test_one_fine_expert(self):
        m,x,r,mask=self.fixture(top_k=1,fine=1,shared=0)
        out=m(x,r,valid_mask=mask)
        torch.testing.assert_close(out.delta[mask],m.fine_experts[0](x[mask]))
        self.assertAlmostEqual(out.balance_loss.item(),1,places=6)

    def test_router_load_detects_collapse(self):
        m,x,r,mask=self.fixture()
        with torch.no_grad():
            m.router[0].weight.zero_();m.router[0].bias.zero_()
            m.router[2].weight.zero_()
            m.router[2].bias.copy_(torch.tensor([1000.0, 900.0, -900.0,-1000.0]))
        out=m(x,r,valid_mask=mask)
        self.assertEqual(out.expert_counts[0].item(),mask.sum().item())
        self.assertEqual(out.expert_counts[1].item(),mask.sum().item())
        self.assertEqual(out.expert_counts[2].item(),0)
        self.assertEqual(out.expert_counts[3].item(),0)
        self.assertTrue(torch.isfinite(out.router_probs).all().item())
        self.assertGreater(out.balance_loss.item(),1.5)

    def test_router_entropy_vs_policy_entropy(self):
        m,x,r,mask=self.fixture()
        out=m(x,r,valid_mask=mask)
        self.assertTrue(0<=out.router_entropy.item()<=math.log(4)+1e-5)
        self.assertFalse(out.router_entropy.requires_grad) # 仅诊断项

    def test_no_input_mutation(self):
        m,x,r,mask=self.fixture()
        copies=[i.clone() for i in (x,r,mask)]
        m(x,r,valid_mask=mask)
        for actual,old in zip((x,r,mask),copies):
            self.assertTrue(torch.equal(actual,old))

    def test_without_mask_all_valid(self):
        m,x,r,mask=self.fixture()
        m(x,r,valid_mask=None)
        self.assertEqual(m(x,r).expert_counts.sum().item(),x.numel()//16*2)

    def test_noncontiguous(self):
        m,x,r,mask=self.fixture()
        # 非连续内存不影响纯逐 token 的 MoE；这里保持所有位置有效。
        orig=m(x,r)
        perm=m(x.transpose(0,1),r.transpose(0,1))
        torch.testing.assert_close(orig.delta.transpose(0,1),perm.delta)
        torch.testing.assert_close(orig.router_probs.transpose(0,1),perm.router_probs)

    def test_2d_3d_equivalence(self):
        m,x,r,mask=self.fixture()
        full=m(x,r,valid_mask=mask)
        parts=[m(x[:,t],r[:,t],valid_mask=mask[:,t]) for t in range(x.shape[1])]
        torch.testing.assert_close(full.delta,torch.stack([z.delta for z in parts],1))
        torch.testing.assert_close(full.router_probs,torch.stack([z.router_probs for z in parts],1))

    def test_train_eval_equal(self):
        m,x,r,mask=self.fixture()
        a=m.train()(x,r,valid_mask=mask)
        b=m.eval()(x,r,valid_mask=mask)
        torch.testing.assert_close(a.delta,b.delta,atol=0,rtol=0)

    def test_save_load_checkpoint(self):
        m,x,r,mask=self.fixture()
        file=io.BytesIO();torch.save(m.state_dict(),file);file.seek(0)
        restored=make_moe();restored.load_state_dict(torch.load(file,weights_only=True))
        torch.testing.assert_close(restored(x,r,valid_mask=mask).delta,
                                   m(x,r,valid_mask=mask).delta,atol=0,rtol=0)

    def test_mismatched_checkpoint_config_rejected(self):
        m,x,r,mask=self.fixture()
        state=m.state_dict()
        for kw in (dict(top_k=1),dict(shared=0),dict(fine=3)):
            new=make_moe(**kw)
            with self.subTest(config=kw),self.assertRaises((ValueError,RuntimeError)):
                new.load_state_dict(state)
        corrupted=copy.deepcopy(state);corrupted['_extra_state']['schema']=99
        with self.assertRaises(ValueError):m.load_state_dict(corrupted)

    def test_invalid_configurations(self):
        for kw in (
            dict(d_model=0),dict(num_fine_experts=0),dict(top_k=0),
            dict(num_shared_experts=-1),dict(top_k=4,num_fine_experts=3),
            dict(router_hidden_dim=0),dict(expert_hidden_dim=0),
            dict(check_values=5),dict(top_k=True),dict(num_fine_experts=True),
        ):
            with self.subTest(kw=kw),self.assertRaises((ValueError,TypeError)):
                ReferenceRoutedMoE(**kw)

    def test_invalid_input_shapes(self):
        m,x,r,mask=self.fixture()
        for xx,rr in ((x[0],r),(x[:1],r),(x[:,:,:-1],r[:,:,:-1]),
                      (x.unsqueeze(1),r.unsqueeze(1))):
            with self.subTest(shape=xx.shape), self.assertRaises((ValueError,TypeError)):
                m(xx,rr)

    def test_nonfinite_values_rejected(self):
        m,x,r,mask=self.fixture()
        for value in (float('nan'),float('inf'),float('-inf')):
            for which in ('x','router_context'):
                xx=x.clone();rr=r.clone()
                if which=='x':xx[0,0,0]=value
                else:rr[0,0,0]=value
                with self.subTest(which=which,value=value),self.assertRaises(ValueError):
                    m(xx,rr,valid_mask=mask)

    def test_invalid_masks(self):
        m,x,r,mask=self.fixture()
        for bad in (mask[:1],mask.float(),mask.unsqueeze(-1),mask.to('meta')):
            with self.subTest(shape=bad.shape,device=str(bad.device)),self.assertRaises(ValueError):
                m(x,r,valid_mask=bad)
        bad=mask.clone();bad[0]=torch.tensor([True,False,True,False,False])
        with self.assertRaises(ValueError):m(x,r,valid_mask=bad)

    def test_wrong_dtype(self):
        m,x,r,mask=self.fixture()
        with self.assertRaises(TypeError):m(x.double(),r.double(),valid_mask=mask)
        with self.assertRaises(TypeError):m(x,r.long(),valid_mask=mask)
        with self.assertRaises(TypeError):m.half()(x.half(),r.half(),valid_mask=mask)

    def test_float64(self):
        m,x,r,mask=self.fixture(dtype=torch.float64)
        result=m(x,r,valid_mask=mask)
        self.assertEqual(result.delta.dtype,torch.float64)
        self.assertEqual(result.router_probs.dtype,torch.float64)

    def test_cpu_autocast_bfloat16(self):
        m,x,r,mask=self.fixture()
        with torch.autocast('cpu',dtype=torch.bfloat16):
            out=m(x,r,valid_mask=mask)
            loss=out.delta.square().mean()+.01*out.balance_loss
            loss.backward()
        self.assertTrue(torch.isfinite(out.delta).all().item())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                            for p in m.router.parameters()))

    def test_mask_zero_gradients(self):
        m,x,r,mask=self.fixture()
        x.requires_grad_();r.requires_grad_()
        out=m(x,r,valid_mask=mask)
        out.delta.square().sum().backward()
        self.assertEqual(x.grad[~mask].abs().sum().item(),0)
        self.assertEqual(r.grad[~mask].abs().sum().item(),0)

    def test_new_robot_semantic_independence(self):
        # 网络只接收统一 D 的特征；不硬编码 K1/G1 的关节号。
        m,x,r,mask=self.fixture()
        self.assertEqual(m(x,r).delta.shape,x.shape)

    def test_three_robot_end_to_end_integration(self):
        robots={key:get_robot_contract(key) for key in IDS}
        encoder=MultiRobotEncoder(robots,d_model=16,hidden_dim=32)
        attention=CausalSelfAttention(d_model=16,n_heads=4,n_kv_heads=2,context_len=4)
        fusion=ReferenceFusion(d_model=16,n_heads=4,n_kv_heads=2)
        moe=make_moe()
        heads=nn.ModuleDict({key:nn.Linear(16,robot.dof) for key,robot in robots.items()})
        opts=list(encoder.parameters())+list(attention.parameters())+list(fusion.parameters())+list(moe.parameters())+list(heads.parameters())
        optim=torch.optim.Adam(opts,lr=1e-3)
        total=0.
        for key,robot in robots.items():
            obs, ref, *_ = build_inputs(robot,leading=(2,4),h=3)
            encoded=encoder(key,obs,ref)
            history=encoded.state_token+attention(encoded.state_token)
            fused=fusion(history,encoded.reference_tokens,encoded.reference_mask)
            moe_out=moe(fused.policy_features, fused.reference_summary, valid_mask=fused.time_mask)
            action_head_input=fused.policy_features+moe_out.delta # 外层加一次残差
            action=heads[key](action_head_input)
            self.assertEqual(action.shape,(2,4,robot.dof))
            total=total+action.square().mean()+.01*moe_out.balance_loss
        total.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                            for p in moe.router.parameters()))
        self.assertTrue(any(p.grad is not None for p in moe.fine_experts.parameters()))
        optim.step()

    def test_large_batch(self):
        m=make_moe(check_values=False)
        x=torch.randn(256,16,16)
        ref=torch.randn_like(x)
        with torch.no_grad():
            out=m(x,ref)
        self.assertEqual(out.delta.shape,x.shape)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA 不可用，跳过 GPU 测试')
    def test_cuda_training(self):
        m,x,r,mask=self.fixture(leading=(4,6))
        m=m.cuda();x=x.cuda().requires_grad_();r=r.cuda().requires_grad_();mask=mask.cuda()
        result=m(x,r,valid_mask=mask)
        (result.delta.square().mean()+.01*result.balance_loss).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA 不可用，跳过 GPU 测试')
    def test_cuda_autocast(self):
        m,x,r,mask=self.fixture(leading=(4,6))
        m=m.cuda();x=x.cuda();r=r.cuda();mask=mask.cuda()
        with torch.autocast('cuda',dtype=torch.float16):
            y=m(x,r,valid_mask=mask)
            (y.delta.float().square().mean()+.01*y.balance_loss).backward()
        self.assertTrue(torch.isfinite(y.delta).all())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)

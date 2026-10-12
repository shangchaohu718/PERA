"""第 05 个模块的回归测试：合成数据，不启动物理仿真或 PPO。"""
import copy
import io
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from unified_motion.models.attention import CausalSelfAttention, AttentionCache
from unified_motion.models.encoder import MultiRobotEncoder
from unified_motion.robot_contract import get_robot_contract
from test_encoder import build_inputs


def naive_attention(model, x, valid, starts):
    """独立的慢速数学实现：不调用模块的 _project/_rope/_attend 或 SDPA。"""
    batch = []
    groups = model.n_heads // model.n_kv_heads
    e = model.head_dim

    def rotary(vec, position):
        out = []
        for pair in range(e // 2):
            angle = position * model.rope_base ** (-2 * pair / e)
            a, b = vec[2 * pair], vec[2 * pair + 1]
            out.extend((a * math.cos(angle) - b * math.sin(angle),
                        a * math.sin(angle) + b * math.cos(angle)))
        return torch.stack(out)

    for env in range(x.shape[0]):
        outputs, segment_start = [], 0
        for i in range(x.shape[1]):
            if not valid[env, i]:
                outputs.append(x[env, i] * 0)
                continue
            if starts[env, i]:
                segment_start = i
            query_all = model.q_proj(x[env, i]).reshape(model.n_heads, e)
            head_outputs = []
            for head in range(model.n_heads):
                query = rotary(query_all[head], i - segment_start)
                keys, values = [], []
                for j in range(max(segment_start, i-model.context_len+1), i+1):
                    key = model.k_proj(x[env, j]).reshape(model.n_kv_heads, e)[head // groups]
                    keys.append(rotary(key, j-segment_start))
                    values.append(model.v_proj(x[env, j]).reshape(model.n_kv_heads, e)[head // groups])
                score = (torch.stack(keys) @ query) / math.sqrt(e)
                head_outputs.append(score.softmax(-1) @ torch.stack(values))
            joined = torch.cat(head_outputs)
            if model.gate_proj is not None:
                joined = joined * model.gate_proj(x[env, i]).sigmoid()
            outputs.append(model.out_proj(joined))
        batch.append(torch.stack(outputs))
    return torch.stack(batch)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)

    def model(self, **kwargs):
        cfg = dict(d_model=16, n_heads=4, n_kv_heads=2, context_len=4)
        cfg.update(kwargs)
        return CausalSelfAttention(**cfg)

    def streamed(self, model, x, starts=None):
        cache, output = None, []
        for t in range(x.shape[1]):
            y, cache = model.step(x[:, t], cache, episode_start=None if starts is None else starts[:, t])
            output.append(y)
        return torch.stack(output, 1), cache

    def test_shape_mha_gqa_mqa_and_gate_variants(self):
        for heads in (1, 2, 4):
            for gated in (False, True):
                m = self.model(n_kv_heads=heads, output_gate=gated)
                self.assertEqual(m(torch.randn(2, 9, 16)).shape, (2, 9, 16))

    def test_independent_mathematical_oracle(self):
        for heads in (1, 2, 4):
            for gated in (False, True):
                with self.subTest(kv=heads, gated=gated):
                    m = self.model(n_kv_heads=heads, output_gate=gated).double()
                    if gated:
                        with torch.no_grad(): m.gate_proj.weight.normal_(0, .2)
                    x = torch.randn(2, 7, 16, dtype=torch.float64)
                    valid = torch.ones(2, 7, dtype=torch.bool); valid[1, 5:] = False
                    starts = torch.zeros_like(valid); starts[0, 3] = True
                    result = m(x, valid_mask=valid, episode_start=starts)
                    expected = naive_attention(m, x, valid, starts)
                    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-10)

    def test_parameter_gradients_against_independent_oracle(self):
        m = self.model().double()
        with torch.no_grad(): m.gate_proj.weight.normal_(0, .1)
        other = copy.deepcopy(m)
        x = torch.randn(1, 5, 16, dtype=torch.float64, requires_grad=True)
        x2 = x.detach().clone().requires_grad_()
        valid = torch.tensor([[True, True, True, True, False]])
        starts = torch.tensor([[False, False, True, False, False]])
        target = torch.randn_like(x)
        (m(x, valid_mask=valid, episode_start=starts) * target).sum().backward()
        (naive_attention(other, x2, valid, starts) * target).sum().backward()
        torch.testing.assert_close(x.grad, x2.grad, atol=1e-11, rtol=1e-9)
        for p, p2 in zip(m.parameters(), other.parameters()):
            torch.testing.assert_close(p.grad, p2.grad, atol=1e-11, rtol=1e-9)

    def test_no_future_value_leakage(self):
        m = self.model()
        x = torch.randn(2, 12, 16)
        a = m(x)
        changed = x.clone(); changed[:, 5:] += torch.randn_like(changed[:, 5:]) * 100
        torch.testing.assert_close(a[:, :5], m(changed)[:, :5])

    def test_no_future_gradient_leakage(self):
        x = torch.randn(2, 9, 16, requires_grad=True)
        m = self.model()
        m(x)[:, 3].square().sum().backward()
        self.assertEqual(torch.count_nonzero(x.grad[:, 4:]).item(), 0)
        self.assertGreater(torch.count_nonzero(x.grad[:, :3]).item(), 0)

    def test_no_cross_environment_leakage(self):
        m = self.model()
        x = torch.randn(2, 7, 16)
        before = m(x)
        x[1] *= 100
        torch.testing.assert_close(before[0], m(x)[0])

    def test_window_excludes_distant_history(self):
        m = self.model(context_len=3)
        x = torch.randn(2, 10, 16)
        before = m(x)
        x[:, :7] += 100
        torch.testing.assert_close(before[:, -1], m(x)[:, -1])

    def test_window_one_is_memoryless(self):
        m = self.model(context_len=1)
        x = torch.randn(2, 6, 16)
        full = m(x)
        single = torch.cat([m(x[:, i:i+1]) for i in range(6)], dim=1)
        torch.testing.assert_close(full, single)

    def test_padding_zero_no_value_or_gradient_effect(self):
        m = self.model()
        x = torch.randn(2, 8, 16, requires_grad=True)
        valid = torch.tensor([[True]*3+[False]*5, [True]*6+[False]*2])
        out = m(x, valid_mask=valid)
        self.assertTrue(torch.equal(out[~valid], torch.zeros_like(out[~valid])))
        modified = torch.where(valid.unsqueeze(-1), x.detach(), torch.finfo(x.dtype).max)
        torch.testing.assert_close(out.detach(), m(modified, valid_mask=valid))
        out.square().sum().backward()
        self.assertEqual(torch.count_nonzero(x.grad[~valid]).item(), 0)

    def test_all_padding_has_finite_zero_forward_and_backward(self):
        m = self.model()
        x = torch.randn(3, 5, 16, requires_grad=True)
        out = m(x, valid_mask=torch.zeros(3, 5, dtype=torch.bool))
        self.assertEqual(torch.count_nonzero(out).item(), 0)
        out.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for p in m.parameters(): self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(torch.count_nonzero(x.grad).item(), 0)

    def test_episode_reset_blocks_old_history(self):
        m = self.model(context_len=20)
        x = torch.randn(2, 9, 16, requires_grad=True)
        starts = torch.zeros(2, 9, dtype=torch.bool); starts[0, 4] = True
        out = m(x, episode_start=starts)
        torch.testing.assert_close(out[0, 4:], m(x[0:1, 4:])[0])
        torch.testing.assert_close(out[1], m(x[1:2])[0])
        out[0, 5].sum().backward()
        self.assertEqual(torch.count_nonzero(x.grad[0, :4]).item(), 0)

    def test_consecutive_episode_starts(self):
        m = self.model(context_len=20)
        x = torch.randn(2, 5, 16)
        a = m(x, episode_start=torch.ones(2, 5, dtype=torch.bool))
        b = torch.cat([m(x[:, i:i+1]) for i in range(5)], dim=1)
        torch.testing.assert_close(a, b)

    def test_rope_pair_norm_and_position_sensitivity(self):
        m = self.model()
        x = torch.randn(2, 4, 7, 4)
        pos = torch.arange(7).repeat(2, 1)
        y = m._rope(x, pos)
        torch.testing.assert_close(x.norm(dim=-1), y.norm(dim=-1))
        torch.testing.assert_close(x[:, :, 0], y[:, :, 0])
        self.assertFalse(torch.allclose(x[:, :, 1:], y[:, :, 1:]))

    def test_rope_relative_shift_invariance(self):
        m = self.model().double()
        x = torch.randn(2, 5, 16, dtype=torch.float64)
        p = torch.arange(5).repeat(2, 1)
        q, k, _ = m._project(x, p)
        q2, k2, _ = m._project(x, p+71)
        k = k.repeat_interleave(2, 1); k2 = k2.repeat_interleave(2, 1)
        torch.testing.assert_close(q @ k.transpose(-1, -2), q2 @ k2.transpose(-1, -2),
                                   atol=1e-11, rtol=1e-10)

    def test_full_equals_stream_with_window_and_resets(self):
        for dtype in (torch.float32, torch.float64):
            for window in (1, 4, 40):
                for heads in (1, 2, 4):
                    for gate in (False, True):
                        with self.subTest(dtype=dtype, window=window, kv=heads, gate=gate):
                            m = self.model(context_len=window, n_kv_heads=heads, output_gate=gate).to(dtype).eval()
                            x = torch.randn(3, 19, 16, dtype=dtype)
                            starts = torch.zeros(3, 19, dtype=torch.bool)
                            starts[0, [0, 6, 7, 15]] = True; starts[1, [3, 12]] = True
                            with torch.no_grad(): full = m(x, episode_start=starts)
                            stream, _ = self.streamed(m, x, starts)
                            torch.testing.assert_close(full, stream, atol=2e-6, rtol=2e-5)

    def test_cache_storage_constant_and_has_only_kv_heads(self):
        m = self.model().eval()
        _, cache = m.step(torch.randn(2, 16))
        ptrs = (cache.keys.data_ptr(), cache.values.data_ptr())
        self.assertEqual(cache.keys.shape, (2, 2, 4, 4))
        for _ in range(30): _, cache = m.step(torch.randn(2, 16), cache)
        self.assertEqual(ptrs, (cache.keys.data_ptr(), cache.values.data_ptr()))
        self.assertTrue(cache.valid.all())
        self.assertEqual(cache.slot, 31 % 4)
        torch.testing.assert_close(cache.next_position, torch.full((2,), 31, dtype=torch.long))

    def test_reset_cache_is_per_environment(self):
        m = self.model(context_len=4).eval()
        _, cache = self.streamed(m, torch.randn(2, 6, 16))
        x = torch.randn(2, 16)
        result, cache = m.step(x, cache, episode_start=torch.tensor([True, False]))
        torch.testing.assert_close(result[0], m(x[0:1, None])[0, 0])
        self.assertEqual(cache.valid[0].sum().item(), 1)
        self.assertEqual(cache.valid[1].sum().item(), 4)
        self.assertEqual(cache.next_position.tolist(), [1, 7])

    def test_step_no_grad_and_eval_required(self):
        m = self.model()
        x = torch.randn(2, 16, requires_grad=True)
        with self.assertRaises(RuntimeError): m.step(x)
        y, cache = m.eval().step(x)
        self.assertFalse(y.requires_grad)
        self.assertFalse(cache.keys.requires_grad)
        self.assertFalse(cache.values.requires_grad)

    def test_caches_not_shared_between_layers(self):
        a, b = self.model().eval(), self.model().eval()
        x = torch.randn(2, 16)
        _, cache = a.step(x)
        with self.assertRaises(ValueError): b.step(x, cache)

    def test_explicit_cache_invalidation_after_optimizer_update(self):
        m = self.model()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        m.eval(); _, cache = m.step(torch.randn(2, 16))
        m.train(); m(torch.randn(2, 4, 16)).square().mean().backward(); opt.step()
        m.invalidate_caches(); m.eval()
        with self.assertRaises(ValueError): m.step(torch.randn(2, 16), cache)
        y, fresh = m.step(torch.randn(2, 16))
        self.assertTrue(torch.isfinite(y).all())
        self.assertIsNot(fresh._owner, cache._owner)

    def test_cache_invalidated_on_checkpoint_load(self):
        m = self.model().eval()
        _, cache = m.step(torch.randn(2, 16))
        m.load_state_dict(copy.deepcopy(m.state_dict()))
        with self.assertRaises(ValueError): m.step(torch.randn(2, 16), cache)

    def test_state_dict_round_trip_and_config_guard(self):
        m = self.model()
        x = torch.randn(2, 5, 16)
        buffer = io.BytesIO(); torch.save(m.state_dict(), buffer); buffer.seek(0)
        state = torch.load(buffer, weights_only=True)
        b = self.model(); b.load_state_dict(state)
        torch.testing.assert_close(m(x), b(x))
        self.assertFalse(any('cache' in key or 'pair_dims' in key for key in state))
        for cfg in ({'context_len':5}, {'rope_base':20000}, {'n_heads':2, 'n_kv_heads':1}):
            with self.subTest(cfg=cfg), self.assertRaises((ValueError, RuntimeError)):
                self.model(**cfg).load_state_dict(state)

    def test_train_eval_forward_same_without_dropout(self):
        m = self.model()
        x = torch.randn(2, 6, 16)
        torch.testing.assert_close(m.train()(x), m.eval()(x))

    def test_noncontiguous_sequence(self):
        m = self.model()
        x = torch.randn(7, 3, 16).transpose(0, 1)
        self.assertFalse(x.is_contiguous())
        torch.testing.assert_close(m(x), m(x.contiguous()))

    def test_no_input_mutation(self):
        m = self.model().eval()
        x = torch.randn(2, 5, 16)
        valid = torch.ones(2, 5, dtype=torch.bool); valid[0, -1] = False
        starts = torch.zeros_like(valid); starts[1, 3] = True
        originals = tuple(v.clone() for v in (x, valid, starts))
        m(x, valid_mask=valid, episode_start=starts)
        self.streamed(m, x, starts)
        for old, now in zip(originals, (x, valid, starts)): self.assertTrue(torch.equal(old, now))

    def test_all_attention_parameters_receive_finite_gradient(self):
        m = self.model()
        x = torch.randn(3, 5, 16, requires_grad=True)
        (m(x) * torch.randn_like(x)).sum().backward()
        for name, p in m.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.abs().sum().item(), 0, name)

    def test_invalid_constructor_values(self):
        for key in ('d_model','n_heads','n_kv_heads','context_len'):
            for value in (0, -1, True, 1.5):
                with self.subTest(key=key,value=value), self.assertRaises(ValueError): self.model(**{key:value})
        for config in ({'d_model':15},{'n_kv_heads':3},{'d_model':12}, {'rope_base':float('inf')},
                       {'rope_base':float('nan')},{'rope_base':1}):
            with self.subTest(config=config), self.assertRaises(ValueError): self.model(**config)
        for config in ({'rope_base':True},{'rope_base':'10000'}, {'output_gate':1},{'check_values':1}):
            with self.subTest(config=config), self.assertRaises(TypeError): self.model(**config)

    def test_invalid_input_types_shapes(self):
        m = self.model()
        with self.assertRaises(TypeError): m([])
        for x in (torch.randn(16),torch.randn(2,16),torch.randn(2,3,4,16),
                  torch.randn(0,5,16),torch.randn(2,0,16),torch.randn(2,5,15)):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError): m(x)
        with self.assertRaises(ValueError): m.eval().step(torch.randn(2,3,16))
        with self.assertRaises(TypeError): m.eval().step([])

    def test_input_dtype_and_device_rejected(self):
        m = self.model()
        for dtype in (torch.float64, torch.float16, torch.bfloat16, torch.long):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError): m(torch.ones(2,3,16,dtype=dtype))
        with self.assertRaises(ValueError): m(torch.empty(2,3,16,device='meta'))
        with self.assertRaises(TypeError): m.half()(torch.zeros(2,3,16).half())

    def test_masks_shapes_types_devices(self):
        m = self.model()
        x = torch.randn(2, 5, 16)
        for key in ('valid_mask','episode_start'):
            with self.subTest(key=key), self.assertRaises(TypeError): m(x, **{key:[]})
            with self.subTest(key=key), self.assertRaises(TypeError): m(x, **{key:torch.ones(2,5)})
            with self.subTest(key=key), self.assertRaises(ValueError): m(x, **{key:torch.ones(1,5,dtype=torch.bool)})
            with self.subTest(key=key), self.assertRaises(ValueError): m(x, **{key:torch.ones(2,5,dtype=torch.bool,device='meta')})
        with self.assertRaises(TypeError): m.eval().step(x[:,0], episode_start=torch.ones(2))

    def test_padding_hole_and_start_in_padding_rejected(self):
        m = self.model()
        x = torch.randn(1, 3, 16)
        with self.assertRaises(ValueError): m(x, valid_mask=torch.tensor([[True,False,True]]))
        with self.assertRaises(ValueError):
            m(x, valid_mask=torch.tensor([[True,True,False]]), episode_start=torch.tensor([[False,False,True]]))

    def test_nonfinite_inputs_rejected_even_in_padding(self):
        m = self.model().eval()
        for bad in (float('nan'),float('inf'),float('-inf')):
            x = torch.randn(1,3,16);x[0,-1,0]=bad
            with self.subTest(bad=bad), self.assertRaises(ValueError): m(x,valid_mask=torch.tensor([[True,True,False]]))
            with self.subTest(bad=bad), self.assertRaises(ValueError): m.step(x[:,-1])

    def test_output_nonfinite_rejected(self):
        m = self.model()
        with torch.no_grad(): m.out_proj.weight.fill_(float('nan'))
        with self.assertRaises(ValueError): m(torch.randn(1,3,16))

    def test_cache_bad_shape_dtype_device_slot(self):
        m = self.model().eval(); x=torch.randn(2,16)
        _, c = m.step(x)
        invalids = (
            replace(c, keys=c.keys[:1]), replace(c, keys=c.keys.double()),
            replace(c, keys=c.keys.to('meta')), replace(c, valid=c.valid.float()),
            replace(c, slot=4),replace(c, slot=-1),replace(c, slot=True),
            replace(c,next_position=c.next_position.float()),
            replace(c,next_position=-torch.ones_like(c.next_position)),
            replace(c, values=c.values.clone().requires_grad_()),
        )
        for bad in invalids:
            with self.assertRaises(ValueError): m.step(x,bad)
        with self.assertRaises(ValueError): m.step(torch.randn(3,16),c)
        with self.assertRaises(ValueError): m.step(x,{})
        c.keys[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError): m.step(x,c)

    def test_check_values_false_matches_valid_path(self):
        m = self.model()
        b = self.model(check_values=False); b.load_state_dict(m.state_dict())
        x = torch.randn(2,8,16)
        torch.testing.assert_close(m(x),b(x))
        a,_=self.streamed(m.eval(),x); c,_=self.streamed(b.eval(),x)
        torch.testing.assert_close(a,c)
        with self.assertRaises(TypeError): b(x.long())
        with self.assertRaises(ValueError): b(x[:,:,:-1])

    def test_cpu_autocast_and_encoder_autocast_input(self):
        m = self.model().eval()
        x = torch.randn(2, 13, 16)
        starts=torch.zeros(2,13,dtype=torch.bool); starts[0,6]=True
        with torch.autocast('cpu',dtype=torch.bfloat16):
            y=m(x,episode_start=starts)
            streamed,cache=self.streamed(m,x,starts)
            self.assertEqual(y.dtype,torch.bfloat16)
            self.assertEqual(cache.keys.dtype,torch.bfloat16)
            torch.testing.assert_close(y,streamed,atol=.006,rtol=.04)
            robot=get_robot_contract('k1_22dof')
            encoder=MultiRobotEncoder({'k1_22dof':robot},d_model=16,hidden_dim=32)
            obs,ref,*_=build_inputs(robot,leading=(2,5))
            encoded=encoder('k1_22dof',obs,ref)
            self.assertEqual(encoded.state_token.dtype,torch.bfloat16)
            output=m(encoded.state_token)
            self.assertTrue(torch.isfinite(output).all())
        with self.assertRaises(ValueError): m.step(x[:,0],cache)

    def test_autocast_backward_finite(self):
        m=self.model()
        x=torch.randn(2,6,16,requires_grad=True)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            loss=m(x).float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for p in m.parameters(): self.assertTrue(torch.isfinite(p.grad).all())

    def test_all_robot_builders_encoder_shared_attention_pipeline(self):
        ids=('k1_22dof','g1_23dof','g1_29dof')
        robots={i:get_robot_contract(i) for i in ids}
        encoder=MultiRobotEncoder(robots,d_model=16,hidden_dim=32)
        attn=self.model()
        optimizer=torch.optim.Adam(list(encoder.parameters())+list(attn.parameters()),lr=1e-3)
        before=attn.q_proj.weight.detach().clone()
        for robot_id in ids:
            obs,ref,*_=build_inputs(robots[robot_id],leading=(2,5))
            encoded=encoder(robot_id,obs,ref)
            y=attn(encoded.state_token)
            self.assertEqual(y.shape,(2,5,16))
            (y*torch.randn_like(y)).sum().backward()
        self.assertGreater(attn.q_proj.weight.grad.abs().sum().item(),0)
        for branch in encoder.state_encoders.values():
            self.assertTrue(all(p.grad is not None for p in branch.parameters()))
        # 参考融合尚未接入；这里不冒称已经训练参考分支。
        self.assertTrue(all(p.grad is None for p in encoder.reference_encoders.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before,attn.q_proj.weight))
        attn.invalidate_caches()

    def test_pipeline_batch_vs_frame_all_robots(self):
        ids=('k1_22dof','g1_23dof','g1_29dof')
        robots={i:get_robot_contract(i) for i in ids}
        encoder=MultiRobotEncoder(robots,d_model=16,hidden_dim=32).eval()
        attn=self.model().eval()
        with torch.no_grad():
            for robot_id in ids:
                obs,ref,*_=build_inputs(robots[robot_id],leading=(2,11))
                encoded=encoder(robot_id,obs,ref)
                full=attn(encoded.state_token)
                rows,cache=[],None
                for t in range(11):
                    one_ref=replace(ref,features=ref.features[:,t],valid_mask=ref.valid_mask[:,t])
                    one=encoder(robot_id,obs[:,t],one_ref)
                    y,cache=attn.step(one.state_token,cache)
                    rows.append(y)
                torch.testing.assert_close(full,torch.stack(rows,1),atol=2e-6,rtol=2e-5)

    def test_large_batch(self):
        m=self.model(context_len=16)
        x=torch.randn(1024,50,16)
        y=m(x)
        self.assertEqual(y.shape,x.shape)
        self.assertTrue(torch.isfinite(y).all())
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(m.q_proj.weight.grad).all())

    def test_default_parameter_count_and_minimum_even_head(self):
        m=CausalSelfAttention()
        self.assertEqual(sum(p.numel() for p in m.parameters()),262144)
        tiny=CausalSelfAttention(d_model=2,n_heads=1,n_kv_heads=1,context_len=1)
        self.assertEqual(tiny(torch.randn(1,2,2)).shape,(1,2,2))

    @unittest.skipUnless(torch.cuda.is_available(),'当前环境没有 CUDA')
    def test_cuda_float32_and_stream(self):
        m=self.model().cuda().eval()
        x=torch.randn(2,13,16,device='cuda')
        with torch.no_grad(): full=m(x)
        stream,_=self.streamed(m,x)
        torch.testing.assert_close(full,stream,atol=2e-5,rtol=2e-4)

    @unittest.skipUnless(torch.cuda.is_available(),'当前环境没有 CUDA')
    def test_cuda_autocast(self):
        m=self.model().cuda().eval()
        x=torch.randn(2,13,16,device='cuda')
        with torch.autocast('cuda',dtype=torch.float16):
            with torch.no_grad(): full=m(x)
            stream,cache=self.streamed(m,x)
        self.assertEqual(cache.keys.dtype,torch.float16)
        torch.testing.assert_close(full,stream,atol=.003,rtol=.03)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)

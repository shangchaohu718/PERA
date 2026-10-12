"""第 03 个模块的回归测试；仅需 PyTorch 和 Python 标准库。

放到项目 tests/ 后运行：python tests/test_reference_observations.py
或从项目外层运行：python -m unittest discover -s tests -p 'test_*.py' -v
"""
import io
import math
import sys
import unittest
from dataclasses import fields, replace
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_motion.robot_contract import RobotContract, get_robot_contract
from unified_motion.observations import RobotState, ObservationBuilder
from unified_motion.reference_observations import (
    ReferenceWindow, ReferenceObservationBuilder, _quat_to_matrix,
)


class ReferenceTests(unittest.TestCase):
    def fixture(self, robot=None, leading=(2,), h=3, dtype=torch.float32, device='cpu'):
        """构造可手工验证的人工输入，不借用待测代码生成期望结果。"""
        robot = robot or get_robot_contract('k1_22dof')
        n = robot.dof
        names = tuple(reversed(robot.joint_names))
        q0 = torch.linspace(-0.2, 0.3, n, dtype=dtype, device=device)
        builder = ReferenceObservationBuilder(robot, names, q0)
        base = torch.ones(*leading, 3, dtype=dtype, device=device)
        current_quat = torch.zeros(*leading, 4, dtype=dtype, device=device)
        current_quat[..., 0] = 1
        q_delta = torch.arange(h*n, dtype=dtype, device=device).reshape(h, n) * 0.001
        q_delta = q_delta.expand(*leading, h, n).clone()
        vel = q_delta * 2
        delta_pos = torch.arange(h*3, dtype=dtype, device=device).reshape(h, 3) * 0.01
        delta_pos = delta_pos.expand(*leading, h, 3).clone()
        lin = delta_pos * 2
        ang = delta_pos * 3
        quat = current_quat.unsqueeze(-2).expand(*leading, h, 4).clone()
        dt = (torch.arange(h, dtype=dtype, device=device) * 0.1).expand(*leading, h).clone()
        valid = torch.ones(*leading, h, device=device, dtype=torch.bool)
        ref = ReferenceWindow(robot.root_name, names, (q0+q_delta).flip(-1),
                              vel.flip(-1), base.unsqueeze(-2)+delta_pos, quat,
                              lin, ang, dt, valid)
        id6 = q0.new_tensor([1.,0.,0.,0.,1.,0.]).expand(*leading, h, 6)
        expected = torch.cat((dt.unsqueeze(-1), delta_pos, id6, lin, ang, q_delta, vel), -1)
        current = {'current_root_pos_w': base, 'current_root_quat_wxyz': current_quat}
        return builder, ref, current, expected

    def test_all_robots_and_sequence_axes(self):
        for rid in ('k1_22dof','g1_23dof','g1_29dof'):
            for leading in ((2,), (2,4)):
                with self.subTest(robot=rid, leading=leading):
                    b,r,c,e = self.fixture(get_robot_contract(rid), leading)
                    result = b(r, **c)
                    torch.testing.assert_close(result.features, e)
                    self.assertTrue(torch.equal(result.valid_mask, r.valid_mask))
                    self.assertEqual(result.features.shape, (*leading,3,b.feature_dim))

    def test_new_robot_and_fixed_base(self):
        for root_dim in (0,7):
            robot = RobotContract('Other', 'base', ('j0','j1','j2'), root_dim)
            b,r,c,e = self.fixture(robot)
            torch.testing.assert_close(b(r, **c).features, e)
            self.assertEqual(b.feature_dim, 22)

    def test_known_yaw_rotation_and_velocity(self):
        # 当前身体朝世界 +Y；世界 +Y 位移变成身体 +X。
        b,r,c,_ = self.fixture(dtype=torch.float64)
        c['current_root_pos_w'].zero_()
        c['current_root_quat_wxyz'][:] = torch.tensor(
            [math.sqrt(0.5),0.,0.,math.sqrt(0.5)], dtype=torch.float64)
        r.root_pos_w[:] = torch.tensor([0.,2.,0.], dtype=torch.float64)
        r.root_lin_vel_w[:] = torch.tensor([0.,3.,0.], dtype=torch.float64)
        r.root_ang_vel_w[:] = torch.tensor([0.,0.,4.], dtype=torch.float64)
        r.root_quat_wxyz[:] = c['current_root_quat_wxyz'].unsqueeze(-2)
        result = b(r, **c).features
        for name, expected in (
            ('root_pos_b',[2.,0.,0.]), ('root_lin_vel_b',[3.,0.,0.]),
            ('root_ang_vel_b',[0.,0.,4.]), ('root_rot6d_b',[1.,0.,0.,0.,1.,0.]),
        ):
            actual = result[..., b.feature_slices[name]]
            torch.testing.assert_close(actual, actual.new_tensor(expected).expand_as(actual))

    def test_known_roll_uses_full_body_frame(self):
        b,r,c,_ = self.fixture(dtype=torch.float64)
        c['current_root_pos_w'].zero_()
        c['current_root_quat_wxyz'][:] = torch.tensor(
            [math.sqrt(0.5),math.sqrt(0.5),0.,0.], dtype=torch.float64)
        r.root_pos_w[:] = torch.tensor([0.,0.,1.], dtype=torch.float64)
        actual = b(r, **c).features[...,b.feature_slices['root_pos_b']]
        torch.testing.assert_close(actual, actual.new_tensor([0.,1.,0.]).expand_as(actual))

    def test_known_relative_rotation6d_columns(self):
        b,r,c,_ = self.fixture(dtype=torch.float64)
        # 参考相对当前绕 z 转90度：前两列依次为 (0,1,0)、(-1,0,0)。
        r.root_quat_wxyz[:] = torch.tensor(
            [math.sqrt(0.5),0.,0.,math.sqrt(0.5)], dtype=torch.float64)
        actual = b(r, **c).features[...,b.feature_slices['root_rot6d_b']]
        expected = actual.new_tensor([0.,1.,0.,-1.,0.,0.]).expand_as(actual)
        torch.testing.assert_close(actual, expected)

    def test_quaternion_matrix_against_rodrigues(self):
        # 独立的轴角/Rodrigues 公式作为对照，不用被测函数计算期望值。
        gen = torch.Generator().manual_seed(500)
        axis = torch.randn(50,3, generator=gen, dtype=torch.float64)
        axis /= axis.norm(dim=-1,keepdim=True)
        angle = torch.linspace(-math.pi, math.pi,50,dtype=torch.float64)
        q = torch.cat((torch.cos(angle[:,None]/2), axis*torch.sin(angle[:,None]/2)), -1)
        x,y,z = axis.unbind(-1)
        zero = torch.zeros_like(x)
        k = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(-1,3,3)
        eye = torch.eye(3, dtype=torch.float64).expand(50,3,3)
        expected = eye + angle.sin()[:,None,None]*k + (1-angle.cos())[:,None,None]*(k@k)
        torch.testing.assert_close(_quat_to_matrix(q), expected)

    def test_quaternion_sign_invariance(self):
        b,r,c,e = self.fixture()
        for sign_r,sign_c in ((-1,1),(1,-1),(-1,-1)):
            rr = replace(r, root_quat_wxyz=r.root_quat_wxyz*sign_r)
            cc = {**c,'current_root_quat_wxyz': c['current_root_quat_wxyz']*sign_c}
            torch.testing.assert_close(b(rr,**cc).features,e)

    def test_common_translation_and_yaw_invariance(self):
        b,r,c,e = self.fixture(dtype=torch.float64)
        # 把整个场景同时平移并绕 z 旋转90度，局部目标应不变。
        rotation = torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=torch.float64)
        shift = torch.tensor([3.,-2.,5.],dtype=torch.float64)
        quat = torch.tensor([math.sqrt(0.5),0.,0.,math.sqrt(0.5)],dtype=torch.float64)
        rr = replace(r, root_pos_w=r.root_pos_w@rotation.T+shift,
                     root_quat_wxyz=quat.expand_as(r.root_quat_wxyz),
                     root_lin_vel_w=r.root_lin_vel_w@rotation.T,
                     root_ang_vel_w=r.root_ang_vel_w@rotation.T)
        cc = {'current_root_pos_w':c['current_root_pos_w']@rotation.T+shift,
              'current_root_quat_wxyz':quat.expand_as(c['current_root_quat_wxyz'])}
        torch.testing.assert_close(b(rr,**cc).features,e)

    def test_small_quaternion_norm_error_normalized(self):
        b,r,c,e = self.fixture()
        rr = replace(r,root_quat_wxyz=r.root_quat_wxyz*1.0002)
        cc = {**c,'current_root_quat_wxyz':c['current_root_quat_wxyz']*0.9998}
        torch.testing.assert_close(b(rr,**cc).features,e)

    def test_invalid_padding_is_zero_before_and_after_arithmetic(self):
        b,r,c,e = self.fixture()
        r.valid_mask[0,-1] = False
        # 允许有限的零四元数补位；无效大数不能污染运算。
        for f in fields(r):
            value = getattr(r,f.name)
            if isinstance(value,torch.Tensor) and value.dtype != torch.bool:
                value[0,-1] = torch.finfo(value.dtype).max
        r.root_quat_wxyz[0,-1] = 0
        e[0,-1] = 0
        torch.testing.assert_close(b(r,**c).features,e)

    def test_mask_shape_type_and_prefix_validation(self):
        b,r,c,_ = self.fixture()
        with self.assertRaises(TypeError): b(replace(r,valid_mask=r.valid_mask.float()),**c)
        with self.assertRaises(ValueError): b(replace(r,valid_mask=r.valid_mask[:1]),**c)
        bad = r.valid_mask.clone(); bad[0,1] = False
        with self.assertRaises(ValueError): b(replace(r,valid_mask=bad),**c)
        bad.zero_()
        with self.assertRaises(ValueError): b(replace(r,valid_mask=bad),**c)

    def test_negative_and_duplicate_and_decreasing_times_rejected(self):
        b,r,c,_ = self.fixture()
        for times in ((-.1,.1,.2),(0.,.1,.1),(0.,.2,.1)):
            with self.subTest(times=times), self.assertRaises(ValueError):
                b(replace(r,time_offsets_s=r.time_offsets_s.new_tensor(times).expand_as(r.time_offsets_s)),**c)

    def test_single_timepoint_and_per_environment_times(self):
        b,r,c,e = self.fixture(h=1)
        r.time_offsets_s[0,0] = .32; e[0,0,0] = .32
        torch.testing.assert_close(b(r,**c).features,e)
        b,r,c,e = self.fixture()
        r.time_offsets_s[1] *= 3; e[1,:,0] *= 3
        torch.testing.assert_close(b(r,**c).features,e)

    def test_mlp_flatten_contains_mask_and_preserves_history(self):
        b,r,c,e = self.fixture(leading=(2,4))
        r.valid_mask[0,0,-1] = False; e[0,0,-1] = 0
        out = b(r,**c)
        expected = torch.cat((e,r.valid_mask[...,None].float()),-1).flatten(-2)
        torch.testing.assert_close(out.flatten_for_mlp(),expected)
        self.assertEqual(out.flatten_for_mlp().shape,(2,4,3*(b.feature_dim+1)))

    def test_independent_joint_permutations(self):
        gen = torch.Generator().manual_seed(4)
        for rid in ('k1_22dof','g1_23dof','g1_29dof'):
            b,r,c,e = self.fixture(get_robot_contract(rid))
            for _ in range(5):
                p = torch.randperm(b.robot.dof,generator=gen)
                names = tuple(r.joint_names[i] for i in p.tolist())
                rr = replace(r,joint_names=names,joint_pos=r.joint_pos[...,p],joint_vel=r.joint_vel[...,p])
                bb = ReferenceObservationBuilder(b.robot,names,b.default_joint_pos)
                torch.testing.assert_close(bb(rr,**c).features,e)

    def test_root_joint_names_and_reference_type_rejected(self):
        b,r,c,_ = self.fixture()
        with self.assertRaises(TypeError): b({},**c)
        with self.assertRaises(ValueError): b(replace(r,root_name='wrong'),**c)
        with self.assertRaises(ValueError): b(replace(r,joint_names=b.robot.joint_names),**c)
        with self.assertRaises(ValueError): b(replace(r,joint_names=r.joint_names[:-1]),**c)

    def test_constructor_rejects_bad_contract_names_defaults(self):
        b,r,_,_ = self.fixture()
        with self.assertRaises(TypeError): ReferenceObservationBuilder(None,r.joint_names,b.default_joint_pos)
        with self.assertRaises(TypeError): ReferenceObservationBuilder(b.robot,r.joint_names,b.default_joint_pos,check_values=1)
        for names in (r.joint_names[:-1],r.joint_names[:-1]+('wrong',),r.joint_names[:-1]+(r.joint_names[0],)):
            with self.assertRaises(ValueError): ReferenceObservationBuilder(b.robot,names,b.default_joint_pos)
        for default in ([0.]*b.robot.dof, b.default_joint_pos.long(), b.default_joint_pos.half()):
            with self.assertRaises(TypeError): ReferenceObservationBuilder(b.robot,r.joint_names,default)
        with self.assertRaises(ValueError): ReferenceObservationBuilder(b.robot,r.joint_names,b.default_joint_pos[None])
        for bad in (float('nan'),float('inf')):
            invalid = b.default_joint_pos.clone(); invalid[0]=bad
            with self.assertRaises(ValueError): ReferenceObservationBuilder(b.robot,r.joint_names,invalid)

    def test_rejects_wrong_rank_and_empty_axes_and_width(self):
        b,r,c,_ = self.fixture()
        for q in (r.joint_pos[0],r.joint_pos[None,None],r.joint_pos[:0],r.joint_pos[:,:0],r.joint_pos[...,:-1]):
            with self.subTest(shape=q.shape), self.assertRaises(ValueError): b(replace(r,joint_pos=q),**c)
        with self.assertRaises(TypeError): b(replace(r,joint_pos=[]),**c)

    def test_all_float_fields_shape_dtype_device_and_type_checks(self):
        b,r,c,_ = self.fixture()
        names = [f.name for f in fields(r) if isinstance(getattr(r,f.name),torch.Tensor) and f.name!='valid_mask']
        for name in names:
            v = getattr(r,name)
            for bad, error in ((v[:1],ValueError),(v.double(),TypeError),(v.to('meta'),ValueError),([],TypeError)):
                with self.subTest(field=name,error=error), self.assertRaises(error): b(replace(r,**{name:bad}),**c)
        for name,v in c.items():
            for bad, error in ((v[:1],ValueError),(v.double(),TypeError),(v.to('meta'),ValueError),([],TypeError)):
                with self.subTest(field=name,error=error), self.assertRaises(error): b(r,**{**c,name:bad})

    def test_nonfinite_all_fields_even_masked(self):
        b,r,c,_ = self.fixture()
        r.valid_mask[0,-1]=False
        for name in [f.name for f in fields(r) if isinstance(getattr(r,f.name),torch.Tensor) and f.name!='valid_mask']:
            for bad in (float('nan'),float('inf'),float('-inf')):
                invalid = getattr(r,name).clone(); invalid[0,-1]=bad
                with self.subTest(field=name,bad=bad), self.assertRaises(ValueError): b(replace(r,**{name:invalid}),**c)
        for name,v in c.items():
            invalid = v.clone(); invalid[0,0]=float('nan')
            with self.assertRaises(ValueError): b(r,**{**c,name:invalid})

    def test_rejects_invalid_valid_quaternions(self):
        b,r,c,_ = self.fixture()
        for scale in (0.,.5,2.):
            with self.assertRaises(ValueError): b(replace(r,root_quat_wxyz=r.root_quat_wxyz*scale),**c)
            with self.assertRaises(ValueError): b(r,**{**c,'current_root_quat_wxyz':c['current_root_quat_wxyz']*scale})

    def test_output_overflow_rejected(self):
        b,r,c,_ = self.fixture()
        huge = torch.finfo(torch.float32).max
        r.root_pos_w[:]=huge; c['current_root_pos_w'][:]=-huge
        with self.assertRaises(ValueError): b(r,**c)

    def test_default_clone_and_no_input_or_output_aliasing(self):
        b,r,c,e = self.fixture()
        q0 = b.default_joint_pos.clone().requires_grad_()
        bb = ReferenceObservationBuilder(b.robot,r.joint_names,q0)
        with torch.no_grad(): q0.add_(3)
        out = bb(r,**c)
        torch.testing.assert_close(out.features,e)
        self.assertFalse(bb.default_joint_pos.requires_grad)
        r.joint_pos.zero_(); r.valid_mask.zero_(); c['current_root_pos_w'].zero_()
        torch.testing.assert_close(out.features,e)
        self.assertTrue(out.valid_mask.all().item())

    def test_zero_parameters_device_dtype_movement(self):
        b,r,c,e = self.fixture(dtype=torch.float64)
        b = b.float()
        for f in fields(r):
            v = getattr(r,f.name)
            if isinstance(v,torch.Tensor) and v.dtype!=torch.bool: setattr(r,f.name,v.float())
        c={k:v.float() for k,v in c.items()}
        torch.testing.assert_close(b(r,**c).features,e.float())
        self.assertEqual(sum(p.numel() for p in b.parameters()),0)
        self.assertEqual(b.policy_indices.dtype,torch.long)
        b.half()
        with self.assertRaises(TypeError): b(r,**c)

    def test_outer_cpu_autocast_does_not_change_geometry(self):
        b,r,c,e = self.fixture()
        with torch.autocast('cpu',dtype=torch.bfloat16): out=b(r,**c)
        self.assertEqual(out.features.dtype,torch.float32)
        torch.testing.assert_close(out.features,e)

    def test_no_input_mutation_and_noncontiguous_sequences(self):
        b,r,c,e = self.fixture(leading=(2,4))
        rr = replace(r,**{f.name:getattr(r,f.name).transpose(0,1)
                          for f in fields(r) if isinstance(getattr(r,f.name),torch.Tensor)})
        cc={k:v.transpose(0,1) for k,v in c.items()}
        snapshots={f.name:getattr(rr,f.name).clone() for f in fields(rr) if isinstance(getattr(rr,f.name),torch.Tensor)}
        torch.testing.assert_close(b(rr,**cc).features,e.transpose(0,1))
        for name,v in snapshots.items(): self.assertTrue(torch.equal(getattr(rr,name),v))

    def test_gradient_flow_with_padding(self):
        b,r,c,_=self.fixture(dtype=torch.float64)
        r.valid_mask[0,-1]=False
        tensors=[]
        for f in fields(r):
            v=getattr(r,f.name)
            if isinstance(v,torch.Tensor) and v.dtype!=torch.bool: v.requires_grad_(); tensors.append(v)
        for v in c.values(): v.requires_grad_(); tensors.append(v)
        b(r,**c).features.square().sum().backward()
        for v in tensors:
            self.assertIsNotNone(v.grad)
            self.assertTrue(torch.isfinite(v.grad).all().item())
        torch.testing.assert_close(r.joint_pos.grad[0,-1],torch.zeros(b.robot.dof,dtype=torch.float64))

    def test_state_dict_roundtrip_and_schema_rejection(self):
        b,r,c,e=self.fixture()
        data=io.BytesIO(); torch.save(b.state_dict(),data); data.seek(0)
        state=torch.load(data,weights_only=True)
        other=ReferenceObservationBuilder(b.robot,r.joint_names,torch.zeros_like(b.default_joint_pos))
        other.load_state_dict(state)
        torch.testing.assert_close(other(r,**c).features,e)
        changed={**state,'_extra_state':{**state['_extra_state'],'schema_version':99}}
        with self.assertRaises(ValueError): other.load_state_dict(changed)

    def test_layout_covers_every_column_exactly(self):
        b,r,c,_=self.fixture()
        covered=[]
        for s in b.feature_slices.values(): covered.extend(range(s.start,s.stop))
        self.assertEqual(covered,list(range(b.feature_dim)))

    def test_fast_mode_matches_strict_on_valid_inputs(self):
        b,r,c,e=self.fixture()
        b.check_values=False
        torch.testing.assert_close(b(r,**c).features,e)
        with self.assertRaises(ValueError): b(replace(r,root_name='wrong'),**c)
        with self.assertRaises(TypeError): b(replace(r,joint_vel=r.joint_vel.double()),**c)

    def test_both_builders_to_dummy_mlp_backward(self):
        # 只验证输入拼接和反向传播，不是已经训练好机器人策略。
        torch.manual_seed(100)
        for rid in ('k1_22dof','g1_23dof','g1_29dof'):
            b,r,c,_=self.fixture(get_robot_contract(rid),leading=(2,4))
            n=b.robot.dof
            sim_names=b.robot.joint_names
            body_builder=ObservationBuilder(b.robot,sim_names,b.default_joint_pos)
            g=torch.zeros(2,4,3); g[...,2]=-1
            state=RobotState(sim_names,b.default_joint_pos.expand(2,4,n),torch.zeros(2,4,n),torch.zeros(2,4,3),g)
            body=body_builder(state,torch.zeros(2,4,n))
            reference=b(r,**c).flatten_for_mlp()
            inputs=torch.cat((body,reference),-1)
            model=nn.Sequential(nn.Linear(inputs.shape[-1],32),nn.Tanh(),nn.Linear(32,n))
            actions=model(inputs)
            self.assertEqual(actions.shape,(2,4,n))
            actions.square().mean().backward()
            for p in model.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all().item())

    def test_large_sequence_batch(self):
        for rid in ('k1_22dof','g1_23dof','g1_29dof'):
            b,r,c,e=self.fixture(get_robot_contract(rid),leading=(1024,50),h=3)
            torch.testing.assert_close(b(r,**c).features,e)

    @unittest.skipUnless(torch.cuda.is_available(), '当前环境无 CUDA；未验证 GPU 执行')
    def test_cuda_matches_cpu(self):
        b,r,c,e=self.fixture(device='cuda')
        torch.testing.assert_close(b(r,**c).features,e)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)

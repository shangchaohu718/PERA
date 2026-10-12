"""第 04 个文件的回归测试：人工张量，不启动仿真器或训练 PPO。"""
import copy
import io
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_motion.robot_contract import RobotContract, get_robot_contract
from unified_motion.observations import ObservationBuilder, RobotState
from unified_motion.reference_observations import (
    ReferenceObservation, ReferenceObservationBuilder, ReferenceWindow,
)
from unified_motion.models.encoder import MultiRobotEncoder


ROBOT_IDS = ('k1_22dof', 'g1_23dof', 'g1_29dof')


def build_inputs(robot, leading=(2,), h=3, dtype=torch.float32):
    """用真正的第 02/03 个 builder 生成输入，避免只用随机形状冒充集成测试。"""
    n = robot.dof
    sim_names = tuple(reversed(robot.joint_names))
    default = torch.linspace(-0.2, 0.2, n, dtype=dtype)
    body_builder = ObservationBuilder(robot, sim_names, default)
    ref_builder = ReferenceObservationBuilder(robot, sim_names, default)
    joint_pos = default + torch.randn(*leading, n, dtype=dtype) * .05
    joint_vel = torch.randn(*leading, n, dtype=dtype) * .1
    gravity = torch.zeros(*leading, 3, dtype=dtype); gravity[..., 2] = -1
    state = RobotState(sim_names, joint_pos.flip(-1), joint_vel.flip(-1),
                       torch.randn(*leading, 3, dtype=dtype) * .1, gravity)
    observation = body_builder(state, torch.zeros(*leading, n, dtype=dtype))
    root_pos = torch.zeros(*leading, 3, dtype=dtype)
    root_quat = torch.zeros(*leading, 4, dtype=dtype); root_quat[..., 0] = 1
    valid = torch.ones(*leading, h, dtype=torch.bool)
    if h > 1:
        valid[..., -1] = False
    window = ReferenceWindow(
        root_name=robot.root_name, joint_names=sim_names,
        joint_pos=(default + torch.randn(*leading, h, n, dtype=dtype) * .1).flip(-1),
        joint_vel=torch.randn(*leading, h, n, dtype=dtype) * .1,
        root_pos_w=root_pos.unsqueeze(-2).expand(*leading, h, 3).clone(),
        root_quat_wxyz=root_quat.unsqueeze(-2).expand(*leading, h, 4).clone(),
        root_lin_vel_w=torch.randn(*leading, h, 3, dtype=dtype) * .1,
        root_ang_vel_w=torch.randn(*leading, h, 3, dtype=dtype) * .1,
        time_offsets_s=(torch.arange(h, dtype=dtype) * .1).expand(*leading, h).clone(),
        valid_mask=valid,
    )
    ref = ref_builder(window, current_root_pos_w=root_pos,
                      current_root_quat_wxyz=root_quat)
    return observation, ref, body_builder.obs_dim, ref_builder.feature_dim


class EncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def fixture(self, robot_id='k1_22dof', leading=(2,), h=3, dtype=torch.float32,
                check_values=True):
        robots = {key: get_robot_contract(key) for key in ROBOT_IDS}
        encoder = MultiRobotEncoder(robots, d_model=16, hidden_dim=32,
                                    check_values=check_values).to(dtype=dtype)
        obs, ref, *_ = build_inputs(robots[robot_id], leading, h, dtype)
        return encoder, obs, ref

    def test_three_robots_body_reference_integration(self):
        for robot_id in ROBOT_IDS:
            for leading in ((3,), (2, 4)):
                with self.subTest(robot=robot_id, leading=leading):
                    enc, obs, ref = self.fixture(robot_id, leading)
                    out = enc(robot_id, obs, ref)
                    self.assertEqual(out.robot_id, robot_id)
                    self.assertEqual(out.state_token.shape, (*leading, 16))
                    self.assertEqual(out.reference_tokens.shape, (*leading, 3, 16))
                    self.assertEqual(out.flatten_for_mlp().shape, (*leading, 67))
                    self.assertEqual(out.reference_mask.dtype, torch.bool)

    def test_dimensions_match_original_builders(self):
        enc, *_ = self.fixture()
        for robot_id in ROBOT_IDS:
            _, _, obs_dim, ref_dim = build_inputs(get_robot_contract(robot_id))
            self.assertEqual(enc.input_dims(robot_id), (obs_dim, ref_dim))

    def test_new_robot_without_core_changes(self):
        robot = RobotContract('Test robot', 'base', ('a', 'b', 'c'))
        enc = MultiRobotEncoder({'custom.v1': robot}, d_model=16, hidden_dim=32)
        obs, ref, *_ = build_inputs(robot)
        out = enc('custom.v1', obs, ref)
        self.assertEqual(out.state_token.shape, (2,16))
        self.assertEqual(enc.input_dims('custom.v1'), (15, 22))

    def test_no_trainable_weights_are_shared_between_robot_adapters(self):
        enc, *_ = self.fixture()
        parameter_sets = []
        for robot_id in enc.robot_ids:
            key = enc._keys[robot_id]
            params = list(enc.state_encoders[key].parameters()) + list(enc.reference_encoders[key].parameters())
            parameter_sets.append({p.data_ptr() for p in params})
        for i in range(len(parameter_sets)):
            for j in range(i):
                self.assertFalse(parameter_sets[i] & parameter_sets[j])

    def test_only_selected_branch_receives_gradients(self):
        enc, obs, ref = self.fixture()
        out = enc('k1_22dof', obs, ref)
        (out.flatten_for_mlp() * torch.randn_like(out.flatten_for_mlp())).sum().backward()
        for robot_id in enc.robot_ids:
            key = enc._keys[robot_id]
            for p in list(enc.state_encoders[key].parameters()) + list(enc.reference_encoders[key].parameters()):
                self.assertEqual(p.grad is not None, robot_id == 'k1_22dof')

    def test_gradient_flows_to_inputs_and_zero_for_padding(self):
        enc, obs, ref = self.fixture(leading=(2, 4))
        obs = obs.detach().requires_grad_()
        ref.features = ref.features.detach().requires_grad_()
        out = enc('k1_22dof', obs, ref)
        flat = out.flatten_for_mlp()
        (flat * torch.randn_like(flat)).sum().backward()
        self.assertGreater(obs.grad.abs().sum().item(), 0)
        self.assertGreater(ref.features.grad[ref.valid_mask].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(obs.grad).all())
        self.assertEqual(torch.count_nonzero(ref.features.grad[~ref.valid_mask]).item(), 0)

    def test_shared_dummy_backbone_and_robot_heads_optimizer_step(self):
        robots = {key: get_robot_contract(key) for key in ('k1_22dof','g1_29dof')}
        enc = MultiRobotEncoder(robots, d_model=16, hidden_dim=32)
        # 临时输出头只用于检查计算图，不是我们项目已经完成的 Actor。
        trunk = nn.Sequential(nn.Linear(67,32), nn.SiLU())
        heads = nn.ModuleDict({key: nn.Linear(32, robot.dof) for key, robot in robots.items()})
        all_params = list(enc.parameters()) + list(trunk.parameters()) + list(heads.parameters())
        optimizer = torch.optim.Adam(all_params, lr=1e-3)
        before = [p.detach().clone() for p in all_params]
        loss = 0
        for robot_id, robot in robots.items():
            obs, ref, *_ = build_inputs(robot)
            prediction = heads[robot_id](trunk(enc(robot_id, obs, ref).flatten_for_mlp()))
            self.assertEqual(prediction.shape, (2, robot.dof))
            loss = loss + (prediction - .5).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in all_params))
        optimizer.step()
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before, all_params)))

    def test_mask_zero_after_nonzero_bias(self):
        enc, obs, ref = self.fixture()
        key = enc._keys['k1_22dof']
        with torch.no_grad():
            enc.reference_encoders[key][-1].bias.fill_(5)
        out = enc('k1_22dof', obs, ref)
        self.assertEqual(torch.count_nonzero(out.reference_tokens[~ref.valid_mask]).item(), 0)

    def test_padding_values_do_not_change_valid_outputs(self):
        enc, obs, ref = self.fixture()
        a = enc('k1_22dof', obs, ref)
        changed = ref.features.clone(); changed[~ref.valid_mask] = torch.finfo(torch.float32).max
        b = enc('k1_22dof', obs, replace(ref, features=changed))
        torch.testing.assert_close(a.state_token, b.state_token)
        torch.testing.assert_close(a.reference_tokens, b.reference_tokens)

    def test_padding_length_does_not_change_existing_tokens(self):
        enc, obs, ref = self.fixture()
        a = enc('k1_22dof', obs, ref)
        f = torch.cat((ref.features, torch.zeros(2, 2, ref.features.shape[-1])), dim=-2)
        m = torch.cat((ref.valid_mask, torch.zeros(2, 2, dtype=torch.bool)), dim=-1)
        b = enc('k1_22dof', obs, ReferenceObservation(f,m))
        torch.testing.assert_close(a.state_token, b.state_token)
        torch.testing.assert_close(a.reference_tokens, b.reference_tokens[:,:3])

    def test_flatten_contains_mask_and_preserves_all_reference_points(self):
        enc, obs, ref = self.fixture(leading=(2, 4))
        out = enc('k1_22dof', obs, ref)
        flat = out.flatten_for_mlp()
        torch.testing.assert_close(flat[..., :16], out.state_token)
        refs = flat[...,16:].reshape(2,4,3,17)
        torch.testing.assert_close(refs[...,:16], out.reference_tokens)
        torch.testing.assert_close(refs[...,-1], out.reference_mask.float())

    def test_h_one_and_variable_h(self):
        enc, _, _ = self.fixture()
        for h in (1, 2, 7):
            obs, ref, *_ = build_inputs(get_robot_contract('k1_22dof'),h=h)
            out = enc('k1_22dof', obs, ref)
            self.assertEqual(out.reference_tokens.shape,(2,h,16))
            self.assertEqual(out.flatten_for_mlp().shape,(2,16+h*17))

    def test_future_state_does_not_change_past_encoding(self):
        enc, obs, ref = self.fixture(leading=(2, 5))
        a = enc('k1_22dof', obs, ref)
        new_obs = obs.clone(); new_obs[:,3:] += 10
        new_ref = ref.features.clone(); new_ref[:,3:] += 5
        b = enc('k1_22dof', new_obs, replace(ref,features=new_ref))
        torch.testing.assert_close(a.state_token[:,:3], b.state_token[:,:3])
        torch.testing.assert_close(a.reference_tokens[:,:3], b.reference_tokens[:,:3])

    def test_no_batch_mixing(self):
        enc, obs, ref = self.fixture(leading=(3,))
        batch = enc('k1_22dof', obs, ref)
        for i in range(3):
            single = enc('k1_22dof', obs[i:i+1], ReferenceObservation(ref.features[i:i+1],ref.valid_mask[i:i+1]))
            torch.testing.assert_close(single.state_token, batch.state_token[i:i+1])
            torch.testing.assert_close(single.reference_tokens, batch.reference_tokens[i:i+1])

    def test_noncontiguous_sequence_inputs(self):
        enc, obs, ref = self.fixture(leading=(2,4))
        expected = enc('k1_22dof', obs, ref)
        actual = enc('k1_22dof', obs.transpose(0,1), ReferenceObservation(
            ref.features.transpose(0,1), ref.valid_mask.transpose(0,1)))
        torch.testing.assert_close(expected.state_token.transpose(0,1), actual.state_token)
        torch.testing.assert_close(expected.reference_tokens.transpose(0,1), actual.reference_tokens)

    def test_train_eval_deterministic_same_input(self):
        enc, obs, ref = self.fixture()
        enc.train(); a = enc('k1_22dof', obs, ref)
        b = enc('k1_22dof', obs, ref)
        enc.eval(); c = enc('k1_22dof', obs, ref)
        self.assertFalse(any(isinstance(m, (nn.Dropout, nn.BatchNorm1d)) for m in enc.modules()))
        torch.testing.assert_close(a.state_token,b.state_token,rtol=0,atol=0)
        torch.testing.assert_close(a.reference_tokens,c.reference_tokens,rtol=0,atol=0)

    def test_no_input_or_mask_mutation(self):
        enc, obs, ref = self.fixture()
        snapshots = (obs.clone(), ref.features.clone(),ref.valid_mask.clone())
        out = enc('k1_22dof',obs,ref)
        for a,b in zip(snapshots,(obs,ref.features,ref.valid_mask)):
            self.assertTrue(torch.equal(a,b))
        out.reference_mask.zero_()
        self.assertTrue(torch.equal(ref.valid_mask,snapshots[-1]))

    def test_float64_and_module_dtype_conversion(self):
        enc, obs, ref = self.fixture(dtype=torch.float64)
        out = enc('k1_22dof',obs,ref)
        self.assertEqual(out.state_token.dtype,torch.float64)
        out.flatten_for_mlp().sum().backward()
        self.assertTrue(all(p.dtype == torch.float64 for p in enc.parameters()))

    def test_cpu_autocast_forward_backward(self):
        enc, obs, ref = self.fixture()
        with torch.autocast('cpu',dtype=torch.bfloat16):
            out = enc('k1_22dof',obs,ref)
            loss = out.flatten_for_mlp().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(out.state_token).all())
        self.assertEqual(out.reference_tokens.dtype,torch.bfloat16)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in enc.parameters()))

    def test_state_dict_roundtrip_and_input_mapping_order(self):
        enc, obs, ref = self.fixture()
        buffer = io.BytesIO(); torch.save(enc.state_dict(),buffer); buffer.seek(0)
        enc2 = MultiRobotEncoder({key:get_robot_contract(key) for key in reversed(ROBOT_IDS)},
                                  d_model=16,hidden_dim=32)
        enc2.load_state_dict(torch.load(buffer,weights_only=True))
        a = enc('k1_22dof',obs,ref); b = enc2('k1_22dof',obs,ref)
        torch.testing.assert_close(a.state_token,b.state_token,rtol=0,atol=0)
        torch.testing.assert_close(a.reference_tokens,b.reference_tokens,rtol=0,atol=0)

    def test_checkpoint_same_dimensions_wrong_joint_order_rejected(self):
        robot = get_robot_contract('k1_22dof')
        enc = MultiRobotEncoder({'r':robot},d_model=16,hidden_dim=32)
        wrong = replace(robot,joint_names=tuple(reversed(robot.joint_names)))
        other = MultiRobotEncoder({'r':wrong},d_model=16,hidden_dim=32)
        with self.assertRaises(ValueError): other.load_state_dict(enc.state_dict())

    def test_checkpoint_metadata_and_missing_metadata_rejected(self):
        enc, *_ = self.fixture()
        for what in ('schema', 'root', 'width', 'missing'):
            state = copy.deepcopy(enc.state_dict())
            if what == 'schema': state['_extra_state']['encoder_schema'] = 999
            if what == 'root': state['_extra_state']['robots']['k1_22dof'] = ('wrong',7,('a',))
            if what == 'width': state['_extra_state']['d_model'] = 100
            if what == 'missing': del state['_extra_state']
            with self.subTest(what=what), self.assertRaises((ValueError,RuntimeError)):
                enc.load_state_dict(state)

    def test_invalid_constructor_options(self):
        robot = get_robot_contract('k1_22dof')
        for robots in ({}, [], {'':robot}, {1:robot}, {'r':None}):
            with self.subTest(robots=robots), self.assertRaises((TypeError,ValueError)):
                MultiRobotEncoder(robots)
        for key, value in (('d_model',1),('d_model',0),('d_model',True),('hidden_dim',-1),
                           ('hidden_dim',32.0),('check_values',1)):
            with self.subTest(key=key,value=value), self.assertRaises((ValueError,TypeError)):
                MultiRobotEncoder({'r':robot},**{key:value})

    def test_unknown_robot_and_wrong_robot_width(self):
        enc, obs, ref = self.fixture()
        for key in ('g1', None, 42):
            with self.subTest(key=key), self.assertRaises(KeyError): enc(key,obs,ref)
        with self.assertRaises(ValueError): enc('g1_29dof',obs,ref)

    def test_invalid_argument_types(self):
        enc, obs, ref = self.fixture()
        with self.assertRaises(TypeError): enc('k1_22dof',[],ref)
        with self.assertRaises(TypeError): enc('k1_22dof',obs,{})
        with self.assertRaises(TypeError): enc('k1_22dof',obs,replace(ref,features=[]))
        with self.assertRaises(TypeError): enc('k1_22dof',obs,replace(ref,valid_mask=[]))

    def test_wrong_leading_shapes_and_width(self):
        enc, obs, ref = self.fixture(leading=(2,4))
        for name, features in (
            ('batch',ref.features[:1]),('history',ref.features[:,:1]),
            ('width',ref.features[...,:-1]),('rank',ref.features[:,0]),
            ('empty_h',ref.features[...,:0,:]),
        ):
            with self.subTest(name=name),self.assertRaises(ValueError):
                enc('k1_22dof',obs,replace(ref,features=features))
        for bad in (obs[...,:-1],obs[:0],obs[0,0],obs.unsqueeze(0)):
            with self.assertRaises(ValueError): enc('k1_22dof',bad,ref)

    def test_wrong_mask_shape_or_type(self):
        enc, obs, ref = self.fixture()
        for mask in (ref.valid_mask[:1],ref.valid_mask[:,:-1]):
            with self.assertRaises(ValueError): enc('k1_22dof',obs,replace(ref,valid_mask=mask))
        for mask in (ref.valid_mask.float(),ref.valid_mask.long()):
            with self.assertRaises(TypeError): enc('k1_22dof',obs,replace(ref,valid_mask=mask))

    def test_empty_or_nonprefix_valid_mask_rejected(self):
        enc, obs, ref = self.fixture()
        for mask in (torch.zeros(2,3,dtype=torch.bool),
                     torch.tensor([[1,0,1],[1,1,1]],dtype=torch.bool)):
            with self.assertRaises(ValueError): enc('k1_22dof',obs,replace(ref,valid_mask=mask))

    def test_nonfinite_inputs_including_padding_rejected(self):
        enc, obs, ref = self.fixture()
        for bad in (float('nan'),float('inf'),float('-inf')):
            obs2=obs.clone();obs2[0,0]=bad
            with self.assertRaises(ValueError): enc('k1_22dof',obs2,ref)
            for idx in (0,2):
                f=ref.features.clone();f[0,idx,0]=bad
                with self.assertRaises(ValueError): enc('k1_22dof',obs,replace(ref,features=f))

    def test_nonfinite_network_output_rejected(self):
        enc, obs, ref = self.fixture()
        with torch.no_grad(): enc.state_encoders[enc._keys['k1_22dof']][0].weight.fill_(float('inf'))
        with self.assertRaises(ValueError): enc('k1_22dof',obs,ref)

    def test_dtype_mismatch_and_half_weights_rejected(self):
        enc, obs, ref = self.fixture()
        for dtype in (torch.float64,torch.int64,torch.float16):
            with self.assertRaises(TypeError): enc('k1_22dof',obs.to(dtype),ref)
            with self.assertRaises(TypeError): enc('k1_22dof',obs,replace(ref,features=ref.features.to(dtype)))
        enc.half()
        with self.assertRaises(TypeError): enc('k1_22dof',obs.half(),ref)

    def test_device_mismatch_rejected(self):
        enc, obs, ref = self.fixture()
        with self.assertRaises(ValueError): enc('k1_22dof',obs.to('meta'),ref)
        with self.assertRaises(ValueError): enc('k1_22dof',obs,replace(ref,features=ref.features.to('meta')))
        with self.assertRaises(ValueError): enc('k1_22dof',obs,replace(ref,valid_mask=ref.valid_mask.to('meta')))

    def test_check_values_false_preserves_structure_and_valid_outputs(self):
        enc, obs, ref = self.fixture()
        a = enc('k1_22dof',obs,ref)
        enc.check_values=False
        b = enc('k1_22dof',obs,ref)
        torch.testing.assert_close(a.reference_tokens,b.reference_tokens,atol=0,rtol=0)
        with self.assertRaises(ValueError): enc('k1_22dof',obs[:,:-1],ref)

    def test_large_sequence_batch_all_robots_small_width(self):
        enc, *_ = self.fixture()
        with torch.no_grad():
            for robot_id in ROBOT_IDS:
                o,r=enc.input_dims(robot_id)
                obs=torch.randn(1024,50,o)
                ref=ReferenceObservation(torch.randn(1024,50,3,r),torch.ones(1024,50,3,dtype=torch.bool))
                out=enc(robot_id,obs,ref)
                self.assertEqual(out.state_token.shape,(1024,50,16))
                self.assertEqual(out.reference_tokens.shape,(1024,50,3,16))

    def test_default_parameter_count(self):
        enc = MultiRobotEncoder({key:get_robot_contract(key) for key in ROBOT_IDS})
        self.assertEqual(sum(p.numel() for p in enc.parameters()), 1017344)

    @unittest.skipUnless(torch.cuda.is_available(), '当前运行环境没有 CUDA GPU')
    def test_cuda_amp_forward_backward(self):
        enc, obs, ref = self.fixture()
        enc=enc.cuda();obs=obs.cuda();ref=ReferenceObservation(ref.features.cuda(),ref.valid_mask.cuda())
        with torch.autocast('cuda',dtype=torch.float16):
            out=enc('k1_22dof',obs,ref)
            loss=out.flatten_for_mlp().float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(out.reference_tokens).all())
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in enc.parameters()))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)

"""开发用 CPU 回归测试，不依赖仿真器。"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_motion.robot_contract import RobotContract, get_robot_contract
from unified_motion.observations import ObservationBuilder, RobotState


class ObservationTests(unittest.TestCase):
    def fixture(self, robot=None, leading=(2,), dtype=torch.float32):
        robot = robot or get_robot_contract('k1_22dof')
        n = robot.dof
        names = tuple(reversed(robot.joint_names))
        default = torch.linspace(-0.5, 0.5, n, dtype=dtype)
        offset = torch.arange(n, dtype=dtype).expand(*leading, n) * 0.01
        q = (default + offset).flip(-1).clone()
        vel = (offset * 2).flip(-1).clone()
        root = torch.ones(*leading, 3, dtype=dtype) * 0.2
        g = torch.zeros(*leading, 3, dtype=dtype)
        g[..., 2] = -1.0
        prev = offset * 4
        state = RobotState(names, q, vel, root, g)
        builder = ObservationBuilder(robot, names, default)
        expected = torch.cat((root, g, offset, offset * 2, prev), dim=-1)
        return builder, state, prev, expected

    def test_all_robot_shapes_and_fields(self):
        for robot_id in ('k1_22dof', 'g1_23dof', 'g1_29dof'):
            for leading in ((2,), (3, 5)):
                with self.subTest(robot=robot_id, leading=leading):
                    b, s, p, expected = self.fixture(get_robot_contract(robot_id), leading)
                    actual = b(s, p)
                    self.assertEqual(actual.shape, (*leading, b.obs_dim))
                    torch.testing.assert_close(actual, expected)

    def test_new_robot_without_code_changes(self):
        b, s, p, expected = self.fixture(RobotContract('new', 'base', ('a','b','c')))
        self.assertEqual(b.obs_dim, 15)
        torch.testing.assert_close(b(s, p), expected)

    def test_default_cloned_and_detached(self):
        b, s, p, expected = self.fixture()
        initial = b.default_joint_pos.clone().requires_grad_()
        b2 = ObservationBuilder(b.robot, s.joint_names, initial)
        with torch.no_grad():
            initial.add_(5)
        torch.testing.assert_close(b2(s, p), expected)
        self.assertFalse(b2.default_joint_pos.requires_grad)

    def test_no_trainable_parameters(self):
        b, *_ = self.fixture()
        self.assertEqual(sum(p.numel() for p in b.parameters()), 0)
        self.assertEqual(b.policy_indices.dtype, torch.long)

    def test_tensor_type_conversion_with_module(self):
        b, s, p, expected = self.fixture()
        b = b.to(dtype=torch.float64)
        s = replace(s, joint_pos=s.joint_pos.double(), joint_vel=s.joint_vel.double(),
                    root_ang_vel_b=s.root_ang_vel_b.double(), gravity_b=s.gravity_b.double())
        torch.testing.assert_close(b(s, p.double()), expected.double(), atol=1e-7, rtol=1e-5)
        self.assertEqual(b.policy_indices.dtype, torch.long)

    def test_float64_supported(self):
        b, s, p, expected = self.fixture(dtype=torch.float64)
        torch.testing.assert_close(b(s, p), expected)

    def test_no_input_mutation(self):
        b, s, p, _ = self.fixture()
        old = (s.joint_pos.clone(), s.joint_vel.clone(), p.clone())
        b(s, p)
        for left, right in zip(old, (s.joint_pos, s.joint_vel, p)):
            self.assertTrue(torch.equal(left, right))

    def test_autograd(self):
        b, s, p, _ = self.fixture(leading=(2, 3))
        s.joint_pos.requires_grad_()
        s.joint_vel.requires_grad_()
        s.root_ang_vel_b.requires_grad_()
        s.gravity_b.requires_grad_()
        p.requires_grad_()
        b(s, p).sum().backward()
        for value in (s.joint_pos, s.joint_vel, s.root_ang_vel_b, s.gravity_b, p):
            torch.testing.assert_close(value.grad, torch.ones_like(value))

    def test_noncontiguous_inputs(self):
        b, s, p, expected = self.fixture(leading=(2, 4))
        s = replace(s, joint_pos=s.joint_pos.transpose(0,1), joint_vel=s.joint_vel.transpose(0,1),
                    root_ang_vel_b=s.root_ang_vel_b.transpose(0,1), gravity_b=s.gravity_b.transpose(0,1))
        torch.testing.assert_close(b(s, p.transpose(0,1)), expected.transpose(0,1))

    def test_all_joint_permutations_randomized(self):
        generator = torch.Generator().manual_seed(2026)
        for robot_id in ('k1_22dof', 'g1_23dof', 'g1_29dof'):
            robot = get_robot_contract(robot_id)
            n = robot.dof
            for trial in range(10):
                permutation = torch.randperm(n, generator=generator)
                names = tuple(robot.joint_names[i] for i in permutation.tolist())
                default = torch.rand(n, generator=generator)
                q = torch.rand(2, 3, n, generator=generator)
                vel = torch.rand(2, 3, n, generator=generator)
                prev = torch.rand(2, 3, n, generator=generator)
                root = torch.randn(2, 3, 3, generator=generator)
                g = torch.zeros_like(root); g[..., 2] = -1
                s = RobotState(names, q[..., permutation], vel[..., permutation], root, g)
                actual = ObservationBuilder(robot, names, default)(s, prev)
                expected = torch.cat((root, g, q-default, vel, prev), dim=-1)
                torch.testing.assert_close(actual, expected)

    def test_wrong_default_shape_and_dtype(self):
        b, s, *_ = self.fixture()
        with self.assertRaises(ValueError):
            ObservationBuilder(b.robot, s.joint_names, torch.zeros(1, b.robot.dof))
        with self.assertRaises(TypeError):
            ObservationBuilder(b.robot, s.joint_names, torch.zeros(b.robot.dof, dtype=torch.long))
        with self.assertRaises(TypeError):
            ObservationBuilder(b.robot, s.joint_names, [0.0] * b.robot.dof)

    def test_default_nonfinite(self):
        b, s, *_ = self.fixture()
        for value in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                q0 = b.default_joint_pos.clone(); q0[0] = value
                ObservationBuilder(b.robot, s.joint_names, q0)

    def test_wrong_joint_set(self):
        b, s, *_ = self.fixture()
        for names in (s.joint_names[:-1], s.joint_names[:-1]+('other',),
                      s.joint_names[:-1]+(s.joint_names[0],)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                ObservationBuilder(b.robot, names, b.default_joint_pos)

    def test_changed_joint_order(self):
        b, s, p, _ = self.fixture()
        with self.assertRaises(ValueError):
            b(replace(s, joint_names=b.robot.joint_names), p)

    def test_invalid_state_and_field_types(self):
        b, s, p, _ = self.fixture()
        with self.assertRaises(TypeError): b({}, p)
        with self.assertRaises(TypeError): b(replace(s, joint_pos=[]), p)
        with self.assertRaises(TypeError): b(replace(s, joint_vel=[]), p)
        with self.assertRaises(TypeError): b(s, [])

    def test_mismatched_leading_dimensions(self):
        b, s, p, _ = self.fixture(leading=(2,3))
        for field in ('joint_vel', 'root_ang_vel_b', 'gravity_b'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                b(replace(s, **{field: getattr(s, field)[:1]}), p)
        with self.assertRaises(ValueError): b(s, p[:1])

    def test_wrong_joint_width(self):
        b, s, p, _ = self.fixture()
        with self.assertRaises(ValueError): b(replace(s, joint_pos=s.joint_pos[:,:-1]), p)
        with self.assertRaises(ValueError): b(replace(s, joint_vel=s.joint_vel[:,:-1]), p)
        with self.assertRaises(ValueError): b(s, p[:,:-1])

    def test_unbatched_and_extra_axis_rejected(self):
        b, s, p, _ = self.fixture()
        for q in (s.joint_pos[0], s.joint_pos[None,None]):
            with self.subTest(shape=q.shape), self.assertRaises(ValueError):
                b(replace(s, joint_pos=q), p)

    def test_empty_batch_rejected(self):
        b, s, p, _ = self.fixture()
        with self.assertRaises(ValueError): b(replace(s, joint_pos=s.joint_pos[:0]), p)

    def test_wrong_dtype(self):
        b, s, p, _ = self.fixture()
        for dtype in (torch.float64, torch.long):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                b(replace(s, joint_vel=s.joint_vel.to(dtype)), p)

    def test_wrong_device_rejected(self):
        b, s, p, _ = self.fixture()
        with self.assertRaises(ValueError): b(replace(s, joint_vel=s.joint_vel.to('meta')), p)

    def test_all_fields_nonfinite(self):
        b, s, p, _ = self.fixture()
        for field in ('joint_pos', 'joint_vel', 'root_ang_vel_b', 'gravity_b', 'previous_action'):
            for bad in (float('nan'), float('inf'), float('-inf')):
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    if field == 'previous_action':
                        invalid = p.clone(); invalid[0,0] = bad
                        b(s, invalid)
                    else:
                        invalid = getattr(s,field).clone(); invalid[0,0] = bad
                        b(replace(s, **{field:invalid}), p)

    def test_gravity_magnitude_rejected(self):
        b, s, p, _ = self.fixture()
        for scale in (0.0, 9.81, 0.5):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                b(replace(s, gravity_b=s.gravity_b*scale), p)

    def test_output_arithmetic_overflow_rejected(self):
        b, s, p, _ = self.fixture()
        maxval = torch.finfo(torch.float32).max
        b = ObservationBuilder(b.robot, s.joint_names, torch.full((b.robot.dof,), -maxval))
        with self.assertRaises(ValueError):
            b(replace(s, joint_pos=torch.full_like(s.joint_pos, maxval)), p)

    def test_valid_tilt_does_not_get_normalized_away(self):
        b, s, p, _ = self.fixture()
        g = torch.tensor([[0.0, 0.6, -0.8]]).repeat(2,1)
        out = b(replace(s, gravity_b=g), p)
        torch.testing.assert_close(out[:,3:6], g)

    def test_input_output_storage_independent(self):
        b, s, p, _ = self.fixture()
        out = b(s,p)
        expected = out.clone()
        s.joint_pos.add_(100)
        s.gravity_b.zero_()
        p.zero_()
        torch.testing.assert_close(out, expected)

    def test_large_batch_all_robots(self):
        for robot_id in ('k1_22dof', 'g1_23dof', 'g1_29dof'):
            b,s,p,expected = self.fixture(get_robot_contract(robot_id), (1024,50))
            torch.testing.assert_close(b(s,p), expected)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)

"""第11模块无 mjlab/CUDA 时可运行的合同与 mock-env 测试（不能替代真实物理测试）。"""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import torch
from unified_motion.robot_contract import get_robot_contract
from unified_motion.envs.mjlab_k1 import (
    ROBOT_ID, K1MJLabBridge, read_k1_joint_specs, official_xml_path,
    _joint_hz,
)

ROBOT = get_robot_contract(ROBOT_ID)


def create_fixture_xml(path: Path, *, remove_joint: bool = False, duplicate_motor: bool = False,
                       zero_armature: bool = False):
    """仅验证 XML 校验规则；不冒充 Booster 官方几何资产。"""
    mj = ET.Element('mujoco')
    ET.SubElement(mj, 'compiler', angle='radian')
    wb = ET.SubElement(mj, 'worldbody')
    base = ET.SubElement(wb, 'body', name='trunk')
    ET.SubElement(base, 'joint', name='world_joint', type='free')
    actuators = ET.SubElement(mj, 'actuator')
    for i, name in enumerate(ROBOT.joint_names):
        if not (remove_joint and i == 0):
            ET.SubElement(base, 'joint', name=name, range='-1.5 1.5',
                          armature='0' if zero_armature and i == 0 else '0.04')
        ET.SubElement(actuators, 'motor', name=name, joint=name, forcerange='-14 14')
    if duplicate_motor:
        actuators[-1].set('joint', ROBOT.joint_names[0])
    ET.ElementTree(mj).write(path, encoding='unicode')


class FakeData:
    @property
    def root_link_pose_w(self):
        return torch.cat([self.root_link_pos_w, self.root_link_quat_w], dim=-1)

    def __init__(self, b, names):
        self.default_joint_pos = torch.arange(len(names), dtype=torch.float32)[None, :].expand(b, -1).clone() * 0.01
        self.joint_pos = self.default_joint_pos.clone()
        self.joint_vel = torch.zeros_like(self.joint_pos)
        self.root_link_ang_vel_b = torch.zeros(b, 3)
        self.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0]]).repeat(b, 1)
        self.root_link_pos_w = torch.tensor([[0.0, 0.0, 0.57]]).repeat(b, 1)
        self.root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(b, 1)


class FakeTerm:
    def __init__(self, b, names):
        self.target_names = list(names)
        self.raw_action = torch.zeros(b, len(names))


class FakeEnv:
    """只模仿必要接口；不解算碰撞、惯性、马达或接触力。"""
    def __init__(self, b=3):
        self.device = 'cpu'; self.num_envs = b; self.step_dt = 0.02
        self.cfg = SimpleNamespace(auto_reset=False)
        names = tuple(reversed(ROBOT.joint_names))
        self.scene = {'robot': SimpleNamespace(joint_names=names, data=FakeData(b, names))}
        self.term = FakeTerm(b, tuple(ROBOT.joint_names[4:] + ROBOT.joint_names[:4]))
        self.action_manager = SimpleNamespace(get_term=lambda _: self.term, total_action_dim=22)
        self.last_action = None; self.step_index = 0; self.reset_ids = []
        self.closed = False

    def reset(self, *, env_ids):
        ids = env_ids.detach().cpu().tolist()
        self.reset_ids.append(ids)
        d = self.scene['robot'].data
        d.joint_pos[ids] = d.default_joint_pos[ids]
        d.joint_vel[ids] = 0
        d.root_link_pos_w[ids, 2] = 0.57
        self.term.raw_action[ids] = 0
        return {}, {}

    def step(self, action):
        self.last_action = action.clone()
        self.term.raw_action[:] = action
        self.step_index += 1
        d = self.scene['robot'].data
        d.joint_pos += 0.005
        d.root_link_pos_w[:, 2] -= 0.001
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.zeros_like(done)
        if self.step_index == 1:
            done[0] = True
            truncated[0] = True  # 碰巧两种信号都发出，优先真实终止。
            truncated[-1] = True
        reward = torch.arange(self.num_envs, dtype=torch.float32)
        return {}, reward, done, truncated, {'fake': True}

    def close(self):
        self.closed = True


class K1XmlTests(unittest.TestCase):
    def test_xml_parsing_and_order_set(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'K1_22dof.xml'
            create_fixture_xml(path)
            result = read_k1_joint_specs(path)
            self.assertEqual(tuple(x.name for x in result), ROBOT.joint_names)
            self.assertEqual(len(result), 22)
            self.assertAlmostEqual(result[0].effort, 14)
            self.assertEqual(official_xml_path(directory), path)

    def test_reject_missing_wrong_duplicate_or_invalid_actuators(self):
        for kw in ({'remove_joint': True}, {'duplicate_motor': True}, {'zero_armature': True}):
            with self.subTest(case=kw), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'K1_22dof.xml'; create_fixture_xml(path, **kw)
                with self.assertRaises(ValueError): read_k1_joint_specs(path)

    def test_wrong_folder_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError): official_xml_path(directory)

    def test_gain_families_are_explicit(self):
        self.assertEqual(_joint_hz('left_knee_pitch_joint'), (4.0, 1.0))
        self.assertEqual(_joint_hz('aahead_yaw_joint'), (10.0, 2.0))


class K1BridgeTests(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv()
        self.bridge = K1MJLabBridge(self.env)

    def test_must_reset_before_reading(self):
        with self.assertRaises(RuntimeError): self.bridge.observation()
        with self.assertRaises(RuntimeError): self.bridge.standing_probe_reference()

    def test_reset_and_joint_observation_reordering(self):
        self.bridge.reset()
        self.assertEqual(self.bridge.observation().shape, (3, 72))
        self.assertEqual(self.bridge.episode_start.tolist(), [True] * 3)
        # reset 关节角就是 builder 默认值；上一步 action 仍是0。
        torch.testing.assert_close(self.bridge.observation()[:, 6:28], torch.zeros(3,22))
        torch.testing.assert_close(self.bridge.observation()[:, -22:], torch.zeros(3,22))

    def test_reference_is_static_probe_from_true_reset_pose(self):
        self.bridge.reset()
        ref = self.bridge.standing_probe_reference()
        self.assertEqual(ref.features.shape, (3,3,60))
        self.assertTrue(ref.valid_mask.all().item())
        torch.testing.assert_close(ref.features[...,0], torch.tensor([0.,0.02,0.04]).expand(3,-1))
        torch.testing.assert_close(ref.features[..., 16:38], torch.zeros(3,3,22))

    def test_action_mapping_done_terminal_read_reset(self):
        self.bridge.reset()
        raw = torch.randn(3,22)
        result = self.bridge.step(raw)
        expected = ROBOT.to_simulator_order(raw, self.env.term.target_names)
        torch.testing.assert_close(self.env.last_action, expected)
        self.assertEqual(result.episode_start.tolist(), [True, True, True])
        self.assertEqual(result.terminated.tolist(), [True,False,False])
        self.assertEqual(result.truncated.tolist(), [False,False,True])
        # 做 GAE bootstrap 前，允许读取 reset 前 final state。
        self.assertEqual(self.bridge.observation().shape, (3,72))
        self.assertEqual(self.bridge.standing_probe_reference().features.shape, (3,3,60))
        torch.testing.assert_close(self.bridge.observation()[:,-22:], raw)
        # 任一环境待 reset，就拒绝继续 step。
        with self.assertRaises(RuntimeError): self.bridge.step(raw)
        self.bridge.reset([0,2])
        self.assertEqual(self.bridge.episode_start.tolist(), [True,False,True])
        self.assertEqual(self.env.reset_ids[-1], [0,2])
        ret2 = self.bridge.step(torch.zeros(3,22))
        self.assertEqual(ret2.episode_start.tolist(), [True,False,True])
        self.assertFalse((ret2.terminated | ret2.truncated).any().item())

    def test_invalid_physics_state_stops_smoke(self):
        self.bridge.reset()
        original = self.env.step
        def corrupted(action):
            result = original(action)
            self.env.scene['robot'].data.root_link_pos_w[1, 0] = float('nan')
            return result
        self.env.step = corrupted
        with self.assertRaisesRegex(RuntimeError, '物理状态'):
            self.bridge.step(torch.zeros(3,22))

    def test_malformed_policy_action_rejected(self):
        self.bridge.reset()
        for wrong in (torch.zeros(3,23), torch.zeros(3,22,dtype=torch.float64),
                      torch.full((3,22),float('nan'))):
            with self.subTest(wrong=str(wrong.shape)+str(wrong.dtype)), self.assertRaises(ValueError):
                self.bridge.step(wrong)

    def test_wrong_auto_reset_or_joint_list_rejected(self):
        self.env.cfg.auto_reset = True
        with self.assertRaises(ValueError): K1MJLabBridge(self.env)
        self.env.cfg.auto_reset = False
        self.env.term.target_names[0] = 'unknown'
        with self.assertRaises(ValueError): K1MJLabBridge(self.env)

    def test_actor_critic_uses_real_bridge_contract(self):
        # 不启动 MuJoCo：仅验证真实第09模块能消费第11模块的接口。
        from unified_motion.models.actor_critic import MultiRobotActorCritic
        from unified_motion.models.transformer import TransformerConfig
        self.bridge.reset()
        cfg = TransformerConfig(d_model=16, n_layers=1, n_heads=4, n_kv_heads=2,
                                context_len=3, num_fine_experts=4, top_k=2,
                                expert_hidden_dim=32, router_hidden_dim=16,
                                check_values=False)
        model = MultiRobotActorCritic({ROBOT_ID: ROBOT}, transformer_config=cfg,
                                      encoder_hidden_dim=32, critic_hidden_dim=32,
                                      check_values=False).eval()
        obs = self.bridge.observation()
        reference = self.bridge.standing_probe_reference()
        step, cache = model.act_step(ROBOT_ID, obs, reference,
                                      env_ids=self.bridge.env_ids,
                                      episode_start=self.bridge.episode_start,
                                      deterministic=True)
        self.assertEqual(step.action.shape, (3,22))
        self.assertEqual(step.value.shape, (3,))
        result = self.bridge.step(step.action)
        # 终止前观测可供 Critic 计算截断下一状态价值。
        final_value = model.value(ROBOT_ID, self.bridge.observation(),
                                  self.bridge.standing_probe_reference())
        self.assertEqual(final_value.shape, (3,))
        self.bridge.reset([0,2])
        next_step, cache = model.act_step(ROBOT_ID, self.bridge.observation(),
                                          self.bridge.standing_probe_reference(),
                                          env_ids=self.bridge.env_ids, cache=cache,
                                          episode_start=self.bridge.episode_start,
                                          deterministic=True)
        self.assertEqual(next_step.action.shape, (3,22))

    def test_close(self):
        self.bridge.close(); self.assertTrue(self.env.closed)


if __name__ == '__main__':
    unittest.main()

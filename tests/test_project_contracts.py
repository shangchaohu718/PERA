"""公开入口、兼容路径、纯目标任务和checkpoint的CPU集成回归。"""
import ast
import copy
import importlib
import io
import json
from pathlib import Path
import pkgutil
import subprocess
import sys
import unittest

import torch

import unified_motion
from unified_motion.api import (
    build_policy, get_robot_contract, DynamicsModel, PhysicsSequenceCollector,
    PhysicsSequencePPO, DynamicsUpdater, PPOConfig,
)
from unified_motion.models.physics import PhysicsMLPActorCritic, ControlGoal
from unified_motion.models.physics_policy import PhysicsConditionedActorCritic
from unified_motion.robot_contract import RobotContract
from physics_fixtures import make_models


ROOT=Path(__file__).resolve().parents[1]


class ProjectContractTests(unittest.TestCase):
    def test_public_factory_three_architectures(self):
        r=get_robot_contract('k1_22dof')
        for kind in ('mlp','dense','moe'):
            p=build_policy({'k1_22dof':r},backbone=kind,d_model=16,hidden_dim=16,
                n_layers=1,context_len=3,physics_width=8,latent_width=4)
            self.assertIsInstance(p,PhysicsMLPActorCritic if kind=='mlp' else PhysicsConditionedActorCritic)

    def test_factory_reject_ambiguous_mlp_multi_robot(self):
        robots={k:get_robot_contract(k) for k in ('k1_22dof','g1_29dof')}
        with self.assertRaises(ValueError):build_policy(robots,backbone='mlp')
        with self.assertRaises(ValueError):build_policy(robots,backbone='typo')

    def test_old_names_are_aliases_not_duplicate_implementations(self):
        from unified_motion.models import physics_dynamics,physics,physics_policy_v2,physics_policy
        self.assertIs(physics_dynamics.DynamicsModel,physics.DynamicsModel)
        self.assertIs(physics_policy_v2.PhysicsConditionedActorCritic,physics_policy.PhysicsConditionedActorCritic)
        self.assertIs(physics_policy_v2.PhysicsPolicyV2Config,physics_policy.PhysicsPolicyConfig)
        from unified_motion.algorithms.ppo import compute_gae as old_gae
        from unified_motion.algorithms.common import compute_gae
        self.assertIs(old_gae,compute_gae)
        from unified_motion.algorithms import physics_training
        from unified_motion.legacy import mlp_physics_training
        self.assertIs(physics_training.PhysicsPPOUpdater,mlp_physics_training.PhysicsPPOUpdater)

    def test_all_modules_import_without_simulator(self):
        for module in pkgutil.walk_packages(unified_motion.__path__,unified_motion.__name__+'.'):
            with self.subTest(module=module.name):importlib.import_module(module.name)

    def test_production_has_no_embedded_selftests(self):
        for path in (ROOT/'unified_motion').rglob('*.py'):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            self.assertFalse(any(isinstance(x,ast.FunctionDef) and x.name=='_self_test' for x in ast.walk(tree)),str(path))

    def test_cli_help_needs_no_cuda(self):
        result=subprocess.run([sys.executable,'-m','unified_motion.train','--help'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--backbone',result.stdout)

    def test_production_cli_refuses_fake_cpu_fallback(self):
        result=subprocess.run([sys.executable,'-m','unified_motion.train','--asset-root','dummy','--motions','dummy',
                               '--device','cpu'],cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('假 CPU',result.stderr)

    def test_pure_contact_goal_all_policy_update(self):
        for kind in ('mlp','dense','moe'):
            task,dyn,policy=make_models(kind,multimodal=True)
            old_goal=task.goal
            def goal():
                g=old_goal();g.motion=None;g.object_features=None;g.object_mask=None;return g
            task.goal=goal
            col=PhysicsSequenceCollector(task,dyn,policy);roll=col.collect(8)
            PhysicsSequencePPO(policy,PPOConfig(epochs=1,minibatch_envs=2)).update(roll)
            DynamicsUpdater(dyn).update(roll)

    def test_pure_object_goal_all_policy_update(self):
        for kind in ('mlp','dense','moe'):
            task,dyn,policy=make_models(kind,multimodal=True)
            old_goal=task.goal
            def goal():
                g=old_goal();g.motion=None;g.contact_features=None;g.contact_mask=None;return g
            task.goal=goal
            col=PhysicsSequenceCollector(task,dyn,policy);roll=col.collect(8)
            PhysicsSequencePPO(policy,PPOConfig(epochs=1,minibatch_envs=2)).update(roll)

    def test_all_goals_absent_rejected(self):
        for kind in ('mlp','dense','moe'):
            task,dyn,p=make_models(kind);task.goal=lambda:ControlGoal()
            with self.assertRaises(ValueError):PhysicsSequenceCollector(task,dyn,p).collect(2)

    def test_mlp_checkpoint_joint_order_and_metadata_required(self):
        task,_,p=make_models('mlp')
        robot=RobotContract('other',task.robot.root_name,tuple(reversed(task.robot.joint_names)))
        other=PhysicsMLPActorCritic(robot,physics_width=8,latent_width=4,hidden_dim=16)
        with self.assertRaises(ValueError):other.load_state_dict(p.state_dict())
        state=p.state_dict();state.pop('_extra_state')
        with self.assertRaises(RuntimeError):p.load_state_dict(state,strict=True)

    def test_safe_weights_only_full_checkpoint_roundtrip(self):
        task,dyn,p=make_models('moe');c=PhysicsSequenceCollector(task,dyn,p)
        ppo=PhysicsSequencePPO(p,PPOConfig(epochs=1));d= DynamicsUpdater(dyn)
        r=c.collect(4);ppo.update(r);d.update(r)
        stream=io.BytesIO();torch.save({'schema':1,'ppo':ppo.state_dict(),'dynamics':d.state_dict()},stream);stream.seek(0)
        ckpt=torch.load(stream,weights_only=True)
        ppo.load_state_dict(ckpt['ppo']);d.load_state_dict(ckpt['dynamics'])
        c.on_dynamics_updated();r=c.collect(3);self.assertLess(ppo.verify_old_policy(r),1e-3)

    def test_trainable_dynamics_can_reduce_a_fixed_cpu_prediction_loss(self):
        # 这只是梯度方向/学习能力的数学检查，不是环境泛化实验。
        task,m,_=make_models();x=task.read();a=torch.zeros(2,task.robot.dof);z=torch.zeros(2,4)
        optimizer=torch.optim.Adam(m.parameters(),lr=.002)
        losses=[]
        for _ in range(12):
            loss=m.prediction_loss(m.predict_next(x,a,z),x,valid_transition=torch.ones(2,dtype=torch.bool)).total
            losses.append(loss.item());optimizer.zero_grad();loss.backward();optimizer.step()
        self.assertLess(losses[-1],losses[0])

    def test_mlp_critic_nonfinite_rejected(self):
        task,d,p=make_models('mlp');d.eval();enc,_=d.step(task.read(),episode_start=task.episode_start,env_ids=task.env_ids)
        with torch.no_grad():p.value_net[-1].weight.fill_(float('inf'))
        with self.assertRaises(ValueError):p(task.read(),enc,task.goal())


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)

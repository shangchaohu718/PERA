"""第 09 模块的独立测试：不需要 Isaac Lab、mjlab 或 PPO 训练器。"""

import copy
import io
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_motion.robot_contract import RobotContract, get_robot_contract
from unified_motion.observations import ObservationBuilder, RobotState
from unified_motion.reference_observations import ReferenceObservation, ReferenceObservationBuilder, ReferenceWindow
from unified_motion.models.actor_critic import (
    GaussianActionHead, MultiRobotActorCritic, MultiRobotValueCritic,
)
from unified_motion.models.transformer import TransformerConfig

ROBOT_IDS = ('k1_22dof', 'g1_23dof', 'g1_29dof')


def model_fixture(*, robots=None, privileged_dims=None, dtype=torch.float32, **kw):
    if robots is None:
        robots = {k: get_robot_contract(k) for k in ROBOT_IDS}
    config = TransformerConfig(
        d_model=16, n_layers=2, n_heads=4, n_kv_heads=2,
        context_len=4, num_fine_experts=4, num_shared_experts=1,
        top_k=2, expert_hidden_dim=24, router_hidden_dim=16,
        check_values=kw.pop('check_values', True),
    )
    return MultiRobotActorCritic(
        robots, transformer_config=config, encoder_hidden_dim=32,
        critic_hidden_dim=32, init_action_std=kw.pop('init_action_std', 0.9),
        privileged_dims=privileged_dims, check_values=config.check_values,
        **kw,
    ).to(dtype=dtype)


def batch_fixture(robot_id='k1_22dof', *, b=2, t=7, h=3,
                  privileged_dim=0, dtype=torch.float32):
    robot = get_robot_contract(robot_id)
    obs = torch.randn(b, t, 6 + 3*robot.dof, dtype=dtype) * 0.1
    refs = torch.randn(b, t, h, 16 + 2*robot.dof, dtype=dtype) * 0.1
    mask = torch.ones(b, t, h, dtype=torch.bool)
    if h > 1:
        mask[..., -1] = False
    reference = ReferenceObservation(refs, mask)
    privileged = torch.randn(b, t, privileged_dim, dtype=dtype) if privileged_dim else None
    return obs, reference, privileged


def one_step(reference, time):
    return ReferenceObservation(reference.features[:, time], reference.valid_mask[:, time])


class ActorCriticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.model = model_fixture()
        self.obs, self.ref, _ = batch_fixture()

    def test_k1_and_g1_shapes_and_joint_order(self):
        for rid in ROBOT_IDS:
            with self.subTest(robot=rid):
                robot = get_robot_contract(rid)
                obs, ref, _ = batch_fixture(rid)
                out = self.model(rid, obs, ref)
                self.assertEqual(out.mean.shape, (2, 7, robot.dof))
                self.assertEqual(out.std.shape, (2, 7, robot.dof))
                self.assertEqual(out.values.shape, (2, 7))
                self.assertEqual(out.joint_names, robot.joint_names)
                self.assertTrue((out.std > 0).all())
                torch.testing.assert_close(out.std, torch.full_like(out.std, 0.9))

    def test_distribution_log_prob_is_sum_over_joint_axis(self):
        out = self.model('k1_22dof', self.obs, self.ref)
        actions = torch.randn_like(out.mean)
        expected = Normal(out.mean, out.std).log_prob(actions).sum(dim=-1)
        torch.testing.assert_close(out.distribution().log_prob(actions), expected)

    def test_entropy_is_sum_over_joint_axis(self):
        out = self.model('k1_22dof', self.obs, self.ref)
        torch.testing.assert_close(out.distribution().entropy(),
                                   Normal(out.mean, out.std).entropy().sum(-1))

    def test_policy_mean_is_not_action_clipped(self):
        m = self.model.eval()
        head = m.action_heads[m._keys['k1_22dof']]
        with torch.no_grad():
            head.mean_head.weight.zero_()
            head.mean_head.bias.fill_(10.0)
        one, _ = m.act_step('k1_22dof', self.obs[:, 0], one_step(self.ref, 0),
                            env_ids=(1, 2), deterministic=True)
        torch.testing.assert_close(one.action, torch.full_like(one.action, 10.0))
        self.assertTrue((one.action > 1).all().item())

    def test_log_std_is_trainable_and_no_silent_cap(self):
        rid = 'k1_22dof'
        out = self.model(rid, self.obs, self.ref)
        fixed_actions = torch.ones_like(out.mean) * 2.0
        result = self.model.evaluate_actions(rid, self.obs, self.ref, fixed_actions)
        (-result.log_prob.mean()).backward()
        head = self.model.action_heads[self.model._keys[rid]]
        self.assertIsNotNone(head.log_std.grad)
        self.assertGreater(head.log_std.grad.abs().sum().item(), 0.0)
        with torch.no_grad():
            head.log_std.fill_(math.log(2.0))
        new_std = self.model(rid, self.obs, self.ref).std
        torch.testing.assert_close(new_std, torch.full_like(new_std, 2.0))

    def test_invalid_std_stops_instead_of_clipping(self):
        head = self.model.action_heads[self.model._keys['k1_22dof']]
        with torch.no_grad():
            head.log_std.fill_(1000.0)
        with self.assertRaises(ValueError):
            self.model('k1_22dof', self.obs, self.ref)

    def test_invalid_initial_std(self):
        for v in (0.0, -1.0, float('inf'), float('nan')):
            with self.subTest(v=v), self.assertRaises(ValueError):
                GaussianActionHead(16, 22, v)
        with self.assertRaises(TypeError):
            GaussianActionHead(16, 22, True)

    def test_gaussian_head_initial_mean_near_zero(self):
        x = torch.randn(100, 16)
        head = GaussianActionHead(16, 22, 1.0)
        mu, std = head(x)
        self.assertLess(mu.abs().max().item(), 0.3)
        torch.testing.assert_close(std, torch.ones_like(std))

    def test_evaluate_recalculates_stored_raw_actions(self):
        out = self.model('k1_22dof', self.obs, self.ref)
        stored = out.distribution().sample().detach()
        ev = self.model.evaluate_actions('k1_22dof', self.obs, self.ref, stored)
        torch.testing.assert_close(ev.log_prob, out.distribution().log_prob(stored))
        torch.testing.assert_close(ev.entropy, out.distribution().entropy())
        torch.testing.assert_close(ev.policy.values, out.values)

    def test_joint_action_logprob_not_componentwise(self):
        out = self.model('g1_29dof', *batch_fixture('g1_29dof')[:2])
        self.assertEqual(out.distribution().event_shape, (29,))
        self.assertEqual(out.distribution().batch_shape, (2,7))

    def test_trainable_gradient_actor_critic_and_router(self):
        actions = torch.randn(2, 7, 22)
        ev = self.model.evaluate_actions('k1_22dof', self.obs, self.ref, actions)
        loss = -ev.log_prob.mean() + ev.policy.values.square().mean() + 0.05*ev.policy.balance_loss
        loss.backward()
        actor = list(self.model.action_heads[self.model._keys['k1_22dof']].parameters())
        critic = list(self.model.critic.parameters())
        ref_branch = list(self.model.encoder.reference_encoders['robot_2'].parameters())
        # K1 是按 ID 排序后的第 2 个 robot_2；选中分支必须能接收梯度。
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in ref_branch))
        self.assertTrue(all(torch.isfinite(p.grad).all().item()
                            for p in self.model.parameters() if p.grad is not None))

    def test_value_only_loss_does_not_update_actor(self):
        m = self.model
        loss = m('k1_22dof', self.obs, self.ref).values.square().mean()
        loss.backward()
        self.assertFalse(any(p.grad is not None for p in m.encoder.parameters()))
        self.assertFalse(any(p.grad is not None for p in m.backbone.parameters()))
        self.assertFalse(any(p.grad is not None for p in m.action_heads.parameters()))
        self.assertTrue(any(p.grad is not None for p in m.critic.parameters()))

    def test_privileged_changes_critic_not_actor(self):
        m = model_fixture(privileged_dims={'k1_22dof': 4})
        a = torch.zeros(2,7,4)
        b = torch.ones(2,7,4)
        output0 = m('k1_22dof', self.obs, self.ref, privileged=a)
        output1 = m('k1_22dof', self.obs, self.ref, privileged=b)
        torch.testing.assert_close(output0.mean, output1.mean, atol=0, rtol=0)
        torch.testing.assert_close(output0.std, output1.std, atol=0, rtol=0)
        self.assertGreater((output0.values-output1.values).abs().max().item(), 1e-7)

    def test_privileged_contract_requires_correct_shape(self):
        m = model_fixture(privileged_dims={'k1_22dof': 4})
        with self.assertRaises(ValueError):m('k1_22dof',self.obs,self.ref)
        with self.assertRaises(ValueError):m('k1_22dof',self.obs,self.ref,privileged=torch.zeros(2,7,3))
        with self.assertRaises(ValueError):self.model('k1_22dof',self.obs,self.ref,privileged=torch.zeros(2,7,4))

    def test_privileged_bad_dims_configuration(self):
        with self.assertRaises(ValueError):model_fixture(privileged_dims={'unknown': 3})
        with self.assertRaises(ValueError):model_fixture(privileged_dims={'k1_22dof': -1})
        with self.assertRaises(ValueError):model_fixture(privileged_dims={'k1_22dof': True})

    def test_critic_reference_padding_does_not_affect_value(self):
        original = self.model.value('k1_22dof', self.obs, self.ref)
        changed = self.ref.features.clone()
        changed[~self.ref.valid_mask] = 9999.0
        value = self.model.value('k1_22dof', self.obs, replace(self.ref, features=changed))
        torch.testing.assert_close(value, original)

    def test_critic_masked_reference_gradient_zero(self):
        m = self.model
        feat = self.ref.features.detach().clone().requires_grad_()
        value = m.value('k1_22dof', self.obs, replace(self.ref, features=feat))
        value.sum().backward()
        self.assertEqual(feat.grad[~self.ref.valid_mask].abs().sum().item(), 0.0)
        self.assertTrue((feat.grad[self.ref.valid_mask].abs().sum() > 0).item())

    def test_fully_padded_tail_returns_zero(self):
        mask = torch.ones(2,7,dtype=torch.bool);mask[0,5:]=False;mask[1,6:]=False
        refmask = self.ref.valid_mask.clone();refmask[~mask]=False
        ref = replace(self.ref, valid_mask=refmask)
        out = self.model('k1_22dof', self.obs, ref, valid_mask=mask)
        self.assertEqual(out.mean[~mask].count_nonzero().item(), 0)
        self.assertEqual(out.values[~mask].count_nonzero().item(), 0)
        self.assertTrue((out.std[~mask] == 1).all().item())
        action = torch.randn_like(out.mean).requires_grad_()
        ev = self.model.evaluate_actions('k1_22dof', self.obs, ref, action, valid_mask=mask)
        self.assertEqual(ev.log_prob[~mask].count_nonzero().item(), 0)
        self.assertEqual(ev.entropy[~mask].count_nonzero().item(), 0)
        (ev.log_prob.sum()+ev.policy.values.sum()).backward()
        self.assertEqual(action.grad[~mask].count_nonzero().item(), 0)

    def test_invalid_padding_is_immune_to_large_values(self):
        mask = torch.ones(2,7,dtype=torch.bool);mask[:,5:]=False
        refmask = self.ref.valid_mask.clone();refmask[:,5:]=False
        ref = replace(self.ref, valid_mask=refmask)
        base = self.model('k1_22dof', self.obs, ref, valid_mask=mask)
        dirty_obs=self.obs.clone();dirty_obs[:,5:]=9999.0
        dirty_ref=self.ref.features.clone();dirty_ref[:,5:]=9999.0
        dirty = self.model('k1_22dof', dirty_obs, replace(ref,features=dirty_ref),valid_mask=mask)
        torch.testing.assert_close(dirty.mean,base.mean)
        torch.testing.assert_close(dirty.values,base.values)
        torch.testing.assert_close(dirty.std,base.std)

    def test_entire_padded_batch_returns_zero(self):
        valid = torch.zeros(2,7,dtype=torch.bool)
        ref = replace(self.ref, valid_mask=torch.zeros_like(self.ref.valid_mask))
        out = self.model('k1_22dof',self.obs,ref,valid_mask=valid)
        self.assertEqual(out.mean.count_nonzero().item(),0)
        self.assertEqual(out.values.count_nonzero().item(),0)
        self.assertEqual(out.balance_loss.item(),0)

    def test_ref_mask_missing_on_valid_time_rejected(self):
        mask=self.ref.valid_mask.clone();mask[0,2]=False
        with self.assertRaises(ValueError):
            self.model('k1_22dof',self.obs,replace(self.ref,valid_mask=mask))

    def test_valid_mask_holes_rejected(self):
        mask = torch.ones(2,7,dtype=torch.bool);mask[0,3]=False
        with self.assertRaises(ValueError):self.model('k1_22dof',self.obs,self.ref,valid_mask=mask)

    def test_episode_start_on_padding_rejected(self):
        valid = torch.ones(2,7,dtype=torch.bool);valid[:,5:]=False
        starts = torch.zeros(2,7,dtype=torch.bool);starts[1,5]=True
        with self.assertRaises(ValueError):
            self.model('k1_22dof',self.obs,self.ref,valid_mask=valid,episode_start=starts)

    def test_episode_reset_prevents_past_leak(self):
        starts = torch.zeros(2,7,dtype=torch.bool);starts[0,4]=True
        original = self.model('k1_22dof',self.obs,self.ref,episode_start=starts)
        changed_obs = self.obs.clone();changed_obs[0,:4]+=123.0
        changed_ref = self.ref.features.clone();changed_ref[0,:4]+=123.0
        changed = self.model('k1_22dof',changed_obs,replace(self.ref,features=changed_ref),episode_start=starts)
        torch.testing.assert_close(changed.mean[0,4:],original.mean[0,4:])
        torch.testing.assert_close(changed.values[0,4:],original.values[0,4:])

    def test_future_true_obs_and_reference_not_leak(self):
        old = self.model('k1_22dof',self.obs,self.ref)
        changed_obs=self.obs.clone();changed_obs[:,4:]+=1000
        changed_ref=self.ref.features.clone();changed_ref[:,4:]+=1000
        changed=self.model('k1_22dof',changed_obs,replace(self.ref,features=changed_ref))
        torch.testing.assert_close(changed.mean[:,:4],old.mean[:,:4])
        torch.testing.assert_close(changed.values[:,:4],old.values[:,:4])

    def test_deterministic_stream_equal_sequence(self):
        for rid in ROBOT_IDS:
            with self.subTest(robot=rid):
                obs,ref,_=batch_fixture(rid,t=9)
                m=self.model.eval()
                starts=torch.zeros(2,9,dtype=torch.bool);starts[0,5]=True
                with torch.no_grad():
                    whole=m(rid,obs,ref,episode_start=starts)
                    cache=None;actions=[];values=[];std=[]
                    for t in range(9):
                        step,cache=m.act_step(rid,obs[:,t],one_step(ref,t),
                            env_ids=(21,22),cache=cache,episode_start=starts[:,t],
                            deterministic=True)
                        actions.append(step.action);values.append(step.value);std.append(step.std)
                    torch.testing.assert_close(torch.stack(actions,1),whole.mean,atol=3e-6,rtol=3e-5)
                    torch.testing.assert_close(torch.stack(values,1),whole.values,atol=1e-6,rtol=2e-5)
                    torch.testing.assert_close(torch.stack(std,1),whole.std,atol=0,rtol=0)

    def test_stochastic_rollout_action_and_logprob_recompute(self):
        rid='k1_22dof';m=self.model.eval()
        starts=torch.zeros(2,7,dtype=torch.bool);starts[1,3]=True
        acts=[];old_probs=[];cache=None
        for t in range(7):
            step,cache=m.act_step(rid,self.obs[:,t],one_step(self.ref,t),
                                  env_ids=(100,101),cache=cache,episode_start=starts[:,t])
            acts.append(step.action);old_probs.append(step.log_prob)
        actions=torch.stack(acts,dim=1)
        old=torch.stack(old_probs,dim=1)
        m.train()
        ev=m.evaluate_actions(rid,self.obs,self.ref,actions,episode_start=starts)
        torch.testing.assert_close(ev.log_prob,old,atol=4e-6,rtol=4e-5)

    def test_action_step_rejects_train_mode(self):
        with self.assertRaises(RuntimeError):
            self.model.act_step('k1_22dof',self.obs[:,0],one_step(self.ref,0),env_ids=(1,2))

    def test_step_cache_rejects_other_robot_and_env_order(self):
        m=self.model.eval()
        a,cache=m.act_step('k1_22dof',self.obs[:,0],one_step(self.ref,0),env_ids=(1,2))
        with self.assertRaises(ValueError):
            m.act_step('k1_22dof',self.obs[:,1],one_step(self.ref,1),env_ids=(2,1),cache=cache)
        ob,rr,_=batch_fixture('g1_29dof',t=1)
        with self.assertRaises(ValueError):
            m.act_step('g1_29dof',ob[:,0],one_step(rr,0),env_ids=(1,2),cache=cache)

    def test_cache_invalidation_after_weights_update(self):
        m=self.model.eval()
        _,cache=m.act_step('k1_22dof',self.obs[:,0],one_step(self.ref,0),env_ids=(1,2))
        m.invalidate_caches()
        with self.assertRaises(ValueError):
            m.act_step('k1_22dof',self.obs[:,1],one_step(self.ref,1),env_ids=(1,2),cache=cache)

    def test_wrong_robot_names_rejected(self):
        with self.assertRaises(KeyError): self.model('g1',self.obs,self.ref)
        with self.assertRaises(KeyError): self.model.value('g1',self.obs,self.ref)

    def test_wrong_action_shape_and_types_rejected(self):
        rid='k1_22dof'
        with self.assertRaises(ValueError):
            self.model.evaluate_actions(rid,self.obs,self.ref,torch.zeros(2,7,23))
        with self.assertRaises(ValueError):
            self.model.evaluate_actions(rid,self.obs,self.ref,torch.zeros(2,6,22))
        with self.assertRaises(ValueError):
            self.model.evaluate_actions(rid,self.obs,self.ref,torch.zeros(2,7,22,dtype=torch.float64))

    def test_nonfinite_action_rejected(self):
        actions=torch.randn(2,7,22);actions[0,0,0]=float('nan')
        with self.assertRaises(ValueError):
            self.model.evaluate_actions('k1_22dof',self.obs,self.ref,actions)

    def test_nonfinite_observation_and_ref_rejected(self):
        bad=self.obs.clone();bad[0,0,0]=float('nan')
        with self.assertRaises(ValueError):self.model('k1_22dof',bad,self.ref)
        bad_ref=self.ref.features.clone();bad_ref[0,0,0,0]=float('inf')
        with self.assertRaises(ValueError):
            self.model('k1_22dof',self.obs,replace(self.ref,features=bad_ref))

    def test_observation_and_reference_wrong_dims_rejected(self):
        with self.assertRaises(ValueError):self.model('k1_22dof',self.obs[:,:,:-1],self.ref)
        with self.assertRaises(ValueError):self.model('k1_22dof',self.obs,replace(self.ref,features=self.ref.features[:,:,:,:-1]))

    def test_critic_can_bootstrap_single_step(self):
        ref=one_step(self.ref,0)
        prediction = self.model.value('k1_22dof',self.obs[:,0],ref)
        self.assertEqual(prediction.shape,(2,))
        torch.testing.assert_close(prediction,self.model('k1_22dof',self.obs,self.ref).values[:,0])

    def test_save_load_roundtrip(self):
        rid='k1_22dof'
        out=self.model(rid,self.obs,self.ref)
        memory=io.BytesIO();torch.save(self.model.state_dict(),memory);memory.seek(0)
        new=model_fixture()
        new.load_state_dict(torch.load(memory,weights_only=True))
        newout=new(rid,self.obs,self.ref)
        torch.testing.assert_close(newout.mean,out.mean,atol=0,rtol=0)
        torch.testing.assert_close(newout.std,out.std,atol=0,rtol=0)
        torch.testing.assert_close(newout.values,out.values,atol=0,rtol=0)

    def test_wrong_joint_order_same_shape_checkpoint_rejected(self):
        contracts={k:get_robot_contract(k) for k in ROBOT_IDS}
        original=model_fixture(robots=contracts)
        k=contracts['k1_22dof']
        contracts['k1_22dof']=RobotContract(k.name,k.root_name,tuple(reversed(k.joint_names)))
        changed=model_fixture(robots=contracts)
        with self.assertRaises(ValueError):
            changed.load_state_dict(original.state_dict())

    def test_privileged_dims_checkpoint_rejected(self):
        m2=model_fixture(privileged_dims={'k1_22dof': 4})
        with self.assertRaises((RuntimeError,ValueError)):
            m2.load_state_dict(self.model.state_dict())

    def test_new_robot_contract(self):
        robot=RobotContract('new','base',('j1','j2','j3'))
        m=model_fixture(robots={'new':robot})
        obs=torch.randn(2,5,15)
        ref=ReferenceObservation(torch.randn(2,5,2,22),torch.ones(2,5,2,dtype=torch.bool))
        out=m('new',obs,ref)
        self.assertEqual(out.mean.shape,(2,5,3))
        self.assertEqual(out.joint_names,robot.joint_names)

    def test_float64_model(self):
        m=model_fixture(dtype=torch.float64)
        obs,ref,_=batch_fixture(dtype=torch.float64)
        out=m('k1_22dof',obs,ref)
        self.assertEqual(out.mean.dtype,torch.float64)
        self.assertEqual(out.std.dtype,torch.float64)
        self.assertEqual(out.values.dtype,torch.float64)

    def test_cpu_autocast_logprob_float32(self):
        m=self.model
        with torch.autocast('cpu',dtype=torch.bfloat16):
            out=m('k1_22dof',self.obs,self.ref)
            self.assertEqual(out.mean.dtype,torch.float32)
            self.assertEqual(out.std.dtype,torch.float32)
            self.assertEqual(out.values.dtype,torch.float32)
            ev=m.evaluate_actions('k1_22dof',self.obs,self.ref,torch.randn(2,7,22))
            self.assertEqual(ev.log_prob.dtype,torch.float32)
            (ev.log_prob.square().mean()+ev.policy.values.square().mean()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None))

    def test_value_actor_input_not_contaminated_by_privileged(self):
        m=model_fixture(privileged_dims={'k1_22dof':2})
        obs,ref,_=batch_fixture()
        extras = torch.randn(2,7,2,requires_grad=True)
        ev=m('k1_22dof',obs,ref,privileged=extras)
        ev.mean.square().sum().backward()
        self.assertIsNone(extras.grad)

    def test_bad_network_configuration_rejected(self):
        with self.assertRaises(ValueError):model_fixture(init_action_std=0.0)
        with self.assertRaises(ValueError):model_fixture(robots={})
        contracts={k:get_robot_contract(k) for k in ROBOT_IDS}
        with self.assertRaises(ValueError):
            MultiRobotActorCritic(contracts, transformer_config=TransformerConfig(check_values=False), check_values=True)

    def test_real_data_builders_to_actor_critic_three_robots(self):
        """从前面已经实现的 Builder 输出进入 Actor，而不是只测试随机特征。"""
        for rid in ROBOT_IDS:
            with self.subTest(robot=rid):
                robot=get_robot_contract(rid)
                n=robot.dof; b=2; t=4; h=2
                names=tuple(reversed(robot.joint_names))
                default=torch.zeros(n)
                obs_builder=ObservationBuilder(robot,names,default)
                ref_builder=ReferenceObservationBuilder(robot,names,default)
                state=RobotState(
                    joint_names=names,
                    joint_pos=torch.randn(b,t,n)*0.1,
                    joint_vel=torch.randn(b,t,n)*0.1,
                    root_ang_vel_b=torch.zeros(b,t,3),
                    gravity_b=torch.tensor([0.,0.,-1.]).expand(b,t,3).clone(),
                )
                obs=obs_builder(state,torch.zeros(b,t,n))
                quat=torch.zeros(b,t,h,4);quat[...,0]=1.0
                window=ReferenceWindow(
                    root_name=robot.root_name,joint_names=names,
                    joint_pos=torch.randn(b,t,h,n)*0.1,
                    joint_vel=torch.randn(b,t,h,n)*0.1,
                    root_pos_w=torch.zeros(b,t,h,3),
                    root_quat_wxyz=quat,
                    root_lin_vel_w=torch.zeros(b,t,h,3),
                    root_ang_vel_w=torch.zeros(b,t,h,3),
                    time_offsets_s=torch.tensor([0.,0.1]).expand(b,t,h).clone(),
                    valid_mask=torch.ones(b,t,h,dtype=torch.bool),
                )
                root_q=quat[...,0,:].clone()
                ref=ref_builder(window,current_root_pos_w=torch.zeros(b,t,3),
                                current_root_quat_wxyz=root_q)
                pol=self.model(rid,obs,ref)
                self.assertEqual(pol.mean.shape,(b,t,n))
                act=pol.distribution().sample()
                evaluation=self.model.evaluate_actions(rid,obs,ref,act)
                self.assertEqual(evaluation.log_prob.shape,(b,t))
                self.assertTrue(evaluation.log_prob.isfinite().all())

    def test_critic_direct_rejects_reference_holes(self):
        bad_mask=self.ref.valid_mask.clone()
        bad_mask[0,0]=torch.tensor([True,False,True])
        with self.assertRaises(ValueError):
            self.model.value('k1_22dof',self.obs,replace(self.ref,valid_mask=bad_mask))

    def test_critic_direct_rejects_time_holes(self):
        valid=torch.ones(2,7,dtype=torch.bool);valid[0,3]=False
        with self.assertRaises(ValueError):
            self.model.value('k1_22dof',self.obs,self.ref,valid_mask=valid)

    def test_step_sampling_has_no_grad(self):
        model=self.model.eval()
        action, _=model.act_step('k1_22dof', self.obs[:,0], one_step(self.ref,0),
                                 env_ids=(1,2))
        for item in (action.action,action.mean,action.std,action.log_prob,action.value):
            self.assertFalse(item.requires_grad)

    def test_ppo_ratio_matches_one_before_update(self):
        model=self.model.eval()
        acts=[];old=[];cache=None
        for i in range(self.obs.shape[1]):
            batch,cache=model.act_step('k1_22dof',self.obs[:,i],one_step(self.ref,i),
                                       env_ids=(1,2),cache=cache)
            acts.append(batch.action);old.append(batch.log_prob)
        result=model.evaluate_actions('k1_22dof',self.obs,self.ref,torch.stack(acts,1))
        ratio=(result.log_prob-torch.stack(old,1)).exp()
        torch.testing.assert_close(ratio,torch.ones_like(ratio),atol=1e-5,rtol=1e-5)

    def test_optimizer_updates_actor_and_critic_parameters(self):
        model=self.model
        actions=torch.randn(2,7,22)
        initial=model.evaluate_actions('k1_22dof',self.obs,self.ref,actions)
        p_mean=initial.policy.mean.detach().clone()
        p_value=initial.policy.values.detach().clone()
        opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        opt.zero_grad(set_to_none=True)
        # 这里只检查梯度更新链路，并不是完整 PPO 剪裁目标。
        loss=-initial.log_prob.mean()+initial.policy.values.square().mean()
        loss.backward();opt.step();model.invalidate_caches()
        updated=model.evaluate_actions('k1_22dof',self.obs,self.ref,actions)
        self.assertFalse(torch.equal(updated.policy.mean,p_mean))
        self.assertFalse(torch.equal(updated.policy.values,p_value))

    @unittest.skipUnless(torch.cuda.is_available(), '需要 CUDA GPU')
    def test_cuda_all_robots(self):
        m=self.model.cuda()
        for rid in ROBOT_IDS:
            obs,ref,_=batch_fixture(rid)
            ref=ReferenceObservation(ref.features.cuda(),ref.valid_mask.cuda())
            out=m(rid,obs.cuda(),ref)
            self.assertTrue(out.mean.is_cuda)
            self.assertTrue(torch.isfinite(out.values).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), '需要 CUDA GPU')
    def test_cuda_mixed_precision(self):
        m=self.model.cuda()
        ref=ReferenceObservation(self.ref.features.cuda(),self.ref.valid_mask.cuda())
        with torch.autocast('cuda', dtype=torch.float16):
            out=m('k1_22dof',self.obs.cuda(),ref)
            self.assertEqual(out.mean.dtype,torch.float32)
            self.assertTrue(torch.isfinite(out.values).all().item())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)

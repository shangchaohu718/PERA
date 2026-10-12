"""物理反馈版策略/动力学/时序 PPO 的实际 CPU 回归；数学夹具不是机器人仿真。"""
import copy
import io
import unittest
from unittest.mock import patch
from dataclasses import replace
import torch
from physics_fixtures import make_models
from unified_motion.models.physics_dynamics import DynamicsEncoding, DynamicsModel, _core_state, _safe_unit
from unified_motion.algorithms.ppo import PPOConfig
from unified_motion.algorithms.sequence_training import (
    PhysicsSequenceCollector, PhysicsSequencePPO, DynamicsUpdater,
    clone_tree, stack_tree, at_time, parameter_stamp, _history_size,
)
from unified_motion.robot_contract import RobotContract


class DynamicsRegressionTests(unittest.TestCase):
    def test_direct_multi_step_heads_train_and_mask_reset_crossings(self):
        task,_,policy=make_models('mlp',endings=False)
        model=DynamicsModel(task.robot,physics_width=8,memory_width=8,latent_width=4,
                            hidden_dim=16,prediction_horizons=(1,4,8))
        rollout=PhysicsSequenceCollector(task,model,policy).collect(10)
        # Simulate one environment restarting at t=5.  Windows that reach across t=5
        # must never use the new episode as a future label for the old episode.
        rollout.episode_start[0,5]=True
        captured=[]
        from unified_motion.algorithms import sequence_training as training
        original=training._core_prediction_terms
        def record(predicted,target,valid,robot,prefix):
            captured.append((prefix,valid.clone()))
            return original(predicted,target,valid,robot,prefix)
        with patch.object(training,'_core_prediction_terms',side_effect=record):
            stats=DynamicsUpdater(model,multi_step_weight=.1,
                                  latent_slow_weight=.01).update(rollout)
        self.assertIn('multi_step',stats.terms)
        self.assertIn('latent_slow',stats.terms)
        masks=dict(captured)
        self.assertEqual(set(masks),{'h4','h8'})
        self.assertFalse(masks['h4'][0,1:5].any())
        self.assertFalse(masks['h8'][0].any())
        self.assertTrue(masks['h4'][1].all())

    def test_multi_step_configuration_is_explicit_and_checkpointed(self):
        task,model,_=make_models()
        with self.assertRaises(ValueError):
            DynamicsModel(task.robot,physics_width=8,memory_width=8,latent_width=4,
                          hidden_dim=16,prediction_horizons=(1,8,4))
        with self.assertRaises(ValueError):
            DynamicsUpdater(model,multi_step_weight=.1)
        configured=DynamicsModel(task.robot,physics_width=8,memory_width=8,latent_width=4,
                                 hidden_dim=16,prediction_horizons=(1,4,8))
        self.assertEqual(configured.get_extra_state()['prediction_horizons'],(1,4,8))

    def test_invalid_transitions_rejected(self):
        task,m,_=make_models();m.eval()
        obs=clone_tree(task.read()); enc,_=m.step(obs,episode_start=task.episode_start,env_ids=task.env_ids)
        pred=m.predict_next(obs,torch.zeros(2,task.robot.dof),enc.latent)
        with self.assertRaises(ValueError): m.prediction_loss(pred,obs,valid_transition=torch.zeros(2,dtype=torch.bool))
    def test_memory_stale_after_load(self):
        task,m,_=make_models();m.eval();obs=task.read()
        _,memory=m.step(obs,episode_start=task.episode_start,env_ids=task.env_ids)
        m.load_state_dict(copy.deepcopy(m.state_dict()))
        with self.assertRaises(ValueError):m.step(obs,episode_start=task.episode_start,env_ids=task.env_ids,memory=memory)
    def test_wrong_same_width_robot_checkpoint_rejected(self):
        task,m,_=make_models()
        robot=RobotContract('other',task.robot.root_name,tuple(reversed(task.robot.joint_names)))
        other=DynamicsModel(robot,physics_width=8,memory_width=8,latent_width=4,hidden_dim=16)
        with self.assertRaises(ValueError):other.load_state_dict(m.state_dict())
    def test_q_sign_invariant_encoder_and_prediction(self):
        task,m,_=make_models();obs=clone_tree(task.read());other=clone_tree(obs)
        other.object_features[...,3:7]*=-1
        torch.testing.assert_close(m.encoder(obs).context,m.encoder(other).context)
        z=torch.zeros(2,4);a=torch.zeros(2,task.robot.dof)
        p=m.predict_next(obs,a,z);q=m.predict_next(other,a,z)
        torch.testing.assert_close(p.next_object_quat,q.next_object_quat)
    def test_masked_large_finite_garbage_does_not_break_backward(self):
        task,m,_=make_models();obs=clone_tree(task.read())
        obs.contact_mask[:]=False;obs.object_mask[:]=False
        clean=m.encoder(obs).context.detach()
        obs.contact_features[:]=torch.finfo(torch.float32).max/2
        obs.object_features[:]=torch.finfo(torch.float32).max/2
        out=m.encoder(obs).context
        torch.testing.assert_close(out,clean)
        out.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
    def test_empty_slot_dynamics_backward(self):
        task,m,_=make_models(contacts=0,objects=0);obs=task.read()
        m.eval();enc,_=m.step(obs,episode_start=task.episode_start,env_ids=task.env_ids)
        pred=m.predict_next(obs,torch.zeros(2,task.robot.dof),enc.latent)
        loss=m.prediction_loss(pred,obs,valid_transition=torch.ones(2,dtype=torch.bool))
        loss.total.backward();self.assertTrue(torch.isfinite(loss.total))
    def test_safe_unit_never_zero(self):
        x=torch.zeros(2,3); fallback=torch.tensor([[0.,0.,-1.],[0.,0.,0.]])
        y=_safe_unit(x,fallback)
        torch.testing.assert_close(y.norm(dim=-1),torch.ones(2))
    def test_sequence_and_step_predictions_agree(self):
        task,m,_=make_models();m.eval();memory=None;obslist=[];pres=[];pvalid=[];starts=[];latents=[];errors=[]
        for t in range(9):
            obs=clone_tree(task.read());start=task.episode_start
            pres.append(torch.zeros(2,9+2*task.robot.dof) if memory is None or memory.predicted_core is None else memory.predicted_core.clone())
            pvalid.append(torch.zeros(2,dtype=torch.bool) if memory is None or memory.prediction_valid is None else memory.prediction_valid.clone())
            enc,memory=m.step(obs,episode_start=start,env_ids=task.env_ids,memory=memory)
            action=torch.ones(2,task.robot.dof)*.03
            memory=m.remember_prediction(memory,m.predict_next(obs,action,enc.latent))
            obslist.append(obs);starts.append(start);latents.append(enc.latent);errors.append(enc.prediction_error)
            tr=task.step(action);done=tr.terminated|tr.truncated
            if done.any():task.reset(done.nonzero().flatten())
        seq=m.forward_sequence(stack_tree(obslist),episode_start=torch.stack(starts,1),
                 previous_predicted_core=torch.stack(pres,1),previous_prediction_valid=torch.stack(pvalid,1))
        torch.testing.assert_close(seq.latent,torch.stack(latents,1),atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(seq.prediction_error,torch.stack(errors,1))
    def test_predict_next_reuses_existing_physics_encoding(self):
        task,m,_=make_models();m.eval();obs=clone_tree(task.read())
        enc,_=m.step(obs,episode_start=task.episode_start,env_ids=task.env_ids)
        action=torch.zeros(2,task.robot.dof)
        expected=m.predict_next(obs,action,enc.latent)
        with patch.object(m.encoder,'forward',wraps=m.encoder.forward) as forward:
            actual=m.predict_next(obs,action,enc.latent,
                                  physics_encoding=enc.physics_encoding())
        self.assertEqual(forward.call_count,0)
        for name in ('next_core','next_contact_logits','next_object_pos',
                     'next_object_lin_vel','next_object_ang_vel','next_object_quat'):
            torch.testing.assert_close(getattr(actual,name),getattr(expected,name))
    def test_prediction_error_affects_latent_not_future(self):
        task,m,_=make_models();m.eval();obs=task.read()
        enc,mem=m.step(obs,episode_start=task.episode_start,env_ids=task.env_ids)
        pred=m.predict_next(obs,torch.zeros(2,task.robot.dof),enc.latent)
        a=m.remember_prediction(mem,pred);b=copy.copy(a);b.predicted_core=a.predicted_core+1
        one,_=m.step(obs,episode_start=torch.zeros(2,dtype=torch.bool),env_ids=task.env_ids,memory=a)
        two,_=m.step(obs,episode_start=torch.zeros(2,dtype=torch.bool),env_ids=task.env_ids,memory=b)
        self.assertGreater((one.latent-two.latent).abs().max().item(),1e-5)
    def test_reset_forgets_previous_prediction(self):
        task,m,_=make_models();m.eval();obs=task.read()
        enc,mem=m.step(obs,episode_start=task.episode_start,env_ids=task.env_ids)
        mem=m.remember_prediction(mem,m.predict_next(obs,torch.ones(2,task.robot.dof),enc.latent))
        actual,_=m.step(obs,episode_start=task.episode_start,env_ids=task.env_ids,memory=mem)
        torch.testing.assert_close(actual.latent,enc.latent)
        self.assertEqual(actual.prediction_error.count_nonzero().item(),0)


class PhysicsSequenceTests(unittest.TestCase):
    def config(self,**kw):
        d=dict(epochs=2,minibatch_envs=1,learning_rate=1e-4,target_kl=.03);d.update(kw)
        return PPOConfig(**d)
    def test_multimodal_partial_masks_all_policy_types(self):
        for kind in ('mlp','dense','moe'):
            with self.subTest(kind=kind):
                task,d,p=make_models(kind,multimodal=True)
                batch=PhysicsSequenceCollector(task,d,p).collect(4)
                stats=PhysicsSequencePPO(p,self.config()).update(batch)
                self.assertGreater(stats.optimizer_steps,0)
    def test_moe_router_context_directly_depends_on_dynamics(self):
        task,d,p=make_models('moe',endings=False)
        batch=PhysicsSequenceCollector(task,d,p).collect(3)
        valid=torch.ones(2,3,dtype=torch.bool)
        one=batch.inputs.dynamics()
        two=DynamicsEncoding(one.latent+1.0,one.context-1.0,one.hidden,
                             one.prediction_error)
        a=p._build_encoded(task.robot_id,batch.inputs.physics,one,batch.inputs.goal,valid)
        b=p._build_encoded(task.robot_id,batch.inputs.physics,two,batch.inputs.goal,valid)
        self.assertIsNotNone(a.router_context)
        self.assertGreater((a.router_context-b.router_context).abs().max().item(),1e-5)
        # Expert 输入和 Router 物理条件是两条明确通路。
        self.assertEqual(a.router_context.shape,a.state_token.shape)
    def test_motion_only_old_policy_agrees_all_robots(self):
        for rid in ('k1_22dof','g1_23dof','g1_29dof'):
            for kind in ('mlp','dense','moe'):
                with self.subTest(robot=rid,kind=kind):
                    task,d,p=make_models(kind,rid)
                    batch=PhysicsSequenceCollector(task,d,p).collect(8)
                    self.assertLess(PhysicsSequencePPO(p,self.config()).verify_old_policy(batch),1e-4)
    def test_mlp_sequence_forward_matches_frame_loop(self):
        task,d,p=make_models('mlp',endings=False)
        batch=PhysicsSequenceCollector(task,d,p).collect(4)
        whole=p(batch.inputs.physics,batch.inputs.dynamics(),batch.inputs.goal,
                privileged=batch.inputs.privileged)
        frames=[p(at_time(batch.inputs.physics,t),at_time(batch.inputs,t).dynamics(),
                  at_time(batch.inputs.goal,t),privileged=at_time(batch.inputs.privileged,t))
                for t in range(batch.horizon)]
        torch.testing.assert_close(whole.mean,torch.stack([x.mean for x in frames],1))
        torch.testing.assert_close(whole.std,torch.stack([x.std for x in frames],1))
        torch.testing.assert_close(whole.value,torch.stack([x.value for x in frames],1))
    def test_mlp_collector_reuses_next_state_value(self):
        task,d,p=make_models('mlp',endings=False)
        with patch.object(d,'step',wraps=d.step) as step:
            batch=PhysicsSequenceCollector(task,d,p).collect(4)
        # Four current-state encodings plus one final rollout bootstrap.
        self.assertEqual(step.call_count,5)
        torch.testing.assert_close(batch.next_value[:-1],batch.old_value[:,1:].transpose(0,1))
    def test_multibatch_prefix_replay_after_weight_update(self):
        for kind in ('dense','moe'):
            task,d,p=make_models(kind,multimodal=True,endings=False)
            c=PhysicsSequenceCollector(task,d,p);u=PhysicsSequencePPO(p,self.config())
            for i in range(4):
                with self.subTest(kind=kind,iteration=i):
                    batch=c.collect(3)
                    if i:self.assertIsNotNone(batch.prefix)
                    stats=u.update(batch);self.assertLess(stats.old_policy_error,1e-4)
    def test_window_more_layers_than_single_context(self):
        task,d,p=make_models('dense',endings=False)
        self.assertEqual(_history_size(p),4)
        c=PhysicsSequenceCollector(task,d,p);c.collect(9);batch=c.collect(3)
        self.assertEqual(batch.prefix.starts.shape[1],4)
        self.assertLess(PhysicsSequencePPO(p,self.config()).verify_old_policy(batch),1e-4)
    def test_final_value_not_reset_value(self):
        task,d,p=make_models('dense');c=PhysicsSequenceCollector(task,d,p);batch=c.collect(8)
        self.assertEqual(batch.next_value[4,0].item(),0)
        self.assertTrue(batch.terminated[4,0])
        self.assertTrue(batch.truncated[6,1])
        self.assertGreater(abs(batch.next_value[6,1].item()),0)
        self.assertTrue(batch.episode_start[0,5])
        self.assertTrue(batch.episode_start[1,7])
        self.assertGreater(batch.next_observation.body_obs[0,4,-task.robot.dof:].abs().sum().item(),0)
        self.assertEqual(batch.observation.body_obs[0,5,-task.robot.dof:].count_nonzero().item(),0)
    def test_collector_snapshots_not_mutated_by_next_reset(self):
        task,d,p=make_models();c=PhysicsSequenceCollector(task,d,p);batch=c.collect(4)
        original=clone_tree(batch.observation);task.reset()
        torch.testing.assert_close(batch.observation.body_obs,original.body_obs)
        task.object[:]=99
        torch.testing.assert_close(batch.observation.object_features,original.object_features)
    def test_ppo_does_not_update_dynamics(self):
        task,d,p=make_models('moe');batch=PhysicsSequenceCollector(task,d,p).collect(4)
        old=[x.clone() for x in d.parameters()]
        PhysicsSequencePPO(p,self.config()).update(batch)
        for a,b in zip(old,d.parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)
    def test_dynamics_update_changes_predictor_not_actor(self):
        task,d,p=make_models('dense');c=PhysicsSequenceCollector(task,d,p);batch=c.collect(4)
        old=[x.clone() for x in p.parameters()];before=d.state_predictor[-1].weight.clone()
        stats=DynamicsUpdater(d).update(batch)
        self.assertTrue(torch.isfinite(torch.tensor(stats.loss)))
        self.assertFalse(torch.equal(before,d.state_predictor[-1].weight))
        for a,b in zip(old,p.parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)
        with self.assertRaises(RuntimeError):c.collect(2)
        c.on_dynamics_updated();again=c.collect(2);self.assertTrue(again.episode_start[:,0].all())
    def test_same_rollout_cannot_be_consumed_twice(self):
        task,d,p=make_models();batch=PhysicsSequenceCollector(task,d,p).collect(3);u=PhysicsSequencePPO(p,self.config())
        u.update(batch)
        with self.assertRaises(ValueError):u.update(batch)
    def test_changed_policy_rejected_even_without_numeric_gate(self):
        task,d,p=make_models();batch=PhysicsSequenceCollector(task,d,p).collect(3)
        with torch.no_grad():next(p.parameters()).add_(.01)
        with self.assertRaises(RuntimeError):PhysicsSequencePPO(p,self.config(verify_old_policy=False)).update(batch)
    def test_raw_action_mismatch_gate_rejects(self):
        task,d,p=make_models();batch=PhysicsSequenceCollector(task,d,p).collect(3)
        batch.action+=5
        with self.assertRaises(RuntimeError):PhysicsSequencePPO(p,self.config()).update(batch)
        self.assertFalse(batch.consumed)
    def test_prefix_missing_rejected(self):
        task,d,p=make_models(endings=False);c=PhysicsSequenceCollector(task,d,p)
        c.collect(6);batch=c.collect(2);batch.prefix=None
        with self.assertRaises(ValueError):PhysicsSequencePPO(p,self.config()).update(batch)
    def test_balance_prefix_uses_only_current_rollout(self):
        task,d,p=make_models('moe',endings=False);c=PhysicsSequenceCollector(task,d,p)
        c.collect(6);batch=c.collect(2)
        self.assertGreater(
            PhysicsSequencePPO(p,self.config(balance_coef=.01)).update(batch).optimizer_steps,0
        )
    def test_no_prefix_balance_supported(self):
        task,d,p=make_models('moe');batch=PhysicsSequenceCollector(task,d,p).collect(4)
        self.assertGreater(PhysicsSequencePPO(p,self.config(balance_coef=.01)).update(batch).optimizer_steps,0)
    def test_checkpoint_roundtrip_with_optimizer(self):
        for kind in ('mlp','dense','moe'):
            task,d,p=make_models(kind);batch=PhysicsSequenceCollector(task,d,p).collect(3)
            u=PhysicsSequencePPO(p,self.config());u.update(batch)
            buf=io.BytesIO();torch.save(u.state_dict(),buf);buf.seek(0)
            _,_,other=make_models(kind);v=PhysicsSequencePPO(other,self.config())
            v.load_state_dict(torch.load(buf,weights_only=True))
            self.assertEqual(u.total_updates,v.total_updates)
            for a,b in zip(p.parameters(),other.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_dynamics_checkpoint_roundtrip(self):
        task,d,p=make_models();batch=PhysicsSequenceCollector(task,d,p).collect(3)
        u=DynamicsUpdater(d);u.update(batch);state=u.state_dict()
        _,e,_=make_models();v=DynamicsUpdater(e);v.load_state_dict(state)
        self.assertEqual(v.steps,u.steps)
        for a,b in zip(d.parameters(),e.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_dynamics_epochs_repeat_the_same_sequence(self):
        task,d,p=make_models();batch=PhysicsSequenceCollector(task,d,p).collect(3)
        updater=DynamicsUpdater(d,epochs=4);stats=updater.update(batch)
        self.assertEqual(updater.steps,4)
        self.assertEqual(stats.optimizer_steps,4)
    def test_value_does_not_consume_random_numbers_or_cache(self):
        task,d,p=make_models();d.eval();p.eval();f=task.read();enc,_=d.step(f,episode_start=task.episode_start,env_ids=task.env_ids)
        output,_,_,cache=p.step(task.robot_id,f,enc,task.goal(),env_ids=task.env_ids,episode_start=task.episode_start)
        keys=[c.keys.clone() for c in cache.layer_caches];rng=torch.get_rng_state().clone()
        v=p.value(task.robot_id,f,enc,task.goal())
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        torch.testing.assert_close(v,output.value)
        for a,b in zip(keys,cache.layer_caches):torch.testing.assert_close(a,b.keys,rtol=0,atol=0)
    def test_auto_reset_tasks_rejected(self):
        task,d,p=make_models();task.auto_reset=True
        with self.assertRaises(ValueError):PhysicsSequenceCollector(task,d,p)
    def test_ids_reorder_rejected(self):
        task,d,p=make_models();c=PhysicsSequenceCollector(task,d,p);c.collect(2);task.env_ids=(1,0)
        with self.assertRaises(ValueError):c.collect(2)
    def test_foreign_robot_dynamics_rejected(self):
        task,d,p=make_models();_,other,_=make_models(robot_id='g1_29dof')
        with self.assertRaises(ValueError):PhysicsSequenceCollector(task,other,p)
    def test_history_reset_equivalence(self):
        task,d,p=make_models('moe');c=PhysicsSequenceCollector(task,d,p);u=PhysicsSequencePPO(p,self.config())
        c.collect(3)
        for i in range(3):
            b=c.collect(8);self.assertLess(u.verify_old_policy(b),1e-4)
    def test_nan_reward_rejected(self):
        task,d,p=make_models();old=task.step
        def bad(a):
            t=old(a);t.reward[0]=float('nan');return t
        task.step=bad
        with self.assertRaises(FloatingPointError):PhysicsSequenceCollector(task,d,p).collect(2)
    def test_optional_goal_structure_must_stay_consistent(self):
        task,d,p=make_models();old=task.goal;counter=[0]
        def changing():
            counter[0]+=1
            goal=old()
            if counter[0]>3: goal.object_features=torch.zeros(2,1,6);goal.object_mask=torch.ones(2,1,dtype=torch.bool)
            return goal
        task.goal=changing
        with self.assertRaises(ValueError):PhysicsSequenceCollector(task,d,p).collect(3)
    def test_complete_two_training_cycles(self):
        for kind in ('mlp','dense','moe'):
            task,d,p=make_models(kind,multimodal=True);c=PhysicsSequenceCollector(task,d,p)
            u=PhysicsSequencePPO(p,self.config());du=DynamicsUpdater(d)
            for _ in range(2):
                batch=c.collect(8);u.update(batch);du.update(batch);c.on_dynamics_updated()
            self.assertEqual(du.steps,8)

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main(verbosity=2)

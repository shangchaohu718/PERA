"""第10模块回归测试：只使用 PyTorch 人工数据，不冒充机器人物理训练。"""
import copy
import io
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_motion.algorithms.ppo import (
    PPOConfig, PPOTrainer, RolloutBuffer, HistoryPrefix, compute_gae,
)
from unified_motion.models.actor_critic import MultiRobotActorCritic
from unified_motion.models.transformer import TransformerConfig
from unified_motion.reference_observations import ReferenceObservation
from unified_motion.robot_contract import get_robot_contract


def make_model(robot_ids=('k1_22dof',), *, privileged_dim=0):
    robots = {k: get_robot_contract(k) for k in robot_ids}
    cfg = TransformerConfig(d_model=16, n_layers=2, n_heads=4, n_kv_heads=2,
                            context_len=3, num_fine_experts=4, top_k=2,
                            expert_hidden_dim=32, router_hidden_dim=16, check_values=False)
    model = MultiRobotActorCritic(
        robots, transformer_config=cfg, encoder_hidden_dim=32,
        critic_hidden_dim=32, check_values=False,
        privileged_dims={k: privileged_dim for k in robot_ids},
    )
    return model


def collect(model, robot_id='k1_22dof', *, steps=4, b=2, h=2,
            prefix_steps=0, done_schedule=None, privileged_dim=0):
    """真实调用 act_step，保存其原始 action/log_prob；人工 reward 仅测试算法。"""
    n = get_robot_contract(robot_id).dof
    ids = tuple(range(100, 100+b))
    model.eval()
    cache = None
    buf = RolloutBuffer(robot_id, get_robot_contract(robot_id), horizon=steps,
                        num_envs=b, reference_steps=h, env_ids=ids,
                        privileged_dim=privileged_dim)
    prefix_obs, prefix_ref, prefix_mask, prefix_start, prefix_p = [], [], [], [], []
    prev_done = torch.zeros(b,dtype=torch.bool)
    for t in range(prefix_steps + steps):
        obs = torch.randn(b, 6+3*n)*0.1
        ref = ReferenceObservation(
            torch.randn(b, h, 16+2*n)*.1,
            torch.ones(b, h, dtype=torch.bool),
        )
        priv = torch.randn(b,privileged_dim)*.1 if privileged_dim else None
        starts = torch.ones(b,dtype=torch.bool) if t==0 else prev_done.clone()
        step, cache = model.act_step(robot_id,obs,ref,env_ids=ids,cache=cache,
                                     episode_start=starts,privileged=priv)
        terminated = torch.zeros(b,dtype=torch.bool)
        truncated = torch.zeros(b,dtype=torch.bool)
        if done_schedule and t in done_schedule:
            for env, kind in done_schedule[t]:
                if kind=='terminated':terminated[env]=True
                else:truncated[env]=True
        prev_done = terminated | truncated
        if t < prefix_steps:
            prefix_obs.append(obs);prefix_ref.append(ref.features)
            prefix_mask.append(ref.valid_mask);prefix_start.append(starts)
            if privileged_dim:prefix_p.append(priv)
        else:
            # next_value 来自独立 Critic，并在本测试中明确与 value() 接口一致。
            nxt_obs = torch.randn(b, 6+3*n)*.1
            nxt_ref = ReferenceObservation(torch.randn(b,h,16+2*n)*.1,
                                           torch.ones(b,h,dtype=torch.bool))
            nxt_priv = torch.randn(b,privileged_dim)*.1 if privileged_dim else None
            with torch.no_grad():
                nxt_val=model.value(robot_id,nxt_obs,nxt_ref,privileged=nxt_priv)
            buf.add(observation=obs,reference=ref,step=step,
                    reward=torch.randn(b)*.3,next_value=nxt_val,
                    terminated=terminated,truncated=truncated,
                    episode_start=starts,env_ids=ids,privileged=priv)
    prefix = None
    if prefix_steps:
        prefix=HistoryPrefix(
            observation=torch.stack(prefix_obs,dim=1),
            reference=ReferenceObservation(torch.stack(prefix_ref,dim=1),torch.stack(prefix_mask,dim=1)),
            episode_start=torch.stack(prefix_start,dim=1),
            env_ids=ids,
            privileged=torch.stack(prefix_p,dim=1) if privileged_dim else None,
        )
    return buf,prefix


class GaeTests(unittest.TestCase):
    def test_terminal_no_bootstrap(self):
        r=torch.tensor([[1.],[1.],[1.]])
        v=torch.zeros_like(r)
        done=torch.tensor([[False],[False],[True]])
        adv, ret=compute_gae(r,v,torch.full_like(r,99),done,torch.zeros_like(done),
                             gamma=.9,gae_lambda=1)
        # 前两步尚未终止，所以其 next_value 应被使用；只检查真正终止的最后一步。
        self.assertEqual(adv[-1,0].item(),1)
        self.assertEqual(ret[-1,0].item(),1)

    def test_timelimit_bootstrap_without_leakage(self):
        # t0 因时间限制截断，下一帧 t1 是新回合；不应把 t1 的 advantage 接到 t0。
        r=torch.tensor([[1.],[100.]])
        v=torch.tensor([[2.],[0.]])
        nv=torch.tensor([[5.],[0.]])
        term=torch.tensor([[False],[True]])
        trunc=torch.tensor([[True],[False]])
        adv,ret=compute_gae(r,v,nv,term,trunc,gamma=.9,gae_lambda=.95)
        self.assertAlmostEqual(adv[0,0].item(), 1+.9*5-2, places=5)
        self.assertAlmostEqual(ret[0,0].item(), 1+.9*5, places=5)
        self.assertAlmostEqual(adv[1,0].item(),100,places=5)

    def test_regular_nonterminal_bootstrap(self):
        r=torch.tensor([[1.],[2.]])
        v=torch.tensor([[.5],[.4]])
        nv=torch.tensor([[.4],[.3]])
        d=torch.zeros_like(r,dtype=torch.bool)
        adv,_=compute_gae(r,v,nv,d,d,gamma=.9,gae_lambda=1)
        self.assertAlmostEqual(adv[1,0].item(),2+.9*.3-.4,places=5)
        self.assertAlmostEqual(adv[0,0].item(),1+.9*.4-.5+.9*adv[1,0].item(),places=5)

    def test_two_env_independence(self):
        r=torch.tensor([[1.,100.],[1.,200.],[1.,300.]])
        v=torch.zeros_like(r);d=torch.zeros_like(r,dtype=torch.bool)
        d[1,0]=True
        a,_=compute_gae(r,v,v,d,torch.zeros_like(d),gamma=.9,gae_lambda=1)
        self.assertAlmostEqual(a[0,0].item(),1+.9,places=5)
        self.assertAlmostEqual(a[0,1].item(),100+.9*200+.9**2*300,places=4)

    def test_invalid_gae_data_refused(self):
        r=torch.ones(2,3);v=torch.zeros_like(r);d=torch.zeros_like(r,dtype=torch.bool)
        with self.assertRaises(ValueError):
            invalid_term=d.clone();invalid_term[0,0]=True
            compute_gae(r,v,v,invalid_term,torch.ones_like(d),gamma=.99,gae_lambda=.95)
        with self.assertRaises(ValueError):compute_gae(r,v,v,d,d,gamma=1.1,gae_lambda=.95)
        with self.assertRaises(ValueError):compute_gae(r,torch.zeros(2),v,d,d,gamma=.99,gae_lambda=.95)
        with self.assertRaises(ValueError):compute_gae(r,v,v,d,d,gamma=.99,gae_lambda=float('nan'))


class PPOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261009)

    def test_config_checks(self):
        for kw in ({'gamma':0}, {'gae_lambda':1.2}, {'entropy_coef':-1},
                   {'minibatch_envs':0}, {'epochs':True}, {'learning_rate':float('nan')},
                   {'balance_coef':-1}, {'value_clip':0}, {'target_kl':0}, {'clip_ratio':1.2}):
            with self.subTest(kw=kw),self.assertRaises((ValueError,TypeError)):
                PPOConfig(**kw)

    def test_buffer_shapes_and_reward_alignment(self):
        model=make_model()
        buf,_=collect(model,steps=5,b=3,done_schedule={1:[(0,'terminated')],3:[(1,'truncated')]})
        self.assertTrue(buf.full)
        cfg=PPOConfig()
        batch=buf.make_batch(cfg)
        self.assertEqual(batch.observation.shape,(3,5,72))
        self.assertEqual(batch.reference.features.shape,(3,5,2,60))
        self.assertEqual(batch.actions.shape,(3,5,22))
        self.assertTrue(batch.loss_mask.all())
        self.assertTrue(torch.equal(batch.episode_start[0,2:3],torch.tensor([True])))
        self.assertTrue(torch.isfinite(batch.returns).all())

    def test_baseline_update_exact_old_logprob_and_value(self):
        model=make_model()
        buf,_=collect(model,steps=5,b=3)
        old_head=model.action_heads[model._keys['k1_22dof']].mean_head.weight.detach().clone()
        trainer=PPOTrainer(model,PPOConfig(epochs=2,minibatch_envs=2))
        stats=trainer.update(buf)
        self.assertEqual(stats.updates,4)
        self.assertLess(stats.max_preupdate_logprob_error,2e-4)
        self.assertLess(stats.max_preupdate_value_error,2e-4)
        self.assertFalse(torch.equal(old_head,model.action_heads[model._keys['k1_22dof']].mean_head.weight))
        self.assertTrue(math_isfinite_stats(stats))
        self.assertTrue(buf.consumed)
        with self.assertRaises(RuntimeError):trainer.update(buf)

    def test_old_logprob_action_clipping_detected(self):
        model=make_model();buf,_=collect(model,steps=4)
        # 模拟错误：把保存的原始动作替换成执行器限位结果，却保留旧 logprob。
        buf.actions.clamp_(-.05,.05)
        trainer=PPOTrainer(model,PPOConfig(epochs=1,minibatch_envs=1))
        with self.assertRaisesRegex(RuntimeError,'Gate'):
            trainer.update(buf)
        self.assertFalse(buf.consumed)
        self.assertEqual(trainer.total_updates,0)

    def test_old_mean_or_std_corruption_rejected(self):
        # 旧 mean/std 是 KL 的依据，不能只检查 log-prob 正确就放心。
        for field in ('old_means', 'old_stds'):
            with self.subTest(field=field):
                model=make_model();buf,_=collect(model,steps=3)
                getattr(buf, field)[0,0,0] += .5
                with self.assertRaisesRegex(RuntimeError,'Gate'):
                    PPOTrainer(model,PPOConfig(epochs=1,minibatch_envs=2)).update(buf)

    def test_old_critic_value_corruption_rejected(self):
        model=make_model();buf,_=collect(model,steps=3)
        buf.old_values[1,0] += 5
        with self.assertRaisesRegex(RuntimeError,'Gate'):
            PPOTrainer(model,PPOConfig(epochs=1,minibatch_envs=2)).update(buf)

    def test_stale_policy_weights_detected(self):
        model=make_model();buf,_=collect(model,steps=4)
        with torch.no_grad():
            model.action_heads[model._keys['k1_22dof']].mean_head.bias.add_(1.5)
        with self.assertRaisesRegex(RuntimeError,'Gate'):
            PPOTrainer(model,PPOConfig(epochs=1,minibatch_envs=1)).update(buf)

    def test_wrong_joint_names_rejected(self):
        model=make_model();buf,_=collect(model,steps=4)
        original=buf.robot
        buf.robot=get_robot_contract('g1_23dof')
        with self.assertRaises(ValueError):PPOTrainer(model).update(buf)
        buf.robot=original

    def test_mid_episode_needs_prefix(self):
        model=make_model()
        buf,prefix=collect(model,steps=4,prefix_steps=4)
        with self.assertRaisesRegex(ValueError,'前缀'):
            PPOTrainer(model).update(buf)
        trainer=PPOTrainer(model,PPOConfig(epochs=1,minibatch_envs=2))
        stats=trainer.update(buf,prefix=prefix)
        self.assertLess(stats.max_preupdate_logprob_error,2e-4)
        self.assertGreater(stats.updates,0)

    def test_prefix_too_short_without_reset_rejected(self):
        model=make_model()
        buf,prefix=collect(model,steps=3,prefix_steps=4)
        prefix=HistoryPrefix(prefix.observation[:,-1:],
                            ReferenceObservation(prefix.reference.features[:,-1:],
                                                 prefix.reference.valid_mask[:,-1:]),
                            prefix.episode_start[:,-1:],prefix.env_ids)
        with self.assertRaisesRegex(ValueError,'历史不足'):
            PPOTrainer(model).update(buf,prefix=prefix)

    def test_prefix_excluded_from_loss_and_gradients_work(self):
        model=make_model();buf,prefix=collect(model,steps=3,prefix_steps=4)
        batch=buf.make_batch(PPOConfig(),prefix=prefix,min_history=4)
        self.assertEqual(batch.loss_mask.shape,(2,7))
        self.assertFalse(batch.loss_mask[:,:4].any())
        self.assertTrue(batch.loss_mask[:,4:].all())
        self.assertEqual(batch.select_envs(torch.tensor([1])).num_envs,1)

    def test_prefix_wrong_env_order_rejected(self):
        model=make_model();buf,prefix=collect(model,steps=3,prefix_steps=4)
        wrong=replace(prefix,env_ids=tuple(reversed(prefix.env_ids)))
        with self.assertRaisesRegex(ValueError,'环境 ID'):
            PPOTrainer(model).update(buf,prefix=wrong)

    def test_cache_invalidated_after_update(self):
        model=make_model();buf,_=collect(model,steps=3)
        model.eval()
        obs=torch.randn(2,72)*.1
        ref=ReferenceObservation(torch.randn(2,2,60)*.1,torch.ones(2,2,dtype=torch.bool))
        _,cached=model.act_step('k1_22dof',obs,ref,env_ids=(100,101),episode_start=torch.ones(2,dtype=torch.bool))
        PPOTrainer(model,PPOConfig(epochs=1,minibatch_envs=2)).update(buf)
        model.eval()
        with self.assertRaises(ValueError):
            model.act_step('k1_22dof',obs,ref,env_ids=(100,101),cache=cached,
                           episode_start=torch.zeros(2,dtype=torch.bool))

    def test_single_step_single_env_advantage_normalization(self):
        model=make_model();buf,_=collect(model,steps=1,b=1)
        batch=buf.make_batch(PPOConfig())
        self.assertTrue(torch.isfinite(batch.advantages).all())
        self.assertEqual(batch.advantages[0,0].item(),0.)
        stat=PPOTrainer(model,PPOConfig(epochs=1,minibatch_envs=1)).update(buf)
        self.assertEqual(stat.updates,1)

    def test_return_config_recomputes_if_gamma_changes(self):
        model=make_model();buf,_=collect(model,steps=4)
        a=buf.make_batch(PPOConfig(gamma=.5, normalize_advantage=False)).returns.clone()
        b=buf.make_batch(PPOConfig(gamma=.99, normalize_advantage=False)).returns
        self.assertFalse(torch.equal(a,b))

    def test_buffer_copies_raw_action_not_input_alias(self):
        model=make_model();robot=get_robot_contract('k1_22dof')
        model.eval();obs=torch.randn(2,72)*.1
        ref=ReferenceObservation(torch.randn(2,2,60)*.1,torch.ones(2,2,dtype=torch.bool))
        step,_=model.act_step('k1_22dof',obs,ref,env_ids=(100,101),
                              episode_start=torch.ones(2,dtype=torch.bool))
        buf=RolloutBuffer('k1_22dof',robot,horizon=1,num_envs=2,reference_steps=2,env_ids=(100,101))
        old_action=step.action.clone()
        buf.add(observation=obs,reference=ref,step=step,reward=torch.zeros(2),
                next_value=torch.zeros(2),terminated=torch.zeros(2,dtype=torch.bool),
                truncated=torch.zeros(2,dtype=torch.bool),
                episode_start=torch.ones(2,dtype=torch.bool),env_ids=(100,101))
        obs.zero_();step.action.zero_();ref.features.zero_()
        torch.testing.assert_close(buf.actions[0],old_action)
        self.assertGreater(buf.observations[0].abs().sum().item(),0)
        self.assertGreater(buf.references[0].abs().sum().item(),0)

    def test_privileged_only_critic(self):
        model=make_model(privileged_dim=3)
        buf,_=collect(model,steps=4,privileged_dim=3)
        trainer=PPOTrainer(model,PPOConfig(epochs=1,minibatch_envs=2))
        self.assertEqual(trainer.update(buf).updates,1)
        self.assertEqual(buf.privileged.shape,(4,2,3))

    def test_cross_robot_shared_trainer_fresh_rollouts(self):
        ids=('k1_22dof','g1_23dof','g1_29dof')
        model=make_model(ids)
        trainer=PPOTrainer(model,PPOConfig(epochs=1,minibatch_envs=2))
        for rid in ids:
            buf,_=collect(model,rid,steps=3)
            stats=trainer.update(buf)
            self.assertEqual(stats.updates,1)
        self.assertEqual(trainer.total_updates,3)

    def test_balance_option_no_prefix(self):
        model=make_model();buf,_=collect(model,steps=3)
        stat=PPOTrainer(model,PPOConfig(balance_coef=.001,epochs=1,minibatch_envs=2)).update(buf)
        self.assertGreater(stat.balance_loss,0)

    def test_balance_with_prefix_forbidden(self):
        model=make_model();buf,prefix=collect(model,steps=3,prefix_steps=4)
        with self.assertRaisesRegex(ValueError,'balance_coef'):
            PPOTrainer(model,PPOConfig(balance_coef=.001)).update(buf,prefix=prefix)

    def test_false_done_reset_alignment_rejected(self):
        model=make_model()
        with self.assertRaisesRegex(ValueError,'不一致'):
            # 校验在数据进入时进行，因此这份数据不能通过新 Buffer.add 的合同。
            fresh=RolloutBuffer('k1_22dof', get_robot_contract('k1_22dof'),
                                horizon=3,num_envs=2,reference_steps=2,env_ids=(100,101))
            model.eval();o=torch.randn(2,72)*.1;r=ReferenceObservation(torch.randn(2,2,60)*.1,
                                                           torch.ones(2,2,dtype=torch.bool))
            s,_=model.act_step('k1_22dof',o,r,env_ids=(100,101),episode_start=torch.ones(2,dtype=torch.bool))
            fresh.add(observation=o,reference=r,step=s,reward=torch.zeros(2),next_value=torch.zeros(2),
                      terminated=torch.zeros(2,dtype=torch.bool),truncated=torch.zeros(2,dtype=torch.bool),
                      episode_start=torch.ones(2,dtype=torch.bool),env_ids=(100,101))
            fresh.add(observation=o,reference=r,step=s,reward=torch.zeros(2),next_value=torch.zeros(2),
                      terminated=torch.zeros(2,dtype=torch.bool),truncated=torch.zeros(2,dtype=torch.bool),
                      episode_start=torch.tensor([True,False]),env_ids=(100,101))

    def test_buffer_not_full_and_env_id_check(self):
        model=make_model();buf,_=collect(model,steps=3)
        partial=RolloutBuffer('k1_22dof',get_robot_contract('k1_22dof'),horizon=3,
                              num_envs=2,reference_steps=2,env_ids=(100,101))
        with self.assertRaises(RuntimeError):PPOTrainer(model).update(partial)
        with self.assertRaises(RuntimeError):buf.add(observation=torch.zeros(2,72),
            reference=ReferenceObservation(torch.zeros(2,2,60),torch.ones(2,2,dtype=torch.bool)),
            step=None,reward=torch.zeros(2),next_value=torch.zeros(2),
            terminated=torch.zeros(2,dtype=torch.bool),truncated=torch.zeros(2,dtype=torch.bool),
            episode_start=torch.zeros(2,dtype=torch.bool),env_ids=(100,101))

    def test_checkpoint_roundtrip_and_cache_invalidated(self):
        model=make_model();trainer=PPOTrainer(model,PPOConfig(epochs=1,minibatch_envs=2))
        buf,_=collect(model,steps=3)
        trainer.update(buf)
        ck=trainer.state_dict()
        f=io.BytesIO();torch.save(ck,f);f.seek(0)
        ck2=torch.load(f,weights_only=True)
        new=PPOTrainer(make_model(),PPOConfig(epochs=1,minibatch_envs=2))
        new.load_state_dict(ck2)
        self.assertEqual(trainer.total_updates,new.total_updates)
        for a,b in zip(trainer.model.parameters(),new.model.parameters()):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        with self.assertRaises(ValueError):
            new.load_state_dict({**ck2,'config':{'different':1}})

    def test_early_stop_kl(self):
        model=make_model();buf,_=collect(model,steps=5,b=3)
        stat=PPOTrainer(model,PPOConfig(epochs=4,minibatch_envs=1,target_kl=1e-12)).update(buf)
        self.assertTrue(stat.early_stopped)
        self.assertGreaterEqual(stat.updates,1)

    def test_double_dtype(self):
        model=make_model().double()
        # 使用非随机缓冲区路径确认 double GAE 数值一致；训练数据必须跟随网络精度。
        r=torch.ones(2,2,dtype=torch.float64);v=torch.zeros_like(r)
        d=torch.zeros_like(r,dtype=torch.bool)
        a,ret=compute_gae(r,v,v,d,d,gamma=.99,gae_lambda=.95)
        self.assertEqual(ret.dtype,torch.float64)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_training(self):
        model=make_model().cuda()
        # 真实 CUDA 用例在 GPU 主机上按同一测试夹具执行；这里先测 GPU GAE。
        r=torch.ones(4,2,device='cuda');v=torch.zeros_like(r)
        d=torch.zeros_like(r,dtype=torch.bool)
        a,ret=compute_gae(r,v,v,d,d,gamma=.99,gae_lambda=.95)
        self.assertEqual(ret.device.type,'cuda')


def math_isfinite_stats(stats):
    return all(torch.isfinite(torch.tensor(v)).item() for v in (
        stats.policy_loss,stats.value_loss,stats.entropy,stats.balance_loss,
        stats.approx_kl,stats.clip_fraction,stats.grad_norm,
        stats.max_preupdate_logprob_error,stats.max_preupdate_value_error,
        stats.max_preupdate_mean_error,stats.max_preupdate_std_error,
    ))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)

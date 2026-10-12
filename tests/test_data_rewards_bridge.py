"""动作、奖励与 mjlab 接口回归。桥接测试用明确标记的假环境，不冒充物理引擎。"""
from dataclasses import replace, fields
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch

from unified_motion.robot_contract import get_robot_contract, RobotContract
from unified_motion.data.motion_sampler import MotionClip, MotionLibrary, MotionSampler
from unified_motion.reference_observations import ReferenceObservationBuilder
from unified_motion.rewards.motion_tracking import (
    MotionTrackingReward, MotionTrackingRewardConfig, TrackingState, BodyTrackingState,
    FootSiteState, FootSiteTarget, FootPhase,
)
from unified_motion.envs.mjlab_k1 import K1MJLabBridge, read_k1_joint_specs
from unified_motion.envs.mjlab_physics import (
    K1PhysicsBridge, ObjectObservationSpec, CONTACT_SENSOR_NAME, FOOT_NAMES,
    PrivilegedFootContactTruth, _contact_primary_names,
)
from unified_motion.algorithms.sequence_training import (
    MotionTrackingTask, PhysicsSequenceCollector, PhysicsSequencePPO, DynamicsUpdater,
)
from unified_motion.algorithms.ppo import PPOConfig
from unified_motion.models.physics_dynamics import DynamicsModel, PhysicsMLPActorCritic
from unified_motion.models.physics_policy_v2 import PhysicsConditionedActorCritic, PhysicsPolicyV2Config
from test_mjlab_k1 import FakeEnv, create_fixture_xml


ROBOT = get_robot_contract('k1_22dof')


def clip_fixture(robot_id='k1_22dof', name='walk', frames=11, fps=10.0, with_bodies=False):
    """已知线性轨迹用于检查插值/时间/坐标，不表示任何可实现的机器人动作。"""
    r = get_robot_contract(robot_id)
    t = torch.arange(frames, dtype=torch.float32) / fps
    quat = torch.zeros(frames,4); quat[:,0] = 1
    pos = torch.stack((t, 2*t, .6+0*t), -1)
    q = t[:,None].expand(-1,r.dof).clone()
    clip = MotionClip(name,robot_id,r.root_name,tuple(reversed(r.joint_names)),fps,
        q.flip(-1),torch.ones_like(q),pos,quat,
        torch.tensor([1.,2.,0.]).expand(frames,3).clone(),torch.zeros(frames,3))
    if with_bodies:
        clip = replace(clip, body_names=(r.root_name,'foot'),
            body_pos_w=torch.stack((pos,pos+torch.tensor([.2,0.,-.6])),1),
            body_quat_wxyz=quat[:,None].expand(-1,2,-1).clone(),
            body_lin_vel_w=clip.root_lin_vel_w[:,None].expand(-1,2,-1).clone(),
            body_ang_vel_w=torch.zeros(frames,2,3))
    return clip


def sampler_fixture(robot_id='k1_22dof', with_bodies=False, b=2):
    clip = clip_fixture(robot_id,with_bodies=with_bodies)
    lib = MotionLibrary(get_robot_contract(robot_id),[clip])
    sampler = MotionSampler(lib,num_envs=b,step_dt=.02,future_offsets_s=(0.,.04,.12))
    sampler.reset(clip_ids=[0]*b)
    return lib,sampler


def reward_fixture(robot_id='k1_22dof'):
    lib,sampler = sampler_fixture(robot_id)
    w = sampler.reference_window()
    s = TrackingState(w.joint_names,w.joint_pos[:,0].clone(),w.joint_vel[:,0].clone(),
        w.root_pos_w[:,0].clone(),w.root_quat_wxyz[:,0].clone(),
        w.root_lin_vel_w[:,0].clone(),w.root_ang_vel_w[:,0].clone())
    return lib.robot,s,w


class MotionDataTests(unittest.TestCase):
    def test_resample_30_50_60_and_boundaries(self):
        for fps in (30.,50.,60.):
            c=clip_fixture(fps=fps,frames=int(fps)+1)
            lib=MotionLibrary(ROBOT,[c]); times=torch.tensor([[0.,.015,.34,1.,1.02]])
            data,valid=lib._interpolate(torch.tensor([0]),times)
            torch.testing.assert_close(data['root_pos_w'][0,:,0],times[0].clamp_max(1),atol=1e-6,rtol=1e-5)
            self.assertEqual(valid.tolist(),[[True,True,True,True,False]])

    def test_joint_order_and_distinct_clips_not_mixed(self):
        c=clip_fixture();c=replace(c,joint_pos=torch.arange(22.).repeat(11,1).flip(-1))
        d=replace(c,name='other',joint_pos=c.joint_pos+100)
        lib=MotionLibrary(ROBOT,[c,d])
        out,_=lib._interpolate(torch.tensor([0,1]),torch.tensor([[1.],[0.]]))
        torch.testing.assert_close(out['joint_pos'][:,0],torch.stack((torch.arange(22.),100+torch.arange(22.))))

    def test_slerp_shortest_path_sign_and_unit(self):
        c=clip_fixture(frames=2,fps=1.)
        q=torch.tensor([[1.,0.,0.,0.],[0.,0.,0.,1.]])
        a=MotionLibrary(ROBOT,[replace(c,root_quat_wxyz=q)])
        b=MotionLibrary(ROBOT,[replace(c,root_quat_wxyz=-q)])
        x,_=a._interpolate(torch.tensor([0]),torch.tensor([[.5]]))
        y,_=b._interpolate(torch.tensor([0]),torch.tensor([[.5]]))
        torch.testing.assert_close(x['root_quat_wxyz'].abs(),y['root_quat_wxyz'].abs())
        torch.testing.assert_close(x['root_quat_wxyz'][0,0].abs(),torch.tensor([math.sqrt(.5),0.,0.,math.sqrt(.5)]))

    def test_reset_alignment_rotates_velocity_but_preserves_height(self):
        lib,s=sampler_fixture(with_bodies=True,b=1)
        q=torch.tensor([[math.sqrt(.5),0,0,math.sqrt(.5)]])
        s.reset(clip_ids=[0],current_root_pos_w=torch.tensor([[5.,6.,9.]]),current_root_quat_wxyz=q)
        w=s.reference_window();torch.testing.assert_close(w.root_pos_w[0,0],torch.tensor([5.,6.,.6]))
        torch.testing.assert_close(w.root_lin_vel_w[0,0],torch.tensor([-2.,1.,0.]),atol=1e-6,rtol=1e-5)
        b=s.body_reference_window();torch.testing.assert_close(b.pos_w[0,0,1],torch.tensor([5.,6.2,0.]),atol=1e-6,rtol=1e-5)
        s.advance(10);self.assertGreater(s.reference_window().root_pos_w[0,0,1].item(),6.)

    def test_partial_reset_keeps_other_clocks(self):
        _,s=sampler_fixture();s.advance(8);s.reset([0],clip_ids=[0])
        torch.testing.assert_close(s.current_times_s(),torch.tensor([0.,.16]))

    def test_terminal_mask_not_looping(self):
        _,s=sampler_fixture();s.advance(50)
        self.assertTrue(s.finished().all());w=s.reference_window()
        self.assertEqual(w.valid_mask.tolist(),[[True,False,False]]*2)
        self.assertEqual(w.joint_pos[:,1:].count_nonzero().item(),0)
        s.advance(30);torch.testing.assert_close(s.current_times_s(),torch.ones(2))

    def test_finished_reserves_future_reference_lookahead(self):
        _,s=sampler_fixture()
        s.advance(44)  # 0.88 s in a 1.0 s clip.
        self.assertFalse(s.finished().any())
        self.assertTrue(s.finished(min_remaining_s=.12).all())
        with self.assertRaises(ValueError):s.finished(min_remaining_s=-.1)

    def test_unready_unknown_and_duplicate_envs(self):
        lib,_=sampler_fixture();s=MotionSampler(lib,num_envs=2,step_dt=.02,future_offsets_s=(0.,.1))
        with self.assertRaises(RuntimeError):s.reference_window()
        for ids in ([0,0],[2],[-1]):
            with self.assertRaises(ValueError):s.reset(ids)

    def test_random_reset_chooses_eligible_clips(self):
        lib=MotionLibrary(ROBOT,[clip_fixture(name='short',frames=2),clip_fixture(name='long')])
        s=MotionSampler(lib,num_envs=100,step_dt=.02,future_offsets_s=(0.,))
        s.reset(random_start=True,min_remaining_s=.5)
        self.assertTrue((s.clip_ids==1).all());self.assertTrue((s.current_times_s()<=.5).all())
        with self.assertRaises(ValueError):s.reset(min_remaining_s=2.)
        with self.assertRaises(ValueError):s.reset(start_times_s=[0.]*100,random_start=True)

    def test_failed_reset_is_atomic(self):
        _,s=sampler_fixture();s.advance(3);before=s.current_times_s().clone()
        with self.assertRaises(ValueError):s.reset(start_times_s=[5.,0.])
        torch.testing.assert_close(s.current_times_s(),before)

    def test_npz_roundtrip_all_robots_and_bodies(self):
        for rid in ('k1_22dof','g1_23dof','g1_29dof'):
            with tempfile.TemporaryDirectory() as d:
                c=clip_fixture(rid,with_bodies=True);p=Path(d)/'clip.npz';c.save_project_npz(p)
                read=MotionClip.from_project_npz(p)
                lib=MotionLibrary.from_project_folder(d,robot_id=rid)
                torch.testing.assert_close(read.body_pos_w,c.body_pos_w)
                self.assertEqual(lib.robot.dof,c.joint_pos.shape[-1])

    def test_npz_reject_bad_metadata_and_pickle(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'clip.npz';clip_fixture().save_project_npz(p)
            with np.load(p,allow_pickle=False) as z: values=dict(z)
            for key,value in (('quaternion_order','xyzw'),('units','degrees'),('schema_version',2)):
                bad=values.copy();bad[key]=value;np.savez(p,**bad)
                with self.assertRaises(ValueError):MotionClip.from_project_npz(p)
            bad=values.copy();bad['joint_pos']=np.array([object()],dtype=object);np.savez(p,**bad)
            with self.assertRaises(ValueError):MotionClip.from_project_npz(p)

    def test_native_npz_requires_exact_names_and_fps(self):
        c=clip_fixture(with_bodies=True)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'native.npz'
            np.savez(p,joint_pos=c.joint_pos,joint_vel=c.joint_vel,body_pos_w=c.body_pos_w,
                body_quat_w=c.body_quat_wxyz,body_lin_vel_w=c.body_lin_vel_w,
                body_ang_vel_w=c.body_ang_vel_w,fps=c.fps,joint_names=c.joint_names)
            kwargs=dict(robot_id=c.robot_id,joint_names=c.joint_names,body_names=c.body_names,fps=c.fps)
            x=MotionClip.from_mjlab_npz(p,**kwargs);torch.testing.assert_close(x.root_pos_w,c.root_pos_w)
            with self.assertRaises(ValueError):MotionClip.from_mjlab_npz(p,**{**kwargs,'fps':50})
            with self.assertRaises(ValueError):MotionClip.from_mjlab_npz(p,**{**kwargs,'joint_names':ROBOT.joint_names})

    def test_named_holomotion_npz_loads_without_rewriting_source(self):
        c=clip_fixture(with_bodies=True)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'holomotion.npz'
            values=dict(joint_pos=c.joint_pos,joint_vel=c.joint_vel,body_pos_w=c.body_pos_w,
                body_quat_w=c.body_quat_wxyz,body_lin_vel_w=c.body_lin_vel_w,
                body_ang_vel_w=c.body_ang_vel_w,fps=c.fps,
                joint_names=np.asarray(c.joint_names),body_names=np.asarray(c.body_names),
                metadata=json.dumps({'quaternion_order':'wxyz','motionloader_schema':'test'}))
            np.savez(p,**values)
            clip=MotionClip.from_named_mjlab_npz(p,robot_id='k1_22dof')
            lib=MotionLibrary.from_named_mjlab_file(p,robot_id='k1_22dof')
            torch.testing.assert_close(clip.root_pos_w,c.root_pos_w)
            self.assertEqual(lib.body_names,c.body_names)
            values['metadata']=json.dumps({'quaternion_order':'xyzw'})
            np.savez(p,**values)
            with self.assertRaises(ValueError):MotionClip.from_named_mjlab_npz(p,robot_id='k1_22dof')

    def test_body_reorder_duplicates_and_presence(self):
        c=clip_fixture(with_bodies=True)
        kw={name:getattr(c,name).flip(1) for name in ('body_pos_w','body_quat_wxyz','body_lin_vel_w','body_ang_vel_w')}
        d=replace(c,name='b',body_names=tuple(reversed(c.body_names)),**kw)
        lib=MotionLibrary(ROBOT,[c,d]);data,_=lib._interpolate(torch.tensor([0,1]),torch.tensor([[0.],[0.]]))
        torch.testing.assert_close(data['body_pos_w'][0],data['body_pos_w'][1])
        with self.assertRaises(ValueError):MotionLibrary(ROBOT,[replace(c,body_names=('trunk','trunk'))])
        with self.assertRaises(ValueError):MotionLibrary(ROBOT,[c,clip_fixture(name='no_bodies')])

    def test_bad_clip_rejected(self):
        c=clip_fixture()
        for bad in (replace(c,fps=True),replace(c,fps=0),replace(c,root_name='pelvis'),
                    replace(c,root_quat_wxyz=torch.zeros_like(c.root_quat_wxyz)),
                    replace(c,joint_vel=torch.full_like(c.joint_vel,float('nan')))):
            with self.assertRaises((ValueError,TypeError)):MotionLibrary(ROBOT,[bad])

    def test_body_missing_rejected_and_builder_compatible(self):
        for rid in ('k1_22dof','g1_23dof','g1_29dof'):
            lib,s=sampler_fixture(rid)
            with self.assertRaises(RuntimeError):s.body_reference_window()
            q=torch.zeros(2,4);q[:,0]=1
            ref=ReferenceObservationBuilder(lib.robot,lib.robot.joint_names,torch.zeros(lib.robot.dof))(
                s.reference_window(),current_root_pos_w=torch.zeros(2,3),current_root_quat_wxyz=q)
            self.assertEqual(ref.features.shape,(2,3,16+2*lib.robot.dof))


class RewardTests(unittest.TestCase):
    def test_perfect_score_is_sum_weights(self):
        for rid in ('k1_22dof','g1_23dof','g1_29dof'):
            r,s,w=reward_fixture(rid);cfg=MotionTrackingRewardConfig();out=MotionTrackingReward(r,cfg)(s,w)
            total=sum(getattr(cfg,f.name) for f in fields(cfg) if f.name.endswith('_weight'))
            torch.testing.assert_close(out.total,torch.full((2,),total))

    def test_joint_and_root_error_decrease_score(self):
        r,s,w=reward_fixture();fn=MotionTrackingReward(r);base=fn(s,w)
        out=fn(replace(s,joint_pos=s.joint_pos+.3,root_pos_w=s.root_pos_w+.3),w)
        self.assertTrue((out.total<base.total).all())
        self.assertTrue(all(not v.requires_grad for v in out.terms.values()))

    def test_state_order_and_quaternion_sign_invariance(self):
        r,s,w=reward_fixture();fn=MotionTrackingReward(r);out=fn(s,w)
        other=replace(s,joint_names=tuple(reversed(s.joint_names)),joint_pos=s.joint_pos.flip(-1),
                      joint_vel=s.joint_vel.flip(-1),root_quat_wxyz=-s.root_quat_wxyz)
        torch.testing.assert_close(out.total,fn(other,w).total)

    def test_reward_uses_current_not_future(self):
        r,s,w=reward_fixture();fn=MotionTrackingReward(r);before=fn(s,w).total
        q=w.joint_pos.clone();q[:,1:]=100
        torch.testing.assert_close(before,fn(s,replace(w,joint_pos=q)).total)
        dt=w.time_offsets_s.clone();dt[:,0]=.1
        with self.assertRaises(ValueError):fn(s,replace(w,time_offsets_s=dt))

    def test_missing_data_and_invalid_state_rejected(self):
        r,s,w=reward_fixture()
        for cfg in (MotionTrackingRewardConfig(body_pos_weight=1),MotionTrackingRewardConfig(foot_xy_weight=1)):
            with self.assertRaises(ValueError):MotionTrackingReward(r,cfg)(s,w)
        with self.assertRaises(ValueError):MotionTrackingReward(r)(replace(s,joint_pos=s.joint_pos[:,:-1]),w)
        with self.assertRaises(ValueError):MotionTrackingRewardConfig(joint_pos_sigma_rad=0)

    def test_body_tracks_named_parts_not_order(self):
        lib,sm=sampler_fixture(with_bodies=True);w=sm.reference_window();br=sm.body_reference_window()
        _,s,_=reward_fixture();actual=BodyTrackingState(tuple(reversed(br.body_names)),br.pos_w[:,0].flip(1),
                br.quat_wxyz[:,0].flip(1),br.lin_vel_w[:,0].flip(1))
        cfg=MotionTrackingRewardConfig(body_pos_weight=1,body_rot_weight=1,body_lin_vel_weight=1)
        out=MotionTrackingReward(lib.robot,cfg)(s,w,bodies=actual,body_reference=br)
        self.assertTrue(all(torch.allclose(out.terms[k],torch.ones(2)) for k in ('body_pos','body_rot','body_lin_vel')))

    def foot_fixture(self,phase):
        names=('left_sole','right_sole');p=torch.zeros(2,2,3);v=torch.zeros_like(p)
        v[:,1,0]=-.3
        actual=FootSiteState(names,p,vel_w=v,contact=torch.ones(2,2,dtype=torch.bool))
        target=FootSiteTarget(names,p.clone(),vel_w=v.clone(),phase=torch.tensor(phase).repeat(2,1))
        return actual,target

    def test_slide_not_punished_as_anchor(self):
        r,s,w=reward_fixture();a,t=self.foot_fixture([1,2]);cfg=MotionTrackingRewardConfig(
            foot_anchor_still_weight=1,foot_slide_vel_weight=1,foot_contact_weight=1)
        out=MotionTrackingReward(r,cfg)(s,w,feet=a,foot_reference=t)
        for name in ('foot_anchor_still','foot_slide_vel','foot_contact_match'):
            torch.testing.assert_close(out.terms[name],torch.ones(2))

    def test_transition_and_unknown_do_not_get_free_score(self):
        r,s,w=reward_fixture();a,t=self.foot_fixture([0,4]);cfg=MotionTrackingRewardConfig(
            foot_anchor_still_weight=1,foot_slide_vel_weight=1,foot_contact_weight=1)
        out=MotionTrackingReward(r,cfg)(s,w,feet=a,foot_reference=t)
        for name in ('foot_anchor_still','foot_slide_vel','foot_contact_match'):
            self.assertEqual(out.terms[name].count_nonzero().item(),0)

    def test_site_names_and_phase_required(self):
        r,s,w=reward_fixture();a,t=self.foot_fixture([1,2]);fn=MotionTrackingReward(r,MotionTrackingRewardConfig(foot_anchor_still_weight=1))
        with self.assertRaises(ValueError):fn(s,w,feet=a,foot_reference=replace(t,phase=None))
        with self.assertRaises(ValueError):fn(s,w,feet=a,foot_reference=replace(t,site_names=('ankle','other')))

    def test_velocity_fallback_matches_measured(self):
        r,s,w=reward_fixture();a,t=self.foot_fixture([1,2]);fn=MotionTrackingReward(r,MotionTrackingRewardConfig(foot_slide_vel_weight=1))
        base=fn(s,w,feet=a,foot_reference=t).total
        alt=replace(a,vel_w=None,prev_pos_w=a.pos_w-a.vel_w*.02,dt_s=.02)
        torch.testing.assert_close(base,fn(s,w,feet=alt,foot_reference=t).total)

    def test_per_site_metrics_preserve_one_sided_error(self):
        r,s,w=reward_fixture();a,t=self.foot_fixture([1,2]);p=a.pos_w.clone();p[:,1,0]+=.05
        out=MotionTrackingReward(r,MotionTrackingRewardConfig(foot_xy_weight=1))(s,w,feet=replace(a,pos_w=p),foot_reference=t)
        torch.testing.assert_close(out.metrics['foot_xy_error_by_site_m'],torch.tensor([[0.,.05]]).repeat(2,1))


class SceneDict(dict):
    @property
    def entities(self): return self


def make_bridge(b=2):
    """伪造传感器接口便于核对坐标数学；不是实际接触求解。"""
    env=FakeEnv(b);env.scene=SceneDict(env.scene)
    entity=env.scene['robot'];entity.body_names=['trunk',*FOOT_NAMES]
    data=entity.data;data.root_link_vel_w=torch.zeros(b,6)
    data.body_link_pose_w=torch.zeros(b,3,7);data.body_link_pose_w[...,3]=1
    data.body_link_pose_w[:,0,:3]=data.root_link_pos_w
    data.body_link_pose_w[:,1,0]=.1;data.body_link_pose_w[:,2,0]=.2
    data.body_link_vel_w=torch.zeros(b,3,6)
    c=SimpleNamespace(found=torch.ones(b,2),force=torch.zeros(b,2,3),pos=torch.zeros(b,2,3),normal=torch.zeros(b,2,3))
    c.normal[...,2]=-1;c.force[...,2]=10
    c.pos[:,0,0]=.2;c.pos[:,1,0]=.1  # deliberately reversed primary order
    env.scene[CONTACT_SENSOR_NAME]=SimpleNamespace(primary_names=list(reversed(FOOT_NAMES)),data=c,
        cfg=SimpleNamespace(reduce='maxforce',num_slots=1,global_frame=True))
    bridge=K1MJLabBridge(env);bridge.reset()
    return env,bridge,K1PhysicsBridge(bridge)


class PhysicsBridgeTests(unittest.TestCase):
    def test_mjlab_13_contact_primary_names_fallback(self):
        slots = [SimpleNamespace(primary_name=name) for name in FOOT_NAMES for _ in range(5)]
        sensor = SimpleNamespace(_slots=slots)
        self.assertEqual(_contact_primary_names(sensor), FOOT_NAMES)

    def test_reordered_contacts_force_and_snapshot(self):
        env,b,p=make_bridge();x=p.read()
        torch.testing.assert_close(x.contact_features[0,:,0],torch.tensor([.1,.2]))
        torch.testing.assert_close(x.contact_features[...,9],torch.full((2,2),10.))
        env.scene[CONTACT_SENSOR_NAME].data.force.zero_()
        torch.testing.assert_close(x.contact_features[...,9],torch.full((2,2),10.))
        self.assertEqual(x.object_features.shape,(2,0,16))

    def test_tangential_contact_truth_is_separate_from_actor_observation(self):
        env,b,p=make_bridge()
        env.scene[CONTACT_SENSOR_NAME].data.force[...,0]=6
        b.entity.data.body_link_vel_w[:,1:,0]=.25
        observation=p.read()
        tangential,slip,utilization=p.contact_supervision()
        torch.testing.assert_close(tangential,torch.full((2,2),6.))
        torch.testing.assert_close(slip,torch.full((2,2),.25))
        torch.testing.assert_close(utilization,torch.full((2,2),.6))
        # The deployable observation contract remains exactly 12 contact features;
        # simulator-only targets are retrieved through the separate training API.
        self.assertEqual(observation.contact_features.shape[-1],12)

    def test_no_contact_uses_body_point_not_fake_force(self):
        env,b,p=make_bridge();env.scene[CONTACT_SENSOR_NAME].data.found.zero_();out=p.read()
        self.assertEqual(out.contact_features[...,9:11].count_nonzero().item(),0)
        self.assertTrue(out.contact_mask.all());self.assertTrue((out.contact_features[...,11]==1).all())

    def test_point_velocity_accounts_for_angular_motion_and_ground(self):
        env,b,p=make_bridge();data=b.entity.data
        data.body_link_vel_w[:,1,5]=2
        env.scene[CONTACT_SENSOR_NAME].data.pos[:,1,0]=.3  # left point offset .2; omega_z=2
        p=K1PhysicsBridge(b,ground_velocity_fn=lambda x:torch.ones_like(x)*.1)
        out=p.read();torch.testing.assert_close(out.contact_features[:,0,3:6],torch.tensor([-.1,.3,-.1]).repeat(2,1),atol=1e-6,rtol=1e-5)

    def test_object_state_orientation_and_size(self):
        env,b,p=make_bridge();pose=torch.tensor([[1.,0.,.57,1.,0.,0.,0.]]).repeat(2,1)
        env.scene['box']=SimpleNamespace(data=SimpleNamespace(root_link_pose_w=pose,root_link_vel_w=torch.zeros(2,6)))
        p=K1PhysicsBridge(b,object_specs=(ObjectObservationSpec('box',(.2,.3,.4)),))
        out=p.read();torch.testing.assert_close(out.object_features[:,0,:3],torch.tensor([1.,0.,0.]).repeat(2,1))
        torch.testing.assert_close(out.object_features[:,0,13:],torch.tensor([.2,.3,.4]).repeat(2,1))

    def test_deployable_foot_observation_cannot_read_exact_contact_truth(self):
        env,b,p=make_bridge()
        sensor=env.scene[CONTACT_SENSOR_NAME].data
        first=p.read_deployable()
        sensor.found.fill_(1);sensor.pos.fill_(91);sensor.force.fill_(73)
        sensor.normal.fill_(0);sensor.normal[...,2]=1
        second=p.read_deployable()
        torch.testing.assert_close(first.contact_features,second.contact_features)
        self.assertEqual(second.base_lin_vel_b.count_nonzero().item(),0)
        self.assertEqual(second.contact_features[...,6:].count_nonzero().item(),0)
        self.assertEqual(second.object_features.shape,(2,0,16))

    def test_privileged_contact_truth_is_separate_and_reads_sensor(self):
        env,b,p=make_bridge()
        sensor=env.scene[CONTACT_SENSOR_NAME].data
        sensor.found.fill_(1);sensor.force.zero_();sensor.force[...,2]=40
        sensor.normal.zero_();sensor.normal[...,2]=1
        truth=p.privileged_contact_truth()
        self.assertIsInstance(truth,PrivilegedFootContactTruth)
        self.assertTrue(truth.contact_found.all())
        torch.testing.assert_close(truth.normal_force_n,torch.full((2,2),40.))
        teacher=truth.teacher_tensor()
        self.assertEqual(teacher.shape,(2,28))
        # Left/right remain adjacent fixed slots rather than being pooled.
        torch.testing.assert_close(teacher[:,9],torch.full((2,),.4))
        torch.testing.assert_close(teacher[:,23],torch.full((2,),.4))

    def test_wrong_sensor_semantics_and_specs_rejected(self):
        env,b,p=make_bridge();env.scene[CONTACT_SENSOR_NAME].cfg.global_frame=False
        with self.assertRaises(ValueError):K1PhysicsBridge(b)
        with self.assertRaises(TypeError):K1PhysicsBridge(b,object_specs=('box',))
        with self.assertRaises(ValueError):ObjectObservationSpec('box',(0.,1.,1.))

    def test_first_partial_reset_rejected(self):
        b=K1MJLabBridge(FakeEnv(2))
        with self.assertRaises(RuntimeError):b.reset([0])
        b.reset();b.reset([0])

    def test_xml_wrong_joint_type_gear_and_root_rejected(self):
        for what in ('type','gear','root'):
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'K1.xml';create_fixture_xml(path);tree=ET.parse(path);r=tree.getroot()
                if what=='type':r.findall('.//joint')[1].set('type','ball')
                elif what=='gear':r.find('./actuator/motor').set('gear','2')
                else:r.find('.//body').remove(r.findall('.//joint')[0])
                tree.write(path)
                with self.assertRaises(ValueError):read_k1_joint_specs(path)

    def test_actual_bridge_sampler_reward_collector_ppo_connection(self):
        for mode in ('mlp','dense','moe'):
            env,b,phy=make_bridge();lib,sm=sampler_fixture()
            reward=MotionTrackingReward(ROBOT)
            task=MotionTrackingTask(phy,sm,reward)
            dyn=DynamicsModel(ROBOT,physics_width=8,memory_width=8,latent_width=4,hidden_dim=16)
            if mode=='mlp':policy=PhysicsMLPActorCritic(ROBOT,physics_width=8,latent_width=4,hidden_dim=16)
            else:policy=PhysicsConditionedActorCritic({'k1_22dof':ROBOT},PhysicsPolicyV2Config(
                backbone=mode,d_model=16,n_layers=1,n_heads=4,n_kv_heads=2,context_len=3,input_hidden_dim=16,
                dense_hidden_dim=16,num_fine_experts=4,expert_hidden_dim=16,router_hidden_dim=16,
                physics_width=8,latent_width=4,critic_hidden_dim=16))
            c=PhysicsSequenceCollector(task,dyn,policy);r=c.collect(4)
            self.assertEqual(r.horizon,4);self.assertTrue((r.reward>0).all())
            ppo=PhysicsSequencePPO(policy,PPOConfig(epochs=1,minibatch_envs=2))
            ppo.update(r);DynamicsUpdater(dyn).update(r);c.on_dynamics_updated();c.collect(2)

    def test_reference_task_full_adaptation_hides_actor_truth(self):
        env,b,phy=make_bridge();lib,sm=sampler_fixture()
        task=MotionTrackingTask(phy,sm,MotionTrackingReward(ROBOT),
                                actor_observation='deployable',privileged_learning=True)
        task.reset();full=phy.read();visible=task.read();priv=task.privileged()
        self.assertEqual(task.privileged_dim,5)
        self.assertEqual(visible.base_lin_vel_b.count_nonzero().item(),0)
        self.assertEqual(visible.contact_features[...,6:].count_nonzero().item(),0)
        torch.testing.assert_close(priv[:,:3],full.base_lin_vel_b)
        torch.testing.assert_close(priv[:,3:],full.contact_features[...,9]/100.)
        dyn=DynamicsModel(ROBOT,physics_width=8,memory_width=8,latent_width=4,hidden_dim=16,
                          system_id_dim=5,teacher_student=True)
        policy=PhysicsMLPActorCritic(ROBOT,physics_width=8,latent_width=4,hidden_dim=16,
                                     privileged_dim=5)
        collector=PhysicsSequenceCollector(task,dyn,policy);rollout=collector.collect(3)
        update=DynamicsUpdater(dyn,system_id_weight=.1,
            teacher_distillation_weight=.05,teacher_reconstruction_weight=.1)
        stats=update.update(rollout)
        self.assertIn('teacher_distillation',stats.terms)

    def test_extended_contact_supervision_remains_privileged_only(self):
        env,b,phy=make_bridge();lib,sm=sampler_fixture()
        env.scene[CONTACT_SENSOR_NAME].data.force[...,0]=5
        task=MotionTrackingTask(phy,sm,MotionTrackingReward(ROBOT),
            actor_observation='deployable',privileged_learning=True,
            extended_contact_supervision=True)
        task.reset();visible=task.read();truth=task.privileged()
        self.assertEqual(task.privileged_dim,11)
        self.assertEqual(visible.contact_features[...,3:10].count_nonzero().item(),0)
        self.assertGreater(truth[:,5:7].abs().sum().item(),0)
        self.assertEqual(visible.contact_features.shape[-1],12)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)

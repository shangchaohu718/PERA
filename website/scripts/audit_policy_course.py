"""Independently integrate saved joint targets and check the physical path.

No recorded robot pose is inserted after frame 0. Only the tennis ball has its
logged relaunch and published rebound rule applied. The renderer consumes the
audited source, with neither pose interpolation nor spatial retargeting.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'third_party/ATHLETE/athlete/src'))
sys.path.insert(0,str(ROOT/'experiments/athlete_tennis_v1'))
from compat import setup
setup()
from athlete.scripts.tennis_physics import STANDARD_TENNIS_PHYSICS as PHYSICS,tennis_ball_aerodynamic_wrench_numpy
from athlete.scripts.tennis_physics import STANDARD_TENNIS_DOMAIN_RANDOMIZATION
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def audit(run):
    report=json.loads((run/'report.json').read_text())
    f=np.load(run/'physics.npz');model=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));data=mujoco.MjData(model)
    names=json.loads((ROOT/'configs/g1_actor_io_contract.json').read_text())
    aids=[model.actuator('robot/'+n).id for n in names['joint_names']]
    target=np.asarray(names['action_offset'],dtype=np.float32)+np.asarray(names['action_scale'],dtype=np.float32)*f['actions']
    data.qpos[:]=f['qpos'][0];data.qvel[:]=f['qvel'][0];mujoco.mj_forward(model,data)
    first_tennis=np.flatnonzero(f['stage']=='tennis')
    end=len(f['qpos'])-1
    bq=int(model.joint('tennis_ball_freejoint').qposadr[0]);bv=int(model.joint('tennis_ball_freejoint').dofadr[0]);body=model.body('tennis_ball').id
    launches={round(c['start_s']/.02) for c in report['tennis_cycles']}
    height=float(model.geom_pos[model.geom('route_court_platform').id,2]+model.geom_size[model.geom('route_court_platform').id,2])
    ground=height+PHYSICS.ball.radius_m
    retention=sum(STANDARD_TENNIS_DOMAIN_RANDOMIZATION.ground_tangent_speed_retention)/2
    pending=False;impact=np.zeros(3);age=0
    errors=[];joint_errors=[]
    for i in range(end):
        if i in launches:
            data.qpos[bq:bq+7]=f['qpos'][i,bq:bq+7];data.qvel[bv:bv+6]=f['qvel'][i,bv:bv+6]
            pending=False;age=0;mujoco.mj_forward(model,data)
        data.ctrl[aids]=target[i]
        for _ in range(4):
            tennis=f['stage'][i]=='tennis'
            if tennis:
                vel=data.qvel[bv:bv+6];force,torque=tennis_ball_aerodynamic_wrench_numpy(vel[:3],vel[3:])
                data.xfrc_applied[body,:3]=force;data.xfrc_applied[body,3:]=torque
            mujoco.mj_step(model,data)
            if tennis:
                vel=data.qvel[bv:bv+3];z=float(data.xpos[body,2])
                if pending:age+=1
                elif vel[2]<-.05 and z<=ground+.03:
                    pending=True;age=0;impact[:]=vel
                    impact[2]=-math.sqrt(float(vel[2]**2)+2*9.81*max(z-ground,0))
                if pending and z<=ground+.002:
                    data.qpos[bq+2]=ground+.0001;data.qvel[bv:bv+2]=impact[:2]*retention;data.qvel[bv+2]=-impact[2]*PHYSICS.court.restitution
                    pending=False;age=0
                elif age>20:pending=False;age=0
        mujoco.mj_forward(model,data)
        errors.append(float(abs(data.qpos[:7]-f['qpos'][i+1,:7]).max()))
        joint_errors.append(float(abs(data.qpos[7:36]-f['qpos'][i+1,7:36]).max()))
    result=dict(source_sha256=digest(run/'physics.npz'),scene_sha256=digest(run/'scene.mjb'),report_sha256=digest(run/'report.json'),
                replayed_control_steps=end,replayed_physics_steps=end*4,control_dt_s=.02,
                robot_pose_writes_in_independent_replay_after_initialization=0,
                max_root_qpos_error=float(max(errors,default=0)),max_joint_error_rad=float(max(joint_errors,default=0)),
                passed=bool(max(errors,default=0)<1e-5 and max(joint_errors,default=0)<1e-5),
                logged_ball_relaunches=len(launches),
                scope='Independent continuous MuJoCo integration of saved executed joint targets from one initial state, with zero robot pose insertion. Covers the entire skateboard/dismount/walking/rough/stairs/tennis course. Only the ball receives its logged relaunch and the native rebound rule.')
    (run/'physical-replay-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();result=audit(args.run)
    raise SystemExit(0 if result['passed'] else 1)

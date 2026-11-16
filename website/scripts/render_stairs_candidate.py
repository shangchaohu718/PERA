#!/usr/bin/env python3
"""Uncut stair-ascent camera study from recorded real student states.

Does not execute/train a policy or edit physics states. Not a multi-skill rollout.
"""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from PIL import Image
from studio_lighting import configure_lighting,light_direction
WEB=Path(__file__).resolve().parents[1];ROOT=WEB.parent
SOURCE=ROOT/'reports/mimicx_stairs_512_v1/run_20261106_083527/training'
OUT=WEB/'qa/stairs'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');args=p.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 source=SOURCE/'final_student_4004_trajectory.npz';scene=SOURCE/'native_scene.mjb'
 report=SOURCE/'final_student_4004.json';evaluation=json.loads(report.read_text())
 assert not evaluation['robot0_failed'] and evaluation['teacher_action_queries']==0
 with np.load(source) as f:q=f['qpos'];fps=float(f['fps'])
 model=configure_lighting(mujoco.MjModel.from_binary_path(str(scene)),'terrain')
 for i in range(model.ngeom):
  name=model.geom(i).name
  if name=='parkour_ground':model.geom_rgba[i]=[.97,.98,1,1];model.geom_matid[i]=-1
  elif name.startswith('parkour_step_'):
   model.geom_rgba[i]=[.62,.69,.77,1];model.geom_matid[i]=-1
 data=mujoco.MjData(model);cam=mujoco.MjvCamera()
 cam.azimuth=35;cam.elevation=-23;cam.distance=3.5
 opt=mujoco.MjvOption();opt.sitegroup[:]=0;opt.geomgroup[3]=0
 # Smooth only the camera path; every robot frame is an untouched recorded pose.
 target=q[:,:3].copy();target[:,0]+=.35;target[:,2]-=.10
 smooth=target.copy();alpha=1-np.exp(-1/(fps*.18))
 for i in range(1,len(smooth)):smooth[i]=smooth[i-1]+alpha*(target[i]-smooth[i-1])
 chosen=[0,round(4*fps),len(q)-1] if args.preview else range(len(q))
 path=OUT/'one-take-stairs-candidate.mp4';proc=None;frames=[]
 if not args.preview:
  if path.exists():raise FileExistsError(path)
  proc=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','slow','-crf','20','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
 try:
  with mujoco.Renderer(model,height=1080,width=1920) as renderer:
   for i in chosen:
    data.qpos[:]=q[i];mujoco.mj_forward(model,data);cam.lookat[:]=smooth[i]
    light_direction(model,cam);renderer.update_scene(data,camera=cam,scene_option=opt)
    frame=renderer.render()
    if args.preview:frames.append(Image.fromarray(frame).resize((640,360)))
    else:proc.stdin.write(frame.tobytes())
 finally:
  if proc:
   proc.stdin.close();assert proc.wait()==0
 if args.preview:
  sheet=Image.new('RGB',(1920,360))
  for i,f in enumerate(frames):sheet.paste(f,(640*i,0))
  sheet.save(OUT/'stairs-camera-preview.jpg');return
 meta=dict(source=str(source.relative_to(ROOT)),source_sha256=digest(source),scene_sha256=digest(scene),report=str(report.relative_to(ROOT)),report_sha256=digest(report),video_sha256=digest(path),fps=fps,frames=len(q),duration_s=len(q)/fps,continuous_episode=True,edited_montage=False,pose_interpolation=False,speed=1,teacher_action_queries=0,robot0_failed=False,camera=dict(azimuth=35,elevation=-23,distance=3.5,forward_lookahead_m=.35,camera_only_smoothing_time_constant_s=.18),scope='Full recorded robot-0 stair-ascent episode, seed 4004, from a 512-world evaluation. Same frozen public Actor plus the selected stairs expert. One fixed scene/reference, not a composed multi-skill course or an independent generalization test.')
 path.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
if __name__=='__main__':main()

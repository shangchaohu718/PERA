#!/usr/bin/env python3
"""Re-render the complete climbing segment of the archived continuous course.

All terrain geometry and recorded qpos remain intact. This is an excerpt of a
single successful course, not the independent 16-episode climbing evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from PIL import Image
WEB=Path(__file__).resolve().parents[1]
ROOT=WEB.parent
RUN=ROOT/'reports/velocity_course_v11/seed7002'

def main(preview=False):
    model=mujoco.MjModel.from_binary_path(str(RUN/'scene.mjb'))
    data=mujoco.MjData(model)
    model.vis.global_.offwidth=1920;model.vis.global_.offheight=1080
    model.vis.quality.shadowsize=4096
    for i in range(model.ntex):
        address=model.tex_adr[i];size=model.tex_width[i]*model.tex_height[i]*model.tex_nchannel[i]
        color=250 if model.tex_type[i]==mujoco.mjtTexture.mjTEXTURE_SKYBOX else 240
        model.tex_data[address:address+size]=color
    model.vis.headlight.ambient[:]=.35
    model.vis.headlight.diffuse[:]=.25
    model.vis.rgba.haze[:]=[.98,.98,.98,1]
    model.light_diffuse[:]=.35
    options=mujoco.MjvOption();options.geomgroup[3]=0;options.sitegroup[:]=0
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth=135;cam.elevation=-24;cam.distance=4.1
    torso=next(i for i in range(model.nbody) if model.body(i).name.endswith('torso_link'))
    with np.load(RUN/'physical_replay.npz') as d:
        ids=np.flatnonzero((d['task']=='knee_climb_no_wall')&(d['mode']=='skill'))
        poses=d['qpos'][ids].copy();times=d['time_s'][ids].copy()
    assert np.max(np.abs(np.diff(times)-.04))<.001
    duration=float(times[-1]-times[0]+np.median(np.diff(times)))
    output=WEB/'public/media/climb'
    master=WEB/'.render-work/climb.master.mkv'
    if not preview and (master.exists() or output.with_suffix('.mp4').exists()):raise FileExistsError(output)
    frames=np.clip(np.rint(np.arange(0,duration,1/30)/np.median(np.diff(times))).astype(int),0,len(poses)-1)
    selected=[len(poses)//2] if preview else frames
    proc=None if preview else subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','-','-an','-c:v','ffv1','-threads','2',str(master)],stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model,height=1080,width=1920) as renderer:
            for n,i in enumerate(selected):
                data.qpos[:]=poses[i];mujoco.mj_forward(model,data)
                cam.lookat[:]=data.xpos[torso];cam.lookat[2]=max(.9,data.xpos[torso,2]-.1)
                model.stat.center[:]=cam.lookat;model.stat.extent=5
                renderer.update_scene(data,camera=cam,scene_option=options)
                frame=renderer.render()
                if preview or n==len(selected)//2:Image.fromarray(frame).save(output.with_suffix('.webp'),quality=92)
                if proc:proc.stdin.write(frame.tobytes())
                if n%120==0:print(f'climb {n}/{len(selected)}',flush=True)
    finally:
        if proc:
            proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('Master encoding failed')
    if preview:return
    subprocess.run(['ffmpeg','-v','error','-i',str(master),'-an','-c:v','libx264','-preset','slow','-crf','22','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(output.with_suffix('.mp4'))],check=True)
    metadata={'id':'climb','name':'50 cm knee climb, continuous-course excerpt','source':str((RUN/'physical_replay.npz').relative_to(ROOT)),'source_sha256':hashlib.file_digest(open(RUN/'physical_replay.npz','rb'),'sha256').hexdigest(),'scene':str((RUN/'scene.mjb').relative_to(ROOT)),'scene_sha256':hashlib.file_digest(open(RUN/'scene.mjb','rb'),'sha256').hexdigest(),'controller':'Frozen HoloMotion Actor + instinct_climb BC/DAgger expert; no teacher action queries','scope':'Complete skill segment from a single successful fixed course at seed 7002, nominal physics; not the independent held-out 16/16 batch. Original controller includes a 0.6 s entry-action blend.','partial_excerpt':True,'source_start_s':float(times[0]),'source_end_s':float(times[-1]),'source_frames':len(poses),'fps':30,'source_fps':25,'output_frames':len(frames),'duration_s':len(frames)/30,'resolution':[1920,1080],'speed':1,'pose_interpolation':False,'all_terrain_geometry_retained':True,'rendering':'Exact archived qpos, nearest-frame time sampling (max 20 ms). Only camera, texture color and lighting changed. No reference-driven poses or fresh policy inference.','master':str(master.relative_to(WEB)),'video_sha256':hashlib.file_digest(open(output.with_suffix('.mp4'),'rb'),'sha256').hexdigest()}
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Climb render complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preview',action='store_true');args=p.parse_args();main(args.preview)

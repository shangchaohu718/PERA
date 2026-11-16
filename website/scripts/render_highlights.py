#!/usr/bin/env python3
"""Create bright studio re-renders, honest excerpts and a selected montage.

Only recorded actual robot/object qpos are replayed. No simulator stepping,
training, reference-driven animation, pose interpolation or speed changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from render_gallery import studio_model
from studio_lighting import configure_lighting, light_direction

WEB=Path(__file__).resolve().parents[1]
ROOT=WEB.parent
CONFIG=json.loads((WEB/'scripts/highlights.json').read_text())
MEDIA=Path(os.environ.get('PERA_MEDIA_DIRECTORY',WEB/'public/media'))
WORK=Path(os.environ.get('PERA_WORK_DIRECTORY',WEB/'.render-work/highlights-v6'))
QA=WEB/'qa/highlights'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def digest(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()



def load(clip):
    source=ROOT/clip['source']
    with np.load(source,allow_pickle=False) as f:
        if clip['kind']=='isaac':
            xyz=f['robot_global_translation'][:,0];quat=f['robot_global_rotation_quat'][:,0]
            poses=np.concatenate([xyz,quat[:,[3,0,1,2]],f['robot_dof_pos']],axis=1)
            fps=clip['source_fps'];times=np.arange(len(poses))/fps;ids=np.arange(len(poses))
            model=studio_model()
            if clip.get('scene_style')=='skate':
                # The studio material normally omits its source ground texture.
                # Reattach it for moonwalk so foot sliding has a world reference.
                ground=model.mat('groundplane').id
                texture=next(i for i in range(model.ntex) if model.tex_type[i]==mujoco.mjtTexture.mjTEXTURE_2D)
                model.mat_texid[ground,1]=texture;model.mat_texuniform[ground]=True
        elif clip['kind']=='course':
            ids=np.flatnonzero((f['task']==clip['task'])&(f['mode']=='skill'))
            assert np.all(np.diff(ids)==1),'Task segment must be contiguous'
            poses=f['qpos'][ids].copy();times=f['time_s'][ids].astype(float);fps=float(f['fps'])
            model=mujoco.MjModel.from_binary_path(str(ROOT/clip['scene']))
        else:
            poses=f['qpos'].copy();fps=float(f['fps']);times=f['time_s'].astype(float) if clip.get('recorded_times',clip['id']=='tennis') else np.arange(len(poses))/fps;ids=np.arange(len(poses))
            model=mujoco.MjModel.from_binary_path(str(ROOT/clip['scene']))
    assert np.isfinite(poses).all() and np.max(np.abs(np.diff(times)-1/fps))<.001
    return configure_lighting(model,clip.get('scene_style','studio')),poses,times,fps,ids


def camera_target(model,data,clip):
    if 'target' in clip['camera']:return np.array(clip['camera']['target'],dtype=float)
    if clip['kind']=='isaac':return np.array([*data.qpos[:2],.76])
    pelvis=next(i for i in range(model.nbody) if model.body(i).name=='pelvis' or model.body(i).name.endswith('/pelvis'))
    # Fixed vertical target avoids camera bobbing and keeps feet + obstacle visible.
    family=clip.get('family',clip['id'])
    base_z=.82 if family=='tennis' else 0.72 if family in ('roll','vault') else .70
    if family=='steps':base_z=max(.58,min(.85,float(data.xpos[pelvis,2])))
    if clip.get('failure_visualization'):base_z=max(-.5,min(base_z,float(data.xpos[pelvis,2])+.18))
    return np.array([*data.xpos[pelvis,:2],base_z])


def update(model,data,renderer,camera,options,clip,pose):
    light_direction(model,camera)
    data.qpos[:]=pose;mujoco.mj_forward(model,data)
    camera.lookat[:]=camera_target(model,data,clip)
    if 'fall_elevation' in clip['camera']:
        # Reveal the actual fallen body inside the recess; geometry stays intact.
        amount=float(np.clip((clip['camera'].get('fall_height',.6)-float(data.qpos[2]))/clip['camera'].get('fall_range',.65),0,1))
        camera.elevation=clip['camera']['elevation']+amount*(clip['camera']['fall_elevation']-clip['camera']['elevation'])
    model.stat.center[:]=camera.lookat;model.stat.extent=4.5
    renderer.update_scene(data,camera=camera,scene_option=options)
    frame=renderer.render()
    if clip.get('closeup'):
        # Two cameras on the same recorded instant; only the presentation changes.
        image=Image.fromarray(frame)
        close=mujoco.MjvCamera();close.azimuth=125;close.elevation=-24;close.distance=3.2
        close.lookat[:]=[*data.qpos[:2],.82]
        model.stat.center[:]=close.lookat
        renderer.update_scene(data,camera=close,scene_option=options)
        detail=Image.fromarray(renderer.render()).resize((640,360),Image.Resampling.LANCZOS)
        image.paste(detail,(36,36));draw=ImageDraw.Draw(image)
        draw.rectangle((35,35,676,396),outline='#e5e7eb',width=2)
        draw.rectangle((48,48,208,82),fill='#ffffff')
        draw.text((58,52),'ROBOT VIEW',font=ImageFont.truetype(FONT,20),fill='#1d1d1f')
        return np.asarray(image)
    return frame


def encode(master,path,frames=None,start=None,label=None):
    command=['ffmpeg','-v','error']
    if start is not None:command+=['-ss',str(start)]
    command+=['-i',str(master),'-an']
    if frames is not None:command+=['-frames:v',str(frames)]
    if label:
        text=label.upper()
        filt=f"drawtext=fontfile={FONT}:text='{text}':fontsize=26:fontcolor=0x1d1d1f:x=54:y=h-75:box=1:boxcolor=white@0.86:boxborderw=12"
        command+=['-vf',filt]
    command+=['-c:v','libx264','-preset','slow','-crf','21','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(path)]
    subprocess.run(command,check=True)


def render(clip,preview=False):
    model,poses,times,fps,source_ids=load(clip)
    data=mujoco.MjData(model);camera=mujoco.MjvCamera()
    camera.azimuth=clip['camera']['azimuth'];camera.elevation=clip['camera']['elevation'];camera.distance=clip['camera']['distance']
    options=mujoco.MjvOption();options.sitegroup[:]=0
    if clip['kind']=='isaac':options.geomgroup[0]=0  # duplicate collision meshes; actual floor is group 2
    else:options.geomgroup[3]=0  # robot collision proxies; terrain groups untouched
    if clip.get('family',clip['id'])=='tennis':
        # The racket contact proxy is an orange translucent ellipsoid overlay.
        # Hide this debug proxy, retaining the actual racket mesh and all states.
        model.geom_rgba[model.geom('racket_ball_collision').id,3]=0
        for name in ('incoming_ball_launch_region','landing_target_marker','landing_target_center','manual_first_bounce_marker'):
            model.geom_rgba[model.geom(name).id,3]=0
    duration=float(times[-1]-times[0]+1/fps)
    frame_times=np.arange(int(np.ceil(duration*30)))/30
    # Find nearest actual timestamp; no interpolation, 1x wall-clock playback.
    upper=np.minimum(np.searchsorted(times-times[0],frame_times),len(times)-1)
    lower=np.maximum(upper-1,0)
    indices=np.where(abs(times[lower]-times[0]-frame_times)<=abs(times[upper]-times[0]-frame_times),lower,upper)
    master=WORK/(clip['id']+'.master.mkv')
    full=MEDIA/(clip['id']+'-full.mp4');short=MEDIA/(clip['id']+'.mp4')
    if not preview and any(p.exists() for p in (master,full,short)):raise FileExistsError(clip['id'])
    if preview:
        moments=[clip['highlight'][0],clip['poster_time'],clip['highlight'][1]-.1]
        selected=np.minimum(np.rint(np.array(moments)*fps).astype(int),len(poses)-1)
    else:selected=indices
    proc=None
    if not preview:
        proc=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','-','-an','-c:v','ffv1','-threads','2',str(master)],stdin=subprocess.PIPE)
    samples=[]
    try:
        with mujoco.Renderer(model,height=1080,width=1920) as renderer:
            for n,i in enumerate(selected):
                frame=update(model,data,renderer,camera,options,clip,poses[i])
                if preview:samples.append(Image.fromarray(frame).resize((640,360)))
                else:
                    proc.stdin.write(frame.tobytes())
                    if n==int(clip['poster_time']*30):
                        Image.fromarray(frame).save(short.with_suffix('.webp'),quality=93)
                        Image.fromarray(frame).save(full.with_suffix('.webp'),quality=93)
                    if n%150==0:print(f'{clip["id"]} {n}/{len(selected)}',flush=True)
    finally:
        if proc:
            proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('Master encoding failed')
    if preview:
        sheet=Image.new('RGB',(1920,360),'#fafafa')
        for i,frame in enumerate(samples):sheet.paste(frame,(i*640,0))
        sheet.save(QA/(clip['id']+'-preview.jpg'),quality=92)
        print('Preview:',clip['id'],flush=True);return
    encode(master,full,frames=len(indices))
    short_frames=round((clip['highlight'][1]-clip['highlight'][0])*30)
    encode(master,short,frames=short_frames,start=clip['highlight'][0],label=clip['name'])
    common={'name':clip['name'],'expert':clip['expert'],'source':clip['source'],'source_sha256':digest(ROOT/clip['source']),'report':clip['report'],'report_sha256':digest(ROOT/clip['report']),'scope':clip['scope'],'source_fps':fps,'source_frames':len(poses),'source_frame_offset':int(source_ids[0]),'source_recording_start_s':float(times[0]),'fps':30,'resolution':[1920,1080],'speed':1,'pose_interpolation':False,'camera':clip['camera'],'scene_style':clip.get('scene_style','studio'),'lighting':'Studio v4: white sky; contrasting terrain, world-anchored skate grid, or original tennis-court colors according to scene. Original robot albedo retained.','all_terrain_geometry_retained':True,'master':str(master.relative_to(WEB)),'rendering':'Exact recorded actual robot and object states. Nearest timestamp resampling only; maximum quantization is half of one source interval, plus source timestamp precision. No training, simulator stepping, teacher queries or reference-pose playback during rendering.'}
    if clip.get('failure_visualization'):common['failure_visualization']=clip['failure_visualization']
    if clip.get('family',clip['id'])=='tennis':
        common['continuous_incoming_ball_cycles']=True
        common['synchronized_robot_inset']=clip.get('closeup',False)
        common['visual_hands']=json.loads((ROOT/clip.get('hands_validation','website/.render-work/tennis-v3/hands-validation.json')).read_text())
    if 'scene' in clip:common.update(scene=clip['scene'],scene_sha256=digest(ROOT/clip['scene']))
    else:
        model_path='HoloMotion/assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.xml'
        common.update(model=model_path,model_sha256=digest(ROOT/model_path))
    for path,is_full,count,start in [(full,True,len(indices),0),(short,False,short_frames,clip['highlight'][0])]:
        meta=dict(common,id=path.stem,full_recorded_episode=is_full and clip['kind']!='course',full_task_segment=is_full and clip['kind']=='course',partial_excerpt=not is_full or clip['kind']=='course',excerpt_start_s=start,excerpt_end_s=start+count/30,source_start_s=float(times[0])+start,source_end_s=float(times[0])+start+count/30,output_frames=count,duration_s=count/30,video_sha256=digest(path))
        if not is_full:meta['full_video']=full.name
        path.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Complete:',clip['id'],flush=True)


def montage():
    target=MEDIA/'showreel.mp4'
    if target.exists():raise FileExistsError(target)
    listing=WORK/'montage.txt'
    clean=WORK/'hero-without-labels';clean.mkdir(exist_ok=True)
    sources=[]
    for c in CONFIG['clips']:
        meta=json.loads((MEDIA/(c['id']+'.json')).read_text())
        output=clean/(c['id']+'.mp4')
        if output.exists():raise FileExistsError(output)
        encode(WEB/meta['master'],output,frames=meta['output_frames'],start=meta['excerpt_start_s'])
        sources.append(output)
    listing.write_text(''.join(f"file '{p.as_posix()}'\n" for p in sources))
    subprocess.run(['ffmpeg','-v','error','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(target)],check=True)
    chapters=[];start=0
    for c in CONFIG['clips']:
        meta=json.loads((MEDIA/(c['id']+'.json')).read_text())
        chapters.append({'id':c['id'],'name':c['name'],'expert':c['expert'],'start_s':start,'end_s':start+meta['duration_s'],'source':meta['source'],'source_sha256':meta['source_sha256'],'excerpt_start_s':meta['excerpt_start_s'],'excerpt_end_s':meta['excerpt_end_s'],'full_video':meta['full_video'],'scope':meta['scope']});start+=meta['duration_s']
    Image.open(MEDIA/'kick.webp').save(MEDIA/'showreel.webp',quality=93)
    meta={'id':'showreel','name':'PERA selected rollout excerpts','partial_excerpt':True,'edited_montage':True,'burned_in_labels':False,'continuous_episode':False,'speed':1,'pose_interpolation':False,'fps':30,'resolution':[1920,1080],'output_frames':round(start*30),'duration_s':start,'video_sha256':digest(target),'chapters':chapters,'scope':f'{len(chapters)} recorded skill demonstrations from separate recordings, including a continuous experimental tennis sequence with unsuccessful returns. Not one continuous run, a random sample, or evidence of success rate. Full source episode/task segments are provided for every clip.'}
    (MEDIA/'showreel.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Montage:',start,'seconds',flush=True)


if __name__=='__main__':
    WORK.mkdir(parents=True,exist_ok=True);QA.mkdir(parents=True,exist_ok=True)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--clip');p.add_argument('--preview',action='store_true');p.add_argument('--montage-only',action='store_true');a=p.parse_args()
    if a.montage_only:montage();sys.exit(0)
    for clip in CONFIG['clips']:
        if a.clip is None or clip['id']==a.clip:render(clip,a.preview)
    if not a.preview and a.clip is None:montage()

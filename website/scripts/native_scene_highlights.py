"""Four independent native-scene recordings for the homepage.

No course geometry, retargeting, synthetic walking joins, pose interpolation,
or prop hiding. Native scenes contain a racket only in the tennis task.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter1d
from render_highlights import encode
from studio_lighting import configure_lighting,light_direction

WEB=Path(__file__).resolve().parents[1];ROOT=WEB.parent
STAIRS=ROOT/'reports/mimicx_stairs_512_v1/run_20261106_083527/training'
EXPORTED=ROOT/'reports/continuous_four_scene_v1/native_stairs_probe_02/scene-export'
CHOICES=[('skate',0.,5.),('steps',0.,5.),('native-stairs',2.8,8.8),('tennis',0.,15.)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def stair_model():
    tree=ET.parse(EXPORTED/'scene.xml');xml=tree.getroot()
    xml.find('compiler').set('meshdir',str(EXPORTED/'assets'))
    ET.SubElement(xml.find('asset'),'texture',name='homepage_white_sky',type='skybox',builtin='flat',rgb1='1 1 1',rgb2='1 1 1',width='256',height='1536')
    model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
    configure_lighting(model,'terrain')
    model.geom_rgba[model.geom('parkour_ground').id]=[.91,.93,.95,1]
    return model

def render_stairs(out,preview=False):
    model=stair_model();data=mujoco.MjData(model)
    with np.load(STAIRS/'final_student_4004_trajectory.npz') as f:q=f['qpos'];fps=float(f['fps'])
    frames=int(np.ceil(len(q)/fps*30));ids=np.minimum(np.rint(np.arange(frames)*fps/30).astype(int),len(q)-1)
    targets=[]
    for pose in q[ids]:
        data.qpos[:]=pose;mujoco.mj_forward(model,data);targets.append(data.xpos[model.body('robot/pelvis').id].copy())
    targets=gaussian_filter1d(np.array(targets),4.5,axis=0);targets[:,2]-=.05
    cam=mujoco.MjvCamera();cam.azimuth=55;cam.elevation=-24;cam.distance=3.65
    opt=mujoco.MjvOption();opt.sitegroup[:]=0;opt.geomgroup[3:]=0
    selected=[84,174,263] if preview else range(frames)
    proc=None;images=[]
    if not preview:
        proc=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','-','-an','-c:v','ffv1','-threads','2',str(out/'native-stairs.master.mkv')],stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model,height=1080,width=1920) as renderer:
            for i in selected:
                data.qpos[:]=q[ids[i]];mujoco.mj_forward(model,data)
                cam.lookat[:]=targets[i];light_direction(model,cam);renderer.update_scene(data,camera=cam,scene_option=opt);frame=renderer.render()
                if preview:images.append(Image.fromarray(frame).resize((640,360)))
                else:
                    proc.stdin.write(frame.tobytes())
                    if i==84:Image.fromarray(frame).save(out/'native-stairs-full.webp',quality=93)
    finally:
        if proc:proc.stdin.close();assert proc.wait()==0
    if preview:
        sheet=Image.new('RGB',(1920,360),'white')
        for i,im in enumerate(images):sheet.paste(im,(i*640,0))
        sheet.save(out/'native-stairs-preview.jpg',quality=93);return
    full=out/'native-stairs-full.mp4';encode(out/'native-stairs.master.mkv',full,frames=frames)
    meta=dict(id='native-stairs-full',name='MimicX native stair ascent',source=str((STAIRS/'final_student_4004_trajectory.npz').relative_to(ROOT)),
              source_sha256=sha(STAIRS/'final_student_4004_trajectory.npz'),scene=str((STAIRS/'native_scene.mjb').relative_to(ROOT)),scene_sha256=sha(STAIRS/'native_scene.mjb'),
              replay_scene=str((EXPORTED/'scene.xml').relative_to(ROOT)),replay_scene_sha256=sha(EXPORTED/'scene.xml'),
              report=str((STAIRS/'final_student_4004.json').relative_to(ROOT)),report_sha256=sha(STAIRS/'final_student_4004.json'),
              master=str((out/'native-stairs.master.mkv').relative_to(WEB)),source_fps=fps,source_frames=len(q),source_recording_start_s=.02,
              fps=30,resolution=[1920,1080],speed=1,pose_interpolation=False,spatial_retargeting=False,all_terrain_geometry_retained=True,
              source_duration_s=len(q)/fps,endpoint_quantization_s=frames/30-len(q)/fps,
              full_recorded_episode=True,partial_excerpt=False,duration_s=frames/30,output_frames=frames,video_sha256=sha(full),
              public_sha256='898d1a02829f901fbad4ed07178df2b06778620b54e8bccf5e07ffcb4c31d32d',
              scope='Complete recorded robot-0 trajectory of the accepted MimicX stair student, seed 4004, native scene. Camera, lighting and ground color only. Native geometry and all actual qpos retained. No simulator stepping, pose interpolation, prop editing or scene relocation during replay.')
    full.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--preview',action='store_true');args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if not args.preview:assert not (out/'scene-preview.mp4').exists() and not (out/'native-stairs.master.mkv').exists()
    render_stairs(out,args.preview)
    if args.preview:return
    parts=[];chapters=[];cursor=0
    for name,a,b in CHOICES:
        meta=json.loads((out/'native-stairs-full.json' if name=='native-stairs' else WEB/'public/media'/f'{name}.json').read_text())
        part=out/f'{name}-hero.mp4';count=round((b-a)*30)
        encode(WEB/meta['master'],part,frames=count,start=a)
        full='native-stairs-full.mp4' if name=='native-stairs' else meta['full_video']
        chapters.append(dict(id=name,start_s=cursor/30,end_s=(cursor+count)/30,source=meta['source'],source_sha256=meta['source_sha256'],scene=meta['scene'],scene_sha256=sha(ROOT/meta['scene']),report=meta['report'],report_sha256=sha(ROOT/meta['report']),excerpt_start_s=a,excerpt_end_s=b,full_video=full,native_scene_retained=True,racket_present=name=='tennis'))
        cursor+=count;parts.append(part)
    listing=out/'concat.txt';listing.write_text(''.join(f"file '{p}'\n" for p in parts))
    target=out/'scene-preview.mp4';subprocess.run(['ffmpeg','-v','error','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(target)],check=True)
    subprocess.run(['ffmpeg','-v','error','-ss','1','-i',str(target),'-frames:v','1','-quality','93',str(out/'scene-preview.webp')],check=True)
    meta=dict(id='scene-preview',name='Four skills in their native scenes',native_scene_highlights=True,edited_montage=True,selected_excerpts=True,
              separate_recordings=True,single_rendered_scene=False,continuous_episode=False,continuous_policy_rollout=False,animated_transitions=False,
              pose_interpolation=False,spatial_retargeting=False,prop_visibility_edits=False,burned_in_labels=False,racket_present_only_in_tennis=True,
              camera_cuts=3,fps=30,resolution=[1920,1080],speed=1,output_frames=cursor,duration_s=cursor/30,video_sha256=sha(target),chapters=chapters,
              public_sha256='898d1a02829f901fbad4ed07178df2b06778620b54e8bccf5e07ffcb4c31d32d',
              scope='Four edited excerpts from separate actual policy recordings in each task’s own native scene: skateboard, rough high-step terrain, MimicX stair ascent and ATHLETE tennis. Same frozen public Actor with task-specific experts/adapters. No shared course, continuous-episode claim, artificial walking, scene relocation or prop hiding. Tennis shows three continuous incoming-ball cycles including an unsuccessful return; no opponent policy. Complete recordings are linked per chapter.')
    (out/'scene-preview.json').write_text(json.dumps(meta,indent=2)+'\n');print('Complete native-scene highlights:',cursor/30,'seconds',flush=True)
if __name__=='__main__':main()

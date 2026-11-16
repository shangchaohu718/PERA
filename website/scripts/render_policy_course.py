"""Camera-only render of one real physical course, without pose interpolation."""
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
from scipy.ndimage import gaussian_filter1d
from PIL import Image,ImageDraw
from studio_lighting import configure_lighting,light_direction

ROOT=Path(__file__).resolve().parents[2];FPS=30
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def s5(u):u=np.clip(u,0,1);return u*u*u*(10+u*(-15+6*u))

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--preview',action='store_true');p.add_argument('--highlight-props',action='store_true',help='Render the four highlight intervals, showing the racket only in tennis');args=p.parse_args()
    run=args.run.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    report=json.loads((run/'report.json').read_text());f=np.load(run/'physics.npz')
    if not args.preview:
        audit=json.loads((run/'physical-replay-audit.json').read_text())
        assert report['success'] and audit['passed'] and audit['source_sha256']==digest(run/'physics.npz')
        assert not (out/'scene-preview.mp4').exists()
    tree=ET.parse(run/'course.xml');xml=tree.getroot()
    ET.SubElement(xml.find('asset'),'texture',name='website_white_sky',type='skybox',builtin='flat',rgb1='1 1 1',rgb2='1 1 1',width='256',height='1536')
    model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
    original=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'))
    assert model.nq==original.nq and model.nv==original.nv
    for name in ['body_mass','body_pos','body_quat','jnt_qposadr','geom_pos','geom_size']:
        np.testing.assert_allclose(getattr(model,name),getattr(original,name),atol=1e-10,rtol=0)
    configure_lighting(model,'skate')
    total=len(f['qpos']);count=int(np.ceil(total/50*FPS));ids=np.minimum(np.rint(np.arange(count)*50/FPS).astype(int),total-1)
    q=f['qpos'][ids];stage=f['stage'][ids]
    targets=gaussian_filter1d(q[:,:3],FPS*.22,axis=0);targets[:,2]-=.04
    distance=np.full(count,3.65);distance[stage=='skate']=3.9;distance[stage=='stairs']=4.0;distance[stage=='tennis']=3.2
    distance=gaussian_filter1d(distance,FPS*.35)
    angle=np.full(count,125.)
    # C2 camera moves are purely visual. Robot/object states are untouched.
    for name,dest in [('walk_rough',100.),('stairs',65.),('walk_court',105.),('tennis',125.)]:
        found=np.flatnonzero(stage==name)
        if not len(found):continue
        start=found[0];old=angle[max(0,start-1)];u=(np.arange(count)-start)/(FPS*2.0)
        angle[start:]=old+s5(u[start:])*(dest-old)
    camera=mujoco.MjvCamera();camera.elevation=-22
    opt=mujoco.MjvOption();opt.sitegroup[:]=0;opt.geomgroup[3:]=0
    data=mujoco.MjData(model);proc=None;frames=[]
    render_ids=np.arange(count)
    if args.highlight_props:
        from cut_policy_highlights import SELECTION
        render_ids=np.concatenate([np.arange(a,b) for _,a,b in SELECTION])
    selected=render_ids[np.unique(np.rint(np.linspace(0,len(render_ids)-1,24)).astype(int))] if args.preview else render_ids
    racket=model.geom('racket_visual').id
    if not args.preview:
        proc=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','slow','-crf','21','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(out/'scene-preview.mp4')],stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model,height=1080,width=1920) as renderer:
            for frame_no,i in enumerate(selected):
                data.qpos[:]=q[i];mujoco.mj_forward(model,data)
                if args.highlight_props:model.geom_rgba[racket,3]=1 if stage[i]=='tennis' else 0
                camera.lookat[:]=targets[i];camera.distance=float(distance[i]);camera.azimuth=float(angle[i]);light_direction(model,camera)
                renderer.update_scene(data,camera=camera,scene_option=opt);frame=renderer.render()
                if args.preview:frames.append((i,Image.fromarray(frame).resize((480,270))))
                else:
                    proc.stdin.write(frame.tobytes())
                    if frame_no==30:Image.fromarray(frame).save(out/'scene-preview.webp',quality=93)
                    if frame_no%180==0:print(f'{frame_no}/{len(render_ids)}',flush=True)
    finally:
        if proc:proc.stdin.close();assert proc.wait()==0
    if args.preview:
        sheet=Image.new('RGB',(1440,2400),'white');draw=ImageDraw.Draw(sheet)
        for j,(i,frame) in enumerate(frames):
            x=j%3*480;y=j//3*300;sheet.paste(frame,(x,y));draw.text((x+10,y+276),f'{i/FPS:.2f}s {stage[i]}',fill='black')
        sheet.save(out/'contact-sheet.jpg',quality=92)
    elif args.highlight_props:
        # Same physical samples and camera as the audited complete recording;
        # edit only prop visibility in the independently cut highlight film.
        meta=json.loads((ROOT/'website/public/media/scene-preview.json').read_text())
        assert meta['policy_course_highlights'] and meta['source_sha256']==digest(run/'physics.npz')
        assert meta['output_frames']==len(render_ids)
        meta.update(video_sha256=digest(out/'scene-preview.mp4'),racket_visible_only_in_tennis=True,
                    visual_prop_changes='Racket mesh and its shadow hidden in non-tennis highlight clips only. Original recorded robot/object states, mounted racket inertia and collision physics remain unchanged. No pickup animation is added; tennis begins at a hard cut.',
                    complete_source_preserves_original_prop_visibility=True)
        (out/'scene-preview.json').write_text(json.dumps(meta,indent=2)+'\n')
        np.save(out/'source-frame-indices.npy',ids[render_ids]);print('Rendered racket visibility fix:',len(render_ids)/FPS,'seconds',flush=True)
    else:
        chapters=[]
        for name in dict.fromkeys(stage):
            rows=np.flatnonzero(stage==name);chapters.append(dict(id=str(name),start_s=float(rows[0]/FPS),end_s=float((rows[-1]+1)/FPS)))
        meta=dict(id='scene-preview',name='Four-scene physical policy course',physical_policy_course=True,policy_executed=True,
                  design_previsualization=False,continuous_policy_rollout=True,continuous_episode=True,single_rendered_scene=True,camera_cuts=0,
                  animated_transitions=False,spatially_retargeted_recorded_motions=False,pose_interpolation=False,burned_in_labels=False,
                  source=str((run/'physics.npz').relative_to(ROOT)),source_sha256=digest(run/'physics.npz'),report=str((run/'report.json').relative_to(ROOT)),report_sha256=digest(run/'report.json'),
                  physical_audit=str((run/'physical-replay-audit.json').relative_to(ROOT)),physical_audit_sha256=digest(run/'physical-replay-audit.json'),
                  scene_sha256=digest(run/'scene.mjb'),video_sha256=digest(out/'scene-preview.mp4'),duration_s=count/FPS,output_frames=count,
                  fps=FPS,resolution=[1920,1080],speed=1,source_fps=50,actor_instances=1,robot_state_writes_after_start=0,robot_resets_after_start=0,
                  teacher_action_queries=0,training_updates=0,attention_cache_resets_after_start=report['attention_cache_resets_after_start'],
                  public_sha256=report['public_sha256'],plugins=report['plugins'],chapters=chapters,tennis_cycles=report['tennis_cycles'],
                  scope=report['scope'],rendering='Camera/lighting-only nearest-frame replay. All robot, board and ball poses come from one physical source trajectory. No world retargeting, pose interpolation, cuts, animated dismount or animated pickup.')
        (out/'scene-preview.json').write_text(json.dumps(meta,indent=2)+'\n');np.save(out/'source-frame-indices.npy',ids)
        print('Rendered',count/FPS,'seconds',flush=True)

if __name__=='__main__':main()

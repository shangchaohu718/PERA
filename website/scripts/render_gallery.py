#!/usr/bin/env python3
"""Render archived closed-loop G1 states. Never runs training or generates poses.

Run with the existing .venvs/instinct-supermimic/bin/python. Originals remain
untouched; lossless masters live in .render-work, web exports in public/media.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image
from studio_lighting import configure_lighting, light_direction

WEB = Path(__file__).resolve().parents[1]
ROOT = WEB.parent
CONFIG = json.loads((WEB / 'scripts/render_config.json').read_text())
MODEL = ROOT / 'HoloMotion/assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.xml'


def digest(p):
    return hashlib.file_digest(open(p, 'rb'), 'sha256').hexdigest()


def studio_model():
    tree = ET.parse(MODEL)
    root = tree.getroot()
    root.find('compiler').set('meshdir', str((MODEL.parent / '../meshes').resolve()))
    for tex in root.findall('.//texture'):
        if tex.get('type') == 'skybox':
            tex.set('rgb1', '0.980392 0.980392 0.980392')
            tex.set('rgb2', '0.980392 0.980392 0.980392')
    for mat in root.findall('.//material'):
        if mat.get('name') == 'groundplane':
            mat.attrib.pop('texture', None)
            mat.set('rgba', '0.92 0.92 0.92 1')
            mat.set('reflectance', '0')
            mat.set('emission', '0.19')
    for visual in root.findall('visual'):
        visual.find('rgba').set('haze', '0.980392 0.980392 0.980392 1')
        visual.find('headlight').set('ambient', '0.24 0.24 0.24')
        visual.find('headlight').set('diffuse', '0.35 0.35 0.35')
        visual.find('headlight').set('specular', '0.2 0.2 0.2')
    for floor in root.findall('.//geom[@name="floor"]'):
        floor.set('group', '2')
    for light in root.findall('.//light'):
        light.set('diffuse', '0.5 0.5 0.5')
        light.set('ambient', '0.05 0.05 0.05')
        light.set('pos', '-2 -3 5')
        light.set('dir', '0.12 0.16 -1')
        light.set('castshadow', 'true')
    ET.SubElement(root.findall('worldbody')[-1], 'light', name='website_fill', directional='true', dir='0.2 0.2 -1', castshadow='false')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    model.vis.global_.offwidth = CONFIG['width']
    model.vis.global_.offheight = CONFIG['height']
    model.vis.quality.shadowsize = 4096
    model.stat.extent = 3.5
    return configure_lighting(model)


def render(clip, preview=False):
    matches = list((ROOT / clip['directory']).glob(clip['pattern']))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one recorded trajectory: {clip['id']}")
    source = matches[0]
    with np.load(source, allow_pickle=False) as f:
        xyz = f['robot_global_translation'].copy()
        quat = f['robot_global_rotation_quat'][:, 0].copy()  # xyzw
        joints = f['robot_dof_pos'].copy()
    model = studio_model()
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    q = quat[0]
    yaw = np.degrees(np.arctan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2)))
    cam.azimuth = yaw + CONFIG['camera']['azimuth_relative_to_initial_heading']
    cam.elevation = CONFIG['camera']['elevation']
    cam.distance = clip.get('camera_distance',CONFIG['camera']['distance'])
    cam.lookat[:] = [*xyz[0, 0, :2], CONFIG['camera']['lookat_height']]
    # Matched camera follows the reference root, not the failing/successful controller.
    with np.load(source, allow_pickle=False) as f:
        reference = f['ref_global_translation'][:, 0].copy()
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    options.geomgroup[0] = 0
    output = WEB / 'public/media' / clip['id']
    work = WEB / '.render-work'
    work.mkdir(exist_ok=True)
    frames = np.minimum(np.rint(np.arange(int(np.ceil(len(joints)/clip['source_fps']*CONFIG['fps']))) * clip['source_fps']/CONFIG['fps']).astype(int), len(joints)-1)
    selected = [int(len(joints)*0.45)] if preview else frames
    master = work / (clip['id'] + '.master.mkv')
    if not preview and (master.exists() or output.with_suffix('.mp4').exists()):
        raise FileExistsError(f"Refusing to overwrite existing render: {clip['id']}")
    command = ['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f"{CONFIG['width']}x{CONFIG['height']}",'-r',str(CONFIG['fps']),'-i','-','-an','-c:v','ffv1','-threads','2',str(master)]
    proc = None if preview else subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model, height=CONFIG['height'], width=CONFIG['width']) as renderer:
            for n,i in enumerate(selected):
                data.qpos[:3] = xyz[i,0]
                data.qpos[3:7] = quat[i][[3,0,1,2]]
                data.qpos[7:] = joints[i]
                light_direction(model,cam)
                mujoco.mj_forward(model,data)
                cam.lookat[:2] = reference[i,:2]
                model.stat.center[:] = [*reference[i,:2], 0.6]
                renderer.update_scene(data,camera=cam,scene_option=options)
                frame = renderer.render()
                if preview or n == min(30, len(selected)-1):
                    Image.fromarray(frame).save(output.with_suffix('.webp'), quality=92)
                if proc:
                    proc.stdin.write(frame.tobytes())
                if n % 150 == 0:
                    print(f"{clip['id']} {n}/{len(selected)}",flush=True)
    finally:
        if proc:
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError('Lossless master encoding failed')
    if preview:
        return
    subprocess.run(['ffmpeg','-v','error','-i',str(master),'-an','-c:v','libx264','-preset','slow','-crf','22','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(output.with_suffix('.mp4'))], check=True)
    manifest = dict(clip, source=str(source.relative_to(ROOT)), source_sha256=digest(source), model=str(MODEL.relative_to(ROOT)), model_sha256=digest(MODEL), source_frames=len(joints), output_frames=len(frames), duration_s=len(frames)/CONFIG['fps'], fps=CONFIG['fps'], resolution=[CONFIG['width'],CONFIG['height']], controller_rate_hz=clip['source_fps'], pose_interpolation=False, speed=1, full_recorded_episode=True, video_sha256=digest(output.with_suffix('.mp4')), lighting='Studio v3: white studio floor and sky; bright camera fill plus directional key; original robot albedo retained.', rendering='Archived Isaac Lab closed-loop robot states re-rendered in MuJoCo; no policy inference or reference-pose playback. Nearest-frame 50-to-30 Hz sampling, maximum time quantization 10 ms. Camera, lighting and ground appearance only.', camera={**CONFIG['camera'], 'distance':cam.distance}, master=str(master.relative_to(WEB)))
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f"Complete: {clip['id']}",flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', choices=[c['id'] for c in CONFIG['clips']])
    parser.add_argument('--preview', action='store_true')
    args=parser.parse_args()
    for clip in CONFIG['clips']:
        if args.clip is None or args.clip == clip['id']:
            render(clip,args.preview)

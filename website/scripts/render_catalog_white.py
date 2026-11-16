"""White-stage replay of archived native states; no physics or pose edits.

Use .venvs/mimicx/bin/python for the original MuJoCo 3.5 binary scenes.
"""
from pathlib import Path
import argparse
import os
import json
import math
import subprocess
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from studio_lighting import configure_lighting, light_direction
from build_expert_catalog import ROOT, WEB, finalize_record, digest

OUT = WEB / 'public/media/catalog31-white-v2'
SUITE = ROOT / 'reports/distillation_suite_v1/run_20261105_173125'


def pmt_scene(report):
    robot = ROOT / 'HoloMotion/assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.xml'
    xml = ET.parse(robot).getroot()
    xml.find('compiler').set('meshdir', str((robot.parent / '../meshes').resolve()))
    # The native task uses only its supplied terrain mesh, not the robot XML's
    # optional studio plane. Preserve the mesh coordinates and triangle faces.
    for parent in xml.iter():
        for geom in list(parent):
            if geom.tag == 'geom' and geom.get('type') == 'plane':
                parent.remove(geom)
    terrain = Path(json.loads(report.read_text())['terrain'])
    ET.SubElement(xml.find('asset'), 'mesh', name='pmt_native_terrain_mesh', file=str(terrain))
    ET.SubElement(xml.find('worldbody'), 'geom', name='pmt_native_terrain',
                  type='mesh', mesh='pmt_native_terrain_mesh', group='2',
                  contype='0', conaffinity='0', rgba='.66 .75 .83 1')
    scene = report.parent / 'white-scene.xml'
    scene.write_text(ET.tostring(xml, encoding='unicode'))
    return mujoco.MjModel.from_xml_path(str(scene)), scene


def white_stage(model):
    textures = model.tex_data.copy()
    configure_lighting(model, 'studio')
    # Preserve object textures and robot colors; recolor only the stage.
    for i in range(model.ntex):
        if model.tex_type[i] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            continue
        if not any(word in model.tex(i).name.lower() for word in ['ground', 'floor']):
            start = int(model.tex_adr[i])
            size = int(model.tex_height[i] * model.tex_width[i] * model.tex_nchannel[i])
            model.tex_data[start:start + size] = textures[start:start + size]
    for i in range(model.nmat):
        if any(word in model.mat(i).name.lower() for word in ['ground', 'floor']):
            model.mat_rgba[i] = [1, 1, 1, 1]
            model.mat_reflectance[i] = 0
            model.mat_emission[i] = .14
            model.mat_texid[i] = -1
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            mat = int(model.geom_matid[i])
            model.geom_rgba[i] = [1, 1, 1, 1]
            if mat >= 0:
                model.mat_texid[mat] = -1
                model.mat_rgba[mat] = [1, 1, 1, 1]
                model.mat_reflectance[mat] = 0
                model.mat_emission[mat] = .14
        if int(model.geom_bodyid[i]) == 0:
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
                model.geom_rgba[i] = [1, 1, 1, 1]
            elif model.geom(i).name.startswith('parkour_'):
                model.geom_rgba[i] = [.63, .73, .82, 1]
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    model.vis.quality.shadowsize = 2048
    return model


def render(task, mode, preview=False, bright=False):
    output_dir = WEB / 'public/media/gallery-bright-v3' if bright else OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    folder = SUITE / task / 'attempt_1'
    role = 'student' if mode == 'on' else 'public'
    if task == 'mimicx_stairs':
        folder = ROOT / 'reports/mimicx_stairs_512_v1/run_20261106_083527/training'
        source = folder / f'final_{role}_4004_trajectory.npz'
        report = folder / f'final_{role}_4004.json'
        scene = folder / 'native_scene.mjb'
        model = mujoco.MjModel.from_binary_path(str(scene))
    elif task == 'velocity_walk':
        folder = WEB / '.render-work/catalog31-v1' / task / mode
        source = folder / 'physics.npz'
        report = folder / 'report.json'
        scene = folder / 'scene.mjb'
        model = mujoco.MjModel.from_binary_path(str(scene))
    elif task == 'pmt_optimized_scene0':
        folder = WEB / '.render-work/catalog31-white-v2' / task / mode
        source = folder / 'physics.npz'
        report = folder / 'report.json'
        model, scene = pmt_scene(report)
    elif task == 'ultra_largebox':
        source = folder / f'final_{role}_trajectory.npz'
        report = folder / f'final_{role}.json'
        scene = ROOT / 'third_party/ULTRA/ultra/data/assets/merge/largebox.xml'
        model = mujoco.MjModel.from_xml_path(str(scene))
    else:
        source = folder / f'final_{role}_4004_trajectory.npz'
        report = folder / f'final_{role}_4004.json'
        scene = folder / 'native_scene.mjb'
        model = mujoco.MjModel.from_binary_path(str(scene))
    if task == 'velocity_walk':
        configure_lighting(model, 'terrain')
        model.vis.global_.offwidth = 1280
        model.vis.global_.offheight = 720
        model.vis.quality.shadowsize = 2048
    else:
        white_stage(model)
    base_light = [model.vis.headlight.ambient.copy(), model.vis.headlight.diffuse.copy(), model.vis.headlight.specular.copy()]
    bright_light = [np.full(3, .62), np.full(3, .85), np.full(3, .10)]
    with np.load(source) as f:
        if 'qpos' in f:
            poses = f['qpos'].copy()
            fps = float(f['fps'])
        else:
            poses = f['states'][:, :model.nq].copy()
            fps = float(f['control_hz']) / float(f['control_step_stride'])
        if task == 'pmt_optimized_scene0':
            names = f['joint_names'].tolist()
            order = [names.index(model.joint(i).name) for i in range(1, model.njnt)]
            poses[:, 7:] = poses[:, 7:][:, order]
    assert poses.shape[1] == model.nq and np.isfinite(poses).all()
    target = output_dir / f'{task}-{mode}.mp4'
    if target.with_suffix('.json').exists() and not preview:
        return
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    is_stair = task in ['mimicx_descent', 'mimicx_bench', 'mimicx_forest', 'mimicx_stairs']
    cam.azimuth = 55 if is_stair else 135
    cam.elevation = -24 if task == 'mimicx_stairs' else -21
    cam.distance = 3.5 if is_stair else 3.1
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    options.geomgroup[3:] = 0
    if task == 'pmt_optimized_scene0':
        options.geomgroup[0] = 0  # Duplicate collision meshes, not native terrain.
    pelvis = next(i for i in range(model.nbody) if model.body(i).name in ['pelvis', 'robot/pelvis'])
    robot_geoms = []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        while body and body != pelvis:
            body = int(model.body_parentid[body])
        if body == pelvis:
            robot_geoms.append(geom)
    count = math.ceil(len(poses) / fps * 30)
    indices = np.minimum(np.rint(np.arange(count) / 30 * fps).astype(int), len(poses) - 1)
    if preview:
        indices = [min(round(.7 * fps), len(poses) - 1)]
    process = None if preview else subprocess.Popen([
        'ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '1280x720', '-r', '30', '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '22', '-pix_fmt', 'yuv420p', '-threads', '2',
        '-movflags', '+faststart', str(target)], stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model, height=720, width=1280) as renderer:
            for n, i in enumerate(indices):
                data.qpos[:] = poses[i]
                mujoco.mj_forward(model, data)
                cam.lookat[:] = data.xpos[pelvis]
                # Keep the complete fall in view, including the contact plane.
                cam.lookat[2] = max(.36, float(cam.lookat[2]))
                if task == 'pmt_optimized_scene0':
                    model.stat.extent = 4
                    model.stat.center[:] = cam.lookat
                if task == 'ultra_largebox':
                    cam.lookat[:2] = (data.xpos[pelvis, :2] + data.xpos[model.body('largebox').id, :2]) / 2
                light_direction(model, cam)
                renderer.update_scene(data, camera=cam, scene_option=options)
                frame = renderer.render().copy()
                robot_frame = None
                if bright:
                    for name, value in zip(['ambient', 'diffuse', 'specular'], bright_light):
                        getattr(model.vis.headlight, name)[:] = value
                    renderer.update_scene(data, camera=cam, scene_option=options)
                    robot_frame = renderer.render().copy()
                    for name, value in zip(['ambient', 'diffuse', 'specular'], base_light):
                        getattr(model.vis.headlight, name)[:] = value
                # Original native MJB scenes have no skybox. Identify empty sky
                # with the renderer's exact object-ID buffer, never robot pixels.
                renderer.enable_segmentation_rendering()
                renderer.update_scene(data, camera=cam, scene_option=options)
                segmentation = renderer.render()
                renderer.disable_segmentation_rendering()
                if robot_frame is not None:
                    robot_mask = np.isin(segmentation[:, :, 0], robot_geoms) & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                    frame[robot_mask] = robot_frame[robot_mask]
                frame[segmentation[:, :, 1] < 0] = 255
                if preview:
                    from PIL import Image
                    preview_dir = WEB / 'qa/catalog31/white-state-previews'
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(frame).save(preview_dir / f'{task}-{mode}{"-bright" if bright else ""}-preview.png')
                else:
                    process.stdin.write(frame.tobytes())
                if n % 300 == 0:
                    print(task, mode, n, '/', len(indices), flush=True)
    finally:
        if process:
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('Encoding failed')
    if not preview:
        finalize_record(task, mode, target, source, report, {
            'id': str(target.relative_to(WEB / 'public/media').with_suffix('')),
            'scene': str(scene.relative_to(ROOT)), 'scene_sha256': digest(scene),
            'background_style': 'white-studio', 'all_native_geometry_retained': True,
            'lighting_profile': 'bright-robot-fill-v3' if bright else 'studio-v2',
            'headlight_ambient': bright_light[0].tolist() if bright else base_light[0].tolist(),
            'headlight_diffuse': bright_light[1].tolist() if bright else base_light[1].tolist(),
            'robot_only_lighting_pass': bright,
            'rendering': 'Exact archived actual qpos, nearest sample at original speed. Native scene and failure retained. Camera, lighting and stage color only; no physics stepping or pose interpolation.',
            'resets_after_initialization': 0})
        print('Complete', target.name, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task')
    p.add_argument('--preview', action='store_true')
    p.add_argument('--bright', action='store_true')
    args = p.parse_args()
    tasks = [args.task] if args.task else [f'mimicx_{name}' for name in ['descent', 'bench', 'forest', 'football', 'tennis', 'dance', 'kungfu']] + ['ultra_largebox']
    for task in tasks:
        for mode in ['on', 'off']:
            render(task, mode, args.preview, args.bright)

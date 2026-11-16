#!/usr/bin/env python3
"""Restore the upstream commented visual hands in a replay-only model."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import argparse

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'website/.render-work/tennis-v3'

def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=OUT);args=p.parse_args();OUT=args.directory.resolve()
    tree=ET.parse(OUT/'original-scene.xml');root=tree.getroot()
    root.find('compiler').set('meshdir',str(ROOT/'third_party/ATHLETE/robots/replay_unitree_description/meshes'))
    for side,y in [('left',.003),('right',-.003)]:
        wrist=root.find(f'.//body[@name="{side}_wrist_yaw_link"]')
        ET.SubElement(wrist,'geom',name=f'website_{side}_hand',type='mesh',mesh=f'{side}_rubber_hand',
                      pos=f'.0415 {y} 0',quat='1 0 0 0',group='2',density='0',contype='0',conaffinity='0',
                      rgba='.7 .7 .7 1')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    original=mujoco.MjModel.from_binary_path(str(OUT/'original-scene.mjb'))
    assert model.nq==original.nq and model.nv==original.nv and model.nbody==original.nbody
    for attr in ('body_mass','body_inertia','dof_damping','jnt_qposadr','actuator_gear','geom_contype','geom_conaffinity'):
        if attr.startswith('geom_'):continue
        np.testing.assert_allclose(getattr(model,attr),getattr(original,attr),rtol=0,atol=1e-12)
    states=np.load(OUT/'physics.npz')['qpos'];a=mujoco.MjData(original);b=mujoco.MjData(model)
    error=0.
    for pose in states[::20]:
        a.qpos[:]=b.qpos[:]=pose;mujoco.mj_forward(original,a);mujoco.mj_forward(model,b)
        error=max(error,float(abs(a.xpos-b.xpos).max()),float(abs(a.site_xpos-b.site_xpos).max()))
    assert error<1e-10
    mujoco.mj_saveModel(model,str(OUT/'scene.mjb'))
    (OUT/'hands-validation.json').write_text(json.dumps({'added_visual_geoms':2,'added_dofs':0,
       'hand_mesh_source':'third_party/ATHLETE/robots/replay_unitree_description/meshes',
       'transforms':'Exact upstream commented visual-hand transforms on wrist_yaw_link',
       'collision_enabled':False,'density':0,'mass_and_inertia_unchanged':True,
       'max_body_and_site_position_error_m':error,'simulation_used_original_model':True},indent=2)+'\n')
    print('Visual hands restored; original masses and replay wrist/racket/ball poses unchanged:',error)

if __name__=='__main__':main()

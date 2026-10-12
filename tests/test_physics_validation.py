"""R18验收器的CPU单元测试。

XML均为明示的人工结构样本；mjModel布局用命名空间模拟。没有运行MuJoCo求解器。
真实模型编译和步进只通过 tools/verify_physics.py 单独记录，不与这里的通过数量混淆。
"""
from dataclasses import replace
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from unified_motion.envs import physics_validation as v
from unified_motion.envs.mjlab_k1 import read_k1_joint_specs, k1_control_profile, make_k1_mjlab_cfg
from test_mjlab_k1 import create_fixture_xml, ROBOT


ROOT = Path(__file__).resolve().parents[1]


class PhysicsAssetValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.xml = self.base/'K1_22dof.xml'
        create_fixture_xml(self.xml)
        root = ET.parse(self.xml).getroot()
        ET.SubElement(root.find('worldbody'), 'geom', name='ground', type='plane')
        asset = ET.SubElement(root, 'asset')
        ET.SubElement(asset, 'mesh', name='test_mesh', file='test.STL')
        root.find('compiler').set('meshdir', 'meshes')
        ET.ElementTree(root).write(self.xml)
        (self.base/'meshes').mkdir()
        # 任意字节只能用来测试“文件存在及指纹”，不是合法STL，不会拿去编译。
        (self.base/'meshes/test.STL').write_bytes(b'NOT A PHYSICS MESH; UNIT TEST BYTES')

    def edit(self, function):
        root = ET.parse(self.xml).getroot(); function(root); ET.ElementTree(root).write(self.xml)

    def test_manifest_and_no_source_mutation(self):
        original = self.xml.read_bytes()
        report = v.audit_k1_assets(self.base)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(len(report['controls']), 22)
        self.assertEqual(len(report['files']), 2)
        self.assertEqual(self.xml.read_bytes(), original)
        self.assertEqual(len(report['fingerprint']), 64)

    def test_missing_mesh_is_blocked_not_passed(self):
        (self.base/'meshes/test.STL').unlink()
        report = v.audit_k1_assets(self.base)
        self.assertEqual(report['status'], 'BLOCKED')
        self.assertEqual(report['missing_files'], ['meshes/test.STL'])

    def test_empty_mesh_is_not_validated(self):
        (self.base/'meshes/test.STL').write_bytes(b'')
        self.assertEqual(v.audit_k1_assets(self.base)['status'], 'BLOCKED')

    def test_mesh_change_changes_fingerprint(self):
        before = v.audit_k1_assets(self.base)['fingerprint']
        (self.base/'meshes/test.STL').write_bytes(b'DIFFERENT TEST BYTES')
        self.assertNotEqual(before, v.audit_k1_assets(self.base)['fingerprint'])

    def test_outside_model_directory_rejected(self):
        self.edit(lambda r: r.find('./asset/mesh').set('file', '../../outside.STL'))
        with self.assertRaisesRegex(ValueError, '越过'):
            v.audit_k1_assets(self.base)

    def test_ground_required(self):
        self.edit(lambda r: r.find('worldbody').remove(r.find('./worldbody/geom')))
        with self.assertRaisesRegex(ValueError, 'ground'):
            v.audit_k1_assets(self.base)

    def test_no_implicit_angle_unit(self):
        self.edit(lambda r: r.remove(r.find('compiler')))
        with self.assertRaisesRegex(ValueError, 'radian'):
            read_k1_joint_specs(self.xml)

    def test_degree_unit_is_rejected(self):
        self.edit(lambda r: r.find('compiler').set('angle', 'degree'))
        with self.assertRaisesRegex(ValueError, 'radian'):
            read_k1_joint_specs(self.xml)

    def test_include_is_not_silently_ignored(self):
        self.edit(lambda r: ET.SubElement(r, 'include', file='other.xml'))
        with self.assertRaisesRegex(ValueError, 'include'):
            read_k1_joint_specs(self.xml)

    def test_child_free_joint_is_not_root(self):
        def change(r):
            trunk = r.find('./worldbody/body'); free = trunk.find('joint'); trunk.remove(free)
            ET.SubElement(trunk, 'body', name='fake_floating_part').append(free)
        self.edit(change)
        with self.assertRaisesRegex(ValueError, 'trunk'):
            read_k1_joint_specs(self.xml)

    def test_disabled_joint_limit_rejected(self):
        self.edit(lambda r: r.findall('./worldbody/body/joint')[1].set('limited', 'false'))
        with self.assertRaisesRegex(ValueError, '关闭'):
            read_k1_joint_specs(self.xml)

    def test_disabled_effort_limit_rejected(self):
        self.edit(lambda r: r.find('./actuator/motor').set('forcelimited', 'false'))
        with self.assertRaisesRegex(ValueError, '关闭'):
            read_k1_joint_specs(self.xml)

    def test_unknown_inheritance_rejected(self):
        self.edit(lambda r: r.find('./worldbody/body').set('childclass', 'other'))
        with self.assertRaisesRegex(ValueError, '继承'):
            v.audit_k1_assets(self.base)

    def test_zero_action_preserves_legal_reset_pose(self):
        def change(r):
            for j in r.findall('./worldbody/body/joint'):
                if 'knee_pitch' in j.get('name',''):
                    j.set('range', '0 2.321')
        self.edit(change)
        report = v.audit_k1_assets(self.base)
        self.assertEqual(report['zero_action_changes'], [])
        self.assertEqual(report['warnings'], [])
        knees = [row for row in report['controls'] if 'knee_pitch' in row['name']]
        self.assertEqual(len(knees), 2)
        for knee in knees:
            self.assertEqual(knee['default_pos'], 0.0)
            self.assertEqual(knee['zero_target'], 0.0)
            self.assertEqual(knee['soft_lower'], 0.0)

    def test_control_profile_formula_and_joint_permutation(self):
        specs = read_k1_joint_specs(self.xml)
        rows = k1_control_profile(specs)
        opposite = k1_control_profile(tuple(reversed(specs)))
        self.assertEqual(rows, tuple(reversed(opposite)))
        for c in rows:
            self.assertAlmostEqual(c.kp*c.action_scale, .25*c.effort)
            self.assertLessEqual(c.soft_lower, c.zero_target)
            self.assertLessEqual(c.zero_target, c.soft_upper)

    def test_invalid_initial_pose_fails_not_clamped_silently(self):
        specs = read_k1_joint_specs(self.xml)
        specs = tuple(replace(s, lower=0., upper=.1) if s.name=='left_shoulder_roll_joint' else s for s in specs)
        with self.assertRaisesRegex(ValueError, '默认角度'):
            k1_control_profile(specs)

    def test_boolean_physics_dt_rejected_without_loading_sim(self):
        with self.assertRaises(ValueError):
            make_k1_mjlab_cfg(self.base, physics_dt=True)


class ValidationFlowTests(unittest.TestCase):
    def test_compatible_version_ranges(self):
        for name, version, accepted in [('mjlab','1.6.0',True),('mjlab','1.7.0',False),
            ('mujoco','3.11.0',True),('mujoco','3.11.1',True),('mujoco-warp','3.10.0',False),
            ('mujoco-warp','3.12.0',False),('warp-lang','1.14.0',True),('warp-lang','1.13.0',False),
            ('mjlab','1.6.0rc1',False)]:
            with self.subTest(name=name, version=version):
                self.assertEqual(v._compatible(name,version),accepted)

    def test_missing_dependencies_blocked(self):
        with patch.object(v.metadata, 'version', side_effect=metadata.PackageNotFoundError):
            report = v.dependency_report()
        self.assertEqual(report['status'],'BLOCKED')
        self.assertEqual(len(report['packages']),4)

    def test_installed_and_importable_only_passes_version_check(self):
        versions={'mjlab':'1.6.0','mujoco':'3.11.0','mujoco-warp':'3.11.0','warp-lang':'1.14.0'}
        with patch.object(v.metadata,'version',side_effect=versions.__getitem__), patch.object(v.importlib,'import_module'):
            self.assertEqual(v.dependency_report()['status'],'PASS')

    def test_incompatible_or_broken_package_is_fail(self):
        for broken in (False,True):
            with patch.object(v.metadata,'version',return_value='1.6.0'), patch.object(v.importlib,'import_module',
                side_effect=RuntimeError('failed shared library') if broken else None):
                self.assertEqual(v.dependency_report()['status'],'FAIL')

    def test_blocked_dependencies_do_not_call_solver(self):
        with patch.object(v,'dependency_report',return_value=v._record('dependencies','BLOCKED','missing')), \
             patch.object(v,'audit_k1_assets',return_value=v._record('assets','PASS','fixture')), \
             patch.object(v,'compile_k1') as compile_fn, patch.object(v,'native_probe') as probe:
            report=v.run_validation(phase='native',asset_root='fixture')
        self.assertEqual(report['exit_code'],2)
        compile_fn.assert_not_called(); probe.assert_not_called()

    def test_assets_phase_does_not_require_or_import_sim(self):
        with patch.object(v,'audit_k1_assets',return_value=v._record('assets','PASS','fixture')), \
             patch.object(v,'dependency_report') as deps:
            report=v.run_validation(phase='assets',asset_root='fixture')
        self.assertEqual(report['exit_code'],0);deps.assert_not_called()
        self.assertIn('native',report['not_requested'])

    def test_error_status_not_swallowed(self):
        with patch.object(v,'audit_k1_assets',side_effect=ValueError('bad xml')):
            report=v.run_validation(phase='assets',asset_root='fixture')
        self.assertEqual(report['status'],'FAIL');self.assertEqual(report['exit_code'],1)

    def test_missing_assets_blocked(self):
        report=v.run_validation(phase='assets')
        self.assertEqual(report['status'],'BLOCKED');self.assertEqual(report['exit_code'],2)

    def test_compile_failure_does_not_run_native(self):
        with patch.object(v,'dependency_report',return_value=v._record('dependencies','PASS','mock')), \
             patch.object(v,'audit_k1_assets',return_value=v._record('assets','PASS','mock')), \
             patch.object(v,'compile_k1',side_effect=ValueError('bad gains')), \
             patch.object(v,'native_probe') as probe:
            report=v.run_validation(phase='native',asset_root='mock')
        self.assertEqual(report['status'],'FAIL');probe.assert_not_called()

    def test_cuda_without_device_is_blocked_not_cpu_fallback(self):
        with patch('torch.cuda.is_available',return_value=False):
            self.assertEqual(v.cuda_probe('/unused')['status'],'BLOCKED')

    def test_invalid_phase_or_budget_rejected(self):
        for kwargs in ({'phase':'typo'},{'phase':'dependencies','steps':0},
                       {'phase':'dependencies','steps':True}):
            with self.assertRaises(ValueError): v.run_validation(**kwargs)

    def test_cli_writes_real_blocked_report_and_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'report.json'
            code=v.main(['--phase','assets','--asset-root',str(Path(folder)/'absent'),'--output',str(output)])
            report=json.loads(output.read_text())
            self.assertEqual(code,2);self.assertEqual(report['status'],'BLOCKED')
            self.assertFalse(output.with_suffix('.json.tmp').exists())

    def test_cli_help_no_physics(self):
        r=subprocess.run([sys.executable,'tools/verify_physics.py','--help'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('native',r.stdout)


class CompiledLayoutMathTests(unittest.TestCase):
    """用人工 model 字段验证地址/PD 审计逻辑，明确不声称进行了实际模型编译。"""
    def fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'K1_22dof.xml';create_fixture_xml(path)
            controls=k1_control_profile(read_k1_joint_specs(path))
        n=22;by_name={c.name:c for c in controls}
        names=list(reversed(ROBOT.joint_names))
        mj=SimpleNamespace(mjtTrn=SimpleNamespace(mjTRN_JOINT=0),
            mjtJoint=SimpleNamespace(mjJNT_HINGE=3),mjtGain=SimpleNamespace(mjGAIN_FIXED=0),
            mjtBias=SimpleNamespace(mjBIAS_AFFINE=1),
            mjtObj=SimpleNamespace(mjOBJ_JOINT=0,mjOBJ_BODY=1,mjOBJ_KEY=2),
            mj_id2name=lambda model,obj,j: 'robot/'+ROBOT.joint_names[j-1],
            mj_name2id=lambda model,obj,name: 1 if name=='robot/trunk' else 0)
        jids=[ROBOT.joint_names.index(name)+1 for name in names]
        m=SimpleNamespace(nq=29,nv=28,nu=n,actuator_trntype=np.zeros(n),
            actuator_trnid=np.array([[j,-1] for j in jids]),jnt_type=np.array([0]+[3]*n),
            actuator_gear=np.tile([1,0,0,0,0,0],(n,1)),actuator_gaintype=np.zeros(n),
            actuator_biastype=np.ones(n),actuator_gainprm=np.zeros((n,10)),
            actuator_biasprm=np.zeros((n,10)),actuator_forcelimited=np.ones(n),
            actuator_forcerange=np.zeros((n,2)),jnt_qposadr=np.array([0]+list(range(7,29))),
            jnt_dofadr=np.array([0]+list(range(6,28))),key_qpos=np.zeros((1,29)))
        for i,name in enumerate(names):
            c=by_name[name];m.actuator_gainprm[i,0]=c.kp;m.actuator_biasprm[i,:3]=[0,-c.kp,-c.kd]
            m.actuator_forcerange[i]=[-c.effort,c.effort]
        for j,c in enumerate(controls):m.key_qpos[0,j+7]=c.default_pos
        return mj,m,controls,names

    def test_actuator_order_is_resolved_by_joint(self):
        mj,m,c,names=self.fixture()
        with patch.dict(sys.modules,{'mujoco':mj}): result=v._model_control_layout(m,c)
        self.assertEqual(result['names'],names)
        self.assertEqual(result['qpos_addresses'],list(reversed(range(7,29))))

    def test_wrong_compiled_gain_rejected(self):
        mj,m,c,_=self.fixture();m.actuator_gainprm[0,0]*=2
        with patch.dict(sys.modules,{'mujoco':mj}), self.assertRaisesRegex(ValueError,'Kp'):
            v._model_control_layout(m,c)

    def test_wrong_compiled_damping_sign_rejected(self):
        mj,m,c,_=self.fixture();m.actuator_biasprm[0,2]*=-1
        with patch.dict(sys.modules,{'mujoco':mj}), self.assertRaisesRegex(ValueError,'Kd'):
            v._model_control_layout(m,c)

    def test_unlimited_actuator_rejected(self):
        mj,m,c,_=self.fixture();m.actuator_forcelimited[0]=0
        with patch.dict(sys.modules,{'mujoco':mj}), self.assertRaisesRegex(ValueError,'力矩'):
            v._model_control_layout(m,c)

    def test_joint_initial_pose_mismatch_rejected(self):
        mj,m,c,_=self.fixture();m.key_qpos[0,7]=3
        with patch.dict(sys.modules,{'mujoco':mj}), self.assertRaisesRegex(ValueError,'默认'):
            v._model_control_layout(m,c)

    def test_compile_applies_sim_options_not_just_scene_options(self):
        scene=SimpleNamespace(compile=lambda: 'model')
        calls=[]
        cfg=SimpleNamespace(scene='scene_cfg',sim=SimpleNamespace(mujoco=SimpleNamespace(apply=calls.append)))
        with patch.dict(sys.modules,{'mjlab.scene':SimpleNamespace(Scene=lambda *a,**k:scene)}), \
             patch('unified_motion.envs.mjlab_physics.make_k1_physics_cfg',return_value=cfg):
            _,_,model=v._compile_scene('/fixture')
        self.assertEqual(model,'model');self.assertEqual(calls,['model'])

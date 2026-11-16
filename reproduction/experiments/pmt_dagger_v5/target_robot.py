"""Load canonical HoloMotion nominal robot parameters into the PMT task."""

import ast
import copy
import importlib.util
import json
import sys

from pera.paths import ROOT


def configure_target_robot(cfg):
    asset = ROOT / "integrations/HoloMotion/assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.urdf"
    cfg.scene.robot.spawn = copy.deepcopy(cfg.scene.robot.spawn)
    cfg.scene.robot.spawn.asset_path = str(asset)
    cache = ROOT / "reports/pmt_migration_v1/usd_cache/holo_unitree"
    cache.mkdir(parents=True, exist_ok=True)
    cfg.scene.robot.spawn.usd_dir = str(cache)
    cfg.scene.robot.spawn.merge_fixed_joints = True
    cfg.scene.robot.spawn.root_link_name = "pelvis"
    actuator_source = (
        ROOT / "integrations/HoloMotion/holomotion/src/env/isaaclab_components/unitree_actuators.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pmt_migration_holo_unitree_actuators", actuator_source
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    source = (
        ROOT / "integrations/HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_scene.py"
    )
    assignments = [
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "unitree_actuator_config_hardcoded"
            for t in n.targets
        )
    ]
    assert len(assignments) == 1
    assignment = assignments[0].value
    assert isinstance(assignment, ast.Dict) and len(assignment.values) == 1
    call = assignment.values[0]
    assert (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "UnitreeActuatorCfg"
    )
    parameters = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    parameters.update(min_delay=0, max_delay=0, erfi_enabled=False)
    cfg.scene.robot.actuators = {"all_joints": module.UnitreeErfiActuatorCfg(**parameters)}
    configs = json.loads(
        json.dumps({k: v.to_dict() for k, v in cfg.scene.robot.actuators.items()}, default=str)
    )
    previous = json.loads((ROOT / "reports/pmt_migration_v1/unitree_v3/report.json").read_text())
    if configs != previous["motor_configs"]:
        raise ValueError(
            "Teacher test must use the same nominal motor configuration as student migration"
        )
    if str(asset) != previous["robot_asset"]:
        raise ValueError("Teacher test robot differs from student migration")
    return cfg, configs

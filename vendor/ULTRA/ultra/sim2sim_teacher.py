
import argparse
import sys

import os
import time
import numpy as np

# Check if we need EGL for headless rendering before importing mujoco
if '--record_video' in sys.argv:
    os.environ['MUJOCO_GL'] = 'egl'

import mujoco
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R

from rl_games.algos_torch import torch_ext
from learning import ultra_network_builder, ultra_models
import torch
import trimesh
from utils.obs import MujocoObs, compute_sdf, _gather_dof_slices
import xml.etree.ElementTree as ET
from utils import torch_utils_mujoco

def _randn_like(x, std):
    """Add zero-mean Gaussian noise with per-element std."""
    if std == 0:
        return x
    return x + torch.randn_like(x) * std
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    quat = torch.stack([x, y, z, w], dim=-1).view(shape)
    return quat
def _perturb_quat(q, std):
    """Apply small-angle axis-angle noise then renormalise."""
    if std == 0:
        return q
    # random unit axes
    q = q.unsqueeze(0)
    axis = torch.randn_like(q[..., :3])
    axis = axis / torch.norm(axis, dim=-1, keepdim=True).clamp_(min=1e-6)
    angle = torch.randn(q.shape[0], 1, device=q.device) * std  # radians
    half  = 0.5 * angle
    delta = torch.cat([axis * torch.sin(half), torch.cos(half)], dim=-1)
    qn    = quat_mul(delta, q)
    return (qn / torch.norm(qn, dim=-1, keepdim=True)).squeeze(0)


# _DEFAULT_NOISE_STD = {
#     "root_pos"     : 0.05,  # m
#     "root_rot"     : 0.04,  # rad (axis–angle magnitude)
#     "dof_pos"      : 0.20,  # rad or m
#     "root_vel"     : 0.05,  # m/s
#     "root_ang_vel" : 0.05,  # rad/s
#     "dof_vel"      : 0.05,  # rad/s or m/s
#     "target_pos"   : 0.05,  # m
#     "target_rot"   : 0.05,  # rad
# }

_DEFAULT_NOISE_STD = {
    "root_pos"     : 0.0,  # m
    "root_rot"     : 0.0,  # rad (axis–angle magnitude)
    "dof_pos"      : 0.0,  # rad or m
    "root_vel"     : 0.0,  # m/s
    "root_ang_vel" : 0.0,  # rad/s
    "dof_vel"      : 0.0,  # rad/s or m/s
    "target_pos"   : 0.0,  # m
    "target_rot"   : 0.0,  # rad
}



def to_torch(x, device=None, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.as_tensor(x, dtype=dtype, device=device)

def _first_or_new(parent, tag):
    node = parent.find(tag)
    if node is None:
        node = ET.SubElement(parent, tag)
    return node

def merge_mjcf(humanoid_xml, object_xml, out_xml):
    """
    - Appends object's <asset> (rewriting mesh 'file' to absolute paths to avoid meshdir surprises)
    - Appends object's top-level <worldbody>/<body> into humanoid <worldbody>
    - Appends object's <default> children (keeps your friction class)
    """
    h_tree, o_tree = ET.parse(humanoid_xml), ET.parse(object_xml)
    h_root, o_root = h_tree.getroot(), o_tree.getroot()

    # --- make the humanoid mesh directory absolute so the merged file can live anywhere
    h_compiler = h_root.find('compiler')
    if h_compiler is not None and 'meshdir' in h_compiler.attrib and not os.path.isabs(h_compiler.attrib['meshdir']):
        h_compiler.set('meshdir', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(humanoid_xml)), h_compiler.attrib['meshdir'])))

    # --- asset merge
    h_asset = _first_or_new(h_root, 'asset')
    o_asset = o_root.find('asset')
    if o_asset is not None:
        base = os.path.dirname(os.path.abspath(object_xml))
        for child in list(o_asset):
            if child.tag == 'mesh' and 'file' in child.attrib:
                fp = child.attrib['file']
                if not os.path.isabs(fp):
                    child.set('file', os.path.normpath(os.path.join(base, fp)))
            h_asset.append(child)

    # --- default merge (preserve object default class for friction, etc.)
    o_default = o_root.find('default')
    if o_default is not None:
        h_default = _first_or_new(h_root, 'default')
        for child in list(o_default):
            h_default.append(child)

    # --- worldbody merge
    h_world = _first_or_new(h_root, 'worldbody')
    o_world = o_root.find('worldbody')
    if o_world is not None:
        # Bring each top-level body over (e.g., <body name="largebox" ...>)
        existing_names = {b.get('name') for b in h_world.findall('body')}
        for body in o_world.findall('body'):
            name = body.get('name', 'object')
            if name in existing_names:
                body.set('name', f"{name}_obj")
            h_world.append(body)

    # Write combined file
    os.makedirs(os.path.dirname(os.path.abspath(out_xml)), exist_ok=True)
    h_tree.write(out_xml, encoding='utf-8', xml_declaration=True)
    return out_xml


def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="jit"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint


def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw = quat[0] 
    qx = quat[1] 
    qy = quat[2]
    qz = quat[3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        eulerVec[1] = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    
    return eulerVec


# -------- helpers --------
def xyzw_to_wxyz(q):
    if isinstance(q, torch.Tensor):
        return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)

def _joint_id_by_name_or_body(model, name: str, require_free: bool = True) -> int:
    """Return joint id by joint name; if not found, try body name → its freejoint."""
    # force plain str (numpy.str_, torch scalar, etc. → str)
    name = str(name)

    # Cast enum to int for pybind
    OBJ_JOINT = int(mujoco.mjtObj.mjOBJ_JOINT)
    OBJ_BODY  = int(mujoco.mjtObj.mjOBJ_BODY)
    JNT_FREE  = int(mujoco.mjtJoint.mjJNT_FREE)

    jid = mujoco.mj_name2id(model, OBJ_JOINT, name)
    if jid >= 0:
        if require_free and model.jnt_type[jid] != JNT_FREE:
            raise ValueError(f"Joint '{name}' exists but is not FREE.")
        return jid

    # Try as a body name: find a FREE joint attached to that body
    bid = mujoco.mj_name2id(model, OBJ_BODY, name)
    if bid >= 0:
        adr = model.body_jntadr[bid]
        n   = model.body_jntnum[bid]
        for k in range(n):
            j = adr + k
            if model.jnt_type[j] == JNT_FREE:
                return j
        raise ValueError(f"Body '{name}' has no FREE joint.")
    raise ValueError(f"Neither joint nor body named '{name}' found.")

def _set_freejoint_state(model, data, joint_or_body_name, pos_xyz, quat_xyzw, lin_vel_xyz, ang_vel_xyz):
    """
    Correctly sets a MuJoCo free joint's state using WORLD-frame inputs.
    """
    jid = _joint_id_by_name_or_body(model, joint_or_body_name, require_free=True)
    qpos_adr = model.jnt_qposadr[jid]
    qvel_adr =  model.jnt_dofadr[jid]

    # Set position and orientation (qpos)
    p = pos_xyz.detach().cpu().numpy().astype(np.float64)
    q = xyzw_to_wxyz(quat_xyzw.detach().cpu().numpy().astype(np.float64))
    data.qpos[qpos_adr:qpos_adr+7] = np.concatenate([p, q])

    # CRITICAL: Run mj_forward to update the model's kinematics (like body orientations)
    # This ensures data.body().xmat is correct for the velocity transformation.
    mujoco.mj_forward(model, data)

    # --- Set Velocity (qvel) ---
    # Linear velocity is already in the world frame and can be set directly.
    data.qvel[qvel_adr:qvel_adr+3] = lin_vel_xyz.detach().cpu().numpy().astype(np.float64)
    
    # Angular velocity from Isaac Gym is in the WORLD frame.
    world_ang_vel = ang_vel_xyz.detach().cpu().numpy().astype(np.float64)

    # Get the body's rotation matrix (which was just updated by mj_forward).
    body_id = model.jnt_bodyid[jid]
    body_rot_mat = data.body(body_id).xmat.reshape(3, 3)

    # Transform WORLD angular velocity to the body's LOCAL frame.
    # v_local = R_transpose * v_world
    local_ang_vel = body_rot_mat.T @ world_ang_vel

    # Set the rotational part of qvel with the correct LOCAL velocity.
    data.qvel[qvel_adr+3:qvel_adr+6] = local_ang_vel
    # print(joint_or_body_name, world_ang_vel, local_ang_vel)
        
# Order for the 29 human hinge joints (matches your XML; tweak if your training order differs)
JOINT_NAMES_29 = [
    # Left leg
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
    "left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
    # Right leg
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
    "right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
    # Waist
    "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
    # Left arm
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
    # Right arm
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
]

def _write_hinge_qpos_qvel_by_names(model, data, joint_names, dof_pos, dof_vel, clamp_to_range=False):
    dof_pos = dof_pos.view(-1)  # (29,)
    dof_vel = dof_vel.view(-1)  # (29,)
    assert dof_pos.numel() == len(joint_names) and dof_vel.numel() == len(joint_names), \
        f"Expected {len(joint_names)} dofs, got {dof_pos.numel()}/{dof_vel.numel()}"

    for k, jn in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"Joint '{jn}' not found in model.")
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE, f"Joint '{jn}' not hinge."
        qpos_adr = model.jnt_qposadr[jid]
        qvel_adr = model.jnt_dofadr[jid]
        # print(jid, qpos_adr, qvel_adr)
        q = float(dof_pos[k].detach().cpu().item())
        dq = float(dof_vel[k].detach().cpu().item())

        if clamp_to_range and model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            q = np.clip(q, lo, hi)

        data.qpos[qpos_adr] = q
        data.qvel[qvel_adr] = dq
        
class HumanoidEnv:
    def __init__(self, policy_path, robot_type="g1_29dof", device="cuda", record_video=False, motion_path=None, use_jit=False):
        self.robot_type = robot_type
        self.device = device
        self.record_video = record_video
        self.motion_path = motion_path
        self.use_jit = use_jit
        human_path = f"ultra/data/assets/g1/g1_29dof.xml"
        object_name = motion_path.split('/')[-1].split('_')[1]
        object_path = f"ultra/data/assets/objects/{object_name}.xml"
        self._load_object(object_name, object_path)
        self.object_name = object_name
        self._load_motion(motion_path)
        os.makedirs("ultra/data/assets/merge/", exist_ok=True)
        model_path = f"ultra/data/assets/merge/{object_name}.xml"
        merge_mjcf(human_path, object_path, f"ultra/data/assets/merge/{object_name}.xml")

        self.stiffness = np.array([
            150, 150, 200, 200, 20, 20,
            150, 150, 200, 200, 20, 20,
            200, 200, 200,
            40, 40, 40, 40, 20, 20, 20,
            40, 40, 40, 40, 20, 20, 20,
        ], dtype=np.float32)
        self.damping = np.array([
            5, 5, 5, 5, 4, 4,
            5, 5, 5, 5, 4, 4,
            5, 5, 5,
            10, 10, 10, 10, 0.5, 0.5, 0.5,
            10, 10, 10, 10, 0.5, 0.5, 0.5,
        ], dtype=np.float32)

        self.stiffness = np.array([40.179238, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
                        40.179238, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
                        40.179238, 28.501246, 28.501246,
                        14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327,
                        14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327])

        # Kd (damping) per DOF
        self.damping = np.array([2.557890, 6.308802, 2.557890, 6.308802, 1.814446, 1.814446,
                2.557890, 6.308802, 2.557890, 6.308802, 1.814446, 1.814446,
                2.557890, 1.814446, 1.814446,
                0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142,
                0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142])
        # armature per DOF (NOTE: feet + waist roll/pitch are doubled vs ARMATURE_5020)
        armature = [0.010178, 0.025102, 0.010178, 0.025102, 0.007219, 0.007219,
                    0.010178, 0.025102, 0.010178, 0.025102, 0.007219, 0.007219,
                    0.010178, 0.007219, 0.007219,
                    0.003610, 0.003610, 0.003610, 0.003610, 0.003610, 0.004250, 0.004250,
                    0.003610, 0.003610, 0.003610, 0.003610, 0.003610, 0.004250, 0.004250]

        # effort limits per DOF (used for dof_prop["effort"] and for torque_limits)
        effort = [88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
                88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
                88.0, 50.0, 50.0,
                25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
                25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0]

        self.control_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]) # all dofs
        self.num_actions = len(self.control_indices)
        self.num_dofs = 29
        self.default_dof_pos = np.array([
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # left leg (6)
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # right leg (6)
            0.0, 0.0, 0.0,  # torso (1)
            0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
        ])
        self.torque_limits = np.array([88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
                88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
                88.0, 50.0, 50.0,
                25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
                25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0])
        self.cycle_time = 0.64

        self.torque_limits = self.torque_limits.astype(np.float32) * 0.8
        
        self.obs_indices = np.arange(self.num_dofs)
        
        if self.record_video:
            self.sim_duration = 10.0
        else:
            self.sim_duration = 60.0
        self.sim_dt = 1.0 / (60.0 * 17.0)
        self.sim_decimation = 17
        # self.sim_dt = 0.001
        # self.sim_decimation = 20
        self.control_dt = self.sim_dt * self.sim_decimation
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)
        if self.record_video:
            # Use MuJoCo's native offscreen rendering instead of mujoco_viewer
            self.viewer = None
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            # Set initial camera parameters
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = 0  # Track the root body (pelvis)
            self.camera.distance = 5.0
            self.camera.elevation = -20
            self.camera.azimuth = 140
        else:
            # self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer.cam.distance = 5.0
        
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.action_scale = 0.3
        
        self.n_priv = 0

        self.n_proprio = 3 + 2 + 3*self.num_dofs
        self.n_priv_latent = 4 + 1 + 2*self.num_dofs + 3
            
        self.history_len = 10
        
        self.dof_pos_scale = 1.0
        # self.dof_vel_scale = 0.05
        self.dof_vel_scale = 1.0
        # self.ang_vel_scale = 0.25
        self.ang_vel_scale = 1.0
        
        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
            
        self.commands = np.zeros(3)
        
        self.set_commands()
        self.obs_builder = MujocoObs(
            self.model, object_name, self.max_episode_length, self.hoi_data, self.object_points, history_step=10
        )

        self.policy_path = policy_path

        # Load policy based on use_jit flag
        if self.use_jit:
            print(f"Loading JIT model from: {policy_path}")
            self.policy_jit = torch.jit.load(policy_path, map_location=self.device)
            self.policy_jit.eval()
            print("✓ JIT model loaded successfully")
            # JIT model doesn't need running_mean/running_var (already included)
            self.policy = None
            self.running_mean = None
            self.running_var = None
        else:
            print(f"Loading checkpoint from: {policy_path}")
            num_envs = 1
            config = {
                'actions_num' : 29,
                'input_shape' : (4052, ),
                'num_seqs' : num_envs,
                'value_size': 1,
            }
            network = ultra_network_builder.UltraBuilder()
            params = {
                "model": {
                    "name": "ultra"
                },
                "network": {
                    "name": "ultra",
                    "separate": True,
                    "space": {
                        "continuous": {
                            "mu_activation": "None",
                            "sigma_activation": "None",
                            "mu_init": {
                                "name": "default"
                            },
                            "sigma_init": {
                                "name": "const_initializer",
                                "val": -2.9
                            },
                            "fixed_sigma": True,
                            "learn_sigma": False
                        }
                    },
                    "mlp": {
                        "units": [1024, 1024, 512],
                        "activation": "relu",
                        "d2rl": False,
                        "initializer": {
                            "name": "default"
                        },
                        "regularizer": {
                            "name": "None"
                        }
                    }
                }
            }
            network.load(params['network'])
            network = ultra_models.ModelUltraContinuous(network)
            ck = torch_ext.load_checkpoint(self.policy_path)
            policy = network.build(config)
            policy.to(self.device)
            policy.load_state_dict(ck['model'])
            self.policy = policy
            running_mean, running_var = ck['running_mean_std']['running_mean'], ck['running_mean_std']['running_var']
            self.running_mean = running_mean
            self.running_var = running_var
            self.policy_jit = None
            print("✓ Checkpoint loaded successfully")

    def _load_motion(self, data_path):
        loaded_dict = {}
        hoi_data = torch.load(data_path)
        hoi_data_expand = torch.zeros((hoi_data.shape[0]*2, hoi_data.shape[1]), device=hoi_data.device)
        hoi_data_expand[0:hoi_data.shape[0]] = hoi_data
        hoi_data_expand[hoi_data.shape[0]:] = hoi_data[-1]
        hoi_data = hoi_data_expand
        # hoi_data[5:] = hoi_data[4:5]
        loaded_dict['hoi_data'] = hoi_data.detach().to('cpu')

        self.max_episode_length = loaded_dict['hoi_data'].shape[0]
        self.fps_data = new_fps = 60.

        loaded_dict['root_pos'] = loaded_dict['hoi_data'][:, 0:3].clone()
        loaded_dict['root_pos_vel'] = loaded_dict['hoi_data'][:, 7:10].clone()

        loaded_dict['root_rot'] = loaded_dict['hoi_data'][:, 3:7]

        loaded_dict['root_rot_vel'] = loaded_dict['hoi_data'][:, 10:13]

        loaded_dict['dof_pos'] = loaded_dict['hoi_data'][:, 13:42]

        loaded_dict['dof_pos_vel'] = loaded_dict['hoi_data'][:, 42:71]

        loaded_dict['body_pos'] = loaded_dict['hoi_data'][:, 84:201]
        loaded_dict['body_pos_vel'] = loaded_dict['hoi_data'][:, 357:474]

        loaded_dict['obj_pos'] = loaded_dict['hoi_data'][:, 71:74]

        loaded_dict['obj_pos_vel'] = loaded_dict['hoi_data'][:, 78:81]

        loaded_dict['obj_rot'] = loaded_dict['hoi_data'][:, 74:78]
        loaded_dict['obj_rot_vel'] = loaded_dict['hoi_data'][:, 81:84]

        obj_rot_extend = loaded_dict['obj_rot'].unsqueeze(1).repeat(1, self.object_points.shape[0], 1).view(-1, 4)
        object_points_extend = self.object_points.unsqueeze(0).repeat(loaded_dict['obj_rot'].shape[0], 1, 1).view(-1, 3)
        obj_points = torch_utils_mujoco.quat_rotate(obj_rot_extend, object_points_extend).view(loaded_dict['obj_rot'].shape[0], self.object_points.shape[0], 3) + loaded_dict['obj_pos'].unsqueeze(1)
        key_body_pose = loaded_dict['body_pos'][:,:].clone()
        ref_ig = compute_sdf(key_body_pose.view(loaded_dict['obj_rot'].shape[0],-1,3), obj_points).view(-1, 3)
        heading_rot = torch_utils_mujoco.calc_heading_quat_inv(loaded_dict['root_rot'])
        heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pose.shape[1] // 3, 1).view(-1, 4)
        ref_ig = torch_utils_mujoco.quat_rotate(heading_rot_extend, ref_ig).view(loaded_dict['obj_rot'].shape[0], -1)    
        loaded_dict['contact_parts'] = loaded_dict['hoi_data'][:, 591:630].clone()

        loaded_dict['human_rot'] = loaded_dict['hoi_data'][:, 201:357]
        loaded_dict['human_rot_vel'] = loaded_dict['hoi_data'][:, 474:591]
        loaded_dict['hoi_data'] = torch.cat([loaded_dict['hoi_data'], ref_ig], dim=-1)

        self.hoi_data_dict = loaded_dict
        self.hoi_data = loaded_dict['hoi_data']
        
        self.hoi_ref = torch.cat((
                            loaded_dict['root_pos'].clone(), # 0:3
                            loaded_dict['root_rot'].clone(), # 3:7
                            loaded_dict['root_pos_vel'].clone(), # 7:10
                            loaded_dict['root_rot_vel'].clone(), # 10:13
                            loaded_dict['dof_pos'].clone(), # 13:42
                            loaded_dict['dof_pos_vel'].clone(), # 42:71
                            loaded_dict['obj_pos'].clone(), # 71:74
                            loaded_dict['obj_rot'].clone(), # 74:78
                            loaded_dict['obj_pos_vel'].clone(), # 78:81
                            loaded_dict['obj_rot_vel'].clone(), # 81:84
                            ),dim=-1)

        return

    def _load_object(self, object_name, object_path): # smplx
        asset_root = "ultra/data/assets/objects/"
        asset_file = object_name + ".xml"
        obj_file = asset_root + 'objects/' + object_name + '/' + object_name + '.obj'
        

        mesh_obj = trimesh.load(obj_file, process=False, force='mesh')
        obj_verts = mesh_obj.vertices
        center = np.mean(obj_verts, 0)
        object_points, object_faces = trimesh.sample.sample_surface_even(mesh_obj, count=256, seed=2024)

        object_points = to_torch(object_points - center)
        

        while object_points.shape[0] < 256:
            object_points = torch.cat([object_points, object_points[:256 - object_points.shape[0]]], dim=0)
        self.object_points = to_torch(object_points)
        return
        
    def set_commands(self):
        self.commands[0] = 0.0
        self.commands[1] = 0.0
        self.commands[2] = 0.0
        
    def extract_data(self):
        dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs:]
        dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs:]
        quat = self.data.sensor('orientation').data.astype(np.float32)
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)
        self.dof_vel = torch.from_numpy(dof_vel).float().unsqueeze(0).to(self.device)
        return (dof_pos, dof_vel, quat, ang_vel)

    # -------- main reset (MuJoCo equivalent of your Isaac reset) --------
    def reset_human_and_object_from_ref(self, motion_t=0,
                                            human_freejoint_name="pelvis",
                                            clamp_to_range=False):
            """
            Sets:
            human root pose/vel from hoi_refs[..., 0:13]
            human hinge joints qpos/qvel from hoi_refs[..., 13:42] and 42:71
            object pose/vel from hoi_refs[..., 71:84] (+ noise on pos/rot like your Isaac _reset_target)
            """
            noise = _DEFAULT_NOISE_STD
            object_freejoint_name=f"{self.object_name}_free"
            # 1) Reset to keyframe 'home' (keeps limbs in sane default) or full reset
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if key_id >= 0: mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            else:           mujoco.mj_resetData(self.model, self.data)
            # 2) Slice references (torch on the right device)
            ref = self.hoi_ref[motion_t]
            if not isinstance(ref, torch.Tensor):
                ref = torch.as_tensor(ref, dtype=torch.float32, device=self.device)
            else:
                ref = ref.to(self.device, torch.float32)
            root_pos      = ref[0:3]
            root_rot_xyzw = ref[3:7]
            root_vel      = torch.zeros_like(ref[7:10])
            root_ang_vel  = torch.zeros_like(ref[10:13])
            dof_pos       = ref[13:42]    # 29
            dof_vel       = torch.zeros_like(ref[42:71])    # 29
            root_pos_new = _randn_like(root_pos, noise["root_pos"])
            root_pos_new[..., 2:3] = (root_pos[..., 2:3] + 0.01)
            root_pos = root_pos_new
            root_rot_xyzw = _perturb_quat(root_rot_xyzw, noise["root_rot"])
            root_vel = _randn_like(root_vel, noise["root_vel"])
            root_ang_vel = _randn_like(root_ang_vel, noise["root_ang_vel"])
            dof_pos = _randn_like(dof_pos, noise["dof_pos"])
            dof_vel = _randn_like(dof_vel, noise["dof_vel"])
            # dof_pos = torch.clamp(dof_pos, min=self.dof_limits_lower, max=self.dof_limits_upper)
            obj_pos       = ref[71:74]
            obj_rot_xyzw  = ref[74:78]
            obj_pos_new  = _randn_like(obj_pos, noise["target_pos"])
            obj_rot_xyzw  = _perturb_quat(obj_rot_xyzw, noise["target_rot"])
            obj_pos[..., :2] = obj_pos_new[..., :2]
            obj_lin       = torch.zeros_like(ref[78:81])
            obj_ang       = torch.zeros_like(ref[81:84])
            # 3) Human: set pelvis freejoint, then all hinge joints
            _set_freejoint_state(self.model, self.data, human_freejoint_name,
                                root_pos, root_rot_xyzw, root_vel, root_ang_vel)
            _write_hinge_qpos_qvel_by_names(self.model, self.data, JOINT_NAMES_29, dof_pos, dof_vel,
                                            clamp_to_range=clamp_to_range)
            _set_freejoint_state(self.model, self.data, object_freejoint_name,
                                obj_pos, obj_rot_xyzw, obj_lin, obj_ang)
            # 5) Finalize
            mujoco.mj_forward(self.model, self.data)
        
    def run(self):
        cnt = 0
        self.reset_human_and_object_from_ref()
        
        desired_rtf = getattr(self, "desired_rtf", 1)      # 1.0 = real-time, 0.5 = half-speed (slow motion), 2.0 = 2x speed
        max_render_fps = getattr(self, "max_render_fps", 60)  # limit onscreen draw rate
        start_wall = time.perf_counter()
        last_render = start_wall
        min_render_dt = 1.0 / max_render_fps
        if self.record_video:
            import imageio
            video_name = f"{self.robot_type}_{''.join(os.path.basename(self.policy_path).split('.')[:-1])}.mp4"
            path = f"./logs/mujoco_videos/"
            if not os.path.exists(path):
                os.makedirs(path)
            video_name = os.path.join(path, video_name)
            # mp4_writer = imageio.get_writer(video_name)
            mp4_writer = imageio.get_writer(
                "out.mp4",
                format="FFMPEG",
                mode="I",
                codec="libx264",
                fps=60,                  # pick your sim/video rate
                pixelformat="yuv420p",   # maximum compatibility
                quality=8,
                macro_block_size=None    # keep your resolution, don’t auto-resize
            )

        self.last_action = torch.zeros((1, 29))
        self.last_torque = torch.zeros((1, 29))
        for i in tqdm(range(int(self.sim_duration / self.sim_dt)), desc="Running simulation..."):
            dof_pos, dof_vel = _gather_dof_slices(
                self.data, self.obs_builder.dof_qpos_ids, self.obs_builder.dof_qvel_ids
            )
            dof_pos = dof_pos.squeeze(0).numpy()
            dof_vel = dof_vel.squeeze(0).numpy()
            if i % self.sim_decimation == 0:
                timestep = i // self.sim_decimation
                if timestep >= self.hoi_data.shape[0] - 1:
                    break
                # print(timestep)
                obs_tensor = self.obs_builder._compute_observations(self.data, timestep, self.last_action, self.last_torque, student_obs=False, episode_length=timestep).to(self.device)
                with torch.no_grad():
                    if self.use_jit:
                        # JIT model: pass raw observation directly (normalization is built-in)
                        teacher_action = self.policy_jit(obs_tensor).squeeze(0)  # (29)
                        raw_action = teacher_action.cpu().numpy()
                    else:
                        # Checkpoint model: manually normalize observation
                        curr_obs = ((obs_tensor - self.running_mean.float().to(self.device)) / torch.sqrt(self.running_var.float().to(self.device) + 1e-05))
                        curr_obs = torch.clamp(curr_obs, min=-5.0, max=5.0)
                        self.policy.eval()
                        input_dict = {
                            'is_train': False,
                            'prev_actions': None,
                            'obs' : curr_obs,
                            'rnn_states' : None
                        }
                        res_dict = self.policy(input_dict)
                        teacher_action = torch.clamp(res_dict['mus'], min=-1.0, max=1.0).squeeze(0)  # (29)
                        raw_action = teacher_action.cpu().numpy()
                
                self.last_action = to_torch(raw_action).unsqueeze(0)
                # raw_action = np.clip(raw_action, -10, 10)
                # scaled_actions = raw_action * self.action_scale
                scaled_actions = raw_action * 3.0
                step_actions = np.zeros(self.num_dofs)
                step_actions[self.control_indices] = scaled_actions
                # step_actions[[0,6]] /= 1.1
                # step_actions[[1,7]] /= 1.1176
                # step_actions[[3,9]] -= 0.5
                
                pd_target = step_actions + self.default_dof_pos * 0.
                # print("Actions: "   , pd_target)
                if self.record_video:
                    # Update camera to follow robot (pelvis position)
                    robot_pos = self.data.qpos[:3]
                    self.camera.lookat[:] = robot_pos

                    # Update scene with data
                    self.renderer.update_scene(self.data, camera=self.camera)

                    # Render and get pixels
                    img = self.renderer.render()
                    img = np.asarray(img)
                    if img.dtype != np.uint8:
                        img = np.clip(img, 0, 255).astype(np.uint8) if img.max() > 1.0 else (img * 255).astype(np.uint8)
                    if img.ndim == 2:
                        img = np.repeat(img[..., None], 3, axis=2)
                    if img.shape[-1] == 4:
                        img = img[..., :3]

                    h, w = img.shape[:2]
                    if (h % 2) or (w % 2):
                        # pad to even dims for yuv420p
                        pad_h = h + (h % 2)
                        pad_w = w + (w % 2)
                        pad = ((0, pad_h - h), (0, pad_w - w), (0, 0))
                        img = np.pad(img, pad, mode="edge")

                    mp4_writer.append_data(np.ascontiguousarray(img))
                else:
                    # --- cap render FPS ---
                    self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
                    now = time.perf_counter()
                    spend = now - last_render
                    if spend < min_render_dt:
                        time.sleep(min_render_dt - spend)
                    last_render = time.perf_counter()
                    self.viewer.render()
                
            # pd_target[[13,14]] *= 0
            # print("RPY: ", rpy)
            torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
            cnt += np.sum(torque < -self.torque_limits) + np.sum(torque > self.torque_limits)
            # print("Torque: ", torque)
            # print("Torque limits exceeded: ", np.where(torque < -self.torque_limits), np.where(torque > self.torque_limits))
            # print("exceeded torques: ", torque[torque < -self.torque_limits], torque[torque > self.torque_limits])
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            # print(timestep, i, pd_target, dof_pos, dof_vel, torque)
            # if i > 2:
            #     break
            # print("Waist dof target: ", pd_target[[12, 13,14]])
            # print("ACTIONS: ", step_actions[[12,13,14]])
            # print("PD_TARGET: ", (pd_target))
            # print("DOF_POS: ", dof_pos)
            # print("Torque: ", torque)
            # print("Dof vel: ", dof_vel)
            # print("Overall cnt: ", cnt)
            self.data.ctrl = torque
            self.last_torque = to_torch(torque).unsqueeze(0)
            mujoco.mj_step(self.model, self.data)
            if not self.record_video:
                target_wall = start_wall + (self.data.time / max(desired_rtf, 1e-6))
                sleep_s = target_wall - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
        
        if not self.record_video:
            self.viewer.close()
        else:
            mp4_writer.close()
            self.renderer.close()
                 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MuJoCo sim-to-sim deployment with student policy")

    parser.add_argument('--robot', type=str, default="g1_29dof",
                        help="Robot type (default: g1_29dof)")
    parser.add_argument('--ckpt', type=str, required=True,
                        help="Path to checkpoint (.pth) or JIT model (.pt)")
    parser.add_argument('--motion_path', type=str, required=True,
                        help="Path to motion data file")
    parser.add_argument('--record_video', action='store_true',
                        help="Record video of the simulation")
    parser.add_argument('--use_jit', action='store_true',
                        help="Use JIT-compiled model (.pt) instead of checkpoint (.pth)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-detect JIT based on file extension if not explicitly specified
    if not args.use_jit and args.ckpt.endswith('.pt'):
        print("Detected .pt file extension, automatically enabling --use_jit")
        args.use_jit = True

    env = HumanoidEnv(
        policy_path=args.ckpt,
        robot_type=args.robot,
        device=device,
        record_video=args.record_video,
        motion_path=args.motion_path,
        use_jit=args.use_jit
    )
    env.run()
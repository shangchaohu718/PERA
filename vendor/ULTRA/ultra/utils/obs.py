"""MuJoCo-side observation builder for the Stage-2 teacher (used by sim2sim_teacher.py).

The student counterpart with modality masks / point clouds lives in utils/obs_vae.py.
"""
import numpy as np
import torch
import mujoco as mj
import os
from utils import torch_utils_mujoco
from collections import OrderedDict

def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:, 0]
    y = quat_angle[:, 1]
    z = quat_angle[:, 2]
    w = quat_angle[:, 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

def build_human_only_mappings(model,
                              pelvis_name="pelvis",
                              human_body_names=None,
                              contact_body_names=None):
    """
    Returns:
      body_ids           : np.int32 [Nb] (root first; expect Nb=39)
      contact_body_ids   : np.int32 [Nb] or subset (aligned to body_ids if you pass the same list)
      joint_ids          : np.int32 [Nu] (from actuators)
      dof_qpos_ids       : np.int32 [Nu] (addresses into qpos)
      dof_qvel_ids       : np.int32 [Nu] (addresses into qvel)
      actuator_ids       : np.int32 [Nu] (0..nu-1)
    """
    # 1) Bodies (prefer user-supplied list to preserve training order)
    human_body_names = ['pelvis', 'imu_in_pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'pelvis_contour_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', 
                        'waist_yaw_link', 'waist_roll_link', 'torso_link', 'd435_link', 'head_link', 'imu_in_torso', 
                        'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 'left_rubber_hand', 
                        'logo_link', 'mid360_link', 
                        'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', 'right_rubber_hand']
    body_ids = np.array([mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n)
                         for n in human_body_names], dtype=np.int32)
    if np.any(body_ids < 0):
        bad = [n for n, i in zip(human_body_names, body_ids) if i < 0]
        raise ValueError(f"Unknown human body names: {bad}")

    # 2) Contact bodies: default to "all human bodies" so you get 39 booleans
    if contact_body_names is None:
        contact_body_names = human_body_names
    contact_body_ids = np.array([mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n)
                                 for n in contact_body_names], dtype=np.int32)

    # 3) Joints/DOFs from actuators (object has none → human-only)
    joint_ids = []
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]  # joint-based motor in your XML
        if jid >= 0:
            joint_ids.append(jid)
    joint_ids = np.array(joint_ids, dtype=np.int32)

    # Addresses
    dof_qpos_ids = model.jnt_qposadr[joint_ids].astype(np.int32)
    dof_qvel_ids = model.jnt_dofadr[joint_ids].astype(np.int32)

    actuator_ids = np.arange(model.nu, dtype=np.int32)

    return body_ids, contact_body_ids, joint_ids, dof_qpos_ids, dof_qvel_ids, actuator_ids

def wxyz_to_xyzw(q):  # q: (..., 4)
    # (w,x,y,z) -> (x,y,z,w)
    if isinstance(q, torch.Tensor):
        w = q[..., 0:1]
        xyz = q[..., 1:4]
        return torch.cat([xyz, w], dim=-1)
    else:
        # numpy path
        w = q[..., 0:1]
        xyz = q[..., 1:4]
        return np.concatenate([xyz, w], axis=-1)

def xyzw_to_wxyz(q):
    # (x,y,z,w) -> (w,x,y,z)
    if isinstance(q, torch.Tensor):
        xyz = q[..., 0:3]
        w = q[..., 3:4]
        return torch.cat([w, xyz], dim=-1)
    else:
        xyz = q[..., 0:3]
        w = q[..., 3:4]
        return np.concatenate([w, xyz], axis=-1)


def _body_world_velocities(model, data, body_ids):
    """
    Compute world-frame linear and angular velocities for each body COM
    using Jacobians:  v = J * qvel
    Returns:
        lin (B, Nb, 3), ang (B, Nb, 3) as torch tensors with B=1
    """
    nv = model.nv
    jacp = np.zeros((3, nv), dtype=np.float64)
    jacr = np.zeros((3, nv), dtype=np.float64)

    lin = []
    ang = []
    for bid in body_ids:
        jacp.fill(0.0); jacr.fill(0.0)
        # Jacobian at body center of mass (world frame)
        mj.mj_jacBody(model, data, jacp, jacr, bid)
        v_lin = jacp @ data.qvel
        v_ang = jacr @ data.qvel
        lin.append(v_lin)
        ang.append(v_ang)

    lin = torch.from_numpy(np.asarray(lin, dtype=np.float64)).unsqueeze(0).to(torch.float32)  # (1, Nb, 3)
    ang = torch.from_numpy(np.asarray(ang, dtype=np.float64)).unsqueeze(0).to(torch.float32)  # (1, Nb, 3)
    return lin, ang


def _single_body_world_velocities(model, data, body_id):
    """World-frame linear & angular COM velocities via Jacobians."""
    nv = model.nv
    jacp = np.zeros((3, nv), dtype=np.float64)
    jacr = np.zeros((3, nv), dtype=np.float64)
    mj.mj_jacBody(model, data, jacp, jacr, int(body_id))
    v_lin = jacp @ data.qvel
    v_ang = jacr @ data.qvel
    v_lin = torch.from_numpy(v_lin).to(torch.float32).unsqueeze(0)  # (1,3)
    v_ang = torch.from_numpy(v_ang).to(torch.float32).unsqueeze(0)  # (1,3)
    return v_lin, v_ang

def _gather_contact_forces(model, data, body_ids):
    """
    MuJoCo external wrenches per body in world frame: data.cfrc_ext[body, 0:3] are forces.
    Returns (1, Nb, 3) torch tensor.
    """
    # cfrc_ext is (nb, 6): [fx, fy, fz, tx, ty, tz]
    f_world = data.cfrc_ext[body_ids, :3].copy()
    return torch.from_numpy(f_world).unsqueeze(0).to(torch.float32)


def _gather_body_poses(model, data, body_ids):
    """
    Returns:
        body_pos: (1, Nb, 3)
        body_rot: (1, Nb, 4)
    """
    pos = data.xpos[body_ids, :].copy()      # (Nb, 3)
    quat_wxyz = data.xquat[body_ids, :].copy()  # (Nb, 4), MuJoCo = (w,x,y,z)
    quat_xyzw = wxyz_to_xyzw(quat_wxyz)         # convert for Isaac-style utils

    body_pos = torch.from_numpy(pos).unsqueeze(0).to(torch.float32)     # (1, Nb, 3)
    body_rot = torch.from_numpy(quat_xyzw).unsqueeze(0).to(torch.float32)  # (1, Nb, 4) in xyzw
    return body_pos, body_rot



def _gather_dof_slices(data, dof_qpos_ids, dof_qvel_ids):
    """
    Map your policy DOFs to qpos/qvel slices.
    - dof_qpos_ids: indices into qpos for your joints (list[int] or np.array)
    - dof_qvel_ids: indices into qvel for your joints
    Returns:
        dof_pos, dof_vel: all shaped (1, Ndof)
    """
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    dof_pos = torch.from_numpy(qpos[dof_qpos_ids].astype(np.float32)).unsqueeze(0)
    dof_vel = torch.from_numpy(qvel[dof_qvel_ids].astype(np.float32)).unsqueeze(0)

    return dof_pos, dof_vel

def compute_sdf(points1, points2):
    dis_mat = points1.unsqueeze(2) - points2.unsqueeze(1)
    dis_mat_lengths = torch.norm(dis_mat, dim=-1)
    min_length_indices = torch.argmin(dis_mat_lengths, dim=-1)
    B_indices, N_indices = torch.meshgrid(torch.arange(points1.shape[0]), torch.arange(points1.shape[1]), indexing='ij')
    min_dis_mat = dis_mat[B_indices, N_indices, min_length_indices].contiguous()
    return min_dis_mat

def contact_indicator_for_bodies(model, data, body_ids, device=None):
    """
    Returns a torch tensor of shape (1, len(body_ids)).
    Each entry is 1.0 if that body is in contact (with ground or any other body),
    else 0.0.

    Args:
        model: mujoco.MjModel
        data: mujoco.MjData
        body_ids: list/array of body ids to check
    """
    body_ids = list(body_ids)
    indicator = torch.zeros((1, len(body_ids)), dtype=torch.float32, device=device)

    # Go through all active contacts
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]

        # If either geom’s parent body is in our target set, mark it
        for j, bid in enumerate(body_ids):
            if b1 == bid or b2 == bid:
                indicator[0, j] = 1.0

    return indicator



class MujocoObs:
    def __init__(self, model, object_body_name, max_episode_length, hoi_data, object_points, actuator_ids=None,
                 local_root_obs=False, root_height_obs=True, history_step=0, device="cpu"):
        self.model = model
        
        # Build human-only mappings (pass your known training order if you have it)
        human_body_names_39 = None  # or a concrete list to lock training order
        contact_body_names  = None  # default = all 39

        (body_ids,
        contact_body_ids,
        joint_ids,
        dof_qpos_ids,
        dof_qvel_ids,
        actuator_ids) = build_human_only_mappings(
            model,
            pelvis_name="pelvis",
            human_body_names=human_body_names_39,
            contact_body_names=contact_body_names
        )
        self.body_ids = np.asarray(body_ids, dtype=np.int32)
        self.contact_body_ids = np.asarray(contact_body_ids, dtype=np.int32)
        self.dof_qpos_ids = np.asarray(dof_qpos_ids, dtype=np.int32)
        self.dof_qvel_ids = np.asarray(dof_qvel_ids, dtype=np.int32)
        # print(self.dof_qpos_ids, self.dof_qvel_ids, actuator_ids)
        self.actuator_ids = None if actuator_ids is None else np.asarray(actuator_ids, dtype=np.int32)
        self.local_root_obs = local_root_obs
        self.root_height_obs = root_height_obs
        self.device = device
        self.object_body_name = object_body_name
        self.max_episode_length = max_episode_length
        self.hoi_data = hoi_data
        self.object_points = object_points
        self.history_step = history_step
        if self.history_step > 0:
            self.obs_history_buf = torch.zeros(
                1, # num_envs
                self.history_step,
                166, # self.cfg['env']['numObsProprio'],
                # device=self.device,
                dtype=torch.float,
            )
        # stateful last_* buffers (1, Ndof)
        self.last_dof_pos = None
        self.last_dof_vel = None

    def _ensure_last_buffers(self, dof_pos, dof_vel):
        if self.last_dof_pos is None:
            self.last_dof_pos = torch.zeros_like(dof_pos.clone())
        if self.last_dof_vel is None:
            self.last_dof_vel = torch.zeros_like(dof_vel.clone())

    def _compute_observations_iter(self, data, curr_t, delta_t=1, actions=None, torques=None, student_obs=False, episode_length=0):
        ts = curr_t
        next_ts = min(ts + delta_t, self.max_episode_length-1)
        ref_obs = self.hoi_data[None, next_ts].clone()
        obs, key_body_pose, key_body_rot, obs_dict = self._compute_humanoid_obs(data, ref_obs, actions, torques, episode_length,  student_obs=student_obs)
        task_obs, obj_points, obj_obs_dict = self._compute_task_obs(data, ref_obs)
        # The teacher is trained with the object task observation kept in only obs_task_keep_prob (0.3) of the
        # environments and the interaction-graph (IG) features in obs_ig_keep_prob (0.1) of them, so the
        # fully-masked variant used here for sim2sim is inside the training distribution.
        task_obs*=0
        obs = torch.cat([obs, task_obs], dim=-1)
        obs_dict.update(obj_obs_dict)
        ig = compute_sdf(key_body_pose, obj_points).view(-1, 3)
        heading_rot = torch_utils_mujoco.calc_heading_quat_inv(key_body_rot[:, 0, :])
        heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pose.shape[1], 1).view(-1, 4)
        ig = torch_utils_mujoco.quat_rotate(heading_rot_extend, ig).view(1, -1, 3)
        ig_norm = ig.norm(dim=-1, keepdim=True)
        ig_all = ig / (ig_norm + 1e-6) * (-5 * ig_norm).exp()
        ig = ig_all.view(1, -1)
        ig_all = ig_all.view(1, -1)    
        ref_ig = ref_obs[:, 630:].view(1, 39, 3)
        ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
        ref_ig = ref_ig / (ref_ig_norm + 1e-6) * (-5 * ref_ig_norm).exp()  
        ref_ig = ref_ig.view(1, -1)
        ref_ig *= 0  # IG features masked, see comment above
        ig_all *= 0
        ig *= 0
        obs_dict.update(OrderedDict([('ig', ig_all), ('diff_ig', ref_ig-ig)]))          
        return torch.cat((obs,ig_all,ref_ig-ig),dim=-1), obs_dict

    def _compute_observations(self, data, curr_t, actions, torques, student_obs=False, episode_length=0):
        obs_1, obs_dict_1 = self._compute_observations_iter(data, curr_t, 1, actions, torques, student_obs=student_obs, episode_length=episode_length)
        if student_obs:
            return obs_1
        obs_2, obs_dict_2 = self._compute_observations_iter(data, curr_t, 16, actions, torques, student_obs=student_obs, episode_length=episode_length)
        # save_json_ordered(obs_dict_1, output_file)
        # return obs_1
        return torch.cat((obs_1, obs_2), dim=-1)

    def _compute_humanoid_obs(self, data, ref_obs, actions, torques, episode_length=0, student_obs=False):
        """
        Equivalent to your Isaac call:
          obs = compute_humanoid_observations_max(
              body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs, self._root_height_obs,
              contact_forces, self._contact_body_ids, ref_obs, self._key_body_ids, self._key_body_ids_gt,
              self._contact_body_ids_gt, actions, dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel,
              student_obs=student_obs
          )
        Returns: obs (1, D)
        """
        # --- Poses
        body_pos, body_rot = _gather_body_poses(self.model, data, self.body_ids)        # (1, Nb, 3/4)
        # --- Velocities (world)
        body_vel, body_ang_vel = _body_world_velocities(self.model, data, self.body_ids) # (1, Nb, 3)
        # --- Contact forces on selected bodies
        contact = contact_indicator_for_bodies(self.model, data, self.body_ids)

        # --- DOF slices and actuator signals
        dof_pos, dof_vel = _gather_dof_slices(
            data, self.dof_qpos_ids, self.dof_qvel_ids
        )

        # --- last_* buffers
        self._ensure_last_buffers(dof_pos, dof_vel)
        last_dof_pos = self.last_dof_pos
        last_dof_vel = self.last_dof_vel

        # Move to device if needed
        tensors = [body_pos, body_rot, body_vel, body_ang_vel, contact, actions,
                   dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel, ref_obs]
        tensors = [t.to(self.device) for t in tensors]
        (body_pos, body_rot, body_vel, body_ang_vel, contact, actions,
         dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel, ref_obs) = tensors

        # Build the proprioceptive observation with the same routine used in IsaacGym training.
        obs, obs_dict = self.compute_humanoid_observations_max(
            body_pos, body_rot, body_vel, body_ang_vel,
            self.local_root_obs, self.root_height_obs,
            contact, ref_obs, actions, dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel, episode_length,student_obs=student_obs
        )

        # update last_* for next step
        self.last_dof_pos = dof_pos.detach()
        self.last_dof_vel = dof_vel.detach()
        return obs, body_pos, body_rot, obs_dict

    def _compute_task_obs(self, data, ref_obs=None,
                        root_body_name="pelvis"):
        """
        Builds Isaac-style root_states and tar_states from MuJoCo, then calls
        compute_obj_observations(root_states, tar_states, object_points, ref_obs).

        Returns:
            obs:        (1, 21)
            obj_points: (1, N, 3) world points (as returned by compute_obj_observations)
        """
        assert ref_obs is not None, "ref_obs is required"
        device = ref_obs.device if isinstance(ref_obs, torch.Tensor) else "cpu"

        # -------- root (pelvis) pose -> root_states[:, :7] = [pos(3), rot_xyzw(4)]
        root_bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, root_body_name)
        root_pos = torch.from_numpy(data.xpos[root_bid].copy()).to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,3)
        root_quat_wxyz = torch.from_numpy(data.xquat[root_bid].copy()).to(device=device, dtype=torch.float32)        # (4,)
        root_rot = wxyz_to_xyzw(root_quat_wxyz).unsqueeze(0)  # (1,4)
        root_states = torch.cat([root_pos, root_rot], dim=-1) # (1,7)

        # -------- object pose + world velocities -> tar_states[:, :13]
        obj_bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.object_body_name)
        tar_pos = torch.from_numpy(data.xpos[obj_bid].copy()).to(device=device, dtype=torch.float32).unsqueeze(0)     # (1,3)
        tar_quat_wxyz = torch.from_numpy(data.xquat[obj_bid].copy()).to(device=device, dtype=torch.float32)           # (4,)
        tar_rot = wxyz_to_xyzw(tar_quat_wxyz).unsqueeze(0)  # (1,4)
        tar_vel, tar_ang_vel = _single_body_world_velocities(self.model, data, obj_bid)  # (1,3), (1,3)
        tar_vel, tar_ang_vel = tar_vel.to(device), tar_ang_vel.to(device)
        tar_states = torch.cat([tar_pos, tar_rot, tar_vel, tar_ang_vel], dim=-1)  # (1,13)

        # -------- object local point cloud -> (1, N, 3)
        # self.object_points should be LOCAL (centered), like in your Isaac code.
        obj_local = self.object_points
        if not isinstance(obj_local, torch.Tensor):
            obj_local = torch.as_tensor(obj_local, dtype=torch.float32, device=device)
        else:
            obj_local = obj_local.to(device=device, dtype=torch.float32)
        if obj_local.dim() == 2:
            obj_local = obj_local.unsqueeze(0)  # (1, N, 3)

        # Same object-observation routine as used in IsaacGym training.
        obs, obj_points, obs_dict = self.compute_obj_observations(root_states, tar_states, obj_local, ref_obs)
        return obs, obj_points, obs_dict

    def compute_humanoid_observations_max(self, body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, contact, ref_obs, actions, dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel, episode_length=0, student_obs=False):
        root_pos = body_pos[:, 0, :]
        root_rot = body_rot[:, 0, :]

        root_h = root_pos[:, 2:3]
        heading_rot = torch_utils_mujoco.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils_mujoco.calc_heading_quat(root_rot)

        if (not root_height_obs):
            root_h_obs = torch.zeros_like(root_h)
        else:
            root_h_obs = root_h

        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand_2 = heading_rot_expand.repeat((1, 39, 1))
        flat_heading_rot_2 = heading_rot_expand_2.reshape(heading_rot_expand_2.shape[0] * heading_rot_expand_2.shape[1], 
                                                heading_rot_expand_2.shape[2])
        
        heading_rot_expand = heading_rot_expand.repeat((1, 39, 1))
        flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                heading_rot_expand.shape[2])

        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, 39, 1))
        flat_heading_inv_rot = heading_inv_rot_expand.reshape(heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                                heading_inv_rot_expand.shape[2])
        
        _ref_body_pos = ref_obs[:,84:201].view(-1, 39, 3)
        _body_pos = body_pos

        diff_global_body_pos = _ref_body_pos - _body_pos
        diff_local_body_pos_flat = torch_utils_mujoco.quat_rotate(flat_heading_rot_2, diff_global_body_pos.view(-1, 3)).view(-1, 39 * 3)

        local_ref_body_pos = _body_pos - root_pos.unsqueeze(1)  # preserves the body position
        local_ref_body_pos = torch_utils_mujoco.quat_rotate(flat_heading_rot_2, local_ref_body_pos.view(-1, 3)).view(-1, 39 * 3)

        root_pos_expand = root_pos.unsqueeze(-2)
        local_body_pos = body_pos - root_pos_expand
        flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
        flat_local_body_pos = torch_utils_mujoco.quat_rotate(flat_heading_rot, flat_local_body_pos)
        local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
        local_body_pos = local_body_pos[..., 3:] # remove root pos

        flat_body_rot = body_rot.reshape(body_rot.shape[0] * 39, body_rot.shape[2])
        flat_local_body_rot = torch_utils_mujoco.quat_mul(flat_heading_rot, flat_body_rot)
        flat_local_body_rot_obs = torch_utils_mujoco.quat_to_tan_norm(flat_local_body_rot)
        local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], 39 * flat_local_body_rot_obs.shape[1])
        
        ref_body_rot = ref_obs[:, 201:357].view(-1, 39, 4)
        ref_body_rot_no_hand = ref_body_rot
        body_rot_no_hand = body_rot

        diff_global_body_rot = torch_utils_mujoco.quat_mul_norm(torch_utils_mujoco.quat_inverse(ref_body_rot_no_hand.reshape(-1, 4)), body_rot_no_hand.reshape(-1, 4))
        diff_local_body_rot_flat = torch_utils_mujoco.quat_mul(torch_utils_mujoco.quat_mul(flat_heading_rot, diff_global_body_rot.view(-1, 4)), flat_heading_inv_rot)
        diff_local_body_rot_obs = torch_utils_mujoco.quat_to_tan_norm(diff_local_body_rot_flat)
        diff_local_body_rot_obs = diff_local_body_rot_obs.view(body_rot_no_hand.shape[0], body_rot_no_hand.shape[1] * diff_local_body_rot_obs.shape[-1])

        local_ref_body_rot = torch_utils_mujoco.quat_mul(flat_heading_rot, ref_body_rot_no_hand.reshape(-1, 4))
        local_ref_body_rot = torch_utils_mujoco.quat_to_tan_norm(local_ref_body_rot).view(ref_body_rot_no_hand.shape[0], -1)

        ref_body_vel = ref_obs[:, 357:474].view(-1, 39, 3)
        _body_vel = body_vel
        diff_global_vel = ref_body_vel - _body_vel
        diff_local_vel = torch_utils_mujoco.quat_rotate(flat_heading_rot_2, diff_global_vel.view(-1, 3)).view(-1, 39 * 3)

        ref_body_ang_vel = ref_obs[:, 474:591]
        ref_body_ang_vel_no_hand = ref_body_ang_vel.view(-1, 39, 3)
        body_ang_vel_no_hand = body_ang_vel
        diff_global_ang_vel = ref_body_ang_vel_no_hand - body_ang_vel_no_hand
        diff_local_ang_vel = torch_utils_mujoco.quat_rotate(flat_heading_rot, diff_global_ang_vel.view(-1, 3)).view(-1, 39 * 3)

        if (local_root_obs):
            root_rot_obs = torch_utils_mujoco.quat_to_tan_norm(root_rot)
            local_body_rot_obs[..., 0:6] = root_rot_obs

        flat_body_vel = body_vel.reshape(body_vel.shape[0] * 39, body_vel.shape[2])
        flat_local_body_vel = torch_utils_mujoco.quat_rotate(flat_heading_rot, flat_body_vel)
        local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], 39 * body_vel.shape[2])
        
        flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * 39, body_ang_vel.shape[2])
        flat_local_body_ang_vel = torch_utils_mujoco.quat_rotate(flat_heading_rot, flat_body_ang_vel)
        local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], 39 * body_ang_vel.shape[2])

        ref_body_contact = ref_obs[:,591:630]
        diff_body_contact = ref_body_contact - contact
        contact_new = torch.zeros_like(contact).to(contact.device)
        diff_body_contact_new = torch.zeros_like(diff_body_contact).to(contact.device)
        # contact_new[:, [6, 7, 13, 14, 28, 38]] = contact[:, [6, 7, 13, 14, 28, 38]]
        # diff_body_contact_new[:, [6, 7, 13, 14, 28, 38]] = diff_body_contact[:, [6, 7, 13, 14, 28, 38]]
        if student_obs:
            # obs_prop shape: [nuv_envs, 320] (1771 - 320 = 1451)
            root_ang_vel = body_ang_vel[:, 0, :]
            base_quat = root_rot # in xyzw
            roll, pitch, yaw = euler_from_quaternion(base_quat)
            ref_base_quat = ref_body_rot[:, 0]
            roll_ref, pitch_ref, yaw_ref = euler_from_quaternion(ref_base_quat)
            imu_obs = torch.stack((roll, pitch), dim=1)

            imu_obs_all = torch.stack((roll, pitch, yaw), dim=1)
            imu_obs_ref_all = torch.stack((roll_ref, pitch_ref, yaw_ref), dim=1)
            diff_imu_obs_all = imu_obs_ref_all - imu_obs_all

            ref_dof_pos = ref_obs[:, 13:42]  # 29 reference DoF positions
            diff_dof_pos = dof_pos - ref_dof_pos


            
            imu_obs_ref = torch.stack((roll_ref, pitch_ref), dim=1)
            diff_imu_obs = imu_obs_ref - imu_obs

            root_pos = _body_pos[:, 0, :].clone()
            ref_root_pos = _ref_body_pos[:, 0, :].clone()
            diff_root_pos = root_pos - ref_root_pos 

            # root_pos *= 0
            # ref_root_pos *= 0
            # diff_root_pos *= 0

            # imu_obs_ref *= 0
            # diff_imu_obs *= 0

            if self.history_step > 0:
                # ref_root_pos = ref_root_pos * 0
                # root_pos = root_pos * 0
                # diff_root_pos = diff_root_pos * 0
                # imu_obs_ref_all = imu_obs_ref_all * 0
                # imu_obs_all[..., 2] = 0
                # diff_imu_obs_all = diff_imu_obs_all * 0
                # # obs shape: [nuv_envs, 2450]
                # obs_prop = torch.cat((ref_dof_pos, diff_dof_pos, root_ang_vel, imu_obs, dof_pos, dof_vel, actions, self.obs_history_buf.view(1, -1)), dim=-1)
                _dof_vel = dof_vel.clone()
                _dof_vel[..., [12, 13, 14]] *= 0.05
                obs_prop = torch.cat((ref_dof_pos, diff_dof_pos, ref_root_pos, root_pos, diff_root_pos, root_ang_vel, imu_obs_ref_all, imu_obs_all, diff_imu_obs_all, dof_pos, _dof_vel, actions, self.obs_history_buf.view(1, -1)), dim=-1)
                
                # obs_prop = torch.cat((ref_dof_pos, diff_dof_pos, ref_root_pos, root_pos, diff_root_pos, root_ang_vel, imu_obs, imu_obs_ref, diff_imu_obs, dof_pos, dof_vel, actions, self.obs_history_buf.view(1, -1)), dim=-1)
                # obs_buf = torch.cat(
                #     (
                #         root_ang_vel, 
                #         imu_obs,  # 2 dims
                #         dof_pos,
                #         dof_vel,
                #         actions,
                #     ),
                #     dim=-1,
                # )
                # obs_buf = torch.cat(
                #     (
                #         ref_dof_pos * 0, 
                #         diff_dof_pos * 0, 
                #         ref_root_pos * 0, 
                #         root_pos * 0, 
                #         diff_root_pos * 0,
                #         root_ang_vel, 
                #         imu_obs_ref_all * 0,  # 2 dims
                #         imu_obs_all, 
                #         diff_imu_obs_all * 0,
                #         dof_pos,
                #         dof_vel,
                #         actions,
                #     ),
                #     dim=-1,
                # )
                obs_buf = torch.cat(
                    (
                        ref_dof_pos, 
                        diff_dof_pos, 
                        ref_root_pos, 
                        root_pos, 
                        diff_root_pos,
                        root_ang_vel, 
                        imu_obs_ref_all,  # 2 dims
                        imu_obs_all, 
                        diff_imu_obs_all,
                        dof_pos,
                        _dof_vel,
                        actions,
                    ),
                    dim=-1,
                )
                cond = torch.tensor(episode_length, device=obs_buf.device)
                cond = cond.view(1,)
                self.obs_history_buf = torch.where(
                    (cond < 1)[:, None, None],
                    torch.stack([obs_buf] * self.history_step, dim=1),
                    torch.cat([self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)], dim=1),
                )

                obs_dict = OrderedDict([
                    ("root_h_obs",               root_h_obs),                # (B, 1)
                    ("local_body_pos",           local_body_pos),            # (B, 39*3-3 = 114)  (root removed)
                    ("local_body_rot_obs",       local_body_rot_obs),        # (B, 39*6)
                    ("local_body_vel",           local_body_vel),            # (B, 39*3)
                    ("local_body_ang_vel",       local_body_ang_vel),        # (B, 39*3)
                    ("contact",                  contact),                   # (B, 39)
                    ("diff_local_body_pos_flat", diff_local_body_pos_flat),  # (B, 39*3)
                    ("diff_local_body_rot_obs",  diff_local_body_rot_obs),   # (B, 39*6)
                    ("diff_body_contact",        diff_body_contact_new),         # (B, 39)
                    ("local_ref_body_pos",       local_ref_body_pos),        # (B, 39*3)
                    ("local_ref_body_rot",       local_ref_body_rot),        # (B, 39*6)
                    ("diff_local_vel",           diff_local_vel),            # (B, 39*3)
                    ("diff_local_ang_vel",       diff_local_ang_vel),        # (B, 39*3)
                    ("actions",                  actions),                   # (B, A)
                    ("dof_pos",                  dof_pos),                   # (B, Q)
                    ("dof_vel",                  dof_vel),                   # (B, Q)
                    ("torques",                  torques),                   # (B, Q)
                    ("last_dof_pos",             last_dof_pos),              # (B, Q)
                    ("last_dof_vel",             last_dof_vel),              # (B, Q)
                ])
                return obs_prop, obs_dict
            else:
                obs = torch.cat((ref_dof_pos, diff_dof_pos, root_ang_vel, imu_obs, dof_pos, dof_vel, actions), dim=-1)
                obs_dict = OrderedDict([
                    ("root_h_obs",               root_h_obs),                # (B, 1)
                    ("local_body_pos",           local_body_pos),            # (B, 39*3-3 = 114)  (root removed)
                    ("local_body_rot_obs",       local_body_rot_obs),        # (B, 39*6)
                    ("local_body_vel",           local_body_vel),            # (B, 39*3)
                    ("local_body_ang_vel",       local_body_ang_vel),        # (B, 39*3)
                    ("contact",                  contact),                   # (B, 39)
                    ("diff_local_body_pos_flat", diff_local_body_pos_flat),  # (B, 39*3)
                    ("diff_local_body_rot_obs",  diff_local_body_rot_obs),   # (B, 39*6)
                    ("diff_body_contact",        diff_body_contact_new),         # (B, 39)
                    ("local_ref_body_pos",       local_ref_body_pos),        # (B, 39*3)
                    ("local_ref_body_rot",       local_ref_body_rot),        # (B, 39*6)
                    ("diff_local_vel",           diff_local_vel),            # (B, 39*3)
                    ("diff_local_ang_vel",       diff_local_ang_vel),        # (B, 39*3)
                    ("actions",                  actions),                   # (B, A)
                    ("dof_pos",                  dof_pos),                   # (B, Q)
                    ("dof_vel",                  dof_vel),                   # (B, Q)
                    ("torques",                  torques),                   # (B, Q)
                    ("last_dof_pos",             last_dof_pos),              # (B, Q)
                    ("last_dof_vel",             last_dof_vel),              # (B, Q)
                ])
                return obs, obs_dict

        else:
            # local_body_vel = local_body_vel * 0
            # local_body_ang_vel = local_body_ang_vel * 0
            # local_body_ang_vel = local_body_ang_vel * 0
            # diff_local_vel = diff_local_ang_vel * 0
            # actions = actions * 0
            # dof_vel = dof_vel * 0
            torques = torques * 0  # the IsaacGym teacher observation zeroes torques (ultra_g1_retarget.py)
            # last_dof_vel = last_dof_vel * 0

            obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, contact_new, diff_local_body_pos_flat, diff_local_body_rot_obs, diff_body_contact_new, local_ref_body_pos, local_ref_body_rot, diff_local_vel, diff_local_ang_vel, actions, dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel), dim=-1)
            obs_dict = OrderedDict([
                    ("root_h_obs",               root_h_obs),                # (B, 1)
                    ("local_body_pos",           local_body_pos),            # (B, 39*3-3 = 114)  (root removed)
                    ("local_body_rot_obs",       local_body_rot_obs),        # (B, 39*6)
                    ("local_body_vel",           local_body_vel),            # (B, 39*3)
                    ("local_body_ang_vel",       local_body_ang_vel),        # (B, 39*3)
                    ("contact",                  contact),                   # (B, 39)
                    ("diff_local_body_pos_flat", diff_local_body_pos_flat),  # (B, 39*3)
                    ("diff_local_body_rot_obs",  diff_local_body_rot_obs),   # (B, 39*6)
                    ("diff_body_contact",        diff_body_contact_new),         # (B, 39)
                    ("local_ref_body_pos",       local_ref_body_pos),        # (B, 39*3)
                    ("local_ref_body_rot",       local_ref_body_rot),        # (B, 39*6)
                    ("diff_local_vel",           diff_local_vel),            # (B, 39*3)
                    ("diff_local_ang_vel",       diff_local_ang_vel),        # (B, 39*3)
                    ("actions",                  actions),                   # (B, A)
                    ("dof_pos",                  dof_pos),                   # (B, Q)
                    ("dof_vel",                  dof_vel),                   # (B, Q)
                    ("torques",                  torques),                   # (B, Q)
                    ("last_dof_pos",             last_dof_pos),              # (B, Q)
                    ("last_dof_vel",             last_dof_vel),              # (B, Q)
                ])

            return obs, obs_dict
    
    def compute_obj_observations(self, root_states, tar_states, object_points, ref_obs):
        root_pos = root_states[:, 0:3]
        root_rot = root_states[:, 3:7]

        tar_pos = tar_states[:, 0:3]
        tar_rot = tar_states[:, 3:7]
        tar_vel = tar_states[:, 7:10]
        tar_ang_vel = tar_states[:, 10:13]
        tar_ang_vel_ori = tar_ang_vel.clone()

        obj_rot_extend = tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        object_points_extend = object_points.view(-1, 3)
        obj_points = torch_utils_mujoco.quat_rotate(obj_rot_extend, object_points_extend).view(tar_rot.shape[0], object_points.shape[1], 3) + tar_pos.unsqueeze(1)

        heading_rot = torch_utils_mujoco.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils_mujoco.calc_heading_quat(root_rot)

        local_tar_pos = tar_pos - root_pos
        local_tar_pos[..., -1] = tar_pos[..., -1]
        local_tar_pos = torch_utils_mujoco.quat_rotate(heading_rot, local_tar_pos)
        local_tar_vel = torch_utils_mujoco.quat_rotate(heading_rot, tar_vel)
        local_tar_ang_vel = torch_utils_mujoco.quat_rotate(heading_rot, tar_ang_vel)

        local_tar_rot = torch_utils_mujoco.quat_mul(heading_rot, tar_rot)
        local_tar_rot_obs = torch_utils_mujoco.quat_to_tan_norm(local_tar_rot)

        _ref_obj_pos = ref_obs[:,71:74]  # G1 reference layout: obj pos 71:74, rot 74:78, vel 78:81, ang vel 81:84
        diff_global_obj_pos = _ref_obj_pos - tar_pos
        diff_local_obj_pos_flat = torch_utils_mujoco.quat_rotate(heading_rot, diff_global_obj_pos)

        local_ref_obj_pos = _ref_obj_pos - root_pos  # preserves the body position
        local_ref_obj_pos = torch_utils_mujoco.quat_rotate(heading_rot, local_ref_obj_pos)

        ref_obj_rot = ref_obs[:,74:78]
        diff_global_obj_rot = torch_utils_mujoco.quat_mul_norm(torch_utils_mujoco.quat_inverse(ref_obj_rot), tar_rot)
        diff_local_obj_rot_flat = torch_utils_mujoco.quat_mul(torch_utils_mujoco.quat_mul(heading_rot, diff_global_obj_rot.view(-1, 4)), heading_inv_rot)  # Need to be change of basis
        diff_local_obj_rot_obs = torch_utils_mujoco.quat_to_tan_norm(diff_local_obj_rot_flat)

        local_ref_obj_rot = torch_utils_mujoco.quat_mul(heading_rot, ref_obj_rot)
        local_ref_obj_rot = torch_utils_mujoco.quat_to_tan_norm(local_ref_obj_rot)

        ref_obj_vel = ref_obs[:, 78:81]
        diff_global_vel = ref_obj_vel - tar_vel
        diff_local_vel = torch_utils_mujoco.quat_rotate(heading_rot, diff_global_vel)

        ref_obj_ang_vel = ref_obs[:, 81:84]
        diff_global_ang_vel = ref_obj_ang_vel - tar_ang_vel
        diff_local_ang_vel = torch_utils_mujoco.quat_rotate(heading_rot, diff_global_ang_vel)

        # local_tar_ang_vel = local_tar_ang_vel * 0
        # local_tar_vel = local_tar_vel * 0
        # diff_local_vel = diff_local_vel * 0
        # diff_local_ang_vel = diff_local_ang_vel * 0
        local_tar_ang_vel = local_tar_ang_vel
        local_tar_vel = local_tar_vel
        diff_local_vel = diff_local_vel
        diff_local_ang_vel = diff_local_ang_vel
        obs = torch.cat([local_tar_vel, local_tar_ang_vel, diff_local_obj_pos_flat, diff_local_obj_rot_obs, diff_local_vel, diff_local_ang_vel], dim=-1)
        obs_dict = OrderedDict([
            ("local_tar_vel",            local_tar_vel),                # (B, 1)
            ("local_tar_ang_vel",        local_tar_ang_vel),            # (B, 39*3-3 = 114)  (root removed)
            ("diff_local_obj_pos_flat",  diff_local_obj_pos_flat),        # (B, 39*6)
            ("diff_local_obj_rot_obs",   diff_local_obj_rot_obs),            # (B, 39*3)
            ("diff_local_vel",           diff_local_vel),        # (B, 39*3)
            ("diff_local_ang_vel",       diff_local_ang_vel),                   # (B, 39)
            ("tar_ang_vel_ori", tar_ang_vel_ori),
            ("heading_rot", heading_rot),
        ])
        return obs, obj_points, obs_dict

def to_plain(x):
    if isinstance(x, OrderedDict):
        return {k: to_plain(v) for k, v in x.items()}
    if isinstance(x, dict):
        return {k: to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(x, np.generic):
        return x.item()
    return x

def save_json_ordered(od, path="out.json"):
    import json
    with open(path, "w") as f:
        json.dump(to_plain(od), f, indent=2, ensure_ascii=False)
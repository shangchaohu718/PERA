# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch

@torch.jit.script
def quat_rotate(q: torch.Tensor, v: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Rotate vector(s) v by quaternion(s) q (IsaacGym convention: xyzw).

    Args:
        q: (..., 4) quaternion(s) in (x, y, z, w) order.
        v: (..., 3) vector(s) to rotate. Leading dims must be broadcastable with q.
        normalize: if True, re-normalizes q before rotation.

    Returns:
        (..., 3) rotated vector(s), same broadcasted shape as v.

    Formula (fast, no matrix build):
        t = 2 * cross(q_xyz, v)
        v' = v + w * t + cross(q_xyz, t)
    """
    if normalize:
        q = q / (torch.norm(q, 2, -1, True) + 1e-8)

    q_xyz = q[..., :3]
    qw    = q[..., 3:4]                  # keep last dim for broadcasting

    # Cross products broadcast over any leading dims as long as shapes are compatible.
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + qw * t + torch.cross(q_xyz, t, dim=-1)

@torch.jit.script
def quat_from_angle_axis(heading, axis, normalize_axis: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """
    Build a quaternion (xyzw) from an angle and axis (IsaacGym convention).

    Args:
        heading: (...,) or (...,1) rotation angle in radians.
        axis:    (...,3) rotation axis (need not be unit length).
        normalize_axis: if True, normalizes the axis before use.
        eps:     small constant for numerical stability.

    Returns:
        q: (...,4) quaternion(s) in (x, y, z, w) order.
    """
    # to tensors
    if not isinstance(axis, torch.Tensor):
        axis = torch.as_tensor(axis, dtype=torch.float32)
    if not isinstance(heading, torch.Tensor):
        heading = torch.as_tensor(heading, dtype=axis.dtype, device=axis.device)
    else:
        heading = heading.to(dtype=axis.dtype, device=axis.device)

    # ensure shapes: axis (...,3), angle (...,1)
    if axis.shape[-1] != 3:
        raise ValueError("axis must have last dimension 3")
    if heading.dim() == axis.dim():
        pass
    elif heading.dim() == axis.dim() - 1:
        heading = heading.unsqueeze(-1)  # (...,1)
    elif heading.dim() > axis.dim():
        raise ValueError("heading has more dims than axis; not broadcastable")

    # normalize axis if requested
    if normalize_axis:
        norm = torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(eps)
        axis_n = axis / norm
    else:
        axis_n = axis

    half = 0.5 * heading
    sh = torch.sin(half)                 # (...,1)
    ch = torch.cos(half)                 # (...,1)

    q_xyz = axis_n * sh                  # (...,3)
    q_w   = ch                           # (...,1)
    q     = torch.cat([q_xyz, q_w], dim=-1)  # (...,4) xyzw

    # handle degenerate axes (||axis|| ~ 0) → identity
    if normalize_axis:
        deg = (torch.linalg.norm(axis, dim=-1, keepdim=True) < eps)
        if deg.any():
            ident = torch.zeros_like(q)
            ident[..., 3] = 1.0
            q = torch.where(deg.expand_as(q), ident, q)

    # optional: re-normalize to be safe
    q = q / (torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps))
    return q

@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw, degrees: bool = False, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert Euler XYZ (roll, pitch, yaw) to quaternion in IsaacGym order (x, y, z, w).

    Args:
        roll, pitch, yaw: angle(s) in radians by default. Any shape; broadcastable.
        degrees: if True, interpret inputs as degrees.
        eps: small constant for safe normalization.

    Returns:
        q: (..., 4) quaternion(s) in (x, y, z, w).
    """

    r, p, y = roll, pitch, yaw
    hr, hp, hy = 0.5 * r, 0.5 * p, 0.5 * y
    cr, sr = torch.cos(hr), torch.sin(hr)
    cp, sp = torch.cos(hp), torch.sin(hp)
    cy, sy = torch.cos(hy), torch.sin(hy)

    # Tait–Bryan XYZ (apply roll about X, then pitch about Y, then yaw about Z)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    q = torch.stack([x, y, z, w], dim=-1)
    # Normalize for safety
    q = q / (torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps))
    return q

@torch.jit.script
def quat_mul(x: torch.Tensor, y: torch.Tensor, normalize: bool = False, eps: float = 1e-8) -> torch.Tensor:
    """
    Quaternion multiply (Hamilton product) with IsaacGym convention (x, y, z, w).

    Args:
        x, y: (..., 4) quaternions in (x, y, z, w) order. Broadcastable over leading dims.
        normalize: if True, renormalizes the result to unit length.
        eps: small value to avoid division by zero when normalizing.

    Returns:
        (..., 4) quaternion product p = x ⊗ y in (x, y, z, w).
    """
    if x.shape[-1] != 4 or y.shape[-1] != 4:
        raise ValueError("quat_mul expects inputs with last dimension 4 (xyzw).")

    x_xyz, x_w = x[..., :3], x[..., 3:4]
    y_xyz, y_w = y[..., :3], y[..., 3:4]

    # Vector part: w1*v2 + w2*v1 + v1 × v2
    v = x_w * y_xyz + y_w * x_xyz + torch.cross(x_xyz, y_xyz, dim=-1)
    # Scalar part: w1*w2 - v1·v2
    w = x_w * y_w - (x_xyz * y_xyz).sum(dim=-1, keepdim=True)

    q = torch.cat([v, w], dim=-1)

    if normalize:
        q = q / (torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps))
    return q

@torch.jit.script
def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """
    Normalize angles (radians) to [-pi, pi], elementwise.
    Works with any shape; preserves dtype/device; fully differentiable.

    IsaacGym-style: atan2(sin x, cos x).
    """
    return torch.atan2(torch.sin(angle), torch.cos(angle))

@torch.jit.script
def quat_angle_axis(x):
    """
    The (angle, axis) representation of the rotation. The axis is normalized to unit length.
    The angle is guaranteed to be between [0, pi].
    """
    s = 2 * (x[..., 3] ** 2) - 1
    angle = s.clamp(-1, 1).arccos()  # just to be safe
    axis = x[..., :3]
    axis /= axis.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
    return angle, axis

@torch.jit.script
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis

@torch.jit.script
def angle_axis_to_exp_map(angle, axis):
    # type: (Tensor, Tensor) -> Tensor
    # compute exponential map from axis-angle
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map

@torch.jit.script
def quat_to_exp_map(q):
    # type: (Tensor) -> Tensor
    # compute exponential map from quaternion
    # q must be normalized
    angle, axis = quat_to_angle_axis(q)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map

@torch.jit.script
def quat_to_tan_norm(q):
    # type: (Tensor) -> Tensor
    # represents a rotation using the tangent and normal vectors
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)
    
    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm)
    
    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan

@torch.jit.script
def euler_xyz_to_exp_map(roll, pitch, yaw):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    q = quat_from_euler_xyz(roll, pitch, yaw)
    exp_map = quat_to_exp_map(q)
    return exp_map

@torch.jit.script
def exp_map_to_angle_axis(exp_map):
    min_theta = 1e-5

    angle = torch.norm(exp_map, dim=-1)
    angle_exp = torch.unsqueeze(angle, dim=-1)
    axis = exp_map / angle_exp
    angle = normalize_angle(angle)

    default_axis = torch.zeros_like(exp_map)
    default_axis[..., -1] = 1

    mask = torch.abs(angle) > min_theta
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)

    return angle, axis

@torch.jit.script
def exp_map_to_quat(exp_map):
    angle, axis = exp_map_to_angle_axis(exp_map)
    q = quat_from_angle_axis(angle, axis)
    return q

@torch.jit.script
def quat_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

@torch.jit.script
def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


@torch.jit.script
def quat_to_6d(q):
    m = quat_to_matrix(q)
    r = matrix_to_rotation_6d(m)
    return r

@torch.jit.script
def exp_map_to_6d(exp_map):
    q = exp_map_to_quat(exp_map)
    r = quat_to_6d(q)
    return r

@torch.jit.script
def slerp(q0, q1, t):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta);
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta);

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta;
    ratioB = torch.sin(t * half_theta) / sin_half_theta; 
    
    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q

@torch.jit.script
def calc_heading(q):
    # type: (Tensor) -> Tensor
    # calculate heading direction from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading

@torch.jit.script
def calc_heading_quat(q):
    # type: (Tensor) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis)
    return heading_q

@torch.jit.script
def calc_heading_quat_inv(q):
    # type: (Tensor) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis)
    return heading_q

@torch.jit.script
def quat_pos(x):
    """
    make all the real part of the quaternion positive
    """
    q = x
    z = (q[..., 3:] < 0).float()
    q = (1 - 2 * z) * q
    return q

@torch.jit.script
def quat_abs(x):
    """
    quaternion norm (unit quaternion represents a 3D rotation, which has norm of 1)
    """
    x = x.norm(p=2, dim=-1)
    return x

@torch.jit.script
def quat_unit(x):
    """
    normalized quaternion with norm of 1
    """
    norm = quat_abs(x).unsqueeze(-1)
    return x / (norm.clamp(min=1e-9))


@torch.jit.script
def quat_normalize(q):
    """
    Construct 3D rotation from quaternion (the quaternion needs not to be normalized).
    """
    q = quat_unit(quat_pos(q))  # normalized to positive and unit quaternion
    return q

@torch.jit.script
def quat_mul_norm(x, y):
    """
    Combine two set of 3D rotations together using \**\* operator. The shape needs to be
    broadcastable
    """
    return quat_normalize(quat_mul(x, y))

@torch.jit.script
def quat_conjugate(x):
    """
    quaternion with its imaginary part negated
    """
    return torch.cat([-x[..., :3], x[..., 3:]], dim=-1)

@torch.jit.script
def quat_inverse(x):
    """
    The inverse of the rotation
    """
    return quat_conjugate(x)
# Three space convert functions: World, Camera and Canonical (human) space
import torch
from core.model.watch.math.angle import extract_yaw_mat_from_rotmat


# ----------------------- Space transform ----------------------- #
def generate_transform(global_orient, transl=None):
    """
    Get the transform factor from canonical space to world space.
    Args:
        global_orient: [*, 3, 3] rotation matrix from source space to target space
        transl: [*, 3] transform in target space
    Examples:
        Human canonical space to world space. We use global_orient and transl to get the location in world coordination.
    Returns:
        Ts: [*, 4, 4]
    """
    device, dtype = global_orient.device, global_orient.dtype
    Ts = torch.eye(4, device=device, dtype=dtype).repeat(global_orient.shape[:-2] + (1, 1))  # [*, 4, 4]
    Ts[..., :3, :3] = global_orient
    if transl is not None:
        Ts[..., :3, 3] = transl
    return Ts


def generate_inverse_transform(Ts):
    """
    Get the inverse transform factor from target space to source space.
    Notes:
        Inverse process: p = R^T @ (p' - t) = R^T @ p' - R^T @ t, so Tinv = [R^T, -R^T @ t].
    Args:
        Ts: [*, 4, 4] transform factor from source space to target space

    Returns:
        Tinv: [*, 4, 4] transform factor from target space to source space
    """
    R = Ts[..., :3, :3]
    t = Ts[..., :3, 3]
    Tinv = torch.eye(4, device=Ts.device, dtype=Ts.dtype).repeat(Ts.shape[:-2] + (1, 1))  # [*, 4, 4]
    Tinv[..., :3, :3] = R.transpose(-1, -2)
    Tinv[..., :3, 3] = -torch.einsum("...ij,...j->...i", R.transpose(-1, -2), t)
    return Tinv


def forward_full_trans(Ts, s, R, t):
    """
    应用完整的对齐变换到外参矩阵（同时处理旋转和平移）
    Args:
        Ts: 外参矩阵 (*,3,4)
        s: 缩放因子
        R: 旋转矩阵 (*,3,3)
        t: 平移向量 (*,3)
    Returns:
        对齐后的外参矩阵 (*,3,4)
    """
    R_orig = Ts[..., :3, :3]  # [*,3,3]
    T_orig = Ts[..., :3, 3]   # [*,3]

    # 计算相机中心位置: center = -R_orig^T @ T_orig
    center = -torch.einsum('...ij,...j->...i', R_orig.transpose(-1, -2), T_orig)

    # 应用变换: center_aligned = s * R @ center + t
    center_aligned = s * torch.einsum('...ij,...j->...i', R, center) + t

    # 计算新的旋转矩阵: R_new = R @ R_orig
    R_new = torch.einsum('...ij,...jk->...ik', R, R_orig)

    # 计算新的平移向量: T_new = -R_new @ center_aligned
    T_new = -torch.einsum('...ij,...j->...i', R_new, center_aligned)

    # 构建新的外参矩阵
    return torch.cat([R_new, T_new.unsqueeze(-1)], dim=-1)


def forward_trans(points, Ts):
    """
    Forward transform points by Ts
    Args:
        points: [*, N, 3] in source space
        Ts: [*, 4, 4] or [*, 3, 4], source to target.
    Examples:
        Extrinsic is the `Ts` from world space to camera space.
        >>> extrinsic = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, -1, 0], [2, 1, 0.5, 1]])
        >>> points_w = torch.tensor([[0, 0, 0]])
        >>> print(forward_trans(points_w, extrinsic))
    Returns:
        points: [*, N, 3] in target space
    """
    R = Ts[..., :3, :3]
    t = Ts[..., None, :3, 3]
    points = torch.einsum("...ij,...kj->...ki", R, points) + t
    return points


def inverse_trans(points, Ts):
    """
    Inverse transform points by Ts.
    Notes:
        Forword process: p' = R @ p + t
        Inverse process: p = R^T @ (p' - t)
        inverse_trans(points, Ts) == forward_trans(points, generate_inverse_transform(Ts))
    Args:
        points: [*, N, 3] in target space
        Ts: [*, 4, 4] or [*, 3, 4], source to target.
    Examples:
        You can use this factor to get camera trajectory.
        >>> extrinsic = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, -1, 0], [2, 1, 0.5, 1]])
        >>> points_c = torch.tensor([[0, 0, 0]])
        >>> print(inverse_trans(points_c, extrinsic))
    Returns:
        points: [*, N, 3] in source space
    """
    R = Ts[..., :3, :3]
    t = Ts[..., None, :3, 3]
    points = torch.einsum("...ij,...kj->...ki", R.transpose(-1, -2), points - t)
    return points


def trans_canon2cam(T_canon2w, T_w2cam):
    """
    Get the transform factor from canonical space to camera space.
    Note that this function is general for any space transform.
    Notes:
        T_canon2cam = R_w2cam @ (R_canon2w @ p + t_canon2w) + t_w2cam
                    = R_w2cam @ R_canon2w @ p + R_w2cam @ t_canon2w + t_w2cam
        => T_canon2cam = [R_w2cam @ R_canon2w, R_w2cam @ t_canon2w + t_w2cam]
    Args:
        T_canon2w: [*, 4, 4] canonical to world transform matrix
        T_w2cam: [*, 4, 4] camera extrinsic matrix
    Returns:
        T_canon2cam: [*, 4, 4]
    """
    R_w2cam = T_w2cam[..., :3, :3]
    t_w2cam = T_w2cam[..., :3, 3]

    R_canon2w = T_canon2w[..., :3, :3]
    t_canon2w = T_canon2w[..., :3, 3]

    R_canon2cam = torch.einsum("...ij,...jk->...ik", R_w2cam, R_canon2w)
    t_canon2cam = torch.einsum("...ij,...j->...i", R_w2cam, t_canon2w) + t_w2cam

    return generate_transform(R_canon2cam, t_canon2cam)


# ----------------------- Gravity vector transform ----------------------- #
# Gravity is [0, 0, -1] (zdown) in world space,
# so use world2canon and world2cam without transl for space transform.
def gravity_in_canon(global_orient):
    """
    Convert gravity vector in world space to canonical space, gravity vector is [0, -1, 0] (ydown) in world space.
    Args:
        global_orient: [*, 3, 3]
    Returns:
        gravity_canonical: [*, 1, 3]
    """
    device = global_orient.device
    gravity = torch.tensor([0, -1, 0], device=device, dtype=torch.float32)
    gravity = gravity.repeat(global_orient.shape[:-2] + (1, 1))  # [*, 1, 3]
    T_canon2w = generate_transform(global_orient)
    return inverse_trans(gravity, T_canon2w)
    # T_w2canon = generate_inverse_transform(generate_transform(global_orient))
    # return forward_trans(gravity, T_w2canon)


def gravity_in_cam(T_w2c):
    """
    Convert gravity vector in world space to camera space, gravity vector is [0, -1, 0] (ydown) in world space.
    Args:
        T_w2c: [*, 4, 4] camera extrinsic matrix, only rot mat is used
    Returns:
        gravity_camera: [*, 1, 3]
    """
    device = T_w2c.device
    gravity = torch.tensor([0, -1, 0], device=device, dtype=torch.float32)
    gravity = gravity.repeat(T_w2c.shape[:-2] + (1, 1))  # [*, 1, 3]
    rotation_extrinsic = generate_transform(T_w2c[..., :3, :3])
    return forward_trans(gravity, rotation_extrinsic)

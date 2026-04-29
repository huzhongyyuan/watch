import torch


def extract_yaw_mat_from_rotmat(mat):
    """
    从旋转矩阵(R = Rx @ Ry @ Rz)中提取绕z轴的航向旋转矩阵(Rz)和剩余旋转分量(Rx@Ry)

    通过强制z轴为[0,0,1]并重新正交化x轴和y轴，分离出航向旋转分量。

    参数:
        mat: [*, 3, 3] 原始旋转矩阵

    返回:
        Rz: [*, 3, 3] 绕z轴的旋转矩阵（航向角部分）
        rest_rot: [*, 3, 3] 剩余旋转分量（俯仰和横滚）
    """
    # 创建原始矩阵的副本
    Rz = mat.clone()

    # 强制z轴为[0,0,1] (列向量)
    Rz[..., :, 2] = torch.tensor([0, 0, 1], device=mat.device, dtype=mat.dtype)

    # 获取列向量
    x = Rz[..., :, 0]
    y = Rz[..., :, 1]
    z = Rz[..., :, 2]

    # 正交化x轴（移除z分量）
    x_proj_z = torch.einsum('...i,...i->...', x, z).unsqueeze(-1)
    x = x - x_proj_z * z
    x = x / torch.norm(x, dim=-1, keepdim=True)

    # 正交化y轴
    y_proj_z = torch.einsum('...i,...i->...', y, z).unsqueeze(-1)
    y_proj_x = torch.einsum('...i,...i->...', y, x).unsqueeze(-1)
    y = y - y_proj_z * z - y_proj_x * x
    y = y / torch.norm(y, dim=-1, keepdim=True)

    # 重建正交矩阵
    Rz[..., :, 0] = x
    Rz[..., :, 1] = y
    Rz[..., :, 2] = z

    # 计算剩余旋转分量: rest_rot = mat @ Rz^T
    rest_rot = torch.matmul(mat, Rz.transpose(-1, -2))

    return Rz, rest_rot


def yaw_rotmat2angle(mat):
    """
    从旋转矩阵中提取绕z轴的航向角(yaw)，使用正交化方法避免不连续问题

    参数:
        mat: [*, 3, 3] 旋转矩阵

    返回:
        yaw: [*] 绕z轴的旋转角度(弧度)
    """
    Rz, _ = extract_yaw_mat_from_rotmat(mat)
    sin_theta = Rz[..., 1, 0]
    cos_theta = Rz[..., 0, 0]
    yaw = torch.atan2(sin_theta, cos_theta)

    return yaw


def yaw_angle2rotmat(yaw_angle):
    """
    Convert yaw angle to rotation matrix, this yaw is the rotation around z-axis in world space
    Args:
        yaw_angle: [*, 1], in rad
    Returns:
        rotmat: [*, 3, 3]
    """
    rotmat = torch.eye(3, device=yaw_angle.device).repeat(*yaw_angle.shape[:-1], 1, 1)
    rotmat[..., 0, 0] = torch.cos(yaw_angle)
    rotmat[..., 0, 1] = -torch.sin(yaw_angle)
    rotmat[..., 1, 0] = torch.sin(yaw_angle)
    rotmat[..., 1, 1] = torch.cos(yaw_angle)
    return rotmat
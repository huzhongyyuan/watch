"""
DPVO-Guided Translation Fusion

Key insight: Use DPVO camera velocity directly (not model predictions) to guide human translation.
When human appears stationary in camera (bbox stable), human world velocity ≈ camera world velocity.

This avoids:
1. Accumulated prediction errors in pred_cam_transl
2. Scale mismatch between model predictions and DPVO
"""
import torch
import torch.nn.functional as F
# from core.utils.geo.rotation import axis_angle_to_matrix


def gaussian_smooth(x, sigma=5.0, dim=-1):
    """Apply Gaussian smoothing along specified dimension."""
    if sigma <= 0:
        return x
    kernel_size = int(4 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    
    # Create Gaussian kernel
    x_kernel = torch.arange(-pad, pad + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-0.5 * (x_kernel / sigma) ** 2)
    kernel = kernel / kernel.sum()
    
    # Apply convolution
    if dim == -1:
        dim = x.ndim - 1
    
    # Reshape for conv1d: (B, C, L)
    orig_shape = x.shape
    if dim != x.ndim - 1:
        x = x.transpose(dim, -1)
    
    x_flat = x.reshape(-1, 1, x.shape[-1])  # (B*C, 1, L)
    kernel = kernel.view(1, 1, -1)
    
    # Pad and convolve
    x_padded = F.pad(x_flat, (pad, pad), mode='replicate')
    x_smooth = F.conv1d(x_padded, kernel)
    
    # Reshape back
    x_smooth = x_smooth.reshape(*x.shape[:-1], -1)
    if dim != orig_shape.__len__() - 1:
        x_smooth = x_smooth.transpose(dim, -1)
    
    return x_smooth


def apply_dpvo_guided_fusion(inputs, outputs, cfg):
    """
    Apply DPVO-guided translation fusion.
    
    Args:
        inputs: dict with keys:
            - cam_transl_local: (B, L, 3) DPVO camera velocity in local frame
            - cam_orient: (B, L, 3, 3) or (B, L, 6) camera orientation
            - bbx_xys: (B, L, 3) bbox [x, y, scale]
        outputs: dict with keys:
            - pred_smpl_params_global: dict with 'transl' (B, L, 3) or (L, 3)
            - pred_cam_orient_w: (B, L, 3, 3) or (L, 3, 3)
            - pred_human_move: dict with 'human_local_vel' (B, L, 3) or (L, 3)
        cfg: config with fusion parameters
    
    Returns:
        Updated outputs dict with corrected translation
    """
    if not getattr(cfg, 'enabled', False):
        return outputs
    
    # Get human rollout trajectory
    human_roll = outputs['pred_smpl_params_global']['transl']
    squeeze_output = human_roll.ndim == 2
    
    def ensure_batch_dim(x):
        if x is None:
            return None
        return x[None] if x.ndim == 2 else x
    
    # Ensure batch dimension
    human_roll_b = ensure_batch_dim(human_roll)  # (B, L, 3)
    dpvo_cam_vel_local = ensure_batch_dim(inputs['cam_transl_local'])  # (B, L, 3)
    cam_ori_w = ensure_batch_dim(outputs['pred_cam_orient_w'])  # (B, L, 3, 3)
    human_local_vel = ensure_batch_dim(outputs['pred_human_move']['human_local_vel'])  # (B, L, 3)
    bbx_xys = ensure_batch_dim(inputs['bbx_xys'])  # (B, L, 3)
    
    B, L = human_roll_b.shape[:2]
    
    # Convert DPVO camera velocity to world frame
    # dpvo_cam_vel_world[t] = cam_ori_w[t] @ dpvo_cam_vel_local[t]
    dpvo_cam_vel_world = torch.einsum('...ij,...j->...i', cam_ori_w, dpvo_cam_vel_local)  # (B, L, 3)
    
    # Compute human velocity from rollout (for comparison)
    human_vel_from_roll = torch.cat([
        human_roll_b[:, 1:] - human_roll_b[:, :-1],
        torch.zeros_like(human_roll_b[:, :1])
    ], dim=1)  # (B, L, 3)
    
    # Compute speeds
    cam_speed = dpvo_cam_vel_local.norm(dim=-1)  # (B, L)
    human_speed = human_local_vel.norm(dim=-1)  # (B, L)
    
    # Compute bbox stability (small delta = stable)
    bbx_scale = bbx_xys[..., 2]  # (B, L)
    bbx_delta = torch.cat([
        (torch.log(torch.clamp(bbx_scale[:, 1:], min=1e-6)) - 
         torch.log(torch.clamp(bbx_scale[:, :-1], min=1e-6))).abs(),
        torch.zeros_like(bbx_scale[:, :1])
    ], dim=1)  # (B, L)
    
    # Compute sequence-level statistics
    seq_cam_speed = cam_speed.mean(dim=1, keepdim=True).expand(-1, L)  # (B, L)
    seq_human_speed = human_speed.mean(dim=1, keepdim=True).expand(-1, L)
    seq_bbox_delta = bbx_delta.mean(dim=1, keepdim=True).expand(-1, L)
    seq_ratio = seq_cam_speed / torch.clamp(seq_human_speed, min=1e-6)
    
    # Sequence-level gate: only apply correction if camera dominates motion
    seq_gate = (seq_ratio > getattr(cfg, 'seq_ratio_thr', 2.0)).float()
    seq_gate = seq_gate * (seq_cam_speed > getattr(cfg, 'seq_cam_speed_thr', 0.02)).float()
    seq_gate = seq_gate * (seq_bbox_delta < getattr(cfg, 'seq_bbox_delta_thr', 0.016)).float()
    
    # Frame-level alpha: blend weight for camera guidance
    # High alpha when: camera fast, human slow, bbox stable
    alpha = torch.sigmoid(getattr(cfg, 'cam_speed_gain', 10.0) * 
                         (cam_speed - getattr(cfg, 'cam_speed_thr', 0.08)))
    alpha = alpha * torch.sigmoid(getattr(cfg, 'human_speed_gain', 10.0) * 
                                  (getattr(cfg, 'human_speed_thr', 0.12) - human_speed))
    alpha = alpha * torch.sigmoid(getattr(cfg, 'bbox_gain', 18.0) * 
                                  (getattr(cfg, 'bbox_delta_thr', 0.035) - bbx_delta))
    alpha = alpha * seq_gate
    alpha = alpha.clamp(0.0, getattr(cfg, 'alpha_cap', 0.85))  # (B, L)
    
    # Apply velocity correction
    # corrected_vel[t] = (1 - alpha[t]) * human_vel[t] + alpha[t] * dpvo_cam_vel_world[t]
    corrected_vel = (1 - alpha[..., None]) * human_vel_from_roll + alpha[..., None] * dpvo_cam_vel_world
    
    # Only correct horizontal (XZ), keep vertical (Y) from model
    corrected_vel[..., 1] = human_vel_from_roll[..., 1]
    
    # Apply smoothing
    smooth_sigma = getattr(cfg, 'smooth_sigma', 5.0)
    if smooth_sigma > 0:
        corrected_vel_xz = torch.stack([
            gaussian_smooth(corrected_vel[..., 0], sigma=smooth_sigma, dim=-1),
            gaussian_smooth(corrected_vel[..., 2], sigma=smooth_sigma, dim=-1)
        ], dim=-1)
        corrected_vel[..., 0] = corrected_vel_xz[..., 0]
        corrected_vel[..., 2] = corrected_vel_xz[..., 1]
    
    # Re-integrate from first frame
    # human_corrected[0] = human_roll[0]
    # human_corrected[t] = human_corrected[t-1] + corrected_vel[t-1]
    human_corrected = torch.zeros_like(human_roll_b)
    human_corrected[:, 0] = human_roll_b[:, 0]
    for t in range(1, L):
        human_corrected[:, t] = human_corrected[:, t-1] + corrected_vel[:, t-1]
    
    # Update outputs
    outputs['pred_smpl_params_global']['transl'] = human_corrected[0] if squeeze_output else human_corrected
    outputs['dpvo_fusion_alpha'] = alpha[0] if squeeze_output else alpha
    outputs['dpvo_fusion_seq_gate'] = seq_gate[0] if squeeze_output else seq_gate
    
    return outputs

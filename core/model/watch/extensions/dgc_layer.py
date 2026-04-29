"""
Differentiable Geometric Coupling Layer (DGC)

Key idea: Force forward consistency between cam_pose and human_pose:
  human_world = cam_pose ∘ human_in_cam  (geometric path)
  human_world = rollout(local_vel)        (motion path)

The two paths should agree. DGC iteratively refines them via learned gates.

This makes the network ARCHITECTURALLY use camera information (vs just
predicting it as a sibling output).

Usage:
    dgc = DGCLayer(latent_dim=512, K=3)
    refined_human_w, residual_features = dgc(
        cam_orient_w, cam_transl_w,
        human_orient_c, human_transl_c,
        human_orient_w_init, human_transl_w_init,
        local_vel,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class DGCLayer(nn.Module):
    """Differentiable Geometric Coupling Layer.

    Iteratively refines world-frame human pose by enforcing:
        human_w_geometric = cam_orient_w @ human_orient_c (rotation)
        human_t_w_geometric = cam_orient_w @ human_t_c + cam_t_w (translation)
        human_w_motion = rollout(local_vel, ...)

    Args:
        latent_dim: feature dim for residual encoding
        K: number of refinement iterations
        share_weights: whether to share update weights across iterations
    """

    def __init__(self, latent_dim=256, K=3, share_weights=True):
        super().__init__()
        self.K = K
        self.share_weights = share_weights

        # Encode geometric/motion residuals into features
        # Input: 6 dim residual (3 rotation + 3 translation, simplified)
        self.residual_encoder = nn.Sequential(
            nn.Linear(12, latent_dim),  # 6 rot residual + 3 transl + 3 vel residual
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        if share_weights:
            self.update_net = self._build_update_net(latent_dim)
        else:
            self.update_net = nn.ModuleList(
                [self._build_update_net(latent_dim) for _ in range(K)]
            )

    def _build_update_net(self, latent_dim):
        """Build the update prediction network. Zero-init for stable residual training."""
        return nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            zero_module(nn.Linear(latent_dim, 6 + 3 + 1)),
            # 6: rotation update (axis-angle)
            # 3: translation update
            # 1: gate scalar (sigmoid)
        )

    def get_update_net(self, k):
        if self.share_weights:
            return self.update_net
        return self.update_net[k]

    def forward(
        self,
        cam_orient_w,         # (B, L, 3, 3) camera orientation in world
        cam_transl_w,          # (B, L, 3)    camera translation in world
        human_orient_c,        # (B, L, 3, 3) human orientation in camera frame
        human_transl_c,        # (B, L, 3)    human translation in camera frame
        human_orient_w_init,   # (B, L, 3, 3) initial estimate of human in world
        human_transl_w_init,   # (B, L, 3)    initial estimate
        local_vel,             # (B, L, 3)    predicted local velocity (for motion path)
    ):
        """
        Returns:
            human_orient_w: (B, L, 3, 3) refined orientation
            human_transl_w: (B, L, 3) refined translation
            residual_features: (B, L, latent_dim) encoded residuals (can be fed back to backbone)
        """
        B, L = cam_orient_w.shape[:2]

        # Geometric path: world = cam @ in_cam (no learning, pure geometry)
        human_orient_w_geo = torch.einsum('blij,bljk->blik', cam_orient_w, human_orient_c)
        human_transl_w_geo = torch.einsum('blij,blj->bli', cam_orient_w, human_transl_c) + cam_transl_w

        human_orient_w = human_orient_w_init
        human_transl_w = human_transl_w_init

        all_residual_features = []

        for k in range(self.K):
            # Compute residuals (geometric vs current)
            rot_residual = self._rotation_residual(human_orient_w_geo, human_orient_w)  # (B, L, 6)
            transl_residual = human_transl_w_geo - human_transl_w  # (B, L, 3)

            # Motion path residual (rollout(vel) should match transl)
            transl_motion = self._integrate_velocity(local_vel, cam_orient_w, human_transl_w[:, :1])
            vel_residual = transl_motion - human_transl_w  # (B, L, 3)

            # Encode residuals into features
            residual = torch.cat([rot_residual, transl_residual, vel_residual], dim=-1)  # (B, L, 12)
            feat = self.residual_encoder(residual)
            all_residual_features.append(feat)

            # Predict update
            update_out = self.get_update_net(k)(feat)  # (B, L, 10)
            rot_update = update_out[..., :6]
            transl_update = update_out[..., 6:9]
            gate = torch.sigmoid(update_out[..., 9:10])  # (B, L, 1) in [0, 1]

            # Apply update with gate
            human_orient_w = self._apply_rot_update(human_orient_w, rot_update * gate)
            human_transl_w = human_transl_w + transl_update * gate

        # Final residual feature (last iter)
        return human_orient_w, human_transl_w, all_residual_features[-1]

    @staticmethod
    def _rotation_residual(R_target, R_current):
        """Compute rotation residual as relative rotation in 6d form (flatten of two columns)."""
        R_rel = torch.einsum('blij,blkj->blik', R_target, R_current)
        # Take first two columns flattened (6D rep)
        return R_rel[..., :, :2].flatten(-2, -1)  # (B, L, 6)

    @staticmethod
    def _apply_rot_update(R_current, rot_update_6d):
        """Apply rotation update via 6D representation.

        rot_update_6d: (B, L, 6) - additive offset in 6D rep
        """
        # Convert 6D update to rotation matrix via Gram-Schmidt
        # First, get current 6D form, add update, project back
        R_curr_6d = R_current[..., :, :2].flatten(-2, -1)
        R_new_6d = R_curr_6d + rot_update_6d * 0.1  # small step

        # Gram-Schmidt projection back to SO(3)
        a1 = R_new_6d[..., :3]
        a2 = R_new_6d[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)  # (B, L, 3, 3)

    @staticmethod
    def _integrate_velocity(local_vel, cam_orient_w, transl_0):
        """Rollout: transl[t] = transl_0 + sum_{i<t} (R_w[i] @ vel[i]).

        Note: local_vel is in CAMERA-LOCAL frame, R rotates to WORLD.
        """
        # Transform local vel to world via cam orientation
        vel_w = torch.einsum('blij,blj->bli', cam_orient_w, local_vel)  # (B, L, 3)
        # Cumsum (with first frame = transl_0)
        # transl[t] = transl_0 + sum_{i=0}^{t-1} vel_w[i]
        transl = torch.cumsum(vel_w[:, :-1], dim=1)
        transl = torch.cat([torch.zeros_like(transl_0), transl], dim=1)
        return transl + transl_0


class DGCWrapper(nn.Module):
    """Wrapper that integrates DGC into WATCH pipeline output.

    Drop-in replacement for the part of pipeline that produces final human_w.
    """

    def __init__(self, latent_dim=256, K=3):
        super().__init__()
        self.dgc = DGCLayer(latent_dim=latent_dim, K=K)

    def forward(self, outputs):
        """
        Args:
            outputs: dict from pipeline.forward, expected keys:
                pred_cam_orient_w: (B, L, 3, 3)
                pred_cam_transl: (B, L, 3)
                pred_smpl_params_incam: dict with 'global_orient' (axis-angle) and 'transl'
                pred_smpl_params_global: dict with 'global_orient' and 'transl' (initial)
                pred_human_move: dict with 'human_local_vel' (B, L, 3)

        Returns:
            outputs dict with refined pred_smpl_params_global
        """
        from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

        cam_orient_w = outputs["pred_cam_orient_w"]
        cam_transl_w = outputs["pred_cam_transl"]
        # convert axis-angle to matrix
        human_orient_c_aa = outputs["pred_smpl_params_incam"]["global_orient"]
        human_orient_c = axis_angle_to_matrix(human_orient_c_aa)
        human_transl_c = outputs["pred_smpl_params_incam"]["transl"]
        human_orient_w_init_aa = outputs["pred_smpl_params_global"]["global_orient"]
        human_orient_w_init = axis_angle_to_matrix(human_orient_w_init_aa)
        human_transl_w_init = outputs["pred_smpl_params_global"]["transl"]
        local_vel = outputs["pred_human_move"]["human_local_vel"]

        # Add batch dim if missing
        added_batch = False
        if cam_orient_w.ndim == 3:  # (L, 3, 3) — no batch
            for k in [cam_orient_w, cam_transl_w, human_orient_c, human_transl_c,
                       human_orient_w_init, human_transl_w_init, local_vel]:
                pass  # placeholder
            cam_orient_w = cam_orient_w.unsqueeze(0)
            cam_transl_w = cam_transl_w.unsqueeze(0)
            human_orient_c = human_orient_c.unsqueeze(0)
            human_transl_c = human_transl_c.unsqueeze(0)
            human_orient_w_init = human_orient_w_init.unsqueeze(0)
            human_transl_w_init = human_transl_w_init.unsqueeze(0)
            local_vel = local_vel.unsqueeze(0)
            added_batch = True

        refined_orient_w, refined_transl_w, _ = self.dgc(
            cam_orient_w, cam_transl_w,
            human_orient_c, human_transl_c,
            human_orient_w_init, human_transl_w_init,
            local_vel,
        )

        if added_batch:
            refined_orient_w = refined_orient_w.squeeze(0)
            refined_transl_w = refined_transl_w.squeeze(0)

        # Update outputs (in-place)
        outputs["pred_smpl_params_global"]["global_orient"] = matrix_to_axis_angle(refined_orient_w)
        outputs["pred_smpl_params_global"]["transl"] = refined_transl_w
        outputs["dgc_refined"] = True

        return outputs

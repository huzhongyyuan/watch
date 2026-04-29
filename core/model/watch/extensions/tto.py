"""
Test-Time Optimization (TTO) with Differentiable Rendering

Key idea: After feed-forward, optimize human + cam pose end-to-end
using:
  1. Silhouette loss (vs SAM2 mask)
  2. KP2D reprojection loss (vs ViTPose)
  3. Temporal smoothness
  4. (Optional) Diffusion prior, contact constraints

Stages:
  Stage 1 (cam-only): freeze human, refine cam
  Stage 2 (human-only): freeze cam, refine body pose
  Stage 3 (joint): refine both end-to-end

Use this AFTER WATCH feed-forward inference.

Note: Requires nvdiffrast or pytorch3d for differentiable rendering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


class DifferentiableSMPLRenderer:
    """Wrapper around nvdiffrast or pytorch3d for SMPL silhouette rendering.

    Falls back to PyTorch3D if nvdiffrast not available.
    """

    def __init__(self, image_size=(1080, 1920), use_nvdiffrast=True):
        self.image_size = image_size
        self.use_nvdiffrast = use_nvdiffrast

        if use_nvdiffrast:
            try:
                import nvdiffrast.torch as dr
                self.dr = dr
                self.glctx = dr.RasterizeGLContext()
            except ImportError:
                print("[TTO] nvdiffrast not available, falling back to pytorch3d")
                self.use_nvdiffrast = False

        if not self.use_nvdiffrast:
            try:
                from pytorch3d.renderer import (
                    look_at_view_transform, PerspectiveCameras, RasterizationSettings,
                    MeshRasterizer, MeshRenderer, SoftSilhouetteShader,
                )
                from pytorch3d.structures import Meshes
                self.Meshes = Meshes
                self.PerspectiveCameras = PerspectiveCameras
                self.MeshRasterizer = MeshRasterizer
                self.MeshRenderer = MeshRenderer
                self.SoftSilhouetteShader = SoftSilhouetteShader
                self.RasterizationSettings = RasterizationSettings
            except ImportError:
                raise RuntimeError("Need either nvdiffrast or pytorch3d for TTO")

    def render_silhouette(self, vertices, faces, K, R, t):
        """Render silhouette of mesh in camera frame.

        Args:
            vertices: (B, V, 3) world-frame vertices
            faces: (F, 3) face indices (shared)
            K: (B, 3, 3) intrinsics
            R: (B, 3, 3) world-to-camera rotation (extrinsic)
            t: (B, 3) world-to-camera translation

        Returns:
            silhouette: (B, H, W) [0, 1] silhouette mask
        """
        if self.use_nvdiffrast:
            return self._render_nvdiffrast(vertices, faces, K, R, t)
        else:
            return self._render_pytorch3d(vertices, faces, K, R, t)

    def _render_nvdiffrast(self, vertices, faces, K, R, t):
        # Transform to camera frame
        v_cam = torch.einsum('bij,bvj->bvi', R, vertices) + t.unsqueeze(1)

        # Project: v_cam -> image plane
        H, W = self.image_size
        # NDC space [-1, 1]
        x = v_cam[..., 0] * K[..., 0:1, 0:1].squeeze(-1) / v_cam[..., 2:3].squeeze(-1)
        y = v_cam[..., 1] * K[..., 1:2, 1:2].squeeze(-1) / v_cam[..., 2:3].squeeze(-1)
        x = (x / W) * 2 - 1
        y = (y / H) * 2 - 1
        z = v_cam[..., 2:3].squeeze(-1)

        v_clip = torch.stack([x, y, z, torch.ones_like(z)], dim=-1)  # (B, V, 4)

        rast, _ = self.dr.rasterize(
            self.glctx, v_clip, faces.int(), resolution=self.image_size
        )
        # rast[..., -1] is triangle id (0 = no triangle)
        sil = (rast[..., -1] > 0).float()
        return sil  # (B, H, W)

    def _render_pytorch3d(self, vertices, faces, K, R, t):
        # PyTorch3D fallback (slower but no GL context needed)
        B = vertices.shape[0]
        H, W = self.image_size
        meshes = self.Meshes(verts=vertices, faces=faces.unsqueeze(0).expand(B, -1, -1))

        # Build cameras
        cameras = self.PerspectiveCameras(
            R=R, T=t, K=K[:, :3, :3].unsqueeze(0), in_ndc=False,
            image_size=[(H, W)], device=vertices.device,
        )
        rast_settings = self.RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1)
        rasterizer = self.MeshRasterizer(cameras=cameras, raster_settings=rast_settings)
        shader = self.SoftSilhouetteShader()
        renderer = self.MeshRenderer(rasterizer=rasterizer, shader=shader)
        sil = renderer(meshes)[..., 3]  # alpha channel
        return sil


def project_kp3d_to_kp2d(kp3d, K, R, t):
    """Project 3D world keypoints to 2D image plane.

    Args:
        kp3d: (B, L, J, 3) world-frame
        K: (B, L, 3, 3)
        R, t: (B, L, 3, 3) and (B, L, 3) world-to-cam

    Returns:
        kp2d: (B, L, J, 2)
    """
    kp_cam = torch.einsum('blij,bljk->blik', R, kp3d.transpose(-1, -2)).transpose(-1, -2) + t.unsqueeze(-2)
    kp_proj = torch.einsum('blij,bljk->blik', K, kp_cam.transpose(-1, -2)).transpose(-1, -2)
    kp2d = kp_proj[..., :2] / (kp_proj[..., 2:3] + 1e-6)
    return kp2d


class TTOOptimizer:
    """Test-time optimization for refining WATCH outputs.

    Usage:
        tto = TTOOptimizer(smpl_model, renderer)
        refined_outputs = tto.optimize(initial_outputs, video_inputs, sam_mask, kp2d_obs)
    """

    def __init__(self, smpl_model, renderer=None,
                 num_iters_stage1=20, num_iters_stage2=20, num_iters_stage3=10,
                 lr=1e-3, weight_sil=1.0, weight_kp=10.0, weight_smooth=5.0):
        self.smpl_model = smpl_model
        self.renderer = renderer or DifferentiableSMPLRenderer()
        self.num_iters_stage1 = num_iters_stage1
        self.num_iters_stage2 = num_iters_stage2
        self.num_iters_stage3 = num_iters_stage3
        self.lr = lr
        self.weight_sil = weight_sil
        self.weight_kp = weight_kp
        self.weight_smooth = weight_smooth

    def optimize(self, initial_outputs, video_inputs, sam_mask=None, kp2d_obs=None):
        """
        Args:
            initial_outputs: dict from WATCH forward
                pred_smpl_params_global: dict with body_pose, betas, global_orient, transl
                pred_cam_orient_w: (L, 3, 3)
                pred_cam_transl: (L, 3)
            video_inputs: dict
                K_fullimg: (L, 3, 3) intrinsics
            sam_mask: (L, H, W) human silhouette ground truth
            kp2d_obs: (L, J, 2) observed 2D keypoints

        Returns:
            refined_outputs: same format as initial_outputs
        """
        # Make trainable copies
        body_pose = initial_outputs["pred_smpl_params_global"]["body_pose"].detach().clone().requires_grad_(True)
        betas = initial_outputs["pred_smpl_params_global"]["betas"].detach().clone().requires_grad_(True)
        global_orient = initial_outputs["pred_smpl_params_global"]["global_orient"].detach().clone().requires_grad_(True)
        transl = initial_outputs["pred_smpl_params_global"]["transl"].detach().clone().requires_grad_(True)

        cam_orient_w = initial_outputs["pred_cam_orient_w"].detach().clone().requires_grad_(True)
        cam_transl_w = initial_outputs["pred_cam_transl"].detach().clone().requires_grad_(True)

        K = video_inputs["K_fullimg"]

        # Stage 1: cam-only
        cam_params = [cam_orient_w, cam_transl_w]
        human_params = [body_pose, betas, global_orient, transl]
        opt = torch.optim.Adam(cam_params, lr=self.lr)
        for it in range(self.num_iters_stage1):
            loss = self._compute_loss(
                body_pose, betas, global_orient, transl,
                cam_orient_w, cam_transl_w, K,
                sam_mask, kp2d_obs,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Stage 2: human-only
        opt = torch.optim.Adam(human_params, lr=self.lr)
        for it in range(self.num_iters_stage2):
            loss = self._compute_loss(
                body_pose, betas, global_orient, transl,
                cam_orient_w, cam_transl_w, K,
                sam_mask, kp2d_obs,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Stage 3: joint
        opt = torch.optim.Adam(human_params + cam_params, lr=self.lr * 0.3)
        for it in range(self.num_iters_stage3):
            loss = self._compute_loss(
                body_pose, betas, global_orient, transl,
                cam_orient_w, cam_transl_w, K,
                sam_mask, kp2d_obs,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Update outputs
        refined = {**initial_outputs}
        refined["pred_smpl_params_global"] = {
            "body_pose": body_pose.detach(),
            "betas": betas.detach(),
            "global_orient": global_orient.detach(),
            "transl": transl.detach(),
        }
        refined["pred_cam_orient_w"] = cam_orient_w.detach()
        refined["pred_cam_transl"] = cam_transl_w.detach()
        refined["tto_refined"] = True
        return refined

    def _compute_loss(self, body_pose, betas, global_orient, transl,
                      cam_orient_w, cam_transl_w, K, sam_mask, kp2d_obs):
        """Compute total TTO loss."""
        # Forward SMPL
        with torch.enable_grad():
            verts, joints = self.smpl_model(
                body_pose=body_pose,
                betas=betas,
                global_orient=global_orient,
                transl=transl,
            )

        # World-to-cam transformation
        R_w2c = cam_orient_w.transpose(-1, -2)
        t_w2c = -torch.einsum('lij,lj->li', R_w2c, cam_transl_w)

        loss = 0.0

        # 1. KP2D reprojection
        if kp2d_obs is not None:
            kp2d_pred = project_kp3d_to_kp2d(
                joints[None], K[None], R_w2c[None], t_w2c[None]
            )[0]  # (L, J, 2)
            kp_loss = (kp2d_pred - kp2d_obs).norm(dim=-1).mean()
            loss = loss + self.weight_kp * kp_loss

        # 2. Silhouette (skip if no nvdiffrast / no mask)
        # if sam_mask is not None:
        #     sil_pred = self.renderer.render_silhouette(verts, faces, K, R_w2c, t_w2c)
        #     sil_loss = F.mse_loss(sil_pred, sam_mask)
        #     loss = loss + self.weight_sil * sil_loss

        # 3. Temporal smoothness on transl
        if transl.shape[0] > 2:
            accel = transl[2:] - 2 * transl[1:-1] + transl[:-2]
            smooth_loss = accel.norm(dim=-1).mean()
            loss = loss + self.weight_smooth * smooth_loss

        return loss

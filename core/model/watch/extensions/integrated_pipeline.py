"""
GeoPhys-WATCH Pipeline: Integrates all 4 extensions into WATCH

This is the main pipeline class that combines:
  1. DGC layer (forward geometric coupling)
  2. Dual-stream backbone (replaces single backbone)
  3. Physics constraint loss (auxiliary supervision during training)
  4. TTO at test time (separate eval pass)

Usage:
    pipeline:
        _target_: core.model.watch.extensions.integrated_pipeline.GeoPhysPipeline
        args_denoiser3d: ${network}  # should be DualStreamEncoder
        args:
            endecoder_opt: ${endecoder}
            normalize_cam_angvel: True
            use_dgc: True
            use_physics: True
            weights:
                cr_j3d: 500.
                ...
                physics_total: 10.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate

from core.model.watch.pipeline.watch_pipeline import (
    Pipeline as WatchPipelineBase,
    compute_extra_incam_loss,
    compute_extra_global_loss,
)
from core.model.watch.extensions.dgc_layer import DGCWrapper
from core.model.watch.extensions.physics_layer import (
    PhysicsConstraintLoss,
    ContactPredictionHead,
)
from core.model.watch.utils.endecoder import EnDecoder
from core.utils.smplx_utils import make_smplx
from core.configs import MainStore, builds


class GeoPhysPipeline(WatchPipelineBase):
    """WATCH pipeline with DGC + Physics + (optional) Dual-stream.

    Inherits from base WATCH Pipeline. Adds:
        - DGC layer applied after initial pose decode (refine human_w)
        - Physics constraint loss during training
        - Contact prediction head (from backbone features)
    """

    def __init__(self, args, args_denoiser3d, **kwargs):
        super().__init__(args, args_denoiser3d, **kwargs)

        self.use_dgc = getattr(args, "use_dgc", True)
        self.use_physics = getattr(args, "use_physics", True)
        self.use_contact_head = getattr(args, "use_contact_head", True)

        if self.use_dgc:
            dgc_K = getattr(args, "dgc_K", 3)
            dgc_latent_dim = getattr(args, "dgc_latent_dim", 256)
            self.dgc = DGCWrapper(latent_dim=dgc_latent_dim, K=dgc_K)

        if self.use_physics:
            self.physics_loss = PhysicsConstraintLoss(
                ground_y=getattr(args, "ground_y", 0.0),
                weight_ground=getattr(args, "physics_weight_ground", 1.0),
                weight_sliding=getattr(args, "physics_weight_sliding", 10.0),
                weight_smoothness=getattr(args, "physics_weight_smoothness", 0.1),
                weight_balance=getattr(args, "physics_weight_balance", 0.5),
            )
            # SMPL model for FK during physics loss
            self.smplx_lite = make_smplx("supermotion_smpl24")

        if self.use_contact_head:
            latent_dim = getattr(args_denoiser3d, "latent_dim", 512)
            self.contact_head = ContactPredictionHead(latent_dim=latent_dim, n_feet=4)

    def forward(self, inputs, train=False, postproc=False, static_cam=False,
                ablation_mask=None, postproc_version=1):
        """Forward pass with DGC + physics integration."""

        # 1. Run base WATCH forward (without postproc, since DGC needs raw output)
        outputs = super().forward(
            inputs, train=train, postproc=False,  # postproc later
            static_cam=static_cam, ablation_mask=ablation_mask,
            postproc_version=postproc_version,
        )

        # 2. DGC refinement (in forward)
        if self.use_dgc and "pred_smpl_params_global" in outputs:
            outputs = self.dgc(outputs)

        # 3. Contact prediction (auxiliary head from backbone features)
        if self.use_contact_head and "pred_context" in outputs.get("model_output", {}):
            backbone_feat = outputs["model_output"]["pred_context"]
            outputs["contact_logits"] = self.contact_head(backbone_feat)

        # 4. Apply postproc AFTER DGC (if requested)
        if not train and postproc:
            from core.model.watch.utils.postprocess import (
                pp_static_joint, pp_static_joint_cam, process_ik
            )
            if static_cam:
                outputs["pred_smpl_params_global"]["transl"] = pp_static_joint_cam(
                    outputs, self.endecoder
                )
            else:
                if postproc_version == 2:
                    from core.model.watch.utils.postprocess import pp_static_joint_v2
                    outputs["pred_smpl_params_global"]["transl"] = pp_static_joint_v2(
                        outputs, self.endecoder
                    )
                else:
                    outputs["pred_smpl_params_global"]["transl"] = pp_static_joint(
                        outputs, self.endecoder
                    )
            body_pose = process_ik(outputs, self.endecoder)
            outputs["pred_smpl_params_global"]["body_pose"] = body_pose
            outputs["pred_smpl_params_incam"]["body_pose"] = body_pose

        # 5. Physics loss (training only)
        if train and self.use_physics and "loss" in outputs:
            try:
                # Forward SMPL to get joints
                smpl_params_global = outputs["pred_smpl_params_global"]
                # Take only first batch dim — adapt as needed
                smpl_out = self.smplx_lite(
                    body_pose=smpl_params_global["body_pose"],
                    betas=smpl_params_global["betas"],
                    global_orient=smpl_params_global["global_orient"],
                    transl=smpl_params_global["transl"],
                )
                joints_world = smpl_out.joints[..., :22, :]  # body joints

                contact_logits = outputs.get("contact_logits", None)
                physics_losses = self.physics_loss(joints_world, contact_logits)

                # Add to total loss
                physics_total = physics_losses["physics_total"]
                physics_weight = getattr(self.args.weights, "physics_total", 1.0)
                outputs["loss"] = outputs["loss"] + physics_weight * physics_total

                # Log individual physics terms
                for k, v in physics_losses.items():
                    outputs[k] = v

            except Exception as e:
                # Don't crash training if physics computation fails
                print(f"[Physics] Warning: physics loss failed: {e}")

        return outputs


# Register
geophys_pipeline = builds(
    GeoPhysPipeline,
    args="${pipeline_watch.args}",
    args_denoiser3d="${pipeline_watch.args_denoiser3d}",
    populate_full_signature=True,
)
MainStore.store(name="geophys_pipeline", node=geophys_pipeline, group="pipeline")

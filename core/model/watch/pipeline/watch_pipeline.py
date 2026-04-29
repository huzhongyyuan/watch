import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from hydra.utils import instantiate

from core.model.watch.utils.endecoder import EnDecoder, EnDecoderV4
from core.model.watch.utils.postprocess import (
    pp_static_joint,
    process_ik,
    pp_static_joint_cam,
    pp_static_joint_v2,
)
from core.model.watch.utils import stats_compose

from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)
from core.utils.geo.hmr_cam import compute_bbox_info_bedlam, compute_transl_full_cam, get_a_pred_cam, project_to_bi01
from core.model.watch.pipeline.dpvo_guided_fusion import apply_dpvo_guided_fusion
from core.utils.geo.hmr_global import (
    get_static_joint_mask,
)
from core.utils.geo_transform import extract_rotation_components_ydown
from core.utils.net_utils import gaussian_smooth
from core.utils.geo.hmr_global import get_local_transl_vel, get_local_vel_mat, rollout_local_vel_rotmat


class Pipeline(nn.Module):
    def __init__(self, args, args_denoiser3d, **kwargs):
        super().__init__()
        self.args = args
        self.weights = args.weights  # loss weights

        # Networks
        self.denoiser3d = instantiate(args_denoiser3d, _recursive_=False)

        # Normalizer
        self.endecoder: EnDecoder = instantiate(args.endecoder_opt, _recursive_=False)
        # self.endecoder: EnDecoderV4 = instantiate(args.endecoder_opt, _recursive_=False)  # v4
        if self.args.normalize_cam_angvel:
            cam_angvel_stats = stats_compose.cam_angvel["manual"]
            self.register_buffer("cam_angvel_mean", torch.tensor(cam_angvel_stats["mean"]), persistent=False)
            self.register_buffer("cam_angvel_std", torch.tensor(cam_angvel_stats["std"]), persistent=False)

        self.traj_decoder = MoveDecoder()

        transl_fusion = getattr(self.args, "transl_fusion", None)
        self.use_transl_fusion = transl_fusion is not None and getattr(transl_fusion, "enabled", False)
        if self.use_transl_fusion:
            hidden_dim = getattr(transl_fusion, "hidden_dim", 32)
            self.transl_gate_mlp = nn.Sequential(
                nn.Linear(8, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.transl_res_mlp = nn.Sequential(
                nn.Linear(10, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 2),
            )
            nn.init.zeros_(self.transl_gate_mlp[-1].weight)
            nn.init.constant_(self.transl_gate_mlp[-1].bias, getattr(transl_fusion, "gate_bias", -10.0))
            nn.init.zeros_(self.transl_res_mlp[-1].weight)
            nn.init.zeros_(self.transl_res_mlp[-1].bias)

    def apply_learned_transl_fusion(self, inputs, outputs, train=False):
        if not self.use_transl_fusion:
            return outputs["pred_human_transl_rollout"]

        cfg = self.args.transl_fusion
        human_roll = outputs["pred_human_transl_rollout"]
        human_geom = outputs["pred_human_transl_geom"]
        cam_speed = outputs["pred_cam_move"]["cam_local_vel"].norm(dim=-1, keepdim=True)
        human_speed = outputs["pred_human_move"]["human_local_vel"].norm(dim=-1, keepdim=True)
        ratio = cam_speed / torch.clamp(human_speed, min=1e-6)

        bbx_scale = inputs["bbx_xys"][..., 2]
        bbx_delta = (torch.log(torch.clamp(bbx_scale[:, 1:], min=1e-6)) - torch.log(torch.clamp(bbx_scale[:, :-1], min=1e-6))).abs()
        bbx_delta = torch.cat([bbx_delta[:, :1], bbx_delta], dim=1)[..., None]

        geom_gap = human_geom - human_roll
        geom_gap_xz = geom_gap[..., [0, 2]]
        geom_gap_norm = geom_gap_xz.norm(dim=-1, keepdim=True)

        seq_ratio = ratio.mean(dim=1, keepdim=True).expand(-1, ratio.shape[1], -1)
        seq_cam_speed = cam_speed.mean(dim=1, keepdim=True).expand(-1, cam_speed.shape[1], -1)
        seq_bbox_delta = bbx_delta.mean(dim=1, keepdim=True).expand(-1, bbx_delta.shape[1], -1)

        gate_feat = torch.cat([cam_speed, human_speed, ratio, bbx_delta, geom_gap_norm, seq_ratio, seq_cam_speed, seq_bbox_delta], dim=-1)
        gate = torch.sigmoid(self.transl_gate_mlp(gate_feat))
        gate = gate.clamp(0.0, getattr(cfg, "alpha_cap", 1.0))

        res_feat = torch.cat([human_roll[..., [0, 2]], geom_gap_xz, cam_speed, human_speed, ratio, bbx_delta, seq_ratio, seq_bbox_delta], dim=-1)
        residual_xz = torch.tanh(self.transl_res_mlp(res_feat)) * getattr(cfg, "residual_scale", 0.05)

        if not train:
            smooth_sigma = getattr(cfg, "smooth_sigma", 0.0)
            if smooth_sigma and smooth_sigma > 0:
                gate = gaussian_smooth(gate.squeeze(-1), sigma=smooth_sigma, dim=-1)[..., None]
                residual_xz = torch.stack([gaussian_smooth(residual_xz[..., i], sigma=smooth_sigma, dim=-1) for i in range(2)], dim=-1)

        fused = human_roll.clone()
        fused[..., [0, 2]] = human_roll[..., [0, 2]] + gate * geom_gap_xz + residual_xz
        outputs["transl_fusion_gate"] = gate
        outputs["transl_fusion_residual_xz"] = residual_xz
        return fused

    def apply_followcam_fusion(self, inputs, outputs):
        cfg = getattr(self.args, "followcam_cfg", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return outputs

        human_roll = outputs["pred_smpl_params_global"]["transl"]
        squeeze_output = human_roll.ndim == 2

        def ensure_batch_dim(x):
            if x is None:
                return None
            if x.ndim == 2:
                return x[None]
            return x

        human_roll_b = ensure_batch_dim(human_roll)
        cam_trans_w = ensure_batch_dim(outputs["pred_cam_transl"])
        cam_ori_w = ensure_batch_dim(outputs["pred_cam_orient_w"])
        human_trans_c = ensure_batch_dim(outputs["pred_smpl_params_incam"]["transl"])
        human_ori_c = axis_angle_to_matrix(ensure_batch_dim(outputs["pred_smpl_params_incam"]["global_orient"]))
        cam_local_vel = ensure_batch_dim(outputs["pred_cam_move"]["cam_local_vel"])
        human_local_vel = ensure_batch_dim(outputs["pred_human_move"]["human_local_vel"])
        bbx_xys = ensure_batch_dim(inputs["bbx_xys"])

        _, human_geom = self.traj_decoder.infer_human_trans_from_cam_trans(
            cam_ori_w=cam_ori_w,
            cam_trans_w=cam_trans_w,
            human_ori_c=human_ori_c,
            human_trans_c=human_trans_c,
        )

        cam_speed = cam_local_vel.norm(dim=-1)
        human_speed = human_local_vel.norm(dim=-1)
        bbx_scale = bbx_xys[..., 2]
        bbx_delta = (torch.log(torch.clamp(bbx_scale[:, 1:], min=1e-6)) - torch.log(torch.clamp(bbx_scale[:, :-1], min=1e-6))).abs()
        bbx_delta = torch.cat([bbx_delta[:, :1], bbx_delta], dim=1)

        seq_cam_speed = cam_speed.mean(dim=-1, keepdim=True)
        seq_human_speed = human_speed.mean(dim=-1, keepdim=True)
        seq_bbox_delta = bbx_delta.mean(dim=-1, keepdim=True)
        seq_ratio = seq_cam_speed / torch.clamp(seq_human_speed, min=1e-6)

        seq_gate = (seq_ratio > getattr(cfg, "seq_ratio_thr", 2.0)).float()
        seq_gate = seq_gate * (seq_cam_speed > getattr(cfg, "seq_cam_speed_thr", 0.02)).float()
        seq_gate = seq_gate * (seq_bbox_delta < getattr(cfg, "seq_bbox_delta_thr", 0.016)).float()

        alpha = torch.sigmoid(getattr(cfg, "cam_speed_gain", 10.0) * (cam_speed - getattr(cfg, "cam_speed_thr", 0.08)))
        alpha = alpha * torch.sigmoid(getattr(cfg, "human_speed_gain", 10.0) * (getattr(cfg, "human_speed_thr", 0.12) - human_speed))
        alpha = alpha * torch.sigmoid(getattr(cfg, "bbox_gain", 18.0) * (getattr(cfg, "bbox_delta_thr", 0.035) - bbx_delta))
        alpha = alpha * seq_gate
        alpha = alpha.clamp(0.0, getattr(cfg, "alpha_cap", 0.85))[..., None]

        outputs["followcam_seq_ratio"] = seq_ratio[0, 0] if squeeze_output else seq_ratio
        outputs["followcam_seq_gate"] = seq_gate[0, 0] if squeeze_output else seq_gate

        delta = alpha * (human_geom - human_roll_b)
        # Only adjust horizontal trajectory; keep vertical motion from the original rollout.
        delta[..., 1] = 0.0

        smooth_sigma = getattr(cfg, "smooth_sigma", 5.0)
        if smooth_sigma and smooth_sigma > 0:
            delta_xz = torch.stack([gaussian_smooth(delta[..., i], sigma=smooth_sigma, dim=-1) for i in [0, 2]], dim=-1)
            delta[..., 0] = delta_xz[..., 0]
            delta[..., 2] = delta_xz[..., 1]

        max_step = getattr(cfg, "max_delta_step", 0.015)
        if max_step and max_step > 0:
            delta_step = delta[:, 1:] - delta[:, :-1]
            delta_step = delta_step.clamp(min=-max_step, max=max_step)
            delta = torch.cumsum(torch.cat([delta[:, :1], delta_step], dim=1), dim=1)

        max_abs = getattr(cfg, "max_delta_abs", 0.35)
        if max_abs and max_abs > 0:
            delta = delta.clamp(min=-max_abs, max=max_abs)

        fused = human_roll_b + delta
        outputs["pred_smpl_params_global"]["transl"] = fused[0] if squeeze_output else fused
        outputs["followcam_alpha"] = alpha.squeeze(-1)[0] if squeeze_output else alpha.squeeze(-1)
        return outputs

    def apply_sgf_lite_fusion(self, inputs, outputs):
        cfg = getattr(self.args, "sgf_lite", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return outputs
        if "cam_transl" not in inputs:
            return outputs

        human_roll = outputs["pred_smpl_params_global"]["transl"]
        squeeze_output = human_roll.ndim == 2

        def ensure_batch_dim(x):
            if x is None:
                return None
            if x.ndim == 2:
                return x[None]
            return x

        human_roll_b = ensure_batch_dim(human_roll)
        cam_trans_w = ensure_batch_dim(inputs["cam_transl"]).to(human_roll_b)
        if "cam_yaw" in inputs and "cam_roll_pitch" in inputs:
            cam_ori_w = self.traj_decoder.compose_global_orient(
                ensure_batch_dim(inputs["cam_yaw"]).to(human_roll_b),
                ensure_batch_dim(inputs["cam_roll_pitch"]).to(human_roll_b),
            )
        else:
            cam_ori_w = ensure_batch_dim(outputs["pred_cam_orient_w"]).to(human_roll_b)
        human_trans_c = ensure_batch_dim(outputs["pred_smpl_params_incam"]["transl"]).to(human_roll_b)
        human_ori_c = axis_angle_to_matrix(ensure_batch_dim(outputs["pred_smpl_params_incam"]["global_orient"]).to(human_roll_b))
        cam_local_vel = ensure_batch_dim(inputs["cam_transl_local"]).to(human_roll_b)
        human_local_vel = ensure_batch_dim(outputs["pred_human_move"]["human_local_vel"]).to(human_roll_b)
        bbx_xys = ensure_batch_dim(inputs["bbx_xys"]).to(human_roll_b)

        _, human_geom = self.traj_decoder.infer_human_trans_from_cam_trans(
            cam_ori_w=cam_ori_w,
            cam_trans_w=cam_trans_w,
            human_ori_c=human_ori_c,
            human_trans_c=human_trans_c,
        )

        # Reset geometry to the same origin as the rollout, then only use horizontal correction.
        human_geom = human_geom - human_geom[:, :1] + human_roll_b[:, :1]
        delta = human_geom - human_roll_b
        delta[..., 1] = 0.0

        cam_speed = cam_local_vel.norm(dim=-1)
        human_speed = human_local_vel.norm(dim=-1)
        ratio = cam_speed / torch.clamp(human_speed, min=1e-4)
        bbx_scale = torch.clamp(bbx_xys[..., 2], min=1e-6)
        bbx_delta = (torch.log(bbx_scale[:, 1:]) - torch.log(bbx_scale[:, :-1])).abs()
        bbx_delta = torch.cat([bbx_delta[:, :1], bbx_delta], dim=1)
        geom_gap = delta[..., [0, 2]].norm(dim=-1)

        seq_cam_speed = cam_speed.mean(dim=1, keepdim=True)
        seq_human_speed = human_speed.mean(dim=1, keepdim=True)
        seq_ratio = seq_cam_speed / torch.clamp(seq_human_speed, min=1e-4)
        seq_bbx_delta = bbx_delta.mean(dim=1, keepdim=True)

        alpha = torch.sigmoid(getattr(cfg, "cam_speed_gain", 10.0) * (cam_speed - getattr(cfg, "cam_speed_thr", 0.08)))
        alpha = alpha * torch.sigmoid(getattr(cfg, "ratio_gain", 4.0) * (ratio - getattr(cfg, "ratio_thr", 1.3)))
        alpha = alpha * torch.sigmoid(getattr(cfg, "bbox_gain", 20.0) * (getattr(cfg, "bbox_delta_thr", 0.04) - bbx_delta))
        alpha = alpha * torch.sigmoid(getattr(cfg, "geom_gap_gain", 6.0) * (geom_gap - getattr(cfg, "geom_gap_thr", 0.25)))

        seq_gate = (seq_cam_speed > getattr(cfg, "seq_cam_speed_thr", 0.03)).float()
        seq_gate = seq_gate * (seq_ratio > getattr(cfg, "seq_ratio_thr", 1.1)).float()
        seq_gate = seq_gate * (seq_bbx_delta < getattr(cfg, "seq_bbox_delta_thr", 0.035)).float()
        alpha = alpha * seq_gate
        alpha = alpha.clamp(0.0, getattr(cfg, "alpha_cap", 0.35))[..., None]

        smooth_sigma = getattr(cfg, "smooth_sigma", 5.0)
        if smooth_sigma and smooth_sigma > 0:
            alpha = gaussian_smooth(alpha.squeeze(-1), sigma=smooth_sigma, dim=-1)[..., None]
            delta_xz = torch.stack([gaussian_smooth(delta[..., i], sigma=smooth_sigma, dim=-1) for i in [0, 2]], dim=-1)
            delta[..., 0] = delta_xz[..., 0]
            delta[..., 2] = delta_xz[..., 1]

        max_delta = getattr(cfg, "max_delta", 0.6)
        if max_delta and max_delta > 0:
            delta = delta.clamp(min=-max_delta, max=max_delta)

        fused = human_roll_b + alpha * delta
        outputs["pred_smpl_params_global"]["transl"] = fused[0] if squeeze_output else fused
        outputs["sgf_lite_alpha"] = alpha.squeeze(-1)[0] if squeeze_output else alpha.squeeze(-1)
        return outputs

    # ========== Training ========== #
    def forward(self, inputs, train=False, postproc=False, static_cam=False, ablation_mask=None, postproc_version=1):
        """
        Args:
            ablation_mask: dict, keys are condition names to mask out (set to zero).
                          e.g., {"f_cam_angvel": True, "f_cam_transl": True} to mask both camera conditions.
                          Used for zero-cost ablation studies at inference time.
        """
        outputs = dict()
        length = inputs["length"]  # (B,) effective length of each sample

        # 1. Conditions modified from GVHMR, with additional camera SLAM info
        cliff_cam = compute_bbox_info_bedlam(inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)
        f_cam_angvel = inputs["cam_angvel"]
        if self.args.normalize_cam_angvel:
            f_cam_angvel = (f_cam_angvel - self.cam_angvel_mean) / self.cam_angvel_std
        f_condition = {
            "obs": inputs["obs"],  # (B, L, J, 3)
            "f_cliffcam": cliff_cam,  # (B, L, 3)
            "f_cam_angvel": f_cam_angvel,  # (B, L, C=6)
            "f_imgseq": inputs["f_imgseq"],  # (B, L, C=1024)
            "f_cam_transl": inputs["cam_transl_local"],  # (B, L, 3)
        }
        if train:
            f_condition = randomly_set_null_condition(f_condition, 0.1)
        
        # Apply ablation mask at inference time (zero-cost ablation)
        if ablation_mask is not None and not train:
            for key in ablation_mask:
                if key in f_condition and f_condition[key] is not None:
                    f_condition[key] = torch.zeros_like(f_condition[key])

        # 2. Forward & output
        # pred_x, pred_cam, static_conf_logits, and other heads' feature
        model_output = self.denoiser3d(length=length, **f_condition)
        # Prediction heads
        pred_hmr = self.endecoder.decode_hmr(model_output["pred_hmr"])
        pred_human_move = self.endecoder.decode_human_move(model_output["pred_human_move"])
        pred_cam_move = self.endecoder.decode_cam_move(model_output["pred_cam_move"])
        outputs.update({"model_output": model_output,
                        "pred_hmr": pred_hmr,
                        "pred_human_move": pred_human_move,
                        "pred_cam_move": pred_cam_move})

        # Post-processing
        outputs["pred_smpl_params_incam"] = {
            "body_pose": pred_hmr["body_pose"],  # (B, L, 63)
            "betas": pred_hmr["betas"],  # (B, L, 10)
            "global_orient": pred_hmr["global_orient"],  # (B, L, 3)
            "transl": compute_transl_full_cam(model_output["pred_weak_cam"], inputs["bbx_xys"], inputs["K_fullimg"]),
        }

        # Decode cam_yaw and human_yaw
        # Option: use predicted cam_angvel_r_yaw instead of input cam_angvel (for ablation)
        use_predicted_cam_angvel = getattr(self.args, 'use_predicted_cam_angvel', False)
        if use_predicted_cam_angvel:
            cam_angvel_for_rollout = pred_cam_move["cam_angvel_r_yaw"]
        else:
            cam_angvel_for_rollout = inputs["cam_angvel"]

        # Option: switch rollout method (WATCH exact vs GVHMR approximate)
        rollout_method = getattr(self.args, 'rollout_method', 'watch')  # "watch" | "gvhmr"

        if rollout_method == "gvhmr":
            # --- GVHMR-style: view projection + acos + gaussian_smooth ---
            if train:
                human_yaw, _ = extract_rotation_components_ydown(
                    axis_angle_to_matrix(inputs["smpl_params_w"]["global_orient"])
                )
                pred_human_yaw_w, pred_cam_yaw_w, pred_human_ori_use_cam, pred_cam_ori_use_cam = \
                    rollout_human_yaw_gvhmr_style(
                        rotation_6d_to_matrix(pred_cam_move["cam_roll_pitch"]),
                        axis_angle_to_matrix(pred_hmr["global_orient"]),
                        cam_angvel_for_rollout,
                        init_human_yaw=human_yaw[:, 0],
                    )
            else:
                pred_human_yaw_w, pred_cam_yaw_w, pred_human_ori_use_cam, pred_cam_ori_use_cam = \
                    rollout_human_yaw_gvhmr_style(
                        rotation_6d_to_matrix(pred_cam_move["cam_roll_pitch"]),
                        axis_angle_to_matrix(pred_hmr["global_orient"]),
                        cam_angvel_for_rollout,
                    )
        elif rollout_method == "direct":
            # --- 路线二: 直接组合，不做 decompose-recompose ---
            if train:
                init_cam_yaw = rotation_6d_to_matrix(inputs["cam_yaw"][:, 0])
            else:
                init_cam_yaw = None
            pred_human_ori_use_cam, pred_cam_ori_use_cam = rollout_human_yaw_direct(
                rotation_6d_to_matrix(pred_cam_move["cam_roll_pitch"]),
                axis_angle_to_matrix(pred_hmr["global_orient"]),
                cam_angvel_for_rollout,
                init_cam_yaw=init_cam_yaw,
            )
            pred_human_yaw_w, pred_cam_yaw_w = pred_human_ori_use_cam, pred_cam_ori_use_cam
        else:
            # --- WATCH: exact matrix factorization (default) ---
            if train:
                human_yaw, _ = extract_rotation_components_ydown(
                    axis_angle_to_matrix(inputs["smpl_params_w"]["global_orient"])
                )
                pred_human_yaw_w, pred_cam_yaw_w, pred_human_ori_use_cam, pred_cam_ori_use_cam = rollout_human_yaw(
                    rotation_6d_to_matrix(pred_cam_move["cam_roll_pitch"]),
                    axis_angle_to_matrix(pred_hmr["global_orient"]),
                    cam_angvel_for_rollout,
                    init_cam_yaw=rotation_6d_to_matrix(inputs["cam_yaw"][:, 0]),
                    init_human_yaw=human_yaw[:, 0],
                )
            else:
                pred_human_yaw_w, pred_cam_yaw_w, pred_human_ori_use_cam, pred_cam_ori_use_cam = rollout_human_yaw(
                    rotation_6d_to_matrix(pred_cam_move["cam_roll_pitch"]),
                    axis_angle_to_matrix(pred_hmr["global_orient"]),
                    cam_angvel_for_rollout,
                )
        # outputs["pred_human_orient_w"] = torch.einsum(
        #     "...ij,...jk->...ik",
        #     pred_human_yaw_w, rotation_6d_to_matrix(pred_human_move["human_roll_pitch"])
        # )
        # outputs["pred_cam_orient_w"] = torch.einsum(
        #     "...ij,...jk->...ik",
        #     pred_cam_yaw_w, rotation_6d_to_matrix(pred_cam_move["cam_roll_pitch"])
        # )
        outputs["pred_human_orient_w"] = pred_human_ori_use_cam
        outputs["pred_cam_orient_w"] = pred_cam_ori_use_cam

        if not train:
            # Check if exact yaw is enabled
            use_exact_yaw = getattr(self.args, "use_exact_yaw", False)
            
            if use_exact_yaw:
                # Use exact yaw factorization (from GVHMR)
                # Convert 6D rotation to axis-angle for GV frame orientation
                human_roll_pitch_mat = rotation_6d_to_matrix(pred_human_move["human_roll_pitch"])
                global_orient_gv_aa = matrix_to_axis_angle(human_roll_pitch_mat)
                
                pred_smpl_params_global = get_smpl_params_w_exact_yaw(
                    global_orient_gv=global_orient_gv_aa,  # GV frame orientation (roll/pitch only)
                    local_transl_vel=pred_human_move["human_local_vel"],
                    global_orient_c=pred_hmr["global_orient"],  # camera frame orientation
                    cam_angvel=cam_angvel_for_rollout,  # camera angular velocity
                )
            else:
                # Use standard rollout
                pred_smpl_params_global = {
                    'global_orient': matrix_to_axis_angle(outputs["pred_human_orient_w"]),
                    'transl': self.traj_decoder.rollout_local_vel(
                        local_vel=pred_human_move["human_local_vel"],
                        orient=outputs["pred_human_orient_w"],
                    )
                }
            outputs["pred_smpl_params_global"] = {
                "body_pose": pred_hmr["body_pose"],
                "betas": pred_hmr["betas"],
                **pred_smpl_params_global,
            }
            outputs["static_conf_logits"] = model_output["static_conf_logits"]
            outputs["pred_cam_transl"] = self.traj_decoder.rollout_local_vel(
                local_vel=pred_cam_move["cam_local_vel"],
                orient=outputs["pred_cam_orient_w"],
            )

            if self.use_transl_fusion:
                outputs["pred_human_transl_rollout"] = outputs["pred_smpl_params_global"]["transl"]
                human_trans_c = outputs["pred_smpl_params_incam"]["transl"]
                human_ori_c = axis_angle_to_matrix(outputs["pred_smpl_params_incam"]["global_orient"])
                _, outputs["pred_human_transl_geom"] = self.traj_decoder.infer_human_trans_from_cam_trans(
                    cam_ori_w=outputs["pred_cam_orient_w"],
                    cam_trans_w=outputs["pred_cam_transl"],
                    human_ori_c=human_ori_c,
                    human_trans_c=human_trans_c,
                )
                outputs["pred_smpl_params_global"]["transl"] = self.apply_learned_transl_fusion(
                    inputs, outputs, train=False
                )

            if postproc:  # apply post-processing
                if static_cam:  # extra post-processing to utilize static camera prior
                    outputs["pred_smpl_params_global"]["transl"] = pp_static_joint_cam(outputs, self.endecoder)
                else:
                    if postproc_version == 2:
                        outputs["pred_smpl_params_global"]["transl"] = pp_static_joint_v2(outputs, self.endecoder)
                    else:
                        outputs["pred_smpl_params_global"]["transl"] = pp_static_joint(outputs, self.endecoder)
                body_pose = process_ik(outputs, self.endecoder)
                pred_hmr["body_pose"] = body_pose
                outputs["pred_smpl_params_global"]["body_pose"] = body_pose
                outputs["pred_smpl_params_incam"]["body_pose"] = body_pose


            return outputs

        # ========== Compute Loss ========== #
        total_loss = 0
        mask = inputs["mask"]["valid"]  # (B, L)
        valid_mask = inputs["valid_mask"] if "valid_mask" in inputs else torch.ones_like(mask)  # (B, L)

        # 1. Simple loss: MSE
        pred_x = torch.cat([
            model_output["pred_hmr"],
            model_output["pred_human_move"],
            model_output["pred_cam_move"],
        ], dim=-1)  # (B, L, C)
        target_x = torch.cat([
            self.endecoder.encode_hmr(inputs),
            self.endecoder.encode_human_move(inputs),
            self.endecoder.encode_cam_move(inputs),
        ], dim=-1)  # (B, L, C)
        simple_loss = F.mse_loss(pred_x, target_x, reduction="none")
        mask_simple = mask[:, :, None].expand(-1, -1, pred_x.size(2)).clone()  # (B, L, C)
        mask_simple = mask_simple * valid_mask[:, :, None].expand(-1, -1, pred_x.size(2)).clone()  # (B, L, C)
        mask_simple[inputs["mask"]["spv_incam_only"], :, 142:] = False  # 3dpw training
        simple_loss = (simple_loss * mask_simple).mean()
        total_loss += simple_loss
        outputs["simple_loss"] = simple_loss

        # 2. Extra loss
        extra_funcs = [
            compute_extra_incam_loss,
            compute_extra_global_loss,
        ]
        for extra_func in extra_funcs:
            extra_loss, extra_loss_dict = extra_func(inputs, outputs, self)
            total_loss += extra_loss
            outputs.update(extra_loss_dict)

        outputs["loss"] = total_loss

        return outputs


class MoveDecoder:
    def __init__(self):
        """
        This is a toolkit class. Use the predicted roll_pitch, yaw angvel, and local transl vel to decode the movement.
        Note that the angvel is different from GVHMR, it is from R_src2world.
         For example, R_human2world is global_orient, R_c2w is camera orientation.
        Returns:
            None
        """
        pass

    @staticmethod
    def rollout_angvel(angvel, rot0=None):
        """
        Rollout the angular velocity to the global rotation matrix. R1 = R_angvel @ R0
        Args:
            angvel: (B, L, 6)
            rot0: (B, 3, 3) or None
        """
        if rot0 is None:
            rot0 = torch.eye(3).repeat(angvel.shape[0], 1, 1).to(angvel.device)  # 生成(B,3,3)的单位矩阵
        angvel_rot = rotation_6d_to_matrix(angvel)
        rotmats = [rot0]
        for i in range(angvel_rot.shape[1]):
            rotmats.append(torch.einsum("...ij,...jk->...ik", angvel_rot[:, i], rotmats[-1]))
        rotmats = torch.stack(rotmats[:-1], dim=-3)
        return rotmats

    @staticmethod
    def rollout_local_vel(local_vel, orient, transl0=None, use_aa=False):
        """
        Rollout the local velocity to the global translation.
        Args:
            local_vel: (B, L, 3) local translation velocity
            orient: (B, L, 3, 3) or (B, L, 3) global orientation (rotation matrix or axis angle)
            transl0: (B, 3) initial global translation
            use_aa: bool, whether the orientation is in axis angle format
        Returns:
            transl: (B, L, 3) global translation
        """
        if use_aa:
            orient_rot = axis_angle_to_matrix(orient)
        else:
            orient_rot = orient
        global_vel = torch.einsum("...ij,...j->...i", orient_rot, local_vel)

        if transl0 is None:
            transl0 = torch.zeros(local_vel.shape[0], 3).to(local_vel.device)
        transl = torch.cat([transl0.unsqueeze(1), global_vel[:, :-1]], dim=-2)
        global_transl = torch.cumsum(transl, dim=-2)
        return global_transl

    @staticmethod
    def compose_global_orient(yaw, roll_pitch):
        if yaw.shape[-1] == 6:
            yaw_rot = rotation_6d_to_matrix(yaw)
        elif yaw.shape[-2] != 3:
            yaw_rot = axis_angle_to_matrix(yaw)
        else:
            yaw_rot = yaw

        if roll_pitch.shape[-1] == 6:
            roll_pitch_rot = rotation_6d_to_matrix(roll_pitch)
        elif roll_pitch.shape[-2] != 3:
            roll_pitch_rot = axis_angle_to_matrix(roll_pitch)
        else:
            roll_pitch_rot = roll_pitch

        return torch.einsum('...ij,...jk->...ik', yaw_rot, roll_pitch_rot)

    @staticmethod
    def reset_ori_trans(ori, trans, init_yaw=None, init_trans=None):
        if init_yaw is None:
            init_yaw = torch.eye(3).repeat(ori.shape[0], 1, 1).to(ori.device)
        if init_trans is None:
            init_trans = torch.zeros(trans.shape[0], 3).to(trans.device)
        # R_reset = yaw_init @ yaw0.T, (B, 3, 3)
        reset_yaw = torch.einsum("...ij,...kj->...ik", init_yaw, ori[:, 0]).unsqueeze(1)
        ori_reset = torch.einsum("...ij,...jk->...ik", reset_yaw, ori)
        trans_reset = torch.einsum("...ij,...j->...i", reset_yaw, trans)
        trans_reset = trans_reset - trans_reset[:, [0]] + init_trans.unsqueeze(1)
        return ori_reset, trans_reset

    @classmethod
    def infer_human_trans_from_cam_trans(cls, cam_ori_w, cam_trans_w, human_ori_c, human_trans_c,
                                         init_yaw=None, init_trans=None):
        """
        From the definition of translation function p' = Rp + t, we get:
         p_w = R_c2w @ (R_h2c @ p_h + t_h2c) + t_c2w = R_c2w @ R_h2c @ p_h + R_c2w @ t_h2c + t_c2w.
        So we get: R_h2w = R_c2w @ R_h2c, t_h2w = R_c2w @ t_h2c + t_c2w.
        Args:
            cam_ori_w: (B, L, 3, 3) R_c2w Camera orientation in world coordinate.
            cam_trans_w: (B, L, 3) t_c2w Camera translation in world coordinate.
            human_ori_c: (B, L, 3, 3) R_h2c Human orientation in camera space.
            human_trans_c: (B, L, 3) t_h2c Human translation in camera space.
            init_yaw: (B, 3, 3) initial yaw rotation matrix.
            init_trans: (B, 3) initial translation.
        Returns:
            human_ori_w: (B, L, 3, 3) R_h2w Human orientation in world coordinate.
            human_trans_w: (B, L, 3) t_h2w Human translation in world coordinate.
        """
        human_ori_w = torch.einsum("...ij,...jk->...ik", cam_ori_w, human_ori_c)
        human_trans_w = torch.einsum("...ij,...j->...i", cam_ori_w, human_trans_c) + cam_trans_w

        return cls.reset_ori_trans(human_ori_w, human_trans_w, init_yaw, init_trans)

    @classmethod
    def infer_cam_trans_from_human_trans(cls, human_ori_w, human_trans_w, human_ori_c, human_trans_c,
                                         init_yaw=None, init_trans=None):
        """
        Following the method infer_human_trans_from_cam_trans(), we get function:
         R_c2w = R_h2w @ R_h2c.T, t_c2w = t_h2w - R_c2w @ t_h2c.
        Args:
            human_ori_w: [*, 3, 3] R_h2w Human orientation in world coordinate.
            human_trans_w: [*, 3] t_h2w Human translation in world coordinate.
            human_ori_c: [*, 3, 3] R_h2c Human orientation in camera space.
            human_trans_c: [*, 3] t_h2c Human translation in camera space.
            init_yaw: (B, 3, 3) initial yaw rotation matrix.
            init_trans: (B, 3) initial translation.
        Returns:
            cam_ori_w: [*, 3, 3] R_c2w Camera orientation in world coordinate.
            cam_trans_w: [*, 3] t_c2w Camera translation in world coordinate.
        """
        cam_ori_w = torch.einsum("...ij,...kj->...ik", human_ori_w, human_ori_c)
        cam_trans_w = human_trans_w - torch.einsum("...ij,...j->...i", cam_ori_w, human_trans_c)

        return cls.reset_ori_trans(cam_ori_w, cam_trans_w, init_yaw, init_trans)


def randomly_set_null_condition(f_condition, uncond_prob=0.1):
    """Conditions are in shape (B, L, *)"""
    keys = list(f_condition.keys())
    for k in keys:
        if f_condition[k] is None:
            continue
        f_condition[k] = f_condition[k].clone()
        mask = torch.rand(f_condition[k].shape[:2]) < uncond_prob
        f_condition[k][mask] = 0.0
    return f_condition


# 完全继承
def compute_extra_incam_loss(inputs, outputs, ppl):
    model_output = outputs["model_output"]
    # decode_dict = outputs["pred_hmr"]
    endecoder = ppl.endecoder
    weights = ppl.weights
    args = ppl.args

    extra_loss_dict = {}
    extra_loss = 0
    mask = inputs["mask"]["valid"]  # effective length mask
    valid_motion_mask = inputs["valid_mask"] if "valid_mask" in inputs else torch.ones_like(mask)  # actual valid mask for multi-person scenes
    mask_reproj = ~inputs["mask"]["spv_incam_only"]  # do not supervise reproj for 3DPW

    # Incam FK
    # prediction
    pred_c_j3d = endecoder.fk_v2(**outputs["pred_smpl_params_incam"])
    # pred_c_j3d[pred_c_j3d.isnan() | pred_c_j3d.isinf()] = 0.0   # 数值溢出，置0交由valid_motion_mask处理
    pred_cr_j3d = pred_c_j3d - pred_c_j3d[:, :, :1]  # (B, L, J, 3)

    # gt
    gt_c_j3d = endecoder.fk_v2(**inputs["smpl_params_c"])  # (B, L, J, 3)
    gt_cr_j3d = gt_c_j3d - gt_c_j3d[:, :, :1]  # (B, L, J, 3)

    # Root aligned C-MPJPE Loss
    if weights.cr_j3d > 0.0:
        cr_j3d_loss = F.mse_loss(pred_cr_j3d, gt_cr_j3d, reduction="none")
        cr_j3d_loss = (cr_j3d_loss * mask[..., None, None] * valid_motion_mask[..., None, None]).mean()
        extra_loss += cr_j3d_loss * weights.cr_j3d
        extra_loss_dict["cr_j3d_loss"] = cr_j3d_loss

    # Reprojection (to align with image)
    if weights.transl_c > 0.0:
        # pred_transl = decode_dict["transl"]  # (B, L, 3)
        # gt_transl = inputs["smpl_params_c"]["transl"]
        # transl_c_loss = F.l1_loss(pred_transl, gt_transl, reduction="none")
        # transl_c_loss = (transl_c_loss * mask[..., None]).mean()

        # Instead of supervising transl, we convert gt to pred_cam (prevent divide 0)
        pred_cam = model_output["pred_weak_cam"]  # (B, L, 3)
        gt_transl = inputs["smpl_params_c"]["transl"]  # (B, L, 3)
        gt_pred_cam = get_a_pred_cam(gt_transl, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)
        gt_pred_cam[gt_pred_cam.isinf()] = -1  # this will be handled by valid_mask
        # (compute_transl_full_cam(gt_pred_cam, inputs["bbx_xys"], inputs["K_fullimg"]) - gt_transl).abs().max()

        # Skip gts that are not good during random construction
        gt_j3d_z_min = inputs["gt_j3d"][..., 2].min(dim=-1)[0]
        valid_mask = (
            (gt_j3d_z_min > 0.3)
            * (gt_pred_cam[..., 0] > 0.3)
            * (gt_pred_cam[..., 0] < 5.0)
            * (gt_pred_cam[..., 1] > -3.0)
            * (gt_pred_cam[..., 1] < 3.0)
            * (gt_pred_cam[..., 2] > -3.0)
            * (gt_pred_cam[..., 2] < 3.0)
            * (inputs["bbx_xys"][..., 2] > 0)
        )[..., None]
        transl_c_loss = F.mse_loss(pred_cam, gt_pred_cam, reduction="none")
        transl_c_loss = (transl_c_loss * mask[..., None] * valid_mask * valid_motion_mask[..., None]).mean()

        extra_loss_dict["transl_c_loss"] = transl_c_loss
        extra_loss += transl_c_loss * weights.transl_c

    if weights.j2d > 0.0:
        # prevent divide 0 or small value to overflow(fp16)
        reproj_z_thr = 0.3
        pred_c_j3d_z0_mask = pred_c_j3d[..., 2].abs() <= reproj_z_thr
        pred_c_j3d[pred_c_j3d_z0_mask] = reproj_z_thr
        gt_c_j3d_z0_mask = gt_c_j3d[..., 2].abs() <= reproj_z_thr
        gt_c_j3d[gt_c_j3d_z0_mask] = reproj_z_thr

        pred_j2d_01 = project_to_bi01(pred_c_j3d, inputs["bbx_xys"], inputs["K_fullimg"])
        # pred_j2d_01[pred_j2d_01.isnan() | pred_j2d_01.isinf()] = 0.0
        gt_j2d_01 = project_to_bi01(gt_c_j3d, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, J, 2)
        # gt_j2d_01[gt_j2d_01.isnan() | gt_j2d_01.isinf()] = 0.0

        valid_mask = (
            (gt_c_j3d[..., 2] > reproj_z_thr)
            * (pred_c_j3d[..., 2] > reproj_z_thr)  # Be safe
            * (gt_j2d_01[..., 0] > 0.0)
            * (gt_j2d_01[..., 0] < 1.0)
            * (gt_j2d_01[..., 1] > 0.0)
            * (gt_j2d_01[..., 1] < 1.0)
        )[..., None]
        valid_mask[~mask_reproj] = False  # Do not supervise on 3dpw
        j2d_loss = F.mse_loss(pred_j2d_01, gt_j2d_01, reduction="none")
        j2d_loss = (j2d_loss * mask[..., None, None] * valid_mask * valid_motion_mask[..., None, None]).mean()

        extra_loss += j2d_loss * weights.j2d
        extra_loss_dict["j2d_loss"] = j2d_loss

    if weights.cr_verts > 0:
        # SMPL forward
        pred_c_verts437, pred_c_j17 = endecoder.smplx_model(**outputs["pred_smpl_params_incam"])
        # pred_c_verts437[pred_c_verts437.isnan() | pred_c_verts437.isinf()] = 0.0
        # pred_c_j17[pred_c_j17.isnan() | pred_c_j17.isinf()] = 0.0
        root_ = pred_c_j17[:, :, [11, 12], :].mean(-2, keepdim=True)
        pred_cr_verts437 = pred_c_verts437 - root_

        gt_cr_verts437 = inputs["gt_cr_verts437"]  # (B, L, 437, 3)
        cr_vert_loss = F.mse_loss(pred_cr_verts437, gt_cr_verts437, reduction="none")
        cr_vert_loss = (cr_vert_loss * mask[:, :, None, None] * valid_motion_mask[:, :, None, None]).mean()
        extra_loss += cr_vert_loss * weights.cr_verts
        extra_loss_dict["cr_vert_loss"] = cr_vert_loss

    if weights.verts2d > 0:
        gt_c_verts437 = inputs["gt_c_verts437"]  # (B, L, 437, 3)

        # prevent divide 0 or small value to overflow(fp16)
        reproj_z_thr = 0.3
        pred_c_verts437_z0_mask = pred_c_verts437[..., 2].abs() <= reproj_z_thr
        pred_c_verts437[pred_c_verts437_z0_mask] = reproj_z_thr
        gt_c_verts437_z0_mask = gt_c_verts437[..., 2].abs() <= reproj_z_thr
        gt_c_verts437[gt_c_verts437_z0_mask] = reproj_z_thr

        pred_verts2d_01 = project_to_bi01(pred_c_verts437, inputs["bbx_xys"], inputs["K_fullimg"])
        # pred_verts2d_01[pred_verts2d_01.isnan() | pred_verts2d_01.isinf()] = 0.0
        gt_verts2d_01 = project_to_bi01(gt_c_verts437, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 437, 2)
        # gt_verts2d_01[gt_verts2d_01.isnan() | gt_verts2d_01.isinf()] = 0.0

        valid_mask = (
            (gt_c_verts437[..., 2] > reproj_z_thr)
            * (pred_c_verts437[..., 2] > reproj_z_thr)  # Be safe
            * (gt_verts2d_01[..., 0] > 0.0)
            * (gt_verts2d_01[..., 0] < 1.0)
            * (gt_verts2d_01[..., 1] > 0.0)
            * (gt_verts2d_01[..., 1] < 1.0)
        )[..., None]
        valid_mask[~mask_reproj] = False  # Do not supervise on 3dpw
        verts2d_loss = F.mse_loss(pred_verts2d_01, gt_verts2d_01, reduction="none")
        verts2d_loss = (verts2d_loss * mask[..., None, None] * valid_mask * valid_motion_mask[..., None, None]).mean()

        extra_loss += verts2d_loss * weights.verts2d
        extra_loss_dict["verts2d_loss"] = verts2d_loss

    return extra_loss, extra_loss_dict


def compute_extra_global_loss(inputs, outputs, ppl):
    movement = MoveDecoder()
    decode_human_move = outputs["pred_human_move"]
    decode_cam_move = outputs["pred_cam_move"]
    endecoder = ppl.endecoder
    weights = ppl.weights
    args = ppl.args

    extra_loss_dict = {}
    extra_loss = 0
    mask = inputs["mask"]["valid"].clone()  # (B, L)
    valid_motion_mask = inputs["valid_mask"] if "valid_mask" in inputs else torch.ones_like(mask)  # (B, L)
    mask[inputs["mask"]["spv_incam_only"]] = False

    if weights.transl_w_human > 0 or weights.transl_w_cam > 0:
        # Human trans Loss
        gt_transl_w = inputs["smpl_params_w"]["transl"]
        gt_global_orient_w = inputs["smpl_params_w"]["global_orient"]
        gt_local_vel = get_local_transl_vel(gt_transl_w, gt_global_orient_w)
        # gt_yaw_w, _ = extract_rotation_components_ydown(axis_angle_to_matrix(gt_global_orient_w))

        # human_pred_yaw = movement.rollout_angvel(decode_human_move["human_angvel_r_yaw"], gt_yaw_w[:, 0])
        # human_roll_pitch = rotation_6d_to_matrix(decode_human_move["human_roll_pitch"])
        # human_global_orient = torch.einsum('...ij,...jk->...ik', human_pred_yaw, human_roll_pitch)
        human_global_orient = outputs["pred_human_orient_w"]
        human_local_vel = decode_human_move["human_local_vel"]

        human_transl_force_orient = movement.rollout_local_vel(human_local_vel, axis_angle_to_matrix(gt_global_orient_w), gt_transl_w[:, 0])
        human_transl_force_vel = movement.rollout_local_vel(gt_local_vel, human_global_orient, gt_transl_w[:, 0])

        # Compute geometric translation from camera
        cam_global_orient = outputs["pred_cam_orient_w"]
        cam_local_vel = decode_cam_move["cam_local_vel"]
        gt_transl_cam = inputs["cam_transl"]
        gt_orient_cam = movement.compose_global_orient(inputs["cam_yaw"], inputs["cam_roll_pitch"])
        
        cam_transl_rollout = movement.rollout_local_vel(cam_local_vel, cam_global_orient, gt_transl_cam[:, 0])
        
        human_trans_c = outputs["pred_smpl_params_incam"]["transl"]
        human_ori_c = axis_angle_to_matrix(outputs["pred_smpl_params_incam"]["global_orient"])
        
        _, human_transl_geom = movement.infer_human_trans_from_cam_trans(
            cam_ori_w=cam_global_orient,
            cam_trans_w=cam_transl_rollout,
            human_ori_c=human_ori_c,
            human_trans_c=human_trans_c,
        )
        
        # Store for fusion
        outputs["pred_human_transl_rollout"] = human_transl_force_vel
        outputs["pred_human_transl_geom"] = human_transl_geom
        outputs["pred_cam_move"] = decode_cam_move
        outputs["pred_human_move"] = decode_human_move
        
        # Apply learned fusion
        human_transl_fused = ppl.apply_learned_transl_fusion(inputs, outputs, train=True)
        
        # Use fused translation for loss computation
        if ppl.use_transl_fusion:
            human_transl_force_vel = human_transl_fused


        loss_human_trans_pred_vel = F.l1_loss(human_transl_force_orient, gt_transl_w, reduction="none")
        loss_human_trans_pred_vel = (loss_human_trans_pred_vel * mask[..., None] * valid_motion_mask[..., None]).mean()
        extra_loss += loss_human_trans_pred_vel * weights.transl_w_human
        extra_loss_dict["trans_human_pred_vel_loss"] = loss_human_trans_pred_vel

        loss_human_trans_pred_orient = F.l1_loss(human_transl_force_vel, gt_transl_w, reduction="none")
        loss_human_trans_pred_orient = (loss_human_trans_pred_orient * mask[..., None] * valid_motion_mask[..., None]).mean()
        extra_loss += loss_human_trans_pred_orient * weights.transl_w_human
        extra_loss_dict["trans_human_pred_orient_loss"] = loss_human_trans_pred_orient

        # Camera trans Loss
        gt_transl_cam = inputs["cam_transl"]
        gt_orient_cam = movement.compose_global_orient(inputs["cam_yaw"], inputs["cam_roll_pitch"])
        gt_local_vel_cam = get_local_vel_mat(gt_transl_cam, gt_orient_cam)
        # gt_yaw_cam = rotation_6d_to_matrix(inputs["cam_yaw"])

        # cam_pred_yaw = movement.rollout_angvel(decode_cam_move["cam_angvel_r_yaw"], gt_yaw_cam[:, 0])
        # cam_roll_pitch = rotation_6d_to_matrix(decode_cam_move["cam_roll_pitch"])
        # cam_global_orient = movement.compose_global_orient(cam_pred_yaw, cam_roll_pitch)
        cam_global_orient = outputs["pred_cam_orient_w"]
        cam_local_vel = decode_cam_move["cam_local_vel"]

        cam_transl_force_orient = movement.rollout_local_vel(cam_local_vel, gt_orient_cam, gt_transl_cam[:, 0])
        cam_transl_force_vel = movement.rollout_local_vel(gt_local_vel_cam, cam_global_orient, gt_transl_cam[:, 0])

        loss_cam_trans_pred_vel = F.l1_loss(cam_transl_force_orient, gt_transl_cam, reduction="none")
        loss_cam_trans_pred_vel = (loss_cam_trans_pred_vel * mask[..., None] * valid_motion_mask[..., None]).mean()
        extra_loss += loss_cam_trans_pred_vel * weights.transl_w_cam
        extra_loss_dict["trans_cam_pred_vel_loss"] = loss_cam_trans_pred_vel

        loss_cam_trans_pred_orient = F.l1_loss(cam_transl_force_vel, gt_transl_cam, reduction="none")
        loss_cam_trans_pred_orient = (loss_cam_trans_pred_orient * mask[..., None] * valid_motion_mask[..., None]).mean()
        extra_loss += loss_cam_trans_pred_orient * weights.transl_w_cam
        extra_loss_dict["trans_cam_pred_orient_loss"] = loss_cam_trans_pred_orient

        # Inference human traj from cam traj, and camera traj from human traj.
        if weights.transl_infer_human_cam > 0:
            t_human_c = outputs["pred_smpl_params_incam"]["transl"]
            r_human_c = axis_angle_to_matrix(outputs["pred_smpl_params_incam"]["global_orient"])
            t_cam_w_force_ori = cam_transl_force_orient
            t_cam_w_force_vel = cam_transl_force_vel
            r_cam_w = cam_global_orient
            t_human_w_force_ori = human_transl_force_orient
            t_human_w_force_vel = human_transl_force_vel
            r_human_w = human_global_orient

            init_cam_yaw, _ = extract_rotation_components_ydown(gt_orient_cam[:, 0])
            init_cam_trans = gt_transl_cam[:, 0]
            init_human_yaw, _ = extract_rotation_components_ydown(axis_angle_to_matrix(gt_global_orient_w)[:, 0])
            init_human_trans = gt_transl_w[:, 0]

            # Loss for human transl from cam transl
            _, human_transl_from_cam_force_ori = movement.infer_human_trans_from_cam_trans(
                r_cam_w, t_cam_w_force_ori, r_human_c, t_human_c, init_cam_yaw, init_cam_trans
            )
            loss_human_trans_from_cam_force_ori = F.l1_loss(human_transl_from_cam_force_ori, gt_transl_w, reduction="none")
            loss_human_trans_from_cam_force_ori = (loss_human_trans_from_cam_force_ori * mask[..., None] * valid_motion_mask[..., None]).mean()
            extra_loss += loss_human_trans_from_cam_force_ori * weights.transl_infer_human_cam
            extra_loss_dict["trans_human_from_cam_force_ori_loss"] = loss_human_trans_from_cam_force_ori

            _, human_transl_from_cam_force_vel = movement.infer_human_trans_from_cam_trans(
                r_cam_w, t_cam_w_force_vel, r_human_c, t_human_c, init_cam_yaw, init_cam_trans
            )
            loss_human_trans_from_cam_force_vel = F.l1_loss(human_transl_from_cam_force_vel, gt_transl_w, reduction="none")
            loss_human_trans_from_cam_force_vel = (loss_human_trans_from_cam_force_vel * mask[..., None] * valid_motion_mask[..., None]).mean()
            extra_loss += loss_human_trans_from_cam_force_vel * weights.transl_infer_human_cam
            extra_loss_dict["trans_human_from_cam_force_vel_loss"] = loss_human_trans_from_cam_force_vel

            # Loss for cam transl from human transl
            _, cam_transl_from_human_force_ori = movement.infer_cam_trans_from_human_trans(
                r_human_w, t_human_w_force_ori, r_cam_w, t_cam_w_force_ori, init_human_yaw, init_human_trans
            )
            loss_cam_trans_from_human_force_ori = F.l1_loss(cam_transl_from_human_force_ori, gt_transl_cam, reduction="none")
            loss_cam_trans_from_human_force_ori = (loss_cam_trans_from_human_force_ori * mask[..., None] * valid_motion_mask[..., None]).mean()
            extra_loss += loss_cam_trans_from_human_force_ori * weights.transl_infer_human_cam
            extra_loss_dict["trans_cam_from_human_force_ori_loss"] = loss_cam_trans_from_human_force_ori

            _, cam_transl_from_human_force_vel = movement.infer_cam_trans_from_human_trans(
                r_human_w, t_human_w_force_vel, r_cam_w, t_cam_w_force_vel, init_human_yaw, init_human_trans
            )
            loss_cam_trans_from_human_force_vel = F.l1_loss(cam_transl_from_human_force_vel, gt_transl_cam, reduction="none")
            loss_cam_trans_from_human_force_vel = (loss_cam_trans_from_human_force_vel * mask[..., None] * valid_motion_mask[..., None]).mean()
            extra_loss += loss_cam_trans_from_human_force_vel * weights.transl_infer_human_cam
            extra_loss_dict["trans_cam_from_human_force_vel_loss"] = loss_cam_trans_from_human_force_vel

    # Static-Conf loss
    if weights.static_conf_bce > 0:
        # Compute gt by thresholding velocity
        vel_thr = args.static_conf.vel_thr
        assert vel_thr > 0
        joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
        gt_w_j3d = endecoder.fk_v2(**inputs["smpl_params_w"])  # (B, L, J=22, 3)
        static_gt = get_static_joint_mask(gt_w_j3d, vel_thr=vel_thr, repeat_last=True)  # (B, L, J)
        static_gt = static_gt[:, :, joint_ids].float()  # (B, L, J')
        pred_static_conf_logits = outputs["model_output"]["static_conf_logits"]

        static_conf_loss = F.binary_cross_entropy_with_logits(pred_static_conf_logits, static_gt, reduction="none")
        static_conf_loss = (static_conf_loss * mask[..., None] * valid_motion_mask[..., None]).mean()
        extra_loss += static_conf_loss * weights.static_conf_bce
        extra_loss_dict["static_conf_loss"] = static_conf_loss


    # Transl fusion regularization losses
    if ppl.use_transl_fusion:
        if hasattr(weights, "transl_fusion_gate_reg") and weights.transl_fusion_gate_reg > 0:
            gate = outputs.get("transl_fusion_gate", None)
            if gate is not None:
                # Encourage sparse gate activation
                gate_reg_loss = gate.mean()
                extra_loss += gate_reg_loss * weights.transl_fusion_gate_reg
                extra_loss_dict["transl_fusion_gate_reg_loss"] = gate_reg_loss
        
        if hasattr(weights, "transl_fusion_residual_reg") and weights.transl_fusion_residual_reg > 0:
            residual_xz = outputs.get("transl_fusion_residual_xz", None)
            if residual_xz is not None:
                # Encourage small residuals
                residual_reg_loss = (residual_xz ** 2).mean()
                extra_loss += residual_reg_loss * weights.transl_fusion_residual_reg
                extra_loss_dict["transl_fusion_residual_reg_loss"] = residual_reg_loss


    # Accumulated position loss: directly penalize trajectory drift
    if getattr(weights, 'transl_accum', 0) > 0:
        gt_transl_w = inputs["smpl_params_w"]["transl"]
        human_local_vel = outputs["pred_human_move"]["human_local_vel"]
        human_global_orient = outputs["pred_human_orient_w"]
        pred_traj = movement.rollout_local_vel(
            human_local_vel, human_global_orient, gt_transl_w[:, 0]
        )
        loss_transl_accum = F.l1_loss(pred_traj, gt_transl_w, reduction="none")
        loss_transl_accum = (loss_transl_accum * mask[..., None] * valid_motion_mask[..., None]).mean()
        extra_loss += loss_transl_accum * weights.transl_accum
        extra_loss_dict["transl_accum_loss"] = loss_transl_accum

    return extra_loss, extra_loss_dict


@autocast(enabled=False)
def get_smpl_params_w_yaw(cam_orient_w, human_orient_c, human_roll_pitch, human_local_vel):
    """
    Get human global orientation and translation from camera orientation,
     human orientation in camera space, and human roll/pitch.
    R_h2w = R_h2c @ R_c2w, so we get human_yaw.
    Notes:
        - yaw is the fixed axis angle of g in world space.
        - yaw is initialized with first frame, yaw0.T @ yaw.
    Args:
        cam_orient_w: Camera orientation in world space (B, L, 3, 3)
        human_orient_c: Human orientation in camera space (B, L, 3, 3)
        human_roll_pitch: Human roll/pitch angles (B, L, 3, 3)
        human_local_vel: Human local velocity (B, L, 3)
    Returns:
        {'global_orient': Human global orientation (B, L, 3, 3)
         'transl': Human global translation (B, L, 3)
         }
    """
    human_global_orient_from_cam = torch.einsum('...ij,...jk->...ik', human_orient_c, cam_orient_w)
    human_yaw, _ = extract_rotation_components_ydown(human_global_orient_from_cam)
    human_yaw = torch.einsum('...ij,...jk->...ik', human_yaw[:, [0]].transpose(-1, -2), human_yaw)
    human_global_orient = torch.einsum('...ij,...jk->...ik', human_yaw, human_roll_pitch)
    # human_transl = torch.einsum('...ij,...j->...i', human_global_orient, human_local_vel)
    human_transl = rollout_local_vel_rotmat(human_local_vel, human_global_orient)
    return {'global_orient': matrix_to_axis_angle(human_global_orient), 'transl': human_transl}


@autocast(enabled=False)
def rollout_human_yaw(cam_roll_pitch, global_orient_mat_c, cam_angvel, init_cam_yaw=None, init_human_yaw=None):
    """
    Rollout human yaw from camera yaw.
    Notes:
        cam_angvel is defined as R @ R_{w2c}^{t} = R_{w2c}^{t+1}, R_h2w is defined as R_c2w @ R_h2c.
        So we get yaw_vel = cam_yaw0.T @ cam_yaw1 = r_roll_pitch0 @ angvel.T @ r_roll_pitch1.T
        Then we get human_ori = cam_yaw0 @ prob{yaw_vel} @ cam_roll_pitch @ global_orient_c.
    Args:
        cam_roll_pitch: Camera roll/pitch angles (B, L, 3, 3)
        global_orient_mat_c: Human global orientation in camera space (B, L, 3, 3)
        cam_angvel: Camera angular velocity, defined as (B, L, 6)
        init_cam_yaw: Initial yaw angle of camera in world (B, 3, 3), default is I.
        init_human_yaw: Initial yaw angle of human in world (B, 3, 3), default is I.

    Returns:
        human_yaw: Human yaw angle in world (B, L, 3, 3)
        cam_yaw: Camera yaw angle in world (B, L, 3, 3)
    """
    B, L, device, dtype = cam_angvel.shape[0], cam_angvel.shape[1], cam_angvel.device, cam_angvel.dtype
    cam_angvel_rot = rotation_6d_to_matrix(cam_angvel)  # (B, L, 3, 3)

    if init_cam_yaw is None:
        init_cam_yaw = torch.eye(3).repeat(B, 1, 1).float().to(device)  # (B, 3, 3)

    # cam yaw_vel = cam_roll_pitch0 @ cam_angvel_rot.T @ cam_roll_pitch1.T
    yaw_vel_tensor = torch.einsum(
        '...ij,...kj,...lk->...il',
        cam_roll_pitch[:, :-1],
        cam_angvel_rot[:, :-1],
        cam_roll_pitch[:, 1:]
    )
    yaw_vel = torch.unbind(yaw_vel_tensor, dim=1)

    # Recursively compute cam_yaw sequence
    cam_yaw_list = [init_cam_yaw]
    for i in range(L - 1):
        cam_yaw_next = torch.einsum('...ij,...jk->...ik', cam_yaw_list[-1], yaw_vel[i])
        cam_yaw_list.append(cam_yaw_next)
    cam_yaw = torch.stack(cam_yaw_list, dim=1)  # (B, L, 3, 3)

    # human_ori = cam_yaw @ cam_roll_pitch @ global_orient_c
    human_ori = torch.einsum('...ij,...jk,...kl->...il', cam_yaw, cam_roll_pitch, global_orient_mat_c)
    human_yaw, human_roll_pitch = extract_rotation_components_ydown(human_ori)

    # Align human_yaw with init_yaw, yaw_init_true @ yaw0.T @ yaw
    if init_human_yaw is None:
        init_human_yaw = torch.eye(3).repeat(B, 1, 1).float().to(device)
    human_yaw_rest = torch.einsum('...ij,...kj,...kl->...il',
                                  init_human_yaw.unsqueeze(1), human_yaw[:, [0]], human_yaw)
    reset_human_ori = torch.einsum('...ij,...jk->...ik', human_yaw_rest, human_roll_pitch)
    reset_cam_ori = torch.einsum('...ij,...jk->...ik', cam_yaw, cam_roll_pitch)

    return human_yaw_rest, cam_yaw, reset_human_ori, reset_cam_ori


@autocast(enabled=False)
def rollout_human_yaw_gvhmr_style(cam_roll_pitch, global_orient_mat_c, cam_angvel, init_human_yaw=None):
    """
    GVHMR-style yaw rollout: view-direction projection + acos + gaussian_smooth.
    Uses the SAME network predictions as WATCH, but recovers yaw using GVHMR's approximate method.
    For ablation ONLY: compare decomposition methods with all else identical.

    Math (mirrors GVHMR get_smpl_params_w_Rt_v2):
        1. R_c2gv ≈ cam_roll_pitch (camera-to-GV, removing yaw)
        2. global_orient_gv = R_c2gv @ global_orient_c
        3. view = R_c2gv @ [0,0,1], project to horizontal, acos → yaw change
        4. gaussian_smooth + accumulate
    """
    B, L = cam_angvel.shape[:2]
    device = cam_angvel.device
    cam_angvel_rot = rotation_6d_to_matrix(cam_angvel)  # (B, L, 3, 3)

    # Derive GVHMR-equivalent: R_c2gv ≈ cam_roll_pitch, R_gv = R_c2gv @ R_h2c
    R_c2gv = cam_roll_pitch  # (B, L, 3, 3)
    R_gv = torch.einsum('...ij,...jk->...ik', R_c2gv, global_orient_mat_c)

    # View direction in GV coordinate
    view_axis_gv = R_c2gv[..., :, 2]  # (B, L, 3)
    R_cnext2gv = torch.einsum('...ij,...kj->...ik', R_c2gv, cam_angvel_rot)
    view_axis_gv_next = R_cnext2gv[..., 2]

    # Project to horizontal (zero Y) — GVHMR's approximation
    vec1 = view_axis_gv.clone()
    vec1[..., 1] = 0
    vec1 = F.normalize(vec1, dim=-1)
    vec2 = view_axis_gv_next.clone()
    vec2[..., 1] = 0
    vec2 = F.normalize(vec2, dim=-1)

    # acos to get angle — GVHMR's approximation
    aa = vec2.cross(vec1, dim=-1)
    angle = torch.acos(torch.clamp((vec1 * vec2).sum(dim=-1, keepdim=True), -1.0, 1.0))
    aa = F.normalize(aa, dim=-1) * angle

    # Gaussian smooth — GVHMR's heuristic
    aa = gaussian_smooth(aa, dim=-2)
    R_tp1_to_t = axis_angle_to_matrix(aa).mT

    # Accumulate R_t_to_0
    def as_identity(R):
        is_I = matrix_to_axis_angle(R).norm(dim=-1) < 1e-5
        R = R.clone()
        R[is_I] = torch.eye(3)[None].expand(is_I.sum(), -1, -1).to(R)
        return R

    R_t_to_0_list = [torch.eye(3)[None].expand(B, -1, -1).to(device)]
    for i in range(1, L):
        R_t_to_0_list.append(R_t_to_0_list[-1] @ R_tp1_to_t[:, i])
    R_t_to_0 = torch.stack(R_t_to_0_list, dim=1)
    R_t_to_0 = as_identity(R_t_to_0)

    # human_ori = R_t_to_0 @ R_gv (same as GVHMR line 376)
    human_ori = torch.einsum('...ij,...jk->...ik', R_t_to_0, R_gv)
    human_yaw, human_roll_pitch = extract_rotation_components_ydown(human_ori)

    if init_human_yaw is None:
        init_human_yaw = torch.eye(3).repeat(B, 1, 1).float().to(device)
    human_yaw_rest = torch.einsum('...ij,...kj,...kl->...il',
                                  init_human_yaw.unsqueeze(1), human_yaw[:, [0]], human_yaw)
    reset_human_ori = torch.einsum('...ij,...jk->...ik', human_yaw_rest, human_roll_pitch)
    reset_cam_ori = torch.einsum('...ij,...jk->...ik', R_t_to_0, cam_roll_pitch)

    return human_yaw_rest, R_t_to_0, reset_human_ori, reset_cam_ori


@autocast(enabled=False)
def get_smpl_params_w_exact_yaw(
    global_orient_gv,
    local_transl_vel,
    global_orient_c,
    cam_angvel,
):
    """Get global R,t using exact yaw matrix factorization (from GVHMR).
    
    This replaces the approximate view-projection + acos + gaussian_smooth 
    with exact matrix factorization for more accurate yaw estimation.

    Math:
        R_c2gv = R_gv @ R_c^T  (camera-to-GV rotation ≈ roll/pitch)
        yaw_vel = R_c2gv[t] @ R_angvel[t]^T @ R_c2gv[t+1]^T  (exact yaw extraction)
        R_yaw = cumulative product of yaw_vel
        global_orient_w = R_yaw @ R_gv

    Args:
        global_orient_gv: (B, L, 3), human orientation in GV frame (axis-angle)
        local_transl_vel: (B, L, 3), local translation velocity
        global_orient_c:  (B, L, 3), human orientation in camera frame (axis-angle)
        cam_angvel:       (B, L, 6), camera angular velocity (6D rotation)
    """
    from core.utils.geo.hmr_global import rollout_local_transl_vel, get_tgtcoord_rootparam

    def as_identity(R):
        is_I = matrix_to_axis_angle(R).norm(dim=-1) < 1e-5
        R[is_I] = torch.eye(3)[None].expand(is_I.sum(), -1, -1).to(R)
        return R

    B, L = cam_angvel.shape[:2]
    device = cam_angvel.device

    R_t_to_tp1 = rotation_6d_to_matrix(cam_angvel)  # (B, L, 3, 3)
    R_t_to_tp1 = as_identity(R_t_to_tp1)

    R_gv = axis_angle_to_matrix(global_orient_gv)  # (B, L, 3, 3)
    R_c = axis_angle_to_matrix(global_orient_c)     # (B, L, 3, 3)

    # R_c2gv: camera → GV (gravity-view, i.e. world without yaw)
    R_c2gv = R_gv @ R_c.mT  # (B, L, 3, 3)

    # Exact yaw factorization
    yaw_vel_tensor = torch.einsum(
        '...ij,...kj,...lk->...il',
        R_c2gv[:, :-1],          # R_c2gv at frame t
        R_t_to_tp1[:, :-1],      # R_angvel^T
        R_c2gv[:, 1:]            # R_c2gv^T at frame t+1
    )  # (B, L-1, 3, 3)

    # Accumulate yaw rotations
    yaw_vel_list = torch.unbind(yaw_vel_tensor, dim=1)
    R_yaw = [torch.eye(3)[None].expand(B, -1, -1).to(device)]
    for i in range(L - 1):
        R_yaw.append(R_yaw[-1] @ yaw_vel_list[i])
    R_yaw = torch.stack(R_yaw, dim=1)  # (B, L, 3, 3)
    R_yaw = as_identity(R_yaw)

    # Final world orientation
    global_orient = matrix_to_axis_angle(R_yaw @ R_gv)

    # Rollout to global translation
    transl = rollout_local_transl_vel(local_transl_vel, global_orient)
    global_orient, transl, _ = get_tgtcoord_rootparam(global_orient, transl, tsf="any->ay")

    smpl_params_w_Rt = {"global_orient": global_orient, "transl": transl}
    return smpl_params_w_Rt


def rollout_human_yaw_direct(cam_roll_pitch, global_orient_mat_c, cam_angvel, init_cam_yaw=None):
    """Direct composition without decompose-recompose (路线二优化)."""
    B, L = cam_angvel.shape[0], cam_angvel.shape[1]
    device = cam_angvel.device
    cam_angvel_rot = rotation_6d_to_matrix(cam_angvel)
    
    if init_cam_yaw is None:
        init_cam_yaw = torch.eye(3).repeat(B, 1, 1).float().to(device)
    
    yaw_vel_tensor = torch.einsum("...ij,...kj,...lk->...il", cam_roll_pitch[:, :-1], cam_angvel_rot[:, :-1], cam_roll_pitch[:, 1:])
    yaw_vel = torch.unbind(yaw_vel_tensor, dim=1)
    
    cam_yaw_list = [init_cam_yaw]
    for i in range(L - 1):
        cam_yaw_next = torch.einsum("...ij,...jk->...ik", cam_yaw_list[-1], yaw_vel[i])
        cam_yaw_list.append(cam_yaw_next)
    cam_yaw = torch.stack(cam_yaw_list, dim=1)
    
    human_ori_w = torch.einsum("...ij,...jk,...kl->...il", cam_yaw, cam_roll_pitch, global_orient_mat_c)
    R_align = human_ori_w[:, [0]].mT
    human_ori_w_aligned = torch.einsum("...ij,...jk->...ik", R_align, human_ori_w)
    cam_ori_w = torch.einsum("...ij,...jk->...ik", cam_yaw, cam_roll_pitch)
    
    return human_ori_w_aligned, cam_ori_w

import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from core.utils.pylogger import Log
from core.utils.geo.hmr_cam import normalize_kp2d
from core.utils.smplx_utils import make_smplx
from core.configs import MainStore, builds
from hydra.utils import instantiate
from core.utils.pylogger import Log
from einops import rearrange, einsum
from core.configs import MainStore, builds

from core.utils.geo_transform import compute_T_ayfz2ay, apply_T_on_points
from core.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from core.utils.smplx_utils import make_smplx
from core.utils.geo.augment_noisy_pose import (
    get_wham_aug_kp3d,
    get_visible_mask,
    get_invisible_legs_mask,
    randomly_occlude_lower_half,
    randomly_modify_hands_legs,
)
from core.utils.geo.hmr_cam import perspective_projection, normalize_kp2d, safely_render_x3d_K, get_bbx_xys

from core.utils.geo.flip_utils import flip_smplx_params, avg_smplx_aa
from core.model.watch.utils.postprocess import pp_static_joint, pp_static_joint_cam, process_ik


class WatchPL(pl.LightningModule):
    def __init__(self,
                 pipeline,
                 optimizer=None,
                 scheduler_cfg=None,
                 ignored_weights_prefix=['smplx', 'pipeline.encoder'],
                 ablation_mask=None,  # For zero-cost ablation at inference: {"f_cam_angvel": True, "f_cam_transl": True}
                 postproc_version=1,
                 ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.optimizer = instantiate(optimizer)
        self.scheduler_cfg = scheduler_cfg

        # Options
        self.ignored_weights_prefix = ignored_weights_prefix
        self.ablation_mask = ablation_mask  # Store for use in validation_step
        self.postproc_version = postproc_version
        self.force_postproc = False  # Can be set externally to force postproc during validation

        # The test step is the same as validation
        self.test_step = self.predict_step = self.validation_step

        # SMPLX
        self.smplx = make_smplx("supermotion_v437coco17")

    def training_step(self, batch, batch_idx):
        B, F = batch["smpl_params_c"]["body_pose"].shape[:2]

        # Create augmented noisy-obs : gt_j3d(coco17)
        with torch.no_grad():
            gt_verts437, gt_j3d = self.smplx(**batch["smpl_params_c"])
            root_ = gt_j3d[:, :, [11, 12], :].mean(-2, keepdim=True)
            batch["gt_j3d"] = gt_j3d
            batch["gt_cr_coco17"] = gt_j3d - root_
            batch["gt_c_verts437"] = gt_verts437
            batch["gt_cr_verts437"] = gt_verts437 - root_

        # bbx_xys
        i_x2d = safely_render_x3d_K(gt_verts437, batch["K_fullimg"], thr=0.3)
        bbx_xys = get_bbx_xys(i_x2d, do_augment=True)
        if False:  # trust image bbx_xys seems better
            batch["bbx_xys"] = bbx_xys
        else:
            mask_bbx_xys = batch["mask"]["bbx_xys"]
            batch["bbx_xys"][~mask_bbx_xys] = bbx_xys[~mask_bbx_xys]

        # noisy_j3d -> project to i_j2d -> compute a bbx -> normalized kp2d [-1, 1]
        noisy_j3d = gt_j3d + get_wham_aug_kp3d(gt_j3d.shape[:2])
        if True:
            noisy_j3d = randomly_modify_hands_legs(noisy_j3d)
        obs_i_j2d = perspective_projection(noisy_j3d, batch["K_fullimg"])  # (B, L, J, 2)
        j2d_visible_mask = get_visible_mask(gt_j3d.shape[:2]).cuda()  # (B, L, J)
        j2d_visible_mask[noisy_j3d[..., 2] < 0.3] = False  # Set close-to-image-plane points as invisible
        if True:  # Set both legs as invisible for a period
            legs_invisible_mask = get_invisible_legs_mask(gt_j3d.shape[:2]).cuda()  # (B, L, J)
            j2d_visible_mask[legs_invisible_mask] = False
        obs_kp2d = torch.cat([obs_i_j2d, j2d_visible_mask[:, :, :, None].float()], dim=-1)  # (B, L, J, 3)
        obs = normalize_kp2d(obs_kp2d, batch["bbx_xys"])  # (B, L, J, 3)
        obs[~j2d_visible_mask] = 0  # if not visible, set to (0,0,0)
        batch["obs"] = obs

        if True:  # Use some detected vitpose (presave data)
            prob = 0.5
            mask_real_vitpose = (torch.rand(B).to(obs_kp2d) < prob) * batch["mask"]["vitpose"]
            batch["obs"][mask_real_vitpose] = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])[mask_real_vitpose]

        # Set untrusted frames to False
        batch["obs"][~batch["mask"]["valid"]] = 0

        # Augment cam local vel (batch["cam_transl_local"]) here.
        #   Two augmentations:
        #   1. Gaussian noise, std use the statistic value: [0.239, 0.07, 0.08]
        #   2. Scale augment, scale is 0.85~1.15;
        # noise = (torch.randn_like(batch["cam_transl_local"], device=batch["cam_transl_local"].device) *
        #          torch.tensor([0.239, 0.07, 0.08], device=batch["cam_transl_local"].device))
        # batch["cam_transl_local"] += noise.float()
        # scale = (torch.randn_like(batch["cam_transl_local"], device=batch["cam_transl_local"].device) * 0.3 + 0.85).float()
        # batch["cam_transl_local"] *= scale

        # Forward and get loss
        outputs = self.pipeline.forward(batch, train=True)

        # Guard against non-finite losses to prevent poisoning validation metrics.
        if not torch.isfinite(outputs["loss"]):
            # sync_dist=False: this is a rank-local event; sync_dist=True would emit a
            # scalar AllReduce that only rank-N calls, deadlocking against rank-0's
            # gradient AllReduce when losses differ across ranks.
            self.log("train/nonfinite_loss", 1.0, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=B, sync_dist=False)
            for k, v in list(outputs.items()):
                if torch.is_tensor(v) and (not torch.isfinite(v).all()):
                    outputs[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            if not torch.isfinite(outputs["loss"]):
                # Keep the loss connected to all parameters so DDP still triggers
                # AllReduce for every gradient bucket (with zero values).  A detached
                # leaf tensor would cause no AllReduce on this rank, deadlocking peers.
                outputs["loss"] = sum(p.sum() * 0 for p in self.parameters() if p.requires_grad)

        # Log
        log_kwargs = {
            "on_epoch": True,
            "prog_bar": True,
            "logger": True,
            "batch_size": B,
            "sync_dist": True,
        }
        self.log("train/loss", outputs["loss"], **log_kwargs)
        for k, v in outputs.items():
            if "_loss" in k:
                self.log(f"train/{k}", v, **log_kwargs)

        return outputs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Options & Check
        do_postproc = self.trainer.state.stage == "test" or getattr(self, "force_postproc", False)  # Apply postproc in test or when forced
        # do_postproc = False
        do_flip_test = "flip_test" in batch
        do_postproc_not_flip_test = do_postproc and not do_flip_test  # later pp when flip_test
        assert batch["B"] == 1, "Only support batch size 1 in evalution."

        # ROPE inference
        obs = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])
        if "mask" in batch:
            obs[0, ~batch["mask"][0]] = 0

        batch_ = {
            "length": batch["length"],
            "obs": obs,
            "bbx_xys": batch["bbx_xys"],
            "K_fullimg": batch["K_fullimg"],
            "cam_angvel": batch["cam_angvel"],
            "cam_transl_local": batch["cam_transl_local"],
            "cam_transl": batch["cam_transl"],  # Used for camera ate
            "cam_yaw": batch["cam_yaw"],
            "cam_roll_pitch": batch["cam_roll_pitch"],
            "f_imgseq": batch["f_imgseq"],
        }
        eval_inputs = batch_
        # Get ablation mask from config (for zero-cost ablation studies)
        ablation_mask = getattr(self, 'ablation_mask', None)
        if False:
            import time
            start = time.time()
            outputs = self.pipeline.forward(batch_, train=False, postproc=do_postproc_not_flip_test, ablation_mask=ablation_mask, postproc_version=self.postproc_version)
            elapsed = time.time() - start
            print(f"[Inference] forward took {elapsed:.4f}s")
        else:
            outputs = self.pipeline.forward(batch_, train=False, postproc=do_postproc_not_flip_test, ablation_mask=ablation_mask, postproc_version=self.postproc_version)
        outputs["pred_smpl_params_global"] = {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()}
        outputs["pred_smpl_params_incam"] = {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()}

        if do_flip_test:
            flip_test = batch["flip_test"]
            obs = normalize_kp2d(flip_test["kp2d"], flip_test["bbx_xys"])
            if "mask" in batch:
                obs[0, ~batch["mask"][0]] = 0

            batch_ = {
                "length": batch["length"],
                "obs": obs,
                "bbx_xys": flip_test["bbx_xys"],
                "K_fullimg": batch["K_fullimg"],
                "cam_angvel": flip_test["cam_angvel"],
                "cam_transl_local": flip_test["cam_transl_local"],
                "f_imgseq": flip_test["f_imgseq"],
            }
            flipped_outputs = self.pipeline.forward(batch_, train=False)

            # First update incam results
            flipped_outputs["pred_smpl_params_incam"] = {
                k: v[0] for k, v in flipped_outputs["pred_smpl_params_incam"].items()
            }
            smpl_params1 = outputs["pred_smpl_params_incam"]
            smpl_params2 = flip_smplx_params(flipped_outputs["pred_smpl_params_incam"])

            smpl_params_avg = smpl_params1.copy()
            smpl_params_avg["betas"] = (smpl_params1["betas"] + smpl_params2["betas"]) / 2
            smpl_params_avg["body_pose"] = avg_smplx_aa(smpl_params1["body_pose"], smpl_params2["body_pose"])
            smpl_params_avg["global_orient"] = avg_smplx_aa(
                smpl_params1["global_orient"], smpl_params2["global_orient"]
            )
            outputs["pred_smpl_params_incam"] = smpl_params_avg

            # Then update global results
            outputs["pred_smpl_params_global"]["betas"] = smpl_params_avg["betas"]
            outputs["pred_smpl_params_global"]["body_pose"] = smpl_params_avg["body_pose"]

            # Finally, apply postprocess
            if do_postproc:
                # temporarily recover the original batch-dim
                outputs["pred_smpl_params_global"] = {k: v[None] for k, v in outputs["pred_smpl_params_global"].items()}
                if self.postproc_version == 2:
                    from core.model.watch.utils.postprocess import pp_static_joint_v2
                    outputs["pred_smpl_params_global"]["transl"] = pp_static_joint_v2(outputs, self.pipeline.endecoder)
                else:
                    outputs["pred_smpl_params_global"]["transl"] = pp_static_joint(outputs, self.pipeline.endecoder)
                body_pose = process_ik(outputs, self.pipeline.endecoder)
                outputs["pred_smpl_params_global"] = {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()}

                outputs["pred_smpl_params_global"]["body_pose"] = body_pose[0]
                # outputs["pred_smpl_params_incam"]["body_pose"] = body_pose[0]

        outputs = self.pipeline.apply_sgf_lite_fusion(eval_inputs, outputs)
        outputs = self.pipeline.apply_followcam_fusion(eval_inputs, outputs)
        return outputs

    def configure_optimizers(self):
        params = []
        for k, v in self.pipeline.named_parameters():
            if v.requires_grad:
                params.append(v)
        optimizer = self.optimizer(params=params)

        if self.scheduler_cfg["scheduler"] is None:
            return optimizer

        scheduler_cfg = dict(self.scheduler_cfg)
        scheduler_cfg["scheduler"] = instantiate(scheduler_cfg["scheduler"], optimizer=optimizer)
        return [optimizer], [scheduler_cfg]

    # ============== Utils ================= #
    def on_save_checkpoint(self, checkpoint) -> None:
        for ig_keys in self.ignored_weights_prefix:
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith(ig_keys):
                    # Log.info(f"Remove key `{ig_keys}' from checkpoint.")
                    checkpoint["state_dict"].pop(k)

    def load_pretrained_model(self, ckpt_path):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"[PL-Trainer] Loading ckpt: {ckpt_path}")

        state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        real_missing = []
        for k in missing:
            ignored_when_saving = any(k.startswith(ig_keys) for ig_keys in self.ignored_weights_prefix)
            if not ignored_when_saving:
                real_missing.append(k)

        if len(real_missing) > 0:
            Log.warn(f"Missing keys: {real_missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")


watch_pl = builds(
    WatchPL,
    pipeline="${pipeline_watch}",
    optimizer="${optimizer}",
    scheduler_cfg="${scheduler_cfg}",
    populate_full_signature=True,  # Adds all the arguments to the signature
)
MainStore.store(name="watch_pl", node=watch_pl, group="model/watch")

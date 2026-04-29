import os.path

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from core.configs import MainStore, builds

from core.utils.comm.gather import all_gather
from core.utils.pylogger import Log

from core.utils.eval.eval_utils import (
    compute_camcoord_metrics,
    compute_global_metrics,
    compute_camcoord_perjoint_metrics,
    rearrange_by_mask,
    as_np_array,
)
from core.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from core.utils.smplx_utils import make_smplx
from einops import einsum, rearrange

from core.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from core.utils.vis.renderer import Renderer, get_global_cameras_static
from core.utils.geo.hmr_cam import estimate_focal_length
from core.utils.video_io_utils import read_video_np, save_video
from core import PROJ_CORE
import imageio
from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2


class MetricMocap(pl.Callback):
    def __init__(self, emdb_split=1):
        """
        Args:
            emdb_split: 1 to evaluate incam, 2 to evaluate global
        """
        super().__init__()
        # vid->result
        if emdb_split == 1:
            self.target_dataset_id = "EMDB_1"
            self.metric_aggregator = {
                "pa_mpjpe": {},
                "mpjpe": {},
                "pve": {},
                "accel": {},
            }
        elif emdb_split == 2:
            self.target_dataset_id = "EMDB_2"
            self.metric_aggregator = {
                "wa2_mpjpe": {},
                "waa_mpjpe": {},
                "rte": {},
                "jitter": {},
                "fs": {},
                # "wa2_cam_ate": {},
                # "waa_cam_ate": {},
                # "cam_ate": {},
                # "cam_scale": {},
            }
        else:
            raise ValueError(f"Unknown emdb_split: {emdb_split}")

        # SMPL
        self.smplx = make_smplx("supermotion")
        self.smpl_model = {"male": make_smplx("smpl", gender="male"), "female": make_smplx("smpl", gender="female")}

        self.J_regressor = torch.load(PROJ_CORE / "utils/body_model/smpl_neutral_J_regressor.pt")
        self.smplx2smpl = torch.load(PROJ_CORE / "utils/body_model/smplx2smpl_sparse.pt")
        self.faces_smpl = self.smpl_model["male"].faces
        self.faces_smplx = self.smplx.faces

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end

        # Only validation record the metrics with logger
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        if dataset_id != self.target_dataset_id:
            return

        # Move to cuda if not
        self.smplx = self.smplx.cuda()
        for g in ["male", "female"]:
            self.smpl_model[g] = self.smpl_model[g].cuda()
        self.J_regressor = self.J_regressor.cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

        vid = batch["meta"][0]["vid"]
        seq_length = batch["length"][0].item()
        gender = batch["gender"][0]
        T_w2c = batch["T_w2c"][0]
        mask = batch["mask"][0]

        # Groundtruth (world, cam)
        target_w_params = {k: v[0] for k, v in batch["smpl_params"].items()}
        target_w_output = self.smpl_model[gender](**target_w_params)
        target_w_verts = target_w_output.vertices
        target_w_j3d = torch.matmul(self.J_regressor, target_w_verts)
        target_c_verts = apply_T_on_points(target_w_verts, T_w2c)
        target_c_j3d = apply_T_on_points(target_w_j3d, T_w2c)

        # + Prediction -> Metric
        if self.target_dataset_id == "EMDB_1":  # in camera metrics
            # 1. cam
            pred_smpl_params_incam = outputs["pred_smpl_params_incam"]
            smpl_out = self.smplx(**pred_smpl_params_incam)
            pred_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
            pred_c_j3d = einsum(self.J_regressor, pred_c_verts, "j v, l v i -> l j i")
            del smpl_out  # Prevent OOM

            batch_eval = {
                "pred_j3d": pred_c_j3d,
                "target_j3d": target_c_j3d,
                "pred_verts": pred_c_verts,
                "target_verts": target_c_verts,
            }
            camcoord_metrics = compute_camcoord_metrics(batch_eval, mask=mask)
            for k in camcoord_metrics:
                self.metric_aggregator[k][vid] = as_np_array(camcoord_metrics[k])

            # # save_dir = f'F:/aitviewer/unify_visualize/data/watch_v2_6/emdb_1/{vid}'
            # save_dir = f'F:/aitviewer/unify_visualize/data/watch_v3_1/emdb_1/{vid}'
            # os.makedirs(save_dir, exist_ok=True)
            # save_data = {
            #     'poses': np.concatenate([outputs['pred_smpl_params_global']['global_orient'].detach().cpu().numpy(),
            #                              outputs['pred_smpl_params_global']['body_pose'].detach().cpu().numpy(),
            #                              np.zeros((outputs['pred_smpl_params_global']['body_pose'].shape[0], 6))],
            #                             axis=-1),
            #     'trans': outputs['pred_smpl_params_global']['transl'].detach().cpu().numpy(),
            #     'betas': outputs['pred_smpl_params_global']['betas'].detach().cpu().numpy(),
            #     'mocap_framerate': np.array(30),
            # }
            # np.savez(os.path.join(save_dir, '0.npz'), **save_data)

        elif self.target_dataset_id == "EMDB_2":  # global metrics
            # 2. global (align-y axis)
            pred_smpl_params_global = outputs["pred_smpl_params_global"]
            smpl_out = self.smplx(**pred_smpl_params_global)
            pred_ay_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
            pred_ay_j3d = einsum(self.J_regressor, pred_ay_verts, "j v, l v i -> l j i")
            del smpl_out  # Prevent OOM

            # # SLAM ATE
            # from pytorch3d.transforms import rotation_6d_to_matrix
            # from core.model.watch.pipeline.watch_pipeline import MoveDecoder
            # target_cam_trans = batch["cam_transl"]
            # pred_cam_ori = outputs["pred_cam_orient_w"]
            # init_cam_yaw = rotation_6d_to_matrix(batch["cam_yaw"][..., [0], :])  # (..., 1, 6) -> (..., 1, 3, 3)
            # pred_cam_ori = torch.einsum("...ij,...jk->...ik", init_cam_yaw, pred_cam_ori)
            # pred_cam_vel = outputs["pred_cam_move"]["cam_local_vel"]
            # move_decoder = MoveDecoder()
            # pred_cam_trans = move_decoder.rollout_local_vel(pred_cam_vel, pred_cam_ori, target_cam_trans[..., 0, :])

            batch_eval = {
                "pred_j3d_glob": pred_ay_j3d,
                "target_j3d_glob": target_w_j3d,
                "pred_verts_glob": pred_ay_verts,
                "target_verts_glob": target_w_verts,
                # "pred_cam_trans": pred_cam_trans.squeeze(0),
                # "target_cam_trans": target_cam_trans.squeeze(0),
            }
            global_metrics = compute_global_metrics(batch_eval, mask=mask)
            for k in global_metrics:
                self.metric_aggregator[k][vid] = as_np_array(global_metrics[k])

        # torch.save({"outputs": outputs, "batch": batch}, f"F:/VIDEO/GVHMR/outputs/P4_37_outdoor_run_circle_data.pt")
        # torch.save({"outputs": outputs, "batch": batch}, f"F:/VIDEO/GVHMR/outputs/P7_56_outdoor_stairs_up_down_data.pt")
        # torch.save({"outputs": outputs, "batch": batch}, f"F:/VIDEO/GVHMR/outputs/P0_09_outdoor_walk_data.pt")
        # torch.save({"outputs": outputs, "batch": batch}, f"F:/VIDEO/GVHMR/outputs/P6_49_outdoor_big_stairs_down_data.pt")
        # torch.save({"outputs": outputs, "batch": batch}, f"F:/VIDEO/GVHMR/outputs/P3_33_outdoor_soccer_warmup_b_data.pt")

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        """Without logger"""
        local_rank, world_size = trainer.local_rank, trainer.world_size
        if "mpjpe" in self.metric_aggregator:
            monitor_metric = "mpjpe"
        else:
            monitor_metric = list(self.metric_aggregator.keys())[0]

        # Reduce metric_aggregator across all processes
        metric_keys = list(self.metric_aggregator.keys())
        with torch.inference_mode(False):  # allow in-place operation of all_gather
            metric_aggregator_gathered = all_gather(self.metric_aggregator)  # list of dict
        for metric_key in metric_keys:
            for d in metric_aggregator_gathered:
                self.metric_aggregator[metric_key].update(d[metric_key])

        total = len(self.metric_aggregator[monitor_metric])
        Log.info(f"{total} sequences evaluated in {self.__class__.__name__}")
        if total == 0:
            return

        # print monitored metric per sequence
        mm_per_seq = {k: v.mean() for k, v in self.metric_aggregator[monitor_metric].items()}
        if len(mm_per_seq) > 0:
            sorted_mm_per_seq = sorted(mm_per_seq.items(), key=lambda x: x[1], reverse=True)
            n_worst = len(sorted_mm_per_seq)  # show all sequences
            if local_rank == 0:
                Log.info(
                    f"monitored metric {monitor_metric} per sequence\n"
                    + "\n".join([f"{m:5.1f} : {s}" for s, m in sorted_mm_per_seq[:n_worst]])
                    + "\n------"
                )

        # average over all batches
        metrics_avg = {k: np.concatenate(list(v.values())).mean() for k, v in self.metric_aggregator.items()}
        if local_rank == 0:
            Log.info(
                f"[Metrics] {self.target_dataset_id}:\n"
                + "\n".join(f"{k}: {v:.1f}" for k, v in metrics_avg.items())
                + "\n------"
            )

        # save to logger if available
        if pl_module.logger is not None:
            cur_epoch = pl_module.current_epoch
            for k, v in metrics_avg.items():
                pl_module.logger.log_metrics({f"val_metric_{self.target_dataset_id}/{k}": v}, step=cur_epoch)

        # reset
        for k in self.metric_aggregator:
            self.metric_aggregator[k] = {}


emdb1_node = builds(MetricMocap, emdb_split=1)
emdb2_node = builds(MetricMocap, emdb_split=2)
MainStore.store(name="metric_emdb1", node=emdb1_node, group="callbacks", package="callbacks.metric_emdb1")
MainStore.store(name="metric_emdb2", node=emdb2_node, group="callbacks", package="callbacks.metric_emdb2")

import torch
# from hmr4d.network.sam3d import load_sam3d, SAM3D
from core.network.sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

from core.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
from core.utils.video_io_utils import read_video_np
import cv2
import numpy as np

from core.network.hmr2.utils.preproc import crop_and_resize, IMAGE_MEAN, IMAGE_STD
from tqdm import tqdm



class SAM3DExtractor:
    def __init__(self, tqdm_leave=True):
        model, model_cfg = load_sam_3d_body(
            checkpoint_path="inputs/checkpoints/sam3d/model.ckpt", device="cuda", mhr_path="inputs/checkpoints/sam3d/assets/mhr_model.pt",
        )
        self.estimator: SAM3DBodyEstimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
        )
        self.tqdm_leave = tqdm_leave

    def extract_video_features(self, video_path, bbx_xys):
        """
        img_ds makes the image smaller, which is useful for faster processing
        """
        imgs = read_video_np(video_path)
        bbx_xyxy = torch.zeros((len(bbx_xys), 4))
        bbx_xyxy[:, :2] = bbx_xys[:, :2] - bbx_xys[:, 2:3] / 2.0
        bbx_xyxy[:, 2:4] = bbx_xys[:, :2] + bbx_xys[:, 2:3] / 2.0

        features = []
        for i in range(len(imgs)):
            feature = self.estimator.process_one_image(
                imgs[i], 
                bbx_xyxy[i],
                inference_type="body",
                feat_mode=True,
            )
            features.append(feature.detach().cpu())
        
        features = torch.stack(features, dim=0).clone()  # (F, 1024)
        return features

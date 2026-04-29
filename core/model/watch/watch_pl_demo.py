import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from core.utils.pylogger import Log
from core.configs import MainStore, builds

from core.utils.geo.hmr_cam import normalize_kp2d


class WatchDemo(pl.LightningModule):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)

    def load_pretrained_model(self, ckpt_path):
        """Load pretrained watch model."""
        Log.info(f"Loading pretrained watch model from {ckpt_path}")

        state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            Log.warn(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")

    @torch.no_grad()
    def predict(self, data, static_cam=False):
        """
        Args:
            data: {
                "length": Torch.Tensor as int,
                "kp2d": (F, 3)
                "bbx_xys": (F, 3)
                "K_fullimg": (F, 3, 3)
                "cam_angvel": (F, 3)
                "cam_transl_local": (F, 3)
                "f_imgseq": (F, 3, 256, 256)
            }
            static_cam: bool, whether to use static camera
        """
        for key, value in data.items():
            print(key, value.shape)
        batch = {
            "length": data["length"][None],
            "obs": normalize_kp2d(data["kp2d"], data["bbx_xys"])[None],
            "bbx_xys": data["bbx_xys"][None],
            "K_fullimg": data["K_fullimg"][None],
            "cam_angvel": data["cam_angvel"][None],
            "cam_transl_local": torch.zeros_like(data["bbx_xys"])[None],
            "f_imgseq": data["f_imgseq"][None],
        }

        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = self.pipeline.forward(batch, train=False, postproc=False, static_cam=static_cam)

        pred = {
            "smpl_params_global": {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()},
            "smpl_params_incam": {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()},
            "K_fullimg": data["K_fullimg"],
            "net_outputs": outputs,  # intermediate outputs
        }
        # todo: 保留预测的相机轨迹和方向，用来和VGGT的外参做Kabsch对齐
        return pred


watch_pl = builds(WatchDemo, pipeline="${pipeline_watch}")
MainStore.store(name="watch_pl_demo", node=watch_pl, group="model/watch")

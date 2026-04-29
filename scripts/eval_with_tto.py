"""Evaluate WATCH ckpt with Test-Time Optimization (TTO).

Usage:
    python scripts/eval_with_tto.py \
      exp=watch/mixed/mixed_v3_2_geophys \
      task=watch/test_3dpw_emdb_rich \
      ckpt_path=outputs/.../e399.ckpt \
      pl_trainer.devices=1 \
      output_dir=outputs/eval_with_tto
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks.checkpoint import Checkpoint

from core.utils.pylogger import Log
from core.configs import register_store_watch
from core.utils.net_utils import load_pretrained_model
from core.utils.smplx_utils import make_smplx
from core.model.watch.extensions.tto import TTOOptimizer


def get_callbacks(cfg):
    if not hasattr(cfg, 'callbacks') or cfg.callbacks is None:
        return []
    callbacks = []
    for cb in cfg.callbacks.values():
        if cb is not None:
            callbacks.append(hydra.utils.instantiate(cb, _recursive_=False))
    return callbacks


@hydra.main(version_base='1.3', config_path='../core/configs', config_name='train')
def main(cfg: DictConfig):
    Log.info(f'[Exp Name]: {cfg.exp_name} (with TTO)')
    pl.seed_everything(cfg.seed)

    datamodule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    if cfg.ckpt_path:
        Log.info(f'Loading ckpt: {cfg.ckpt_path}')
        load_pretrained_model(model, cfg.ckpt_path)

    # Force postproc + enable TTO
    model.force_postproc = True
    model.use_tto = True  # WatchPL needs to check this flag in validation_step

    # Patch validation_step to apply TTO
    smpl_model = make_smplx("supermotion_v437coco17")
    tto = TTOOptimizer(
        smpl_model=smpl_model,
        num_iters_stage1=20,
        num_iters_stage2=20,
        num_iters_stage3=10,
        lr=1e-3,
    )

    original_val_step = model.validation_step

    def tto_val_step(batch, batch_idx, dataloader_idx=0):
        # Get initial outputs
        outputs = original_val_step(batch, batch_idx, dataloader_idx)

        # Apply TTO (skip if already in flip-test mode)
        try:
            video_inputs = {"K_fullimg": batch["K_fullimg"]}
            kp2d_obs = batch.get("kp2d", None)
            outputs = tto.optimize(outputs, video_inputs, sam_mask=None, kp2d_obs=kp2d_obs)
        except Exception as e:
            Log.warn(f"TTO failed for batch {batch_idx}: {e}")

        return outputs

    model.validation_step = tto_val_step

    callbacks = [cb for cb in get_callbacks(cfg) if not isinstance(cb, Checkpoint)]
    cfg.pl_trainer = {**cfg.pl_trainer, 'precision': 32, 'enable_checkpointing': False}
    trainer = pl.Trainer(**cfg.pl_trainer, callbacks=callbacks, logger=False)

    trainer.test(model, datamodule.test_dataloader())


if __name__ == "__main__":
    register_store_watch()
    main()

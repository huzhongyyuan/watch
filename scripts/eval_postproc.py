import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks.checkpoint import Checkpoint
from core.utils.pylogger import Log
from core.configs import register_store_watch
from core.utils.net_utils import load_pretrained_model


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
    Log.info(f'[Exp Name]: {cfg.exp_name}')
    pl.seed_everything(cfg.seed)

    datamodule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    if cfg.ckpt_path:
        Log.info(f'Loading ckpt: {cfg.ckpt_path}')
        load_pretrained_model(model, cfg.ckpt_path)

    # Force postproc during validation
    model.force_postproc = True
    Log.info(f'Postproc version: {model.postproc_version}')

    callbacks = [cb for cb in get_callbacks(cfg) if not isinstance(cb, Checkpoint)]
    cfg.pl_trainer = {**cfg.pl_trainer, 'precision': 32, 'enable_checkpointing': False}

    trainer = pl.Trainer(
        accelerator='gpu',
        logger=False,
        callbacks=callbacks,
        **cfg.pl_trainer,
    )

    Log.info('Start Eval with Postproc...')
    trainer.validate(model, datamodule.val_dataloader())
    Log.info('Done.')


if __name__ == '__main__':
    register_store_watch()
    main()

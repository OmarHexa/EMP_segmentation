from typing import Any, Dict, List, Optional, Tuple
import lightning as L
from lightning import LightningDataModule, LightningModule,Callback,Trainer
from lightning.pytorch.loggers import Logger

import hydra
from omegaconf import DictConfig,OmegaConf
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
)

log = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(config_path="../configs",config_name="config",version_base='1.1')
def main(cfg: DictConfig):
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


if __name__=="__main__":
    main()

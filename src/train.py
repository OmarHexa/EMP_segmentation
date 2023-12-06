
from dataset.EMP_datamodule import EmpDataModule
import pytorch_lightning as pl
from models.UnetModule import UnetLitModule
import hydra
from omegaconf import DictConfig,OmegaConf
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.utils import RankedLogger

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
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.dataset._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.dataset)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

   

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=None, logger=None)

    trainer.fit(model,datamodule)

if __name__=="__main__":
    main()

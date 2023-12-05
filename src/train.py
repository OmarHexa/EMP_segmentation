import torch.optim
import torchvision
from dataset.EMP_datamodule import EmpDataModule
from tqdm import tqdm
from pytorch_lightning import Trainer
from model.UnetModule import UnetLitModule
import hydra
from omegaconf import DictConfig,OmegaConf
   
@hydra.main(config_path="../configs",config_name="config")
def main(cfg: DictConfig):

    datamodule = EmpDataModule(cfg.directory,cfg.batch_size)
    model = UnetLitModule(learning_rate=cfg.learning_rate)

    # if LOAD_MODEL:
    #     loadModel(torch.load("my_checkpoint.pth.tar"), model)
    # ModelSize(model)
    trainer = Trainer(accelerator="gpu",
                      default_root_dir="./src/checkpoints",
                      precision="16-mixed",fast_dev_run=True)
    trainer.fit(model,datamodule)





if __name__=="__main__":
    main()

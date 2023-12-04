import torch.optim
import torchvision
from dataset.EMP_datamodule import EmpDataModule
from tqdm import tqdm
from utils.utils import (saveCheckpoint,loadModel,checkaccuarcy,ModelSize,savePredAsImages)
from pytorch_lightning import Trainer
from model.UnetModule import UnetLitModule
#hyper-parameters
BATCH_SIZE = 5
LEARNING_RATE =0.0001
NUM_EPOCHS = 5
NUM_WORKERS =1
IMAGE_HEIGHT = 256
IMAGE_WEDITH = 256
ITERATION = 2
LOAD_MODEL = False
DIRECTROY = "/home/omar/code/pytorch/EMP_data/"

   

def main(Bilinear=False):

    datamodule = EmpDataModule(DIRECTROY,BATCH_SIZE)
    model = UnetLitModule()

    # if LOAD_MODEL:
    #     loadModel(torch.load("my_checkpoint.pth.tar"), model)
    # ModelSize(model)
    trainer = Trainer(accelerator="gpu",min_epochs=1,max_epochs=3,
                      default_root_dir="./src/checkpoints",
                      precision="16-mixed",fast_dev_run=True)
    trainer.fit(model,datamodule)





if __name__=="__main__":
    main(Bilinear=False)

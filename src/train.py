import torch.optim
import torchvision
from dataset.EMP_datamodule import EmpDataModule
from tqdm import tqdm
from utils.utils import (saveCheckpoint,loadModel,checkaccuarcy,ModelSize,savePredAsImages,DiceBCELoss)
from pytorch_lightning import Trainer
from model.UnetModule import UnetLitModule
#hyper-parameters
BATCH_SIZE = 2
LEARNING_RATE =0.0001
NUM_EPOCHS = 20
NUM_WORKERS =1
IMAGE_HEIGHT = 256
IMAGE_WEDITH = 256
ITERATION = 2
LOAD_MODEL = False
DIRECTROY = "/home/omar/code/pytorch/EMP_data/"


def trainFnCPU(loader, model, optimizer, lossFn, iter, epoch,Writer=False):
    loop = tqdm(loader)
    runningLoss =0
    for idx, (img, seg) in enumerate(loop):
        seg = (seg>0).float().unsqueeze(1)

        # forward
        preds = model(img)
        loss = lossFn(preds, seg)
        loss = loss/iter #mean loss over the iteration
        runningLoss+= loss.item()
        # backward
        loss.backward() #gradient accumulation
        if ((idx+1)%iter==0) or ((idx+1)==len(loader)):
            optimizer.step() 
            optimizer.zero_grad()
            loop.set_postfix(loss=loss.item())



 # this code is quick development pupose only
def quickTrain(loader,model,lossFun,optimizer):
    im,seg =next(iter(loader))
    seg = (seg>0).float()
    optimizer.zero_grad()
    pred = model(im)
    loss =lossFun(pred,seg)
    loss.backward()
    optimizer.step()
    print(f"loss={loss.item()}, Predicted image shape: {pred.shape}")



def trainFnGPU(loader,model,optimizer,lossFun,iter):
    Device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(Device)
    scaler = torch.cuda.amp.GradScaler()
    loop = tqdm(loader)
    for idx, (image,segment) in enumerate(loop):
            image =image.to(Device)
            segment=segment.to(Device)
            segment = (segment>0).float()
            with torch.cuda.amp.autocast():
                predict = model(image)
                loss = lossFun(predict,segment)
                loss =loss/iter
            scaler.scale(loss).backward()
            if ((idx+1)%iter==0) or ((idx+1)==len(loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                loop.set_postfix(loss=loss.item())

    model.to("cpu")

   

def main(Bilinear=False):

    datamodule = EmpDataModule(DIRECTROY,BATCH_SIZE)
    model = UnetLitModule()

    # if LOAD_MODEL:
    #     loadModel(torch.load("my_checkpoint.pth.tar"), model)
    # ModelSize(model)

    trainer = Trainer(accelerator="cpu",min_epochs=1,max_epochs=3)
    trainer.fit(model,datamodule)





if __name__=="__main__":
    main(Bilinear=False)

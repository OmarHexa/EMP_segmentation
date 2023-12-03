




import torch.optim
import torchvision
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from src.dataset.EMP_datamodule import EmpDataset
from model.Unet import UNET, UNETBilinear
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv
from tqdm import tqdm
from utils.utils import (saveCheckpoint,loadModel,checkaccuarcy,ModelSize,savePredAsImages,DiceBCELoss)
# from torch.utils.tensorboard import SummaryWriter
# writer=SummaryWriter("runs/empUnetDice")

#hyper-parameters
BATCH_SIZE = 10
LEARNING_RATE =0.0001
NUM_EPOCHS = 20
NUM_WORKERS =1
IMAGE_HEIGHT = 256
IMAGE_WEDITH = 256
ITERATION = 2
LOAD_MODEL = False
IMG_DIR = "/home/omar/code/pytorch/EMP_data/images"
SEG_DIR = "/home/omar/code/pytorch/EMP_data/segmaps"


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
            # if Writer:  
                # writer.add_scalar("trainingLoss",runningLoss/iter,epoch*len(loader)+idx) #tensorboard loss update
                # runningLoss=0



 # this code is quick development pupose only
def quickTrain(loader,model,lossFun,optimizer):
    im,seg =next(iter(loader))
    seg = (seg>0).float().unsqueeze(1)
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

    transform= A.Compose([A.Resize(IMAGE_HEIGHT,IMAGE_WEDITH),
                                A.Rotate(limit=35, p=1.0,interpolation=cv.BORDER_CONSTANT),
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.1),
                                A.Normalize(mean=(0.47759078,0.47759078,0.47759078),
                                        std=(0.2459953,0.2459953,0.2459953)),
                                        ToTensorV2()])

    data = EmpDataset(IMG_DIR,SEG_DIR,augmentation=transform)

    #split the data set and load the data in batch to dataloader
    trainDataSize = int(len(data)*0.8)
    valDataSize = len(data)-trainDataSize

    print(f"Training Data Size: {trainDataSize}, Validation Data Size: {valDataSize} Effective Batch Size: {BATCH_SIZE*ITERATION}")
    trainData, valData = random_split(data,(trainDataSize,valDataSize))
    trainDataloder = DataLoader(dataset=trainData,batch_size=BATCH_SIZE,shuffle=True)
    valDataloder = DataLoader(dataset=valData,batch_size=BATCH_SIZE,shuffle=False)


    # setup model, loss funciton and optimizer
    model = UNETBilinear(3,1) if Bilinear else UNET(3,1)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    
    exImg,_=next(iter(trainDataloder))
    image_grid = torchvision.utils.make_grid(exImg)
    # writer.add_image("EMP_images",image_grid)
    # writer.add_graph(model,exImg)

    if LOAD_MODEL:
        loadModel(torch.load("my_checkpoint.pth.tar"), model)
    ModelSize(model)


    for epoch in range(NUM_EPOCHS):
        quickTrain(trainDataloder,model,optimizer,criterion,ITERATION,epoch)

        #save model
        checkPoint= {
                    "state_dice":model.state_dict(),
                    "optimizer":optimizer.state_dict,
                    "epoch":epoch
                    }
        saveCheckpoint(checkPoint)
        # quickTrain(trainDataloder,model,criterion,optimizer)

        checkaccuarcy(valDataloder,model)

        savePredAsImages(valDataloder,model)






if __name__=="__main__":
    main(Bilinear=False)

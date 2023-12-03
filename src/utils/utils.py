
import torch
import torchvision
import os
import torch.nn as nn
def saveCheckpoint(state,path="checkpoint.pth.tar"):
    print(f"====> saving checkpoint in directiory: {path}")
    torch.save(state,path)

def loadModel(checkpoint,model):
    print(f"====>loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


# taken from pytorch : https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
def ModelSize(model):
    param_size = sum([param.nelement()*param.element_size() for param in model.parameters()])
    buffer_size = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f}MB'.format(size_all_mb))


def checkaccuarcy(loader, model):
    numCorrect = 0
    numPixels = 0
    diceScore = 0
    model.eval()

    with torch.no_grad():
        for img,seg in loader:
            seg=(seg>0).float().unsqueeze(1)
            predicted = torch.sigmoid(model(img))
            predicted = (predicted>0.5).float()
            numCorrect+=(predicted==seg).sum()
            numPixels+=torch.numel(predicted)
            # print(f"correct pixel:{numCorrect},pixel number:{numPixels}")
            diceScore+= (2*(predicted*seg).sum())/(
                        (predicted+seg).sum()+1e-8
                        )

    print(
        f"Got {numCorrect}/{numPixels} with acc {(numCorrect/numPixels)*100:.2f}"
    )
    print(f"Dice score: {diceScore/len(loader)}")
    model.train()



def savePredAsImages(loader,model,folder="predictionImages/"):
    if not os.path.exists(folder):
        os.mkdir(folder)
    model.eval()
    for idx, (img,seg) in enumerate(loader):
        seg=(seg>0).float().unsqueeze(1)
        with torch.no_grad():
            preds = torch.sigmoid(model(img))
            preds = (preds>0.5).float()
        torchvision.utils.save_image(preds,f"{folder}/preds{idx}.png")
        torchvision.utils.save_image(seg,f"{folder}/{idx}.png")
    model.train()


# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = nn.functional.sigmoid(inputs) 
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = nn.functional.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
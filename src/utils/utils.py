
import torch
import torchvision
import os
import torch.nn as nn
import torch.nn.functional as F



# taken from pytorch : https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
def ModelSize(model):
    param_size = sum([param.nelement()*param.element_size() for param in model.parameters()])
    buffer_size = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f}MB'.format(size_all_mb))





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




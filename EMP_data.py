
import os
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Transfomation is only applied to 

class EmpDataset(Dataset):
    def __init__(self,image_dir,seg_dir,transform=None, augmentation= None) -> None:
        super(EmpDataset,self).__init__()

        self.image_dir =image_dir
        self.seg_dir = seg_dir
        self.images = os.listdir(self.image_dir)
        self.transform = transform


    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir,self.images[index])
        mask_path   = os.path.join(self.seg_dir,self.images[index])
        image = np.array(cv.imread(image_path,cv.IMREAD_UNCHANGED))
        mask = np.array(cv.imread(mask_path,cv.IMREAD_UNCHANGED))
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image,mask


    def __len__(self):
        return len(self.images)



#To test the data class 
if __name__=="__main__":
    dir= os.getcwd()
    image_dir = os.path.join(dir,"archive/images")
    seg_dir = os.path.join(dir,"archive/segmaps")

    data=EmpDataset(image_dir,seg_dir)
    im,seg =data.__getitem__(1)
    print(data.__len__())
    print(seg.shape)
    print(seg[250])
    fig,axes = plt.subplots(1,2,figsize=(16, 8))
    axes = axes.ravel()
    for ax in axes:
        ax.axis('off')
    axes[0].imshow(im)
    axes[1].matshow(seg,cmap="tab20")

    plt.show()

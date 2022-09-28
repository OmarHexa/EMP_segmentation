
import os
from torch.utils.data import Dataset,DataLoader,random_split,IterableDataset
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


# Transfomation is only applied to 

class EmpDataset(Dataset):
    def __init__(self,image_dir,seg_dir, augmentation= None) -> None:
        super(EmpDataset,self).__init__()

        self.image_dir =image_dir
        self.seg_dir = seg_dir
        self.augmentation = augmentation
        self.images = os.listdir(self.image_dir)


    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir,self.images[index])
        mask_path   = os.path.join(self.seg_dir,self.images[index])
        image = np.array(cv.imread(image_path,cv.IMREAD_UNCHANGED),dtype=np.float32)
        mask = np.array(cv.imread(mask_path,cv.IMREAD_UNCHANGED),dtype=np.float32)
        if self.augmentation:
            augmented = self.augmentation(image=image,mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image,mask


    def __len__(self):
        return len(self.images)



#To test the data class 
if __name__=="__main__":
    os.chdir("/home/omar/code/pytorch")
    BATCH_SIZE = 20
    dir= os.getcwd()
    image_dir = os.path.join(dir,"archive/images")
    seg_dir = os.path.join(dir,"archive/segmaps")

    data=EmpDataset(image_dir,seg_dir)

    print(type(data))

    im,seg =data.__getitem__(1)
    print((im.size*im.itemsize)*1e-6)


    print(seg.shape)
    fig,axes = plt.subplots(1,2,figsize=(16, 8))
    axes = axes.ravel()
    for ax in axes:
        ax.axis('off')
    axes[0].imshow(im.astype('uint8'),cmap="gray")
    axes[1].matshow(seg,cmap="tab20")

    plt.show()

    # mean = np.array([0.,0.,0.])
    # stdTemp = np.array([0.,0.,0.])
    # std = np.array([0.,0.,0.])
    # numSamples = data.__len__()
    # print(numSamples)
    # for i in range(numSamples):
    #     im,_ = data.__getitem__(i)
    #     im= im.astype(float)/255

    #     for j in range(3):
    #         mean[j]+= np.mean(im[:,:,j])
    # mean = mean/numSamples
    # for i in range(numSamples):
    #     im,_ = data.__getitem__(i)
    #     im= im.astype(float)/255

    #     for j in range(3):
    #         stdTemp[j] += ((im[:,:,j] - mean[j])**2).sum()/(im.shape[0]*im.shape[1])

    # std = np.sqrt(stdTemp/numSamples)
    # print(mean,std)
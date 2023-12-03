
import os
from torch.utils.data import Dataset,DataLoader,random_split,IterableDataset
import cv2 as cv
import numpy as np
from torchvision import transforms
import pytorch_lightning as pl

# Transfomation is only applied to 

class EmpDataset(Dataset):
    def __init__(self,image_dir,seg_dir, augmentation= None) -> None:
        super(EmpDataset,self).__init__()

        self.image_dir =image_dir
        self.seg_dir = seg_dir
        self.augmentation = augmentation
        self.images = os.listdir(self.image_dir)


    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.seg_dir, image_name)

        image = np.array(cv.imread(image_path, cv.IMREAD_UNCHANGED), dtype=np.float32)
        mask = np.array(cv.imread(mask_path, cv.IMREAD_UNCHANGED), dtype=np.float32)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


    def __len__(self):
        return len(self.images)

class EmpDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32):
        super().__init__()
        self.image_dir = os.path.join(data_dir,"images")
        self.seg_dir = os.path.join(dir,"segmaps")
        self.transform= transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((256, 256)),
                        transforms.RandomRotation(degrees=35),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(
                        mean=(0.47759078, 0.47759078, 0.47759078),
                        std=(0.2459953, 0.2459953, 0.2459953)
                            )
                        ])
        self.data = EmpDataset(image_dir=self.image_dir, seg_dir=self.seg_dir, augmentation=self.transform)
        self.batch_size = batch_size
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            trainDataSize = int(len(self.data)*0.8)
            valDataSize = len(self.data)-trainDataSize
            self.train_dataset,self.val_dataset = random_split(self.data,(trainDataSize,valDataSize))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        pass
    
#To test the data class 
if __name__=="__main__":
    os.chdir("/home/omar/code/pytorch/EMP_data/")
    BATCH_SIZE = 20
    dir= os.getcwd()
    module = EmpDataModule(dir,BATCH_SIZE)
    module.setup(stage="fit")

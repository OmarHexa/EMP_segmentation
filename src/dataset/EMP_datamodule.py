
import os
from torch.utils.data import Dataset,DataLoader,random_split
from PIL import Image
import torch
from torchvision import transforms
import pytorch_lightning as pl
from typing import Optional,List,Tuple

class EmpDataset(Dataset):
    def __init__(self, data_dir: str, img_transform= None, mask_transform = None) -> None:
        super(EmpDataset,self).__init__()
        self.image_dir: str = os.path.join(data_dir,"images")
        self.seg_dir: str = os.path.join(data_dir,"segmaps")
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.images: List[str] = os.listdir(self.image_dir)


    def __getitem__(self, index)-> Tuple[torch.Tensor|Image.Image, torch.Tensor|Image.Image]:
        image_name:str = self.images[index]
        image_path:str = os.path.join(self.image_dir, image_name)
        mask_path: str = os.path.join(self.seg_dir, image_name)

        # Open images as PIL Image
        image:Image.Image = Image.open(image_path)
        mask:Image.Image = Image.open(mask_path)


        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


    def __len__(self) -> int:
        return len(self.images)

class EmpDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int =32) -> None:
        super().__init__()
        self.dir: str = data_dir
        self.img_transform= transforms.Compose([
                        transforms.Resize((256, 256),antialias=None),
                        transforms.RandomRotation(degrees=35),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(
                        mean=(0.47759078, 0.47759078, 0.47759078),
                        std=(0.2459953, 0.2459953, 0.2459953)
                            )
                        ])
        self.mask_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((256,256),antialias=None)
                            ])
        self.batch_size: int = batch_size
        self.train_dataset: Optional[Dataset]= None
        self.val_dataset: Optional[Dataset]= None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            data: Dataset = EmpDataset(data_dir=self.dir,
                              img_transform=self.img_transform,
                              mask_transform=self.mask_transform)
            trainDataSize: int = int(len(data)*0.8)
            valDataSize: int = len(data)-trainDataSize
            self.train_dataset,self.val_dataset = random_split(data,(trainDataSize,valDataSize))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=2)

    def test_dataloader(self):
        pass


# Function to test the data module
def test_data_module(data_module: EmpDataModule) -> None:
    data_module.setup()
    # Access dataloaders
    train_dataloader: DataLoader = data_module.train_dataloader()
    val_dataloader: DataLoader = data_module.val_dataloader()

    # Perform a quick test to check if dataloaders are working
    for batch in train_dataloader:
        images, masks = batch
        print(images.shape, masks.shape)
        # Perform necessary testing logic here
        break  # Only process one batch for testing purposes

    for batch in val_dataloader:
        images, masks = batch
        print(images.shape, masks.shape)
        break  # Only process one batch for testing purposes
#To test the data class 
if __name__=="__main__":
    os.chdir("/home/omar/code/pytorch/EMP_data/")
    BATCH_SIZE = 20
    dir= os.getcwd()
    module = EmpDataModule(dir,BATCH_SIZE)
    test_data_module(module)

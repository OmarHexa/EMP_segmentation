
import os
from torch.utils.data import Dataset,DataLoader,random_split
from PIL import Image
import torch
from torchvision import transforms
import lightning.pytorch as pl
from typing import Optional,List,Tuple,Dict,Any
from dataclasses import dataclass


@dataclass
class EmpDataset(Dataset):
    data_dir:str
    img_transform: callable = None
    mask_transform: callable = None

    def __post_init__(self):
        self.image_dir:str = os.path.join(self.data_dir,"images")
        self.mask_dir:str = os.path.join(self.data_dir,"segmaps")
        self.images: List[str] = os.listdir(self.image_dir)
    def __getitem__(self, index)-> Tuple[torch.Tensor|Image.Image, torch.Tensor|Image.Image]:
        image_name:str = self.images[index]
        image_path:str = os.path.join(self.image_dir, image_name)
        mask_path: str = os.path.join(self.mask_dir, image_name)

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
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        resolution: int = 256,
        train_split: float = 1.0
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.img_transform= transforms.Compose([
                        transforms.Resize((resolution, resolution),antialias=None),
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
                                transforms.Resize((resolution,resolution),antialias=None)
                            ])

        self.data_train: Optional[Dataset]= None
        self.data_val: Optional[Dataset]= None
        self.data_test: Optional[Dataset]= None


    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.

  
        if stage == 'fit' or stage is None:
            data: Dataset = EmpDataset(data_dir=self.hparams.data_dir,
                              img_transform=self.img_transform,
                              mask_transform=self.mask_transform)
            trainDataSize: int = int(len(data)*self.hparams.train_split)
            valDataSize: int = len(data)-trainDataSize
            self.data_train, self.data_val = random_split(data,(trainDataSize,valDataSize))

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.hparams.batch_size,
                           shuffle=True,num_workers=self.hparams.num_workers,
                           pin_memory=self.hparams.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.hparams.batch_size,
                           shuffle=False,num_workers=self.hparams.num_workers,
                           pin_memory=self.hparams.pin_memory)
    def test_dataloader(self):
        pass
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
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

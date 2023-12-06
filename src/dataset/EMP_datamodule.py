import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import lightning.pytorch as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


@dataclass
class EmpDataset(Dataset):
    data_dir: str
    img_transform: callable = None
    mask_transform: callable = None

    def __post_init__(self):
        """Initialize image and mask directories after dataclass instantiation.

        Notes:
            This method is automatically called after the dataclass is initialized.
            It sets the image directory (`self.image_dir`) and mask directory (`self.mask_dir`)
            based on the provided `data_dir`. Additionally, it populates the list of image filenames
            (`self.images`) by listing the contents of the image directory.
        """
        self.image_dir: str = os.path.join(self.data_dir, "images")
        self.mask_dir: str = os.path.join(self.data_dir, "segmaps")
        self.images: List[str] = os.listdir(self.image_dir)

    def __getitem__(self, index) -> Tuple[torch.Tensor | Image.Image, torch.Tensor | Image.Image]:
        """Retrieve the image and its corresponding mask at the specified index.

        Parameters:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor | Image.Image, torch.Tensor | Image.Image]: A tuple containing the processed image
            and its corresponding mask. The image and mask can be either PIL Image objects or PyTorch tensors,
            depending on the transformations applied during initialization.

        Raises:
            IndexError: If the provided index is out of bounds.

        Notes:
            This method opens the image and mask files based on the index, applies any specified transformations,
            and returns the processed image and mask.

        Example:
            >>> dataset = YourDatasetClass(image_dir='path/to/images', mask_dir='path/to/masks', ...)
            >>> image, mask = dataset[0]
        """
        image_name: str = self.images[index]
        image_path: str = os.path.join(self.image_dir, image_name)
        mask_path: str = os.path.join(self.mask_dir, image_name)

        # Open images as PIL Image
        image: Image.Image = Image.open(image_path)
        mask: Image.Image = Image.open(mask_path)

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.

        Notes:
            This method returns the length of the dataset, which is equivalent to the number of images
            available in the dataset. It is automatically called when using the `len()` function on
            an instance of the dataset.

        Example:
            >>> dataset = YourDatasetClass(image_dir='path/to/images', mask_dir='path/to/masks', ...)
            >>> length = len(dataset)
        """
        return len(self.images)


class EmpDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        resolution: int = 256,
        train_split: float = 1.0,
    ) -> None:
        """Lightning DataModule for handling and preparing datasets for the segmentation tasks.

        Args:
            data_dir (str): Root directory containing the dataset.
            batch_size (int): Number of samples in each batch.
            num_workers (int): Number of parallel workers to load data.
            pin_memory (bool): Whether to use pin memory for faster data transfer to GPU.
            resolution (int): Desired resolution for image and mask resizing.
            train_split (float): Percentage of data to be used for training (0.0 to 1.0).

        Notes:
            This DataModule assumes the presence of 'images' and 'segmaps' directories
            inside the provided `data_dir`, containing image and segmentation map files.

        Example:
            >>> data_module = YourDataModuleClass(
            ...     data_dir='path/to/data',
            ...     batch_size=32,
            ...     num_workers=4,
            ...     resolution=128,
            ...     train_split=0.8
            ... )
        """
        super().__init__()
        # save_hyperparameters saves all the input parameters into self.hparams
        self.save_hyperparameters(logger=False)
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), antialias=None),
                transforms.RandomRotation(degrees=35),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.47759078, 0.47759078, 0.47759078),
                    std=(0.2459953, 0.2459953, 0.2459953),
                ),
            ]
        )
        self.mask_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((resolution, resolution), antialias=None)]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.

        if stage == "fit" or stage is None:
            data: Dataset = EmpDataset(
                data_dir=self.hparams.data_dir,
                img_transform=self.img_transform,
                mask_transform=self.mask_transform,
            )
            trainDataSize: int = int(len(data) * self.hparams.train_split)
            valDataSize: int = len(data) - trainDataSize
            self.data_train, self.data_val = random_split(data, (trainDataSize, valDataSize))

    def train_dataloader(self) -> DataLoader:
        """Training DataLoader for the image segmentation task.

        Returns:
            DataLoader: DataLoader for training data with specified batch size and transformations.

        Notes:
            The DataLoader is configured with parameters from the hyperparameters saved during
            DataModule initialization.
        """
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation DataLoader for the image segmentation task.

        Returns:
            DataLoader: DataLoader for validation data with specified batch size and transformations.

        Notes:
            The DataLoader is configured with parameters from the hyperparameters saved during
            DataModule initialization.
        """
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        """Test DataLoader for the image segmentation task.

        Returns:
            DataLoader: DataLoader for validation data with specified batch size and transformations.
        """
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


# # Function to test the data module
# def test_data_module(data_module: EmpDataModule) -> None:
#     data_module.setup()
#     # Access dataloaders
#     train_dataloader: DataLoader = data_module.train_dataloader()
#     val_dataloader: DataLoader = data_module.val_dataloader()

#     # Perform a quick test to check if dataloaders are working
#     for batch in train_dataloader:
#         images, masks = batch
#         print(images.shape, masks.shape)
#         # Perform necessary testing logic here
#         break  # Only process one batch for testing purposes

#     for batch in val_dataloader:
#         images, masks = batch
#         print(images.shape, masks.shape)
#         break  # Only process one batch for testing purposes


# To test the data class
if __name__ == "__main__":
    os.chdir("/home/omar/code/pytorch/EMP_data/")
    BATCH_SIZE = 20
    dir = os.getcwd()
    module = EmpDataModule(dir, BATCH_SIZE)
    # test_data_module(module)

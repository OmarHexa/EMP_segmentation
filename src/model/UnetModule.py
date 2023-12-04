from typing import Any, Dict, Tuple

import torch
from pytorch_lightning import LightningModule

from torchmetrics import MaxMetric, MeanMetric
from model.networks.Unet import UNET
from utils.utils import DiceBCELoss



class UnetLitModule(LightningModule):
    

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.net = UNET(3,1)

        # loss function
        self.criterion = DiceBCELoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.net(x)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        x, y = batch
        y = (y>0).float()
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return loss, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
       
        loss, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss



    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
       
        loss, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)




    def configure_optimizers(self) -> Dict[str, Any]:
        return torch.optim.Adam(self.net.parameters(),lr=0.001)


if __name__ == "__main__":

    model = UnetLitModule()

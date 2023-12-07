from typing import Any, Dict, Tuple

import torch
import torchmetrics
from lightning.pytorch import LightningModule
from torchmetrics import MaxMetric, MeanMetric


class DiceBCELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        """Custom loss module combining Dice loss and Binary Cross-Entropy loss.

        Args:
            weight (torch.Tensor, optional): A manual rescaling weight to be applied to the
                binary cross-entropy loss. Default is None.
            size_average (bool, optional): Deprecated. By default, the losses are averaged
                over each loss element in the batch. Default is True.

        Notes:
            This loss is designed for binary segmentation tasks and utilizes both the Dice loss
            and Binary Cross-Entropy loss to create a combined loss function.

        Example:
            >>> loss_function = DiceBCELoss()
            >>> inputs = torch.randn((batch_size, num_channels, height, width))
            >>> targets = torch.randint(0, 2, size=(batch_size, 1, height, width)).float()
            >>> loss = loss_function(inputs, targets)
        """
        super().__init__()

        # Define the dice metric
        self.dice_metric = torchmetrics.Dice()

    def forward(self, inputs, targets):
        """Calculate the combined Dice loss and Binary Cross-Entropy loss.

        Args:
            inputs (torch.Tensor): Predicted logits from the model.
            targets (torch.Tensor): Ground truth binary segmentation masks.

        Returns:
            torch.Tensor: Combined Dice and Binary Cross-Entropy loss.

        Notes:
            The forward method calculates the Dice loss and Binary Cross-Entropy loss
            separately and combines them to form the final loss.
        """
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate Binary Cross-Entropy using torch.nn.functional
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction="mean"
        )

        # Calculate Dice using torchmetrics function
        dice_loss = 1 - self.dice_metric(torch.sigmoid(inputs), targets)

        # Combine Dice and BCE losses
        dice_bce_loss = bce_loss + dice_loss

        return dice_bce_loss


class UnetLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
    ) -> None:
        """Lightning module for a U-Net based image segmentation task.

        Args:
            net (torch.nn.Module): U-Net model for image segmentation.
            optimizer (torch.optim.Optimizer): Optimizer for training the model.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            compile (bool, optional): Flag indicating whether to compile the model. Default is False.

        Notes:
            This module combines a U-Net model with a custom loss function (DiceBCELoss) and
            several metrics for training and evaluation.

        Example:
            >>> unet_model = UnetLitModule(net=my_unet_model, optimizer=my_optimizer, scheduler=my_scheduler)
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net

        # Loss function
        self.criterion = DiceBCELoss()

        # Metrics
        self.train_loss = MeanMetric()
        self.train_acc = torchmetrics.classification.Accuracy(
            task="binary", num_classes=1, threshold=0.5
        )
        self.val_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

        self.val_acc = torchmetrics.classification.Accuracy(
            task="binary", num_classes=1, threshold=0.5
        )
        self.val_f1 = torchmetrics.classification.F1Score(
            num_classes=1, task="binary", threshold=0.5
        )
        self.val_jaccard = torchmetrics.classification.JaccardIndex(
            num_classes=1, task="binary", threshold=0.5
        )

        # self.test_acc = torchmetrics.classification.Accuracy(
        #     task="binary", num_classes=1, threshold=0.5
        # )
        # self.test_f1 = torchmetrics.classification.F1Score(
        #     num_classes=1, task="binary", threshold=0.5
        # )
        # self.test_jaccard = torchmetrics.classification.JaccardIndex(
        #     num_classes=1, task="binary", threshold=0.5
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor from the U-Net model.
        """
        return self.net(x)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single step of the model during training.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch consisting of images and masks.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing loss, ground truth masks,
            and predicted logits.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y > 0)
        return loss, y, logits

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step for the U-Net model.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch consisting of images and masks.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value for the current training step.
        """
        loss, y, logits = self.model_step(batch)
        # update the log metrics
        self.train_loss(loss)
        self.train_acc(logits, y > 0)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step for the U-Net model.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch consisting of images and masks.
            batch_idx (int): Index of the current batch.
        """
        loss, y, logits = self.model_step(batch)
        self.val_loss(loss)

        y_pred = torch.sigmoid(logits)
        self.val_f1(y_pred, y > 0)
        self.val_jaccard(y_pred, y > 0)
        self.val_acc(y_pred, y > 0)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/jaccard", self.val_jaccard, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=False)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)


if __name__ == "__main__":
    model = UnetLitModule()
    # test_loss()

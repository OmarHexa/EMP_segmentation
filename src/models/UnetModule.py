from typing import Any, Dict, Tuple

import torch
from lightning.pytorch import LightningModule
import torchmetrics
from torchmetrics import MaxMetric, MeanMetric

class DiceBCELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

        # Define the dice metric
        self.dice_metric = torchmetrics.Dice()

    def forward(self, inputs, targets):

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='mean')


        # Calculate Dice using torchmetrics function
        dice_loss = 1 - self.dice_metric(torch.sigmoid(inputs), targets)

        # Calculate Binary Cross-Entropy using torch.nn.functional

        # Combine Dice and BCE losses
        dice_bce_loss = bce_loss + dice_loss

        return dice_bce_loss
    

class UnetLitModule(LightningModule):
    

    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 compile: bool=False
                ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net

        # loss function
        self.criterion = DiceBCELoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        #Validation metrics
        self.val_acc = torchmetrics.classification.Accuracy(task="binary",num_classes=1,threshold=0.5)
        self.val_f1 = torchmetrics.classification.F1Score(num_classes=1, task="binary", threshold=0.5)
        self.val_jaccard = torchmetrics.classification.JaccardIndex(num_classes=1, task="binary", threshold=0.5)
        #Test metrics
        self.test_acc = torchmetrics.classification.Accuracy(task="binary",num_classes=1,threshold=0.5)
        self.test_f1 = torchmetrics.classification.F1Score(num_classes=1, task="binary", threshold=0.5)
        self.test_jaccard = torchmetrics.classification.JaccardIndex(num_classes=1, task="binary", threshold=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.net(x)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y>0)
        return loss, y,logits

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
        ) -> torch.Tensor:
       
        loss, _,_ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:

        loss, y, logits = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss) #using mean metrics 
        
        # Compute F1 Score and Jaccard Index
        y_pred = torch.sigmoid(logits)
        self.val_f1(y_pred, y>0)
        self.val_jaccard(y_pred, y>0)
        self.val_acc(y_pred, y>0)

        # update and log metrics
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/jaccard", self.val_jaccard, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    # def configure_optimizers(self) -> Dict[str, Any]:
    #     return torch.optim.Adam(self.net.parameters(),lr=self.learning_rate)
    
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

def test_loss():
    # Instantiate the loss function
    loss_fn = DiceBCELoss()

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Example usage
    batch_size = 4
    image_size = (1, 256, 256)  # Single-channel images of size 256x256
    inputs = torch.randn(batch_size, *image_size, requires_grad=True)  # Replace with your actual input shape
    targets = torch.randint(0, 2, (batch_size, *image_size))  # Replace with your actual target shape
    targets.requires_grad = False  # Ensure targets do not require gradients

    # Create an optimizer (for illustrative purposes)
    optimizer = torch.optim.Adam([inputs], lr=0.001)

    # Forward pass and compute the loss
    loss = loss_fn(inputs, targets)

    # Backward pass (for illustrative purposes, you may not need this in testing)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check if the loss is finite
    assert torch.isfinite(loss).all().item(), "Loss is not finite"

    # Check if the gradients are finite
    assert torch.isfinite(inputs.grad).all().item(), "Gradients are not finite"

    # Check if the loss matches expectations using torch.testing
    torch.testing.assert_close(loss, torch.tensor(1.306977868), rtol=1e-5, atol=1e-5)

    print("Test passed!")
if __name__ == "__main__":

    # model = UnetLitModule()
    test_loss()

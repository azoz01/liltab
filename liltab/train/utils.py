import pytorch_lightning as pl
import torch.nn.functional as F

from torch import optim, Tensor
from typing import Any, Callable

from ..model.heterogenous_attributes_network import HeterogenousAttributesNetwork


class LightningWrapper(pl.LightningModule):
    """
    Wrapper around pyTorch model, which makes it compatible with pyTorch-lightning
    framework
    """

    def __init__(self, model: HeterogenousAttributesNetwork, loss: Callable):
        """
        Args:
            model (HeterogenousAttributesNetwork): network to be wrapped around.
            loss (Callable): loss function used during training
        """
        super().__init__()
        self.model = model
        self.loss = F.mse_loss

    def training_step(self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor]], batch_idx):
        loss_value = 0.0
        for example in batch:
            X_support, y_support, X_query, y_query = example
            prediction = self.model(X_support, y_support, X_query)
            loss_value += self.loss(prediction, y_query)
        self.log("train_loss", loss_value, prog_bar=True)
        return loss_value

    def validation_step(self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor]], batch_idx):
        loss_value = 0.0
        for example in batch:
            X_support, y_support, X_query, y_query = example
            prediction = self.model(X_support, y_support, X_query)
            loss_value += self.loss(prediction, y_query)
        self.log("val_loss", loss_value, prog_bar=True)
        return loss_value

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters())

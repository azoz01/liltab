import pytorch_lightning as pl
import torch.nn.functional as F

from torch import optim, Tensor
from typing import Any, Callable

from ..model.heterogenous_attributes_network import HeterogenousAttributesNetwork


class LightningWrapper(pl.LightningModule):
    """
    Wrapper around pyTorch model, which makes it compatible with pyTorch-lightning
    framework. It's purpose is to implement pyTorch-lightning hooks.
    """

    def __init__(
        self,
        model: HeterogenousAttributesNetwork,
        learning_rate: float,
        weight_decay: float,
        loss: Callable = F.mse_loss,
    ):
        """
        Args:
            model (HeterogenousAttributesNetwork): network to be wrapped around.
            learning_rate (float): learning rate during training.
            weight_decay (float): weight decay during training.
            loss (Callable): loss function used during training.
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.metrics_history = dict()

    def training_step(self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor]], batch_idx) -> float:
        loss_value = 0.0
        for example in batch:
            X_support, y_support, X_query, y_query = example
            prediction = self.model(X_support, y_support, X_query)
            loss_value += self.loss(prediction, y_query)

        self.log("train_loss", loss_value, prog_bar=True)
        self.metrics_history["train_loss"] = self.metrics_history.get("train_loss", []) + [
            loss_value.item()
        ]
        return loss_value

    def validation_step(
        self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor]], batch_idx
    ) -> float:
        loss_value = 0.0
        for example in batch:
            X_support, y_support, X_query, y_query = example
            prediction = self.model(X_support, y_support, X_query)
            loss_value += self.loss(prediction, y_query)
        self.log("val_loss", loss_value, prog_bar=True)
        self.metrics_history["val_loss"] = self.metrics_history.get("val_loss", []) + [
            loss_value.item()
        ]

        return loss_value

    def test_step(self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor]], batch_idx) -> float:
        loss_value = 0.0
        for example in batch:
            X_support, y_support, X_query, y_query = example
            prediction = self.model(X_support, y_support, X_query)
            loss_value += self.loss(prediction, y_query)
        self.log("test_loss", loss_value, prog_bar=True)
        self.metrics_history["test_loss"] = self.metrics_history.get("test_loss", []) + [
            loss_value.item()
        ]
        return loss_value

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

import pytorch_lightning as pl
import torch
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

        self.save_hyperparameters()

    def training_step(self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor]], batch_idx) -> float:
        sum_loss_value = 0.0
        for i, example in enumerate(batch):
            X_support, y_support, X_query, y_query = example
            prediction = self.model(X_support, y_support, X_query)
            loss = self.loss(prediction, y_query)
            if torch.isnan(loss):
                sum_loss_value = sum_loss_value * (i + 1) / i if i > 0 else 0
            else:
                sum_loss_value += loss

        return sum_loss_value

    def validation_step(
        self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor]], batch_idx
    ) -> float:
        sum_loss_value = 0.0
        for i, example in enumerate(batch):
            X_support, y_support, X_query, y_query = example
            prediction = self.model(X_support, y_support, X_query)
            loss = self.loss(prediction, y_query)
            if torch.isnan(loss):
                sum_loss_value = sum_loss_value * (i + 1) / i if i > 0 else 0
            else:
                sum_loss_value += loss

        return sum_loss_value

    def test_step(self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor]], batch_idx) -> float:
        sum_loss_value = 0.0
        for i, example in enumerate(batch):
            X_support, y_support, X_query, y_query = example
            prediction = self.model(X_support, y_support, X_query)
            loss = self.loss(prediction, y_query)
            if torch.isnan(loss):
                sum_loss_value = sum_loss_value * (i + 1) / i if i > 0 else 0
            else:
                sum_loss_value += loss

        return sum_loss_value

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def calculate_model_weights_norm(self) -> Tensor:
        pass

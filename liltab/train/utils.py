import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torch import optim, Tensor
from typing import Any, Callable, List

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
        representation_penalty_weight: float = 0,
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
        self.representation_penalty_weight = representation_penalty_weight
        self.metrics_history = dict()

        self.save_hyperparameters()

    def training_step(self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor]], batch_idx) -> float:
        sum_loss_value = 0.0
        supports_representations = [None] * len(batch)
        indices = [None] * len(batch)
        for i, example in enumerate(batch):
            idx, (X_support, y_support, X_query, y_query) = example
            full_trace = self.model(X_support, y_support, X_query, return_full_trace=True)
            prediction = full_trace["prediction"]

            supports_representations[i] = full_trace["support_set_representation"]
            indices[i] = idx
            loss = torch.nn.CrossEntropyLoss()(prediction, y_query)
            if torch.isnan(loss):
                sum_loss_value = sum_loss_value * (i + 1) / i if i > 0 else 0
            else:
                sum_loss_value += loss

        if self.representation_penalty_weight != 0:
            rep_loss = (
                self.representation_penalty_weight
                * self._calculate_representation_penalty(supports_representations, indices)
                / 4
            )
            sum_loss_value += rep_loss

        return sum_loss_value

    def validation_step(
        self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor]], batch_idx
    ) -> float:
        sum_loss_value = 0.0
        supports_representations = [None] * len(batch)
        indices = [None] * len(batch)
        for i, example in enumerate(batch):
            idx, (X_support, y_support, X_query, y_query) = example
            full_trace = self.model(X_support, y_support, X_query, return_full_trace=True)
            prediction = full_trace["prediction"]

            supports_representations[i] = full_trace["support_set_representation"]
            indices[i] = idx
            loss = torch.nn.CrossEntropyLoss()(prediction, y_query)
            if torch.isnan(loss):
                sum_loss_value = sum_loss_value * (i + 1) / i if i > 0 else 0
            else:
                sum_loss_value += loss

        if self.representation_penalty_weight != 0:
            rep_loss = (
                self.representation_penalty_weight
                * self._calculate_representation_penalty(supports_representations, indices)
                / 4
            )
            sum_loss_value += rep_loss

        return sum_loss_value

    def test_step(self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor]], batch_idx) -> float:
        sum_loss_value = 0.0
        supports_representations = [None] * len(batch)
        indices = [None] * len(batch)
        for i, example in enumerate(batch):
            idx, (X_support, y_support, X_query, y_query) = example
            full_trace = self.model(X_support, y_support, X_query, return_full_trace=True)
            prediction = full_trace["prediction"]

            supports_representations[i] = full_trace["support_set_representation"]
            indices[i] = idx
            loss = torch.nn.CrossEntropyLoss()(prediction, y_query)
            if torch.isnan(loss):
                sum_loss_value = sum_loss_value * (i + 1) / i if i > 0 else 0
            else:
                sum_loss_value += loss

        if self.representation_penalty_weight != 0:
            rep_loss = (
                self.representation_penalty_weight
                * self._calculate_representation_penalty(supports_representations, indices)
                / 4
            )
            sum_loss_value += rep_loss

        return sum_loss_value

    def _calculate_representation_penalty(
        self, supports_representations: List[Tensor], dataset_indices: List[int]
    ):
        support_size = supports_representations[0].shape[0]
        supports_representations_to_penalty = torch.concat(supports_representations, dim=0)
        dist_matrix = torch.cdist(
            supports_representations_to_penalty, supports_representations_to_penalty
        )
        indices_to_mask = (
            torch.Tensor(dataset_indices).reshape(-1, 1).repeat((1, support_size)).reshape(-1, 1)
        )
        mask = (-1) ** (torch.cdist(indices_to_mask, indices_to_mask) == 0)
        return (dist_matrix * mask).sum()

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


class LightningEncoderWrapper(pl.LightningModule):
    def __init__(
        self,
        model: HeterogenousAttributesNetwork,
        learning_rate: float,
        weight_decay: float,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.metrics_history = dict()

        self.save_hyperparameters()

    def training_step(self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor]], batch_idx) -> float:
        supports_representations = [None] * len(batch)
        indices = [None] * len(batch)
        for i, example in enumerate(batch):
            idx, (X_support, y_support, X_query, _) = example
            supports_representations[i] = self.model.encode_support_set(X_support, y_support)
            indices[i] = idx
        return self._calculate_representation_penalty(supports_representations, indices)

    def validation_step(
        self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor]], batch_idx
    ) -> float:
        supports_representations = [None] * len(batch)
        indices = [None] * len(batch)
        for i, example in enumerate(batch):
            idx, (X_support, y_support, X_query, _) = example
            supports_representations[i] = self.model.encode_support_set(X_support, y_support)
            indices[i] = idx
        return self._calculate_representation_penalty(supports_representations, indices)

    def test_step(self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor]], batch_idx) -> float:
        supports_representations = [None] * len(batch)
        indices = [None] * len(batch)
        for i, example in enumerate(batch):
            idx, (X_support, y_support, X_query, _) = example
            supports_representations[i] = self.model.encode_support_set(X_support, y_support)
            indices[i] = idx
        return self._calculate_representation_penalty(supports_representations, indices)

    def _calculate_representation_penalty(
        self, supports_representations: List[Tensor], dataset_indices: List[int]
    ):
        support_size = supports_representations[0].shape[0]
        supports_representations_to_penalty = torch.concat(supports_representations, dim=0)
        data_length = supports_representations_to_penalty.shape[0]
        dist_matrix = torch.cdist(
            supports_representations_to_penalty, supports_representations_to_penalty
        )
        indices_to_mask = (
            torch.Tensor(dataset_indices).reshape(-1, 1).repeat((1, support_size)).reshape(-1, 1)
        )
        mask = (-1) ** (torch.cdist(indices_to_mask, indices_to_mask) == 0)
        return (dist_matrix * mask).sum() / (data_length * (data_length - 1))

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

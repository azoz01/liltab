import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

from typing import Any, Optional, List
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pathlib import Path

from liltab.model.heterogenous_attributes_network import HeterogenousAttributesNetwork
from liltab.data.dataloaders import (
    ComposedDataLoader,
    RepeatableOutputComposedDataLoader,
)
from .utils import LightningWrapper
from .logger import TensorBoardLogger, FileLogger, CustomLogger
from datetime import datetime


class HeterogenousAttributesNetworkTrainer:
    """
    Class used for traning HeterogenousAttributesNetwork.
    """

    def __init__(
        self,
        n_epochs: int,
        gradient_clipping: bool,
        learning_rate: float,
        weight_decay: float,
        loggers: Optional[List[CustomLogger]] = None,
    ):
        """
        Args:
            n_epochs (int): number of epochs to train
            gradient_clipping (bool): If true, then gradient clipping is applied
            learning_rate (float): learning rate during training.
            weight_decay (float): weight decay during training.
        """
        callbacks = HeterogenousAttributesNetworkCallback(loggers=loggers)
        model_path = Path("results")
        model_path = (
            model_path / "models" / datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        )
        model_checkpoints = ModelCheckpoint(
            dirpath=model_path,
            filename="model-{epoch}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
            every_n_epochs=10,
            save_last=True,
        )

        self.trainer = pl.Trainer(
            max_epochs=n_epochs,
            gradient_clip_val=1 if gradient_clipping else 0,
            check_val_every_n_epoch=n_epochs // 1000 if n_epochs > 1000 else 1,
            callbacks=[callbacks, model_checkpoints],
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def train_and_test(
        self,
        model: HeterogenousAttributesNetwork,
        train_loader: ComposedDataLoader | RepeatableOutputComposedDataLoader,
        val_loader: ComposedDataLoader | RepeatableOutputComposedDataLoader,
        test_loader: ComposedDataLoader | RepeatableOutputComposedDataLoader,
    ) -> tuple[LightningWrapper, list[dict[str, float]]]:
        """
        Method used to train and test model.

        Args:
            model (HeterogenousAttributesNetwork): model to train
            train_loader (ComposedDataLoader | RepeatableOutputComposedDataLoader):
                loader withTrainingData
            val_loader (ComposedDataLoader | RepeatableOutputComposedDataLoader):
                loader with validation data
            test_loader (ComposedDataLoader | RepeatableOutputComposedDataLoader):
                loader with test data

        Returns:
            tuple[HeterogenousAttributesNetwork, list[dict[str, float]]]:
                trained network with metrics on test set.
        """
        model_wrapper = LightningWrapper(
            model, learning_rate=self.learning_rate, weight_decay=self.weight_decay
        )
        self.trainer.fit(model_wrapper, train_loader, val_loader)
        test_results = self.trainer.test(model_wrapper, test_loader)
        return model_wrapper, test_results


class HeterogenousAttributesNetworkCallback(Callback):
    def __init__(self, loggers: CustomLogger) -> None:
        super().__init__()
        self.loggers = loggers

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss_value = outputs["loss"]
        pl_module.log("train_loss", loss_value, prog_bar=True, logger=False)
        for logger in self.loggers:
            logger.log_train_value(loss_value)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        loss_value = outputs
        pl_module.log("val_loss", loss_value, prog_bar=True, logger=False)
        for logger in self.loggers:
            logger.log_validate_value(loss_value)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        loss_value = outputs
        pl_module.log("test_loss", loss_value, prog_bar=True, logger=False)
        for logger in self.loggers:
            logger.log_test_value(loss_value)

    def on_fit_end(self, trainer, pl_module):
        for logger in self.loggers:
            logger.log_model_graph(pl_module.model, pl_module.example_input)

import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pathlib import Path
from torch import nn
from datetime import datetime
from typing import Optional, Callable, Union

from liltab.model.heterogenous_attributes_network import HeterogenousAttributesNetwork
from liltab.data.dataloaders import (
    ComposedDataLoader,
    RepeatableOutputComposedDataLoader,
)
from .utils import LightningWrapper
from .logger import TensorBoardLogger, FileLogger


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
        early_stopping: bool = False,
        loss: Callable = nn.MSELoss(),
        file_logger: bool = True,
        tb_logger: bool = True,
        model_checkpoints: bool = True,
        results_path: Union[str, Path, None] = None,
    ):
        """
        Args:
            n_epochs (int): number of epochs to train
            gradient_clipping (bool): If true, then gradient clipping is applied
            learning_rate (float): learning rate during training.
            weight_decay (float): weight decay during training.
            early_stopping (Optional, bool): if True, then early stopping with
                patience n_epochs // 10 is applied. Defaults to False.
            loss (Callable): Loss used during training. Defaults to MSELoss().
            file_logger (bool): if True, then file logger will write to
                {results_path} directory
            tb_logger (bool): if True, then tensorboard logger will write to
                {results_path} directory
            model_checkpoints (bool): if True, then model checkpoints will
                be loaded to {results_path} directory
            results_path (Optional, str, Path): directory to save logs and
                model checkpoints; required only if any of `file_logger`,
                `tb_logger`, `model_checkpoints` is not None
        """

        if not results_path and (file_logger or tb_logger or model_checkpoints):
            raise ValueError(
                """`results_path` is required if any of (`file_logger`,
                `tb_logger`, `model_checkpoints`) is not None"""
            )

        callbacks = []

        timestamp = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")

        if file_logger:
            file_logger_callback = FileLogger(
                save_dir=Path(results_path) / timestamp, version="flat"
            )
        else:
            file_logger_callback = None

        if tb_logger:
            tb_logger_callback = TensorBoardLogger(
                save_dir=Path(results_path) / timestamp, version="tensorboard"
            )
        else:
            tb_logger_callback = None

        if file_logger or tb_logger:
            loggers_callback = LoggerCallback(
                file_logger=file_logger_callback, tb_logger=tb_logger_callback
            )
            callbacks.append(loggers_callback)

        if early_stopping:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=100,
                min_delta=1e-3,
            )
            callbacks.append(early_stopping)

        if model_checkpoints:
            model_checkpoints_callback = ModelCheckpoint(
                dirpath=Path(results_path) / timestamp / "model_checkpoints",
                filename="model-{epoch}-{val_loss:.2f}",
                save_top_k=1,
                mode="min",
                every_n_epochs=10,
                save_last=True,
            )
            callbacks.append(model_checkpoints_callback)

        check_val_every_n_epoch = n_epochs // 1000 if n_epochs > 1000 else 1

        self.trainer = pl.Trainer(
            max_epochs=n_epochs,
            gradient_clip_val=1 if gradient_clipping else 0,
            check_val_every_n_epoch=check_val_every_n_epoch,
            callbacks=callbacks,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss = loss

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
            model,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            loss=self.loss,
        )
        self.trainer.fit(model_wrapper, train_loader, val_loader)
        test_results = self.trainer.test(model_wrapper, test_loader)
        return model_wrapper, test_results


class LoggerCallback(Callback):
    def __init__(
        self,
        file_logger: Optional[FileLogger] = None,
        tb_logger: Optional[TensorBoardLogger] = None,
    ) -> None:
        """
        Args:
            file_logger (FileLogger|None): csv logger
            tb_logger (TensorBoardLogger|None): tensorboard logger
        """
        super().__init__()
        self.file_logger = file_logger
        self.tb_logger = tb_logger

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss_value = outputs["loss"]
        pl_module.log("train_loss", loss_value, prog_bar=True, logger=False)

        if self.file_logger is not None:
            self.file_logger.log_train_value(loss_value)

        if self.tb_logger is not None:
            self.tb_logger.log_train_value(loss_value)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        loss_value = outputs
        pl_module.log("val_loss", loss_value, prog_bar=True, logger=False)

        if self.file_logger is not None:
            self.file_logger.log_validate_value(loss_value)

        if self.tb_logger is not None:
            self.tb_logger.log_validate_value(loss_value)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        loss_value = outputs
        pl_module.log("test_loss", loss_value, prog_bar=True, logger=False)

        if self.file_logger is not None:
            self.file_logger.log_test_value(loss_value)

        if self.tb_logger is not None:
            self.tb_logger.log_test_value(loss_value)

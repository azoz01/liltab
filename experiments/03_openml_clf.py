import yaml
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime

from liltab.data.datasets import PandasDataset
from liltab.data.dataloaders import (
    FewShotDataLoader,
    ComposedDataLoader,
    RepeatableOutputComposedDataLoader,
)
from liltab.data.factory import ComposedDataLoaderFactory
from liltab.model.heterogenous_attributes_network import HeterogenousAttributesNetwork
from liltab.train.trainer import HeterogenousAttributesNetworkTrainer
from liltab.train.logger import TensorBoardLogger, FileLogger
from loguru import logger
from pathlib import Path
from torch import nn


def main():
    config_path = Path("config/03_openml_clf_data_experiment_config.yaml")
    logger_type = "both"
    use_profiler = "no"

    pl.seed_everything(123)

    logger.info("Loading config")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    logger.info("Loading data")
    train_loader = ComposedDataLoaderFactory.create_composed_dataloader_from_path(
        Path(config["train_data_path"]),
        PandasDataset,
        {"encode_categorical_target": True},
        FewShotDataLoader,
        {"support_size": config["support_size"], "query_size": config["query_size"]},
        ComposedDataLoader,
        batch_size=config["batch_size"],
    )
    val_loader = ComposedDataLoaderFactory.create_composed_dataloader_from_path(
        Path(config["val_data_path"]),
        PandasDataset,
        {"encode_categorical_target": True},
        FewShotDataLoader,
        {"support_size": config["support_size"], "query_size": config["query_size"]},
        RepeatableOutputComposedDataLoader,
        batch_size=config["batch_size"],
    )
    test_loader = ComposedDataLoaderFactory.create_composed_dataloader_from_path(
        Path(config["test_data_path"]),
        PandasDataset,
        {"encode_categorical_target": True},
        FewShotDataLoader,
        {"support_size": config["support_size"], "query_size": config["query_size"]},
        RepeatableOutputComposedDataLoader,
        batch_size=config["batch_size"],
    )

    logger.info("Creating model")
    model = HeterogenousAttributesNetwork(
        hidden_representation_size=config["hidden_representation_size"],
        n_hidden_layers=config["n_hidden_layers"],
        hidden_size=config["hidden_size"],
        dropout_rate=config["dropout_rate"],
        is_classifier=config["is_classifier"],
    )

    results_path = Path("results") / config["name"]

    trainer = HeterogenousAttributesNetworkTrainer(
        n_epochs=config["num_epochs"],
        gradient_clipping=config["gradient_clipping"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        early_stopping=config["early_stopping"],
        loss=nn.CrossEntropyLoss(),
        file_logger=True,
        tb_logger=True,
        model_checkpoints=True,
        results_path=results_path,
    )

    logger.info("Training model")
    trainer.train_and_test(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


if __name__ == "__main__":
    main()

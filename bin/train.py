import datetime
import typer
import yaml
import pytorch_lightning as pl
import warnings

from datetime import datetime
from liltab.data.datasets import PandasDataset, RandomFeaturesPandasDataset
from liltab.data.dataloaders import (
    FewShotDataLoader,
    ComposedDataLoader,
    RepeatableOutputComposedDataLoader,
)
from liltab.data.factory import ComposedDataLoaderFactory
from liltab.model.heterogenous_attributes_network import HeterogenousAttributesNetwork
from liltab.train.trainer import HeterogenousAttributesNetworkTrainer
from loguru import logger
from typing_extensions import Annotated
from pathlib import Path
from utils import save_json, save_pickle, generate_plots

warnings.filterwarnings("ignore")
app = typer.Typer()


@app.command(help="Trains network on heterogenous attribute spaces.")
def main(
    config_path: Annotated[Path, typer.Option(..., help="Path to experiment configuration.")],
    seed: Annotated[int, typer.Option(..., help="Seed")] = 123,
):
    pl.seed_everything(seed)

    logger.info("Loading config")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    logger.info("Loading data")
    train_loader = ComposedDataLoaderFactory.create_composed_dataloader_from_path(
        Path(config["train_data_path"]),
        RandomFeaturesPandasDataset,
        {},
        FewShotDataLoader,
        {"support_size": config["support_size"], "query_size": config["query_size"]},
        ComposedDataLoader,
        batch_size=config["batch_size"],
    )
    val_loader = ComposedDataLoaderFactory.create_composed_dataloader_from_path(
        Path(config["val_data_path"]),
        PandasDataset,
        {},
        FewShotDataLoader,
        {"support_size": config["support_size"], "query_size": config["query_size"]},
        RepeatableOutputComposedDataLoader,
        batch_size=config["batch_size"],
    )
    test_loader = ComposedDataLoaderFactory.create_composed_dataloader_from_path(
        Path(config["test_data_path"]),
        PandasDataset,
        {},
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
    )
    trainer = HeterogenousAttributesNetworkTrainer(
        n_epochs=config["num_epochs"],
        gradient_clipping=config["gradient_clipping"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    logger.info("Training model")
    wrapper, test_results = trainer.train_and_test(
        model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader
    )

    logger.info("Saving results")
    results_path = Path("results") / (config["name"] + "_" +datetime.now().isoformat())
    results_path.mkdir(parents=True)
    save_pickle(results_path / "trainer.pkl", trainer)
    save_pickle(results_path / "wrapper.pkl", wrapper)
    save_json(results_path / "metrics_history.json", wrapper.metrics_history)
    generate_plots(results_path / "plots", wrapper.metrics_history)

if __name__ == "__main__":
    app()

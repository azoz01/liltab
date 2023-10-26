from pathlib import Path
from datetime import datetime

from typing import Any
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.loggers import TensorBoardLogger as TBLogger


class TensorBoardLogger(TBLogger):
    def __init__(
        self,
        save_dir: _PATH = "results/tensorboard",
        version: str = None,
        name: str = "",
        **kwargs: Any,
    ):
        if version is None:
            _version = "experiment_" + name + " " + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        else:
            _version = version

        self.iter = 0

        super().__init__(
            save_dir,
            name=None,
            version=_version,
            log_graph=False,
            default_hp_metric=False,
            prefix="",
            sub_dir=None,
            **kwargs,
        )

    def log_train_value(self, value: float) -> None:
        self.iter += 1
        self.experiment.add_scalar("train loss", value, self.iter)

    def log_test_value(self, value: float) -> None:
        self.experiment.add_text("test loss", str(value))

    def log_validate_value(self, value: float) -> None:
        self.experiment.add_scalar("validate loss", value, self.iter)

    def log_weights(self, weights) -> None:
        self.experiment.add_histogram("weights and biases", weights)


class FileLogger:
    def __init__(self, save_dir: str = "results/flat", name: str = "", version: str = None) -> None:
        super().__init__()
        self.save_dir = save_dir

        if version is None:
            self.version = "experiment " + name + " " + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        else:
            self.version = version

        experiment_path = Path(save_dir)
        experiment_path = experiment_path / self.version
        experiment_path.mkdir(exist_ok=False, parents=True)

        self.train_loss_path = experiment_path / "train.csv"
        self.train_loss_path.touch(exist_ok=False)

        self.validate_loss_path = experiment_path / "validate.csv"
        self.validate_loss_path.touch(exist_ok=False)

        self.test_loss_path = experiment_path / "test.csv"
        self.test_loss_path.touch(exist_ok=False)

    def log_train_value(self, value: float) -> None:
        with open(self.train_loss_path, "a") as f:
            f.write(str(value.item()))
            f.write("\n")

    def log_test_value(self, value: float) -> None:
        with open(self.test_loss_path, "a") as f:
            f.write(str(value.item()))
            f.write("\n")

    def log_validate_value(self, value: float) -> None:
        with open(self.validate_loss_path, "a") as f:
            f.write(str(value.item()))
            f.write("\n")

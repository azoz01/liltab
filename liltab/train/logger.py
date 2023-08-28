from argparse import Namespace
from typing import Any, Dict, Optional, Union, List
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.loggers import TensorBoardLogger as TBLogger, Logger

from liltab.model.heterogenous_attributes_network import HeterogenousAttributesNetwork
from ..model.heterogenous_attributes_network import HeterogenousAttributesNetwork
from abc import abstractmethod, ABC
from torch import Tensor
from pathlib import Path
from datetime import datetime


class CustomLogger(ABC):
    @abstractmethod
    def log_train_value(self, value: float) -> None:
        pass

    @abstractmethod
    def log_test_value(self, value: float) -> None:
        pass

    @abstractmethod
    def log_validate_value(self, value: float) -> None:
        pass

    @abstractmethod
    def log_model_graph(
        self,
        model: HeterogenousAttributesNetwork,
        model_input: List[Tensor],
    ) -> None:
        pass

    @abstractmethod
    def log_weights(self, weights) -> None:
        pass

    @abstractmethod
    def log_hparams(self, hparams) -> None:
        pass

    # def add_hparams(
    # self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None
    # ):


class TensorBoardLogger(TBLogger, CustomLogger):
    def __init__(
        self,
        save_dir: _PATH = "results/tensorboard",
        version: str = None,
        **kwargs: Any
    ):
        if version is None:
            _version = "experiment " + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        else:
            _version = version
        super().__init__(
            save_dir,
            name=None,
            version=_version,
            log_graph=False,
            default_hp_metric=False,
            prefix="",
            sub_dir=None,
            **kwargs
        )

    def log_train_value(self, value: float) -> None:
        self.experiment.add_scalar("train loss", value)

    def log_test_value(self, value: float) -> None:
        self.experiment.add_scalar("test loss", value)

    def log_validate_value(self, value: float) -> None:
        self.experiment.add_text("validate loss", str(value))

    def log_model_graph(
        self, model: HeterogenousAttributesNetwork, model_input: List[Tensor]
    ) -> None:
        self.experiment.add_graph(model, model_input)

    def log_weights(self, weights) -> None:
        self.experiment.add_histogram("weights and biases", weights)


class FileLogger(CustomLogger):
    def __init__(self, save_dir: str = "results/flat", version: str = None) -> None:
        super().__init__()
        self.save_dir = save_dir

        if version is None:
            self.version = "experiment " + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        else:
            self.version = version

        experiment_path = Path(save_dir)
        experiment_path = experiment_path / self.version
        experiment_path.mkdir(exist_ok=True)

        self.train_loss_path = experiment_path / "train.csv"
        self.train_loss_path.touch(exist_ok=True)

        self.validate_loss_path = experiment_path / "validate.csv"
        self.validate_loss_path.touch(exist_ok=True)

        self.test_loss_path = experiment_path / "test.csv"
        self.test_loss_path.touch(exist_ok=True)

        self.hyperparam_path = experiment_path / "hyperparams.yaml"
        self.hyperparam_path.touch(exist_ok=True)

        self.hyperparams = {}

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

    def log_model_graph(
        self, model: HeterogenousAttributesNetwork, model_input: List[Tensor]
    ) -> None:
        return None

    def log_weights(self, weights) -> None:
        return None

import torch

from torch import Tensor
from pathlib import Path
from datetime import datetime
from torch.profiler import profile

from typing import Any, List
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.loggers import TensorBoardLogger as TBLogger

from liltab.model.heterogenous_attributes_network import HeterogenousAttributesNetwork
from ..model.heterogenous_attributes_network import HeterogenousAttributesNetwork


class TensorBoardLogger(TBLogger):
    def __init__(
        self,
        save_dir: _PATH = "results/tensorboard",
        version: str = None,
        use_profiler: bool = False,
        **kwargs: Any
    ):
        if version is None:
            _version = "experiment " + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        else:
            _version = version

        self.use_profiler = use_profiler

        if self.use_profiler:
            self.profiler = profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    Path("./results") / "tensorboard" / _version
                ),
                record_shapes=True,
                with_stack=True,
            )

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

    def profile_start(self):
        if self.use_profiler:
            self.profiler.start()

    def profile_end(self):
        if self.use_profiler:
            self.profiler.stop()

    def profile_step(self):
        if self.use_profiler:
            self.profiler.step()


class FileLogger:
    def __init__(self, save_dir: str = "results/flat", version: str = None) -> None:
        super().__init__()
        self.save_dir = save_dir

        if version is None:
            self.version = "experiment " + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
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

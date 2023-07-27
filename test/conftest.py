from torch import Tensor
from pathlib import Path
from pytest import fixture


@fixture(scope="session")
def resources_path():
    return Path("test/resources")


class Utils:
    @staticmethod
    def tensors_have_common_rows(t1: Tensor, t2: Tensor) -> bool:
        for row_t1 in t1:
            for row_t2 in t2:
                if ((row_t1 - row_t2).abs() < 1e-9).all():
                    return True
        return False

    @staticmethod
    def run_training_iteration():
        pass


@fixture(scope="session")
def utils():
    return Utils

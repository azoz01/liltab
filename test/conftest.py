from liltab.model.heterogenous_attributes_network import HeterogenousAttributesNetwork
from torch import nn, Tensor
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
    def get_inference_adapter(network, X_support, y_support):
        return InferenceAdapter(network, X_support, y_support)


@fixture(scope="session")
def utils():
    return Utils


class InferenceAdapter(nn.Module):
    def __init__(
        self, network: HeterogenousAttributesNetwork, X_support: Tensor, y_support: Tensor
    ):
        super().__init__()
        self.network = network
        self.X_support = X_support
        self.y_support = y_support

    def forward(self, X_query: Tensor):
        return self.network(self.X_support, self.y_support, X_query)

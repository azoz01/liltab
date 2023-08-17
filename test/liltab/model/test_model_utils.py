import numpy as np
import torch.nn.functional as F

from torch import nn, optim, Tensor
from torchtest import assert_vars_change
from liltab.model.utils import NetworkBlock, FeedForwardNetwork


def test_network_block_initializes_properly():
    block = NetworkBlock(
        input_size=12, output_size=30, activation_function=nn.ReLU(), dropput_rate=0.45
    )

    assert len(block.block) == 3
    assert block.block[1].p == 0.45


def test_network_block_accepts_and_returns_propoer_shape():
    block = NetworkBlock(
        input_size=12, output_size=30, activation_function=nn.ReLU(), dropput_rate=0.45
    )
    input = Tensor(np.random.uniform(size=(3, 12)))
    output = block(input)

    assert output.shape == (3, 30)


def test_network_block_is_trained():
    block = NetworkBlock(
        input_size=12, output_size=30, activation_function=nn.ReLU(), dropput_rate=0.45
    )
    batch = (Tensor(np.random.uniform(size=(3, 12))), Tensor(np.random.uniform(size=(3, 30))))

    assert_vars_change(
        model=block,
        loss_fn=F.cross_entropy,
        optim=optim.Adam(block.parameters()),
        batch=batch,
        device="cpu:0",
    )


def test_feed_forward_network_initializes_properly():
    network = FeedForwardNetwork(
        input_size=12,
        output_size=30,
        n_hidden_layers=5,
        hidden_size=20,
        dropout_rate=0.5,
        inner_activation_function=nn.ReLU(),
        output_activation_function=nn.Sigmoid(),
    )

    assert len(network.hidden_layers) == 5

    assert network.input_layer.block[0].in_features == 12
    assert network.input_layer.block[0].out_features == 20
    for block in network.hidden_layers:
        assert block.block[0].in_features == 20
        assert block.block[0].out_features == 20
    assert network.output_layer.block[0].in_features == 20
    assert network.output_layer.block[0].out_features == 30


def test_feed_forward_network_accepts_and_returns_propoer_shape():
    network = FeedForwardNetwork(
        input_size=12,
        output_size=30,
        n_hidden_layers=5,
        hidden_size=20,
        dropout_rate=0.5,
        inner_activation_function=nn.ReLU(),
        output_activation_function=nn.Sigmoid(),
    )
    input = Tensor(np.random.uniform(size=(10, 12)))
    output = network(input)
    assert output.shape == (10, 30)


def test_feed_forward_network_is_trained():
    network = FeedForwardNetwork(
        input_size=12,
        output_size=30,
        n_hidden_layers=5,
        hidden_size=20,
        dropout_rate=0.5,
        inner_activation_function=nn.ReLU(),
        output_activation_function=nn.Sigmoid(),
    )
    batch = (Tensor(np.random.uniform(size=(10, 12))), Tensor(np.random.uniform(size=(10, 30))))

    assert_vars_change(
        model=network,
        loss_fn=F.cross_entropy,
        optim=optim.Adam(network.parameters()),
        batch=batch,
        device="cpu:0",
    )

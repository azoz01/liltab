from __future__ import annotations

from torch import nn, Tensor
from typing import Callable


class NetworkBlock(nn.Module):
    """
    Module representing single neural network block.
    It's composed with Linear + Activation function
    and dropout.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation_function: Callable,
        dropput_rate: float,
    ):
        """
        Args:
            input_size (int): Dimensionality of input.
            output_size (int): Dimensionality of output.
            activation_function (Callable): Activation function
                from torch.nn.
            dropput_rate (float): Dropout rate inside block.
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Dropout(dropput_rate),
            activation_function,
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.block(X)


class FeedForwardNetwork(nn.Module):
    """
    Module representing simple feed-forward neural network
    with only fully-connected layers.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_hidden_layers: int,
        hidden_size: int,
        dropout_rate: float,
        inner_activation_function: Callable = nn.ReLU(),
        output_activation_function: Callable = nn.Identity(),
    ):
        """
        Args:
            input_size (int): Dimensionality of input.
            output_size (int): Dimensionality of output.
            n_hidden_layers (int): Number of hidden layers.
            hidden_size (int): Size of hidden layer.
                Each layer has equal size.
            dropout_rate (float): Dropout rate for each hidden layer.
                Each layer has equal dropout rate.
            inner_activation_function (Callable, optional): Activation function
                from torch.nn which will be used after each hidden layer.
                Defaults to nn.ReLU().
            output_activation_function (Callable, optional):
            Activation function from torch.nn which will
            be used after output layer. Defaults to nn.Identity().
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden_layers = n_hidden_layers
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.inner_activation_function = inner_activation_function
        self.output_activation_function = output_activation_function

        self.input_layer = NetworkBlock(
            self.input_size,
            self.hidden_size,
            self.inner_activation_function,
            self.dropout_rate,
        )
        self.hidden_layers = nn.ModuleList(
            [
                NetworkBlock(
                    self.hidden_size,
                    self.hidden_size,
                    self.inner_activation_function,
                    self.dropout_rate,
                )
                for _ in range(self.n_hidden_layers)
            ]
        )
        self.output_layer = NetworkBlock(
            self.hidden_size,
            self.output_size,
            self.output_activation_function,
            self.dropout_rate,
        )

    def forward(self, X: Tensor) -> Tensor:
        X = self.input_layer(X)
        for layer in self.hidden_layers:
            X = layer(X)
        X = self.output_layer(X)
        return X

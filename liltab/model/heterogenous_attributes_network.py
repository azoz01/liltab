import torch

from torch import nn, Tensor
from typing import Callable
from .utils import FeedForwardNetwork


class HeterogenousAttributesNetwork(nn.Module):
    def __init__(
        self,
        hidden_representation_size: int = 32,
        n_hidden_layers: int = 3,
        hidden_size: int = 32,
        dropout_rate: int = 0.1,
        inner_activation_function: Callable = nn.ReLU(),
        output_activation_function: Callable = nn.Identity(),
    ):
        super().__init__()
        self.initial_support_encoding_network = FeedForwardNetwork(
            1,
            hidden_representation_size,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            output_activation_function,
        )
        self.initial_support_representation_network = FeedForwardNetwork(
            hidden_representation_size,
            hidden_representation_size,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            output_activation_function,
        )

        self.interaction_encoding_network = FeedForwardNetwork(
            hidden_representation_size + 1,
            hidden_representation_size,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            output_activation_function,
        )
        self.interaction_representation_network = FeedForwardNetwork(
            hidden_representation_size,
            hidden_representation_size,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            output_activation_function,
        )
        self.attributes_encoding_network = FeedForwardNetwork(
            hidden_representation_size + 1,
            hidden_representation_size,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            output_activation_function,
        )
        self.attributes_representation_network = FeedForwardNetwork(
            hidden_representation_size,
            hidden_representation_size,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            output_activation_function,
        )
        self.responses_encoding_network = FeedForwardNetwork(
            hidden_representation_size + 1,
            hidden_representation_size,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            output_activation_function,
        )
        self.responses_representation_network = FeedForwardNetwork(
            hidden_representation_size,
            hidden_representation_size,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            output_activation_function,
        )
        self.inference_encoding_network = FeedForwardNetwork(
            hidden_representation_size + 1,
            hidden_representation_size,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            output_activation_function,
        )
        self.inference_network = FeedForwardNetwork(
            2*hidden_representation_size,
            1,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            output_activation_function,
        )

    def forward(
        self, X_support: Tensor, y_support: Tensor, X_query: Tensor
    ) -> Tensor:
        attributes_initial_representation = (
            self._create_initial_features_representation(
                self.initial_support_encoding_network,
                self.initial_support_representation_network,
                X_support,
            )
        )
        responses_initial_representation = (
            self._create_initial_features_representation(
                self.initial_support_encoding_network,
                self.initial_support_representation_network,
                y_support,
            )
        )
        support_set_representation = self._create_support_set_representation(
            self.interaction_encoding_network,
            self.interaction_representation_network,
            X_support,
            attributes_initial_representation,
            y_support,
            responses_initial_representation,
        )
        attributes_representation = self._create_features_representation(
            self.attributes_encoding_network,
            self.attributes_representation_network,
            support_set_representation,
            X_support,
        )
        responses_representation = self._create_features_representation(
            self.responses_encoding_network,
            self.responses_representation_network,
            support_set_representation,
            y_support,
        )
        prediction = self._make_prediction(
            self.inference_encoding_network,
            self.inference_network,
            attributes_representation,
            responses_representation,
            X_query,
        )

        return prediction

    def _create_initial_features_representation(
        self,
        encoder_network: FeedForwardNetwork,
        representation_network: FeedForwardNetwork,
        X: Tensor,
    ) -> Tensor:
        initial_tensor_shape = X.shape
        representation_length = encoder_network.output_size
        network_input = X.reshape(-1, 1)
        encoded_input = encoder_network(network_input)
        encoded_input = encoded_input.reshape(
            *initial_tensor_shape, representation_length
        )
        encoded_input = encoded_input.mean(axis=0)
        features_representation = representation_network(encoded_input)
        return features_representation

    def _create_support_set_representation(
        self,
        interaction_encoding_network: FeedForwardNetwork,
        interaction_representation_network: FeedForwardNetwork,
        X: Tensor,
        attributes_initial_representation: Tensor,
        y: Tensor,
        responses_initial_representation: Tensor,
    ) -> Tensor:
        attributes_encoded = self._create_interaction_encoding(
            interaction_encoding_network, attributes_initial_representation, X
        )
        responses_encoded = self._create_interaction_encoding(
            interaction_encoding_network, responses_initial_representation, y
        )

        return interaction_representation_network(
            attributes_encoded + responses_encoded
        )

    def _create_interaction_encoding(
        self,
        interaction_encoding_network: FeedForwardNetwork,
        representation: Tensor,
        set_: Tensor,
    ) -> Tensor:
        encoding_input = self._enrich_representation_with_set_rows(
            representation, set_
        )
        representation_with_set = encoding_input.reshape(
            -1, representation.shape[1] + 1
        )
        encoding = interaction_encoding_network(representation_with_set)
        encoding = encoding.reshape(
            *set_.shape, interaction_encoding_network.output_size
        ).mean(axis=1)
        return encoding

    def _enrich_representation_with_set_rows(
        self, representation: Tensor, set_: Tensor
    ) -> Tensor:
        n_rows, n_features = set_.shape
        repeated_representation = representation.repeat([n_rows, 1]).reshape(
            n_rows, n_features, -1
        )
        representation_with_set = torch.concat(
            [repeated_representation, torch.unsqueeze(set_, 2)], 2
        )
        return representation_with_set

    def _create_features_representation(
        self,
        features_encoding_network: FeedForwardNetwork,
        features_representation_network: FeedForwardNetwork,
        set_representation: Tensor,
        set_: Tensor,
    ) -> Tensor:
        n_features = set_.shape[1]
        attributes_encoding_input = (
            self._enrich_representation_with_set_features(
                set_representation, set_
            )
        )
        features_encoded = features_encoding_network(
            attributes_encoding_input.reshape(
                -1, set_representation.shape[1] + 1
            )
        )
        features_encoded = features_encoded.reshape(
            n_features, -1, set_representation.shape[1]
        ).mean(axis=1)
        features_representation = features_representation_network(
            features_encoded
        )
        return features_representation

    def _enrich_representation_with_set_features(
        self, representation: Tensor, set_: Tensor
    ) -> Tensor:
        n_examples, n_features = set_.shape
        repeated_representation = representation.repeat(n_features, 1).reshape(
            n_features, n_examples, -1
        )
        representation_with_set = torch.concat(
            [
                repeated_representation,
                torch.unsqueeze(set_.T, 2),
            ],
            axis=2,
        )
        return representation_with_set

    def _make_prediction(
        self,
        inference_encoding_network: FeedForwardNetwork,
        inference_network: FeedForwardNetwork,
        attributes_representation: Tensor,
        responses_representation: Tensor,
        X_query: Tensor,
    ) -> Tensor:
        X_query_encoding_input = self._enrich_representation_with_set_rows(
            attributes_representation, X_query
        )
        X_query_encoding_input = X_query_encoding_input.reshape(
            -1, attributes_representation.shape[1] + 1
        )

        X_query_inference_embedding = inference_encoding_network(
            X_query_encoding_input
        )
        X_query_inference_embedding = X_query_inference_embedding.reshape(
            *X_query.shape, -1
        ).mean(axis=1)

        response_dim = responses_representation.shape[0]
        n_query_rows = X_query.shape[0]
        query_embedding_dim = X_query_inference_embedding.shape[1]
        attributes_representation_dim = attributes_representation.shape[1]

        inference_network_input = X_query_inference_embedding.repeat(
            response_dim, 1
        ).reshape(
            response_dim, n_query_rows, inference_encoding_network.output_size
        )
        inference_network_input = torch.concat(
            [
                inference_network_input,
                responses_representation.unsqueeze(1).repeat(
                    1, n_query_rows, 1
                ),
            ],
            axis=2,
        )
        inference_network_input = inference_network_input.reshape(
            -1, query_embedding_dim + attributes_representation_dim
        )
        response = inference_network(inference_network_input)
        response = response.reshape(n_query_rows, -1)
        return response

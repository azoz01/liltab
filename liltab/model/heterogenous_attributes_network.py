import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import Callable
from .utils import FeedForwardNetwork


class HeterogenousAttributesNetwork(nn.Module):
    """
    Module representing neural network wchich takes as an input:
        * support set attributes
        * support set responses
        * query set attributes
    And returns responses corresponding to query set.
    Support and query set must have same attributes.
    Can take any number of attributes.
    """

    def __init__(
        self,
        hidden_representation_size: int = 32,
        n_hidden_layers: int = 3,
        hidden_size: int = 32,
        dropout_rate: int = 0.1,
        inner_activation_function: Callable = nn.ReLU(),
        output_activation_function: Callable = nn.Identity(),
        is_classifier: bool = False,
    ):
        """
        Args:
            hidden_representation_size (int, optional): Size of hidden
                representation sizes i. e. all intermediate network outputs.
                Defaults to 32.
            n_hidden_layers (int, optional): Number hidden layers of networks
                used during inference. Defaults to 3.
            hidden_size (int, optional): Number of neurons per hidden layer
                in networks using during inference. Defaults to 32.
            dropout_rate (int, optional): Dropout rate of networks
                used during inference. Defaults to 0.1.
            inner_activation_function (Callable, optional): Inner activation function
                of networks used during inference. Should be function from torch.nn.
                Defaults to ReLU.
            output_activation_function (Callable, optional): Output activation function
                of networks used during inference. Should be function from torch.nn.
                Defaults to nn.Identity().
            is_classifier (bool, optional): If true then output of the network will
                generate probabilities of classes for query set Defaults to False.
        """
        super().__init__()
        self.initial_features_encoding_network = FeedForwardNetwork(
            1,
            hidden_representation_size,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            inner_activation_function,
        )
        self.initial_features_representation_network = FeedForwardNetwork(
            hidden_representation_size,
            hidden_representation_size,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            inner_activation_function,
        )
        self.interaction_encoding_network = FeedForwardNetwork(
            hidden_representation_size + 1,
            hidden_representation_size,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            inner_activation_function,
        )
        self.interaction_representation_network = FeedForwardNetwork(
            hidden_representation_size,
            hidden_representation_size,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            inner_activation_function,
        )

        self.features_encoding_network = FeedForwardNetwork(
            hidden_representation_size + 1,
            hidden_representation_size,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            inner_activation_function,
        )
        self.features_representation_network = FeedForwardNetwork(
            hidden_representation_size,
            hidden_representation_size,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            inner_activation_function,
        )

        self.inference_encoding_network = FeedForwardNetwork(
            hidden_representation_size + 1,
            hidden_representation_size,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            inner_activation_function,
        )
        self.inference_embedding_network = FeedForwardNetwork(
            hidden_representation_size,
            hidden_representation_size,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            inner_activation_function,
        )
        self.inference_network = FeedForwardNetwork(
            2 * hidden_representation_size,
            1,
            n_hidden_layers,
            hidden_size,
            dropout_rate,
            inner_activation_function,
            output_activation_function,
        )
        self.is_classifier = is_classifier

    def forward(self, X_support: Tensor, y_support: Tensor, X_query: Tensor) -> Tensor:
        """
        Inference function of network. Inference is done in following steps:
            1. Calculate initial representation for all atrributes and responses
                in support set.
            2. Calculate representations for all observations in support set
                i. e. calculate representation of support set using aforementioned
                initial representations of attributes and responses
            3. Calculate final attributes and responses representations in support set
            4. Calculate representation of query set.attributes.
            5. Make prediction based on representations of support set attributes and
                responses and query set attributes representations.
        All representations calculations are done using feed forward neural networks.
        Args:
            X_support (Tensor): Support set attributes
                with shape (n_support_observations, n_attributes)
            y_support (Tensor): Support set responses
                with shape (n_support_observations, n_responses)
            X_query (Tensor): Query set attributes
                with shape (n_query_observations, n_attributes)
        Returns:
            Tensor: Inferred query set responses shaped (n_query_obervations, n_responses)
        """
        attributes_initial_representation = self._calculate_initial_features_representation(
            self.initial_features_encoding_network,
            self.initial_features_representation_network,
            X_support,
        )
        responses_initial_representation = self._calculate_initial_features_representation(
            self.initial_features_encoding_network,
            self.initial_features_representation_network,
            y_support,
        )
        support_set_representation = self._calculate_support_set_representation(
            self.interaction_encoding_network,
            self.interaction_representation_network,
            X_support,
            attributes_initial_representation,
            y_support,
            responses_initial_representation,
        )
        attributes_representation = self._calculate_features_representation(
            self.features_encoding_network,
            self.features_representation_network,
            support_set_representation,
            X_support,
        )
        responses_representation = self._calculate_features_representation(
            self.features_encoding_network,
            self.features_representation_network,
            support_set_representation,
            y_support,
        )

        if self.is_classifier:
            prediction = self._make_prediction_clf(
                self.inference_encoding_network,
                self.inference_embedding_network,
                attributes_representation,
                X_support,
                X_query,
                y_support,
            )
        else:
            prediction = self._make_prediction_reg(
                self.inference_encoding_network,
                self.inference_embedding_network,
                self.inference_network,
                attributes_representation,
                responses_representation,
                X_query,
            )

        return prediction

    def _calculate_initial_features_representation(
        self,
        encoder_network: FeedForwardNetwork,
        representation_network: FeedForwardNetwork,
        set_: Tensor,
    ) -> Tensor:
        """
        Calculates initial features (can be attributes or resposnes) of given set
        using feed forward neural network.
        Args:
            encoder_network (FeedForwardNetwork): Feed forward network used to encode
                features. Network takes as an input single number and returns
                vector with length hidden_representation_size.
            representation_network (FeedForwardNetwork): Feed forward network used to
                calculate initial representation of features. Network takes as an
                input vector with length hidden_representation_size + 1 and returns
                vector with length hidden_representation_size.
            X (Tensor): set itself. Can be only attributes or only responses.
                Tensor with shape (n_observations, n_features).
        Returns:
            Tensor: calculated initial features representation with shape
                (n_features, hidden_representation_size) containing representations
                of attributes in rows.
        """
        initial_tensor_shape = set_.shape
        representation_length = encoder_network.output_size
        network_input = set_.reshape(-1, 1)
        encoded_input = encoder_network(network_input)
        encoded_input = encoded_input.reshape(*initial_tensor_shape, representation_length)
        encoded_input = encoded_input.mean(axis=0)
        features_representation = representation_network(encoded_input)
        return features_representation

    def _calculate_support_set_representation(
        self,
        interaction_encoding_network: FeedForwardNetwork,
        interaction_representation_network: FeedForwardNetwork,
        X: Tensor,
        attributes_initial_representation: Tensor,
        y: Tensor,
        responses_initial_representation: Tensor,
    ) -> Tensor:
        """
        Calculates representation of observations in given support set
        using feed-forwand neural networks.
        Args:
            interaction_encoding_network (FeedForwardNetwork): Feed forward neural network
                used to encode interactions between features representations and observations
                in X. Takes as an input vector with length hidden_representation_size + 1
                and returns vector with length hidden_representation_size.
            interaction_representation_network (FeedForwardNetwork): Feed forward neural network
                used to calculate representation of interactions between features
                representations and observations in X.
                Takes as an input vector with length hidden_representation_size + 1
                and returns vector with length hidden_representation_size.
            X (Tensor): attributes of support set with shape (n_support_observations, n_attributes).
            attributes_initial_representation (Tensor): initial representation of attributes
                of support set with shape (n_attributes, hidden_representation_size).
            y (Tensor): responses of support set with shape (n_support_observations, n_responses).
            responses_initial_representation (Tensor): initial representation of responses
                of support set with shape (n_responses, hidden_representation_size).
        Returns:
            Tensor: calcualted representaion of support selfset
                with shape (n_support_observations, hidden_representation_size).
        """
        attributes_encoded = self._calculate_interaction_encoding(
            interaction_encoding_network, attributes_initial_representation, X
        )
        responses_encoded = self._calculate_interaction_encoding(
            interaction_encoding_network, responses_initial_representation, y
        )

        return interaction_representation_network(attributes_encoded + responses_encoded)

    def _calculate_interaction_encoding(
        self,
        interaction_encoding_network: FeedForwardNetwork,
        representation: Tensor,
        set_: Tensor,
    ) -> Tensor:
        """
        Method used to calculate encoding of interaction encoding
        between initial representation of features in set and
        observations in this set.
        Args:
            interaction_encoding_network (FeedForwardNetwork): Feed forward neural network
                used to encode interactions between features representations and observations
                in X. Takes as an input vector with length hidden_representation_size + 1
                and returns vector with length hidden_representation_size.
            representation (Tensor): initial representation of features
                of support set with shape (n_features, hidden_representation_size).
            set_ (Tensor): set containing observations to calculate interations encoding with.
                Can be only attributes or only features.
                Tensor with shape (n_observations, n_features).
        Returns:
            Tensor: Calculated interations encoding
                with shape (n_observations, hidden_representation_size).
        """
        encoding_input = self._enrich_representation_with_set_rows(representation, set_)
        representation_with_set = encoding_input.reshape(-1, representation.shape[1] + 1)
        encoding = interaction_encoding_network(representation_with_set)
        encoding = encoding.reshape(*set_.shape, interaction_encoding_network.output_size).mean(
            axis=1
        )
        return encoding

    def _calculate_features_representation(
        self,
        features_encoding_network: FeedForwardNetwork,
        features_representation_network: FeedForwardNetwork,
        set_representation: Tensor,
        set_: Tensor,
    ) -> Tensor:
        """
        Method used to calculate representations of features based on
        set representation and observations from set.
        Args:
            features_encoding_network (FeedForwardNetwork): Feed forward neural
                network used for calculation of encoding of concatenation
                of initial representation and features from set_.
                Takes as an input vector with length hidden_representation_size + 1
                and returns vector with length hidden_representation_size.
            features_representation_network (FeedForwardNetwork): Feed forward neural
                network used for calculation of representation of concatenation
                of calculated encoding.
                Takes as an input vector with length hidden_representation_size
                and returns vector with length hidden_representation_size.
            set_representation (Tensor): representation of set with shape
                (n_observations, hidden_representation_size).
            set_ (Tensor): set itself. Can be only attributes or only features.
                Tensor with shape (n_observations, n_features).
        Returns:
            Tensor: Calculated representation of features
                with shape (n_features, hidden_representation_size).
        """
        n_features = set_.shape[1]
        attributes_encoding_input = self._enrich_representation_with_set_features(
            set_representation, set_
        )
        features_encoded = features_encoding_network(
            attributes_encoding_input.reshape(-1, set_representation.shape[1] + 1)
        )
        features_encoded = features_encoded.reshape(
            n_features, -1, set_representation.shape[1]
        ).mean(axis=1)
        features_representation = features_representation_network(features_encoded)
        return features_representation

    def _enrich_representation_with_set_features(
        self, representation: Tensor, set_: Tensor
    ) -> Tensor:
        """
        Calculates tensor which is repeated representation with feature
        from set as a last column for each repetition e.g.
        representation:
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9]
            ]
        set_:
            [
                [1, 2],
                [3, 4],
                [5, 6]
            ]
        result:
            [
                [ # First feature
                    [0.1, 0.2, 0.3, 1],
                    [0.4, 0.5, 0.6, 3],
                    [0.7, 0.8, 0.9, 5]
                ],
                [ # Second feature
                    [0.1, 0.2, 0.3, 2],
                    [0.4, 0.5, 0.6, 4],
                    [0.7, 0.8, 0.9, 6]
                ]
            ]
        Args:
            representation (Tensor): representation of feautres
                with shape (n_features, hidden_representation_size)
            set_ (Tensor): set to enrich representation with
                with shape (n_observations, n_features)
        Returns:
            Tensor: enriched representations with shape
                (n_observations, n_features, hidden_representation_size + 1)
        """
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

    def _make_prediction_reg(
        self,
        inference_encoding_network: FeedForwardNetwork,
        inference_embedding_network: FeedForwardNetwork,
        inference_network: FeedForwardNetwork,
        attributes_representation: Tensor,
        responses_representation: Tensor,
        X_query: Tensor,
    ) -> Tensor:
        """
        Method responsible for calculation of final prediction based on support
        set representation and query set in regression problem.
        Args:
            inference_encoding_network (FeedForwardNetwork): Network responsible
                for encoding concatenation of support set attributes
                representation with query set attributes.
                Takes as an input vector with length hidden_representation_size + 1
                and returns vector with length hidden_representation_size.
            inference_embedding_network (FeedForwardNetwork): Network responsible
                for calculating representation of encoding created by
                inference_encoding_network.
                Takes as an input vector with length hidden_representation_size
                and returns vector with length hidden_representation_size.
            inference_network (FeedForwardNetwork): Network responsible
                for calculation of responses corresponding to query set based
                on representation of attributes and responses from query set
                and representation of query set observations.
                Takes as an input vector with length 2*hidden_representation_size
                and returns vector with length n_responses.
            attributes_representation (Tensor): representation of attributes of support set
                with shape (n_attributes, hidden_representation_size)
            responses_representation (Tensor): representation of responses of support set
                with shape (n_responses, hidden_representation_size)
            X_query (Tensor): query set with shape (n_query_observations, n_attributes)
        Returns:
            Tensor: responses corresponding to X_query
                with shape (n_query_observations, n_responses)
        """
        X_query_inference_embedding = self._get_inference_embedding_of_set(
            inference_encoding_network,
            inference_embedding_network,
            X_query,
            attributes_representation,
        )
        response_dim = responses_representation.shape[0]
        n_query_observations = X_query.shape[0]
        query_embedding_dim = X_query_inference_embedding.shape[1]
        attributes_representation_dim = attributes_representation.shape[1]

        inference_network_input = (
            X_query_inference_embedding.repeat(response_dim, 1)
            .reshape(response_dim, n_query_observations, inference_encoding_network.output_size)
            .transpose(1, 0)
        )

        responses_representation_expanded = (
            responses_representation.unsqueeze(1)
            .repeat(n_query_observations, 1, 1)
            .reshape(n_query_observations, response_dim, -1)
        )
        inference_network_input = torch.concat(
            [
                responses_representation_expanded,
                inference_network_input,
            ],
            axis=2,
        )
        inference_network_input = inference_network_input.reshape(
            -1, query_embedding_dim + attributes_representation_dim
        )
        response = inference_network(inference_network_input)
        response = response.reshape(n_query_observations, -1)
        return response

    def _make_prediction_clf(
        self,
        inference_encoding_network: FeedForwardNetwork,
        inference_embedding_network: FeedForwardNetwork,
        attributes_representation: Tensor,
        X_support: Tensor,
        X_query: Tensor,
        y_support: Tensor,
    ) -> Tensor:
        """
        Method responsible for calculation of final prediction based on support
        set representation and query set in classification problem.
        Args:
            inference_encoding_network (FeedForwardNetwork): Network responsible
                for encoding concatenation of support set attributes
                representation with query set attributes.
                Takes as an input vector with length hidden_representation_size + 1
                and returns vector with length hidden_representation_size.
            inference_embedding_network (FeedForwardNetwork): Network responsible
                for calculating representation of encoding created by
                inference_encoding_network.
                Takes as an input vector with length hidden_representation_size
                and returns vector with length hidden_representation_size.
            inference_network (FeedForwardNetwork): Network responsible
                for calculation of responses corresponding to query set based
                on representation of attributes and responses from query set
                and representation of query set observations.
                Takes as an input vector with length 2*hidden_representation_size
                and returns vector with length n_responses.
            attributes_representation (Tensor): representation of attributes of support set
                with shape (n_attributes, hidden_representation_size)
            responses_representation (Tensor): representation of responses of support set
                with shape (n_responses, hidden_representation_size)
            X_query (Tensor): query set with shape (n_query_observations, n_attributes)
            X_support (Tensor): support set with shape (n_support_observations, n_attributes)
        Returns:
            Tensor: responses corresponding to X_query
                with shape (n_query_observations, n_responses)
        """
        X_support_inference_embedding = self._get_inference_embedding_of_set(
            inference_encoding_network,
            inference_embedding_network,
            X_support,
            attributes_representation,
        )
        X_query_inference_embedding = self._get_inference_embedding_of_set(
            inference_encoding_network,
            inference_embedding_network,
            X_query,
            attributes_representation,
        )

        classes_representations = self._calculate_classes_representations(
            X_support_inference_embedding, y_support
        )
        response = F.softmax(
            -(torch.cdist(X_query_inference_embedding, classes_representations) ** 2), dim=1
        )

        return response

    def _calculate_classes_representations(self, X: Tensor, y: Tensor) -> Tensor:
        """
        Calculates classes representations by averaging their observations.

        Args:
            X (Tensor): Observations.
            y (Tensor): Categorical responses, one-hot encoded.

        Returns:
            Tensor: Calculated representations ordered by corresponding response value.
        """
        response_values = torch.arange(y.shape[1])
        y = y.argmax(axis=1)
        classes_representations = torch.zeros((response_values.shape[0], X.shape[1]))
        for val in response_values:
            if (y == val).sum() != 0:
                classes_representations[val] = X[y == val].mean(axis=0)
        return classes_representations

    def _get_inference_embedding_of_set(
        self,
        inference_encoding_network: FeedForwardNetwork,
        inference_embedding_network: FeedForwardNetwork,
        set_: Tensor,
        attributes_representation: Tensor,
    ) -> Tensor:
        """
        Calculates embedding of set used during prediction generation
        based on attributes representation.

        Args:
            inference_encoding_network (FeedForwardNetwork): Network responsible
                for encoding concatenation of support set attributes
                representation with query set attributes.
                Takes as an input vector with length hidden_representation_size + 1
                and returns vector with length hidden_representation_size.
            inference_embedding_network (FeedForwardNetwork): Network responsible
                for calculating representation of encoding created by
                inference_encoding_network.
                Takes as an input vector with length hidden_representation_size
                and returns vector with length hidden_representation_size.
            set_ (Tensor): Set to generate embedding with shape
                (n_observations, n_attributes)
            attributes_representation (Tensor): representation of attributes of support set
                with shape (n_attributes, hidden_representation_size)

        Returns:
            Tensor: calculated embedding with shape (n_observations, hidden_representation_size)
        """
        X_query_encoding_input = self._enrich_representation_with_set_rows(
            attributes_representation, set_
        )
        X_query_encoding_input = X_query_encoding_input.reshape(
            -1, attributes_representation.shape[1] + 1
        )

        X_query_inference_embedding = inference_encoding_network(X_query_encoding_input)
        X_query_inference_embedding = X_query_inference_embedding.reshape(*set_.shape, -1).mean(
            axis=1
        )
        X_query_inference_embedding = inference_embedding_network(X_query_inference_embedding)
        return X_query_inference_embedding

    def _enrich_representation_with_set_rows(self, representation: Tensor, set_: Tensor) -> Tensor:
        """
        Calculates tensor which is repeated representation with observaion
        from set as a last column for each repetition e.g.
        representation:
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ]
        set_:
            [
                [1, 2],
                [3, 4],
                [5, 6]
            ]
        result:
            [
                [ # First observation
                    [0.1, 0.2, 0.3, 1],
                    [0.4, 0.5, 0.6, 2]
                ],
                [ # Second observation
                    [0.1, 0.2, 0.3, 3],
                    [0.4, 0.5, 0.6, 4]
                ],
                [ # Third observation
                    [0.1, 0.2, 0.3, 5],
                    [0.4, 0.5, 0.6, 6]
                ]
            ]
        Args:
            representation (Tensor): representation of feautres
                with shape (n_features, hidden_representation_size)
            set_ (Tensor): set to enrich representation with
                with shape (n_observations, n_features)
        Returns:
            Tensor: enriched representations with shape
                (n_observations, n_features, hidden_representation_size + 1)
        """
        n_rows, n_features = set_.shape
        repeated_representation = representation.repeat([n_rows, 1]).reshape(n_rows, n_features, -1)
        representation_with_set = torch.concat(
            [repeated_representation, torch.unsqueeze(set_, 2)], 2
        )
        return representation_with_set

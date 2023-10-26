import numpy as np
import torch
import torch.nn.functional as F

from liltab.model.heterogenous_attributes_network import HeterogenousAttributesNetwork
from torch import Tensor, nn, optim, testing
from torchtest import assert_vars_change


def test_network_initializes_properly():
    network = HeterogenousAttributesNetwork(
        hidden_representation_size=20,
        n_hidden_layers=5,
        hidden_size=20,
        dropout_rate=0.1,
        inner_activation_function=nn.ReLU(),
        output_activation_function=nn.Identity(),
    )

    assert network.initial_features_encoding_network.input_size == 1
    assert network.initial_features_encoding_network.output_size == 20
    assert network.initial_features_encoding_network.n_hidden_layers == 5
    assert network.initial_features_encoding_network.hidden_size == 20
    assert network.initial_features_encoding_network.dropout_rate == 0.1

    assert network.initial_features_representation_network.input_size == 20
    assert network.initial_features_representation_network.output_size == 20
    assert network.initial_features_representation_network.n_hidden_layers == 5
    assert network.initial_features_representation_network.hidden_size == 20
    assert network.initial_features_representation_network.dropout_rate == 0.1

    assert network.interaction_encoding_network.input_size == 21
    assert network.interaction_encoding_network.output_size == 20
    assert network.interaction_encoding_network.n_hidden_layers == 5
    assert network.interaction_encoding_network.hidden_size == 20
    assert network.interaction_encoding_network.dropout_rate == 0.1

    assert network.interaction_representation_network.input_size == 20
    assert network.interaction_representation_network.output_size == 20
    assert network.interaction_representation_network.n_hidden_layers == 5
    assert network.interaction_representation_network.hidden_size == 20
    assert network.interaction_representation_network.dropout_rate == 0.1

    assert network.features_encoding_network.input_size == 21
    assert network.features_encoding_network.output_size == 20
    assert network.features_encoding_network.n_hidden_layers == 5
    assert network.features_encoding_network.hidden_size == 20
    assert network.features_encoding_network.dropout_rate == 0.1

    assert network.features_representation_network.input_size == 20
    assert network.features_representation_network.output_size == 20
    assert network.features_representation_network.n_hidden_layers == 5
    assert network.features_representation_network.hidden_size == 20
    assert network.features_representation_network.dropout_rate == 0.1

    assert network.inference_encoding_network.input_size == 21
    assert network.inference_encoding_network.output_size == 20
    assert network.inference_encoding_network.n_hidden_layers == 5
    assert network.inference_encoding_network.hidden_size == 20
    assert network.inference_encoding_network.dropout_rate == 0.1

    assert network.inference_network.input_size == 40
    assert network.inference_network.output_size == 1
    assert network.inference_network.n_hidden_layers == 5
    assert network.inference_network.hidden_size == 20
    assert network.inference_network.dropout_rate == 0.1


def test_forward_returns_proper_shape(utils):
    X_support = Tensor(np.random.uniform(size=(5, 10)))
    y_support = Tensor(np.random.uniform(size=(5, 3)))
    X_query = Tensor(np.random.uniform(size=(27, 10)))

    network = HeterogenousAttributesNetwork(
        hidden_representation_size=32,
        n_hidden_layers=3,
        hidden_size=32,
        dropout_rate=0.1,
        inner_activation_function=nn.ReLU(),
        output_activation_function=nn.Identity(),
    )

    prediction = network(X_support, y_support, X_query)

    assert prediction.shape == (27, 3)


def test_forward_returns_probabilities_when_classifier():
    X_support = Tensor(np.random.uniform(size=(5, 10)))
    y_support = Tensor([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]])
    X_query = Tensor(np.random.uniform(size=(27, 10)))

    network = HeterogenousAttributesNetwork(
        hidden_representation_size=32,
        n_hidden_layers=3,
        hidden_size=32,
        dropout_rate=0.1,
        inner_activation_function=nn.ReLU(),
        output_activation_function=nn.Identity(),
        is_classifier=True,
    )

    prediction = network(X_support, y_support, X_query)

    assert prediction.shape == (X_query.shape[0], y_support.shape[1])
    assert (prediction >= 0).all()
    testing.assert_close(prediction.sum(axis=1), torch.ones(27))


def test_all_network_params_are_trained(utils):
    X_support = Tensor(np.random.uniform(size=(5, 10)))
    y_support = Tensor(np.random.uniform(size=(5, 3)))
    X_query = Tensor(np.random.uniform(size=(27, 10)))
    y_query = Tensor(np.random.uniform(size=(27, 3)))

    network = HeterogenousAttributesNetwork(
        hidden_representation_size=32,
        n_hidden_layers=3,
        hidden_size=32,
        dropout_rate=0.1,
        inner_activation_function=nn.ELU(),
        output_activation_function=nn.Identity(),
    )

    inference_adapter = utils.get_inference_adapter(network, X_support, y_support)

    assert_vars_change(
        model=inference_adapter,
        loss_fn=F.cross_entropy,
        optim=optim.Adam(inference_adapter.parameters()),
        batch=(X_query, y_query),
        device="cpu:0",
    )


def test_calculate_initial_features_representation():
    X_support = Tensor(np.random.uniform(size=(5, 10)))

    network = HeterogenousAttributesNetwork(
        hidden_representation_size=32,
        n_hidden_layers=3,
        hidden_size=32,
        dropout_rate=0,
        inner_activation_function=nn.ReLU(),
        output_activation_function=nn.Identity(),
    )

    actual_initial_features_representation = network._calculate_initial_features_representation(
        network.initial_features_encoding_network,
        network.initial_features_representation_network,
        X_support,
    )

    expected_initial_features_representation_example = X_support[:, 1].unsqueeze(1)
    expected_initial_features_representation_example = network.initial_features_encoding_network(
        expected_initial_features_representation_example
    ).mean(axis=0)
    expected_initial_features_representation_example = (
        network.initial_features_representation_network(
            expected_initial_features_representation_example
        )
    )

    assert actual_initial_features_representation.shape == (10, 32)
    testing.assert_close(
        actual_initial_features_representation[1], expected_initial_features_representation_example
    )


def test_calculate_support_set_representation():
    X_support = Tensor(np.random.uniform(size=(5, 10)))
    attributes_initial_representation = Tensor(np.random.uniform(size=(10, 32)))
    y_support = Tensor(np.random.uniform(size=(5, 3)))
    responses_initial_representation = Tensor(np.random.uniform(size=(3, 32)))

    network = HeterogenousAttributesNetwork(
        hidden_representation_size=32,
        n_hidden_layers=3,
        hidden_size=32,
        dropout_rate=0,
        inner_activation_function=nn.ReLU(),
        output_activation_function=nn.Identity(),
    )

    actual_representation_shape = network._calculate_support_set_representation(
        network.features_encoding_network,
        network.features_representation_network,
        X_support,
        attributes_initial_representation,
        y_support,
        responses_initial_representation,
    ).shape

    assert actual_representation_shape == (5, 32)


def test_calculate_interaction_encoding():
    X_support = Tensor(np.random.uniform(size=(5, 10)))
    features_representation = Tensor(np.random.uniform(size=(10, 32)))

    network = HeterogenousAttributesNetwork(
        hidden_representation_size=32,
        n_hidden_layers=3,
        hidden_size=32,
        dropout_rate=0,
        inner_activation_function=nn.ReLU(),
        output_activation_function=nn.Identity(),
    )

    actual = network._calculate_interaction_encoding(
        network.interaction_encoding_network, features_representation, X_support
    )
    expected_example = network._enrich_representation_with_set_rows(
        features_representation, X_support
    )[1]
    expected_example = network.interaction_encoding_network(expected_example)
    expected_example = expected_example.mean(axis=0)

    testing.assert_close(actual[1], expected_example)


def test_calculate_features_representation():
    X_support = Tensor(np.random.uniform(size=(5, 10)))
    set_representation = Tensor(np.random.uniform(size=(5, 32)))

    network = HeterogenousAttributesNetwork(
        hidden_representation_size=32,
        n_hidden_layers=3,
        hidden_size=32,
        dropout_rate=0,
        inner_activation_function=nn.ReLU(),
        output_activation_function=nn.Identity(),
    )

    actual_representation_shape = network._calculate_features_representation(
        network.features_encoding_network,
        network.features_representation_network,
        set_representation,
        X_support,
    ).shape

    assert actual_representation_shape == (10, 32)


def test_enrich_representation_with_set_features():
    network = HeterogenousAttributesNetwork(
        hidden_representation_size=32,
        n_hidden_layers=3,
        hidden_size=32,
        dropout_rate=0,
        inner_activation_function=nn.ReLU(),
        output_activation_function=nn.Identity(),
    )

    representation = Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    set_ = Tensor([[1, 2], [3, 4], [5, 6]])

    expected = Tensor(
        [
            [[0.1, 0.2, 0.3, 1], [0.4, 0.5, 0.6, 3], [0.7, 0.8, 0.9, 5]],
            [[0.1, 0.2, 0.3, 2], [0.4, 0.5, 0.6, 4], [0.7, 0.8, 0.9, 6]],
        ]
    )
    actual = network._enrich_representation_with_set_features(representation, set_)

    testing.assert_close(actual, expected)


def test_make_prediction_reg():
    attributes_representation = Tensor(np.random.uniform(size=(10, 32)))
    responses_representation = Tensor(np.random.uniform(size=(3, 32)))
    X_query = Tensor(np.random.uniform(size=(5, 10)))

    network = HeterogenousAttributesNetwork(
        hidden_representation_size=32,
        n_hidden_layers=3,
        hidden_size=32,
        dropout_rate=0,
        inner_activation_function=nn.ReLU(),
        output_activation_function=nn.Identity(),
    )

    actual_prediction = network._make_prediction_reg(
        network.inference_encoding_network,
        network.inference_embedding_network,
        network.inference_network,
        attributes_representation,
        responses_representation,
        X_query,
    )

    query_example_embedding = X_query[1].unsqueeze(0)
    query_example_embedding = network._enrich_representation_with_set_rows(
        attributes_representation, query_example_embedding
    )
    query_example_embedding = query_example_embedding.reshape(-1, 33)
    query_example_embedding = network.inference_encoding_network(query_example_embedding).mean(
        axis=0
    )
    query_example_embedding = network.inference_embedding_network(query_example_embedding)

    inference_network_example_input = query_example_embedding.repeat(3, 1)
    inference_network_example_input = torch.concat(
        [responses_representation, inference_network_example_input], axis=1
    )
    expected_example_response = network.inference_network(inference_network_example_input).reshape(
        3
    )

    assert actual_prediction.shape == (5, 3)
    testing.assert_close(actual_prediction[1], expected_example_response)


def test_get_inference_embedding_of_set():
    attributes_representation = Tensor(np.random.uniform(size=(10, 32)))
    X_query = Tensor(np.random.uniform(size=(5, 10)))

    network = HeterogenousAttributesNetwork(
        hidden_representation_size=32,
        n_hidden_layers=3,
        hidden_size=32,
        dropout_rate=0,
        inner_activation_function=nn.ReLU(),
        output_activation_function=nn.Identity(),
    )

    expected_query_example_embedding = X_query[1].unsqueeze(0)
    expected_query_example_embedding = network._enrich_representation_with_set_rows(
        attributes_representation, expected_query_example_embedding
    )
    expected_query_example_embedding = expected_query_example_embedding.reshape(10, 33)
    expected_query_example_embedding = network.inference_encoding_network(
        expected_query_example_embedding
    ).mean(axis=0)
    expected_query_example_embedding = network.inference_embedding_network(
        expected_query_example_embedding
    )

    actual_query_example_embedding = network._get_inference_embedding_of_set(
        network.inference_encoding_network,
        network.inference_embedding_network,
        X_query,
        attributes_representation,
    )

    testing.assert_close(actual_query_example_embedding[1], expected_query_example_embedding)


def test_calculate_classes_representations():
    X = torch.Tensor([[1, 2, 3], [4, 5, 6], [6, 5, 4], [3, 2, 1]])
    y = torch.Tensor([[0, 1], [1, 0], [1, 0], [0, 1]])
    expected_representations = Tensor([[5.0, 5.0, 5.0], [2.0, 2.0, 2.0]])

    network = HeterogenousAttributesNetwork(
        hidden_representation_size=32,
        n_hidden_layers=3,
        hidden_size=32,
        dropout_rate=0,
        inner_activation_function=nn.ReLU(),
        output_activation_function=nn.Identity(),
    )

    actual_representations = network._calculate_classes_representations(X, y)

    testing.assert_close(expected_representations, actual_representations)


def test_enrich_representation_with_set_rows():
    network = HeterogenousAttributesNetwork(
        hidden_representation_size=32,
        n_hidden_layers=3,
        hidden_size=32,
        dropout_rate=0,
        inner_activation_function=nn.ReLU(),
        output_activation_function=nn.Identity(),
    )

    representation = Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    set_ = Tensor([[1, 2], [3, 4], [5, 6]])

    expected = Tensor(
        [
            [[0.1, 0.2, 0.3, 1], [0.4, 0.5, 0.6, 2]],
            [[0.1, 0.2, 0.3, 3], [0.4, 0.5, 0.6, 4]],
            [[0.1, 0.2, 0.3, 5], [0.4, 0.5, 0.6, 6]],
        ]
    )
    actual = network._enrich_representation_with_set_rows(representation, set_)

    testing.assert_close(actual, expected)

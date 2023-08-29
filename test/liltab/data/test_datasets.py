import numpy as np
import pandas as pd
import pytest

from liltab.data.datasets import PandasDataset, RandomFeaturesPandasDataset
from numpy.testing import assert_almost_equal
from torch import Tensor, float32


def test_dataset_initializes_default_columns(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    df = pd.read_csv(frame_path)
    frame_columns = df.columns.tolist()

    dataset = PandasDataset(frame_path)

    assert dataset.feature_columns == frame_columns[:-1]
    assert dataset.target_columns == [frame_columns[-1]]


def test_dataset_assigns_non_default_columns(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    df = pd.read_csv(frame_path)
    frame_columns = df.columns.tolist()

    dataset = PandasDataset(
        frame_path,
        feature_columns=frame_columns[1:3],
        target_columns=frame_columns[4:],
    )

    assert dataset.feature_columns == frame_columns[1:3]
    assert dataset.target_columns == frame_columns[4:]


def test_indexing_dataset_returns_proper_data(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    df = pd.read_csv(frame_path)
    index = [2, 3, 6, 8, 10, 11]

    dataset = PandasDataset(frame_path, preprocess_data=False)

    expected_records = df.loc[index]
    actual_X, actual_y = dataset[index]

    assert_almost_equal(actual_X.numpy(), expected_records[dataset.feature_columns].values)
    assert_almost_equal(actual_y.numpy(), expected_records[dataset.target_columns].values)


def test_indexing_dataset_returns_proper_data_with_preprocessing(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    df = pd.read_csv(frame_path)
    df = (df - df.mean(axis=0)) / df.std(axis=0)
    index = [2, 3, 6, 8, 10, 11]

    dataset = PandasDataset(frame_path)

    expected_records = df.loc[index]
    actual_X, actual_y = dataset[index]

    assert_almost_equal(
        actual_X.numpy(), expected_records[dataset.feature_columns].values, decimal=2
    )
    assert_almost_equal(
        actual_y.numpy(), expected_records[dataset.target_columns].values, decimal=2
    )


def test_class_forbids_one_hot_with_multiple_targets(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    df = pd.read_csv(frame_path)
    feture_columns = df.columns[:-2]
    target_columns = df.columns[-2:]
    with pytest.raises(ValueError):
        PandasDataset(frame_path, feture_columns, target_columns, encode_categorical_target=True)


def test_preprocessing_when_target_categorical(resources_path):
    frame_path = resources_path / "random_df_3.csv"
    df = pd.read_csv(frame_path)
    expected_X = df.drop(columns=["class"])
    expected_X = (expected_X - expected_X.mean(axis=0)) / expected_X.std(axis=0)

    dataset = PandasDataset(frame_path, encode_categorical_target=True)

    assert dataset.y.shape == (df.shape[0], df["class"].max())
    assert_almost_equal(dataset.y.sum(axis=1).numpy(), np.ones(df.shape[0]))
    assert_almost_equal(dataset.X.numpy(), expected_X.values, decimal=2)


def test_indexing_returns_dataset_with_proper_type(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    index = [1, 2, 3]
    dataset = PandasDataset(frame_path)

    X, y = dataset[index]

    assert type(X) is Tensor
    assert X.dtype == float32
    assert type(y) is Tensor
    assert y.dtype == float32


def test_random_features_pandas_dataset_has_no_null(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    dataset = RandomFeaturesPandasDataset(frame_path, preprocess_data=True)
    assert dataset.df.isnull().sum().sum() == 0


def test_random_features_pandas_dataset_is_standarized(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    dataset = RandomFeaturesPandasDataset(frame_path, preprocess_data=True)
    assert (np.abs(dataset.df.std() - 1) < 1e-1).all()
    assert (np.abs(dataset.df.mean()) < 1e-6).all()


def test_random_features_pandas_dataset_change_features(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    persist_features_iter = 3
    dataset = RandomFeaturesPandasDataset(
        frame_path, preprocess_data=True, persist_features_iter=persist_features_iter
    )

    previous_features = np.ndarray([])
    previous_target = np.ndarray([])

    features_change_cnt = 0
    target_change_cnt = 0

    for _ in range(int(1e2)):
        dataset[0]
        features_change_cnt += int(not np.array_equal(previous_features, dataset.features))
        target_change_cnt += int(not np.array_equal(previous_target, dataset.target))

        previous_features = dataset.features
        previous_target = dataset.target

    assert np.abs(features_change_cnt / int(1e2) - 1 / persist_features_iter) < 2e-1
    assert np.abs(target_change_cnt / int(1e2) - 1 / persist_features_iter) < 2e-1

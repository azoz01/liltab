import pandas as pd

from liltab.data.datasets import PandasDataset
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

    dataset = PandasDataset(frame_path)

    expected_records = df.loc[index]
    actual_X, actual_y = dataset[index]

    assert_almost_equal(actual_X.numpy(), expected_records[dataset.feature_columns].values)
    assert_almost_equal(actual_y.numpy(), expected_records[dataset.target_columns].values)


def test_indexing_returns_dataset_with_proper_type(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    index = [1, 2, 3]
    dataset = PandasDataset(frame_path)

    X, y = dataset[index]

    assert type(X) is Tensor
    assert X.dtype == float32
    assert type(y) is Tensor
    assert y.dtype == float32

import numpy as np
import pandas as pd
import pytest

from liltab.data.datasets import PandasDataset, RandomFeaturesPandasDataset
from numpy.testing import assert_almost_equal
from torch import Tensor, float32


def test_dataset_works_when_path_given(resources_path):
    frame_path = resources_path / "random_df_1.csv"

    dataset = PandasDataset(frame_path)

    assert dataset.df is not None


def test_dataset_works_dataframe_given(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    df = pd.read_csv(frame_path)

    dataset = PandasDataset(df)

    assert dataset.df is not None


def test_dataset_raises_error_with_incorrect_data():
    with pytest.raises(ValueError):
        PandasDataset(1)


def test_dataset_initializes_default_columns(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    df = pd.read_csv(frame_path)
    frame_columns = df.columns.tolist()

    dataset = PandasDataset(frame_path)

    assert (
        dataset.attribute_columns
        == ["pipeline-2__col_1", "pipeline-2__col_2", "pipeline-2__col_3", "pipeline-2__col_4"]
    ).all()
    assert (dataset.response_columns == [frame_columns[-1]]).all()


def test_dataset_assigns_non_default_columns(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    df = pd.read_csv(frame_path)
    frame_columns = df.columns.tolist()

    dataset = PandasDataset(
        frame_path,
        attribute_columns=frame_columns[1:3],
        response_columns=frame_columns[4:],
    )

    assert (dataset.attribute_columns == ["pipeline-2__col_2", "pipeline-2__col_3"]).all()
    assert (dataset.response_columns == frame_columns[4:]).all()


def test_indexing_dataset_returns_proper_data(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    df = pd.read_csv(frame_path)
    index = [2, 3, 6, 8, 10, 11]

    dataset = PandasDataset(frame_path, preprocess_data=False)

    expected_records = df.loc[index]
    actual_X, actual_y = dataset[index]

    assert_almost_equal(actual_X.numpy(), expected_records[dataset.attribute_columns].values)
    assert_almost_equal(actual_y.numpy(), expected_records[dataset.response_columns].values)


def test_indexing_dataset_returns_proper_data_with_preprocessing(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    df = pd.read_csv(frame_path)
    df = (df - df.mean(axis=0)) / df.std(axis=0)
    index = [2, 3, 6, 8, 10, 11]

    dataset = PandasDataset(frame_path)

    expected_records = df.loc[index]
    actual_X, actual_y = dataset[index]

    assert_almost_equal(actual_X.numpy(), expected_records.iloc[:, :4].values, decimal=2)
    assert_almost_equal(
        actual_y.numpy(), expected_records[dataset.response_columns].values, decimal=2
    )


def test_dataset_encodes_categorical_columns():
    df = df = pd.DataFrame(
        data=[
            [1, "A", "E", 0.1],
            [3, "B", "E", 0.5],
            [3, "A", "F", 0.4],
            [1, "C", "E", 0.3],
        ],
        columns=["int1", "cat1", "cat2", "target"],
    )
    df["cat2"].astype("category")
    dataset = PandasDataset(df)

    expected_attribute_columns = [
        "pipeline-2__int1",
        "pipeline-1__cat1_A",
        "pipeline-1__cat1_B",
        "pipeline-1__cat1_C",
        "pipeline-1__cat2_E",
        "pipeline-1__cat2_F",
    ]

    assert (dataset.attribute_columns == expected_attribute_columns).all()
    assert (
        dataset.df[
            [
                "pipeline-1__cat1_A",
                "pipeline-1__cat1_B",
                "pipeline-1__cat1_C",
                "pipeline-1__cat2_E",
                "pipeline-1__cat2_F",
            ]
        ].values
        == np.array(
            [
                [1, 0, 0, 1, 0],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 1, 1, 0],
            ]
        )
    ).all()


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
        frame_path,
        preprocess_data=True,
        persist_features_iter=persist_features_iter,
        total_random_feature_sampling=True,
    )

    previous_features = np.ndarray([])
    previous_target = np.ndarray([])

    features_change_cnt = 0
    target_change_cnt = 0

    for _ in range(int(1e2)):
        dataset[0]
        features_change_cnt += int(not np.array_equal(previous_features, dataset.attributes))
        target_change_cnt += int(not np.array_equal(previous_target, dataset.responses))

        previous_features = dataset.attributes
        previous_target = dataset.responses

    assert np.abs(features_change_cnt / int(1e2) - 1 / persist_features_iter) < 2e-1
    assert np.abs(target_change_cnt / int(1e2) - 1 / persist_features_iter) < 2e-1


def test_random_features_pandas_dataset_returns_proper_subset(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    persist_features_iter = 1
    dataset = RandomFeaturesPandasDataset(
        frame_path,
        preprocess_data=True,
        persist_features_iter=persist_features_iter,
        attribute_columns=["col_1", "col_2"],
        response_columns=["col_3", "col_4", "col_5"],
    )
    for _ in range(int(1e2)):
        X, y = dataset[0:10]
        assert X.shape[1] <= 2
        assert y.shape[1] <= 3

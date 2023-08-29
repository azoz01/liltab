import numpy as np
import pandas as pd
import torch

from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset

from .preprocessing import get_preprocessing_pipeline


class PandasDataset(Dataset):
    """
    Torch wrapper to pandas DataFrame which makes it usable
    as torch.Dataset. Data from this class is returned
    as Tensor.
    """

    def __init__(
        self,
        data_path: Path,
        feature_columns: list[str] = None,
        target_columns: list[str] = None,
        preprocess_data: bool = True,
        encode_categorical_target: bool = False,
    ):
        """
        Args:
            data_path (Path): Path to data to be loaded
            feature_columns (list[str], optional): Columns from frame
                which will be used as features.
                Defaults to all columns without last.
            target_columns (list[str], optional): Columns from frame
                to be used as features. Defaults to last column from frame.
            preprocess_data (bool, optional): If true, then imputes data
                using mean strategy and standardizes using StandardScaler.
                Defaults to True.
            encode_categorical_target(bool, optional): if True, then target column
                will be encoded using one-hot. Works only with single target variable.
                Default to False.
        """
        self.data_path = data_path
        self.encode_categorical_target = encode_categorical_target
        self.df = pd.read_csv(data_path)
        self.feature_columns = (
            feature_columns if feature_columns is not None else self.df.columns.tolist()[:-1]
        )
        self.target_columns = (
            target_columns if target_columns is not None else [self.df.columns.tolist()[-1]]
        )

        if len(self.target_columns) > 1 and self.encode_categorical_target:
            raise ValueError("One-hot encoding is supported only for single target")

        if preprocess_data:
            self.preprocessing_pipeline = get_preprocessing_pipeline()
            if self.encode_categorical_target:
                self.df.loc[:, self.feature_columns] = self.preprocessing_pipeline.fit_transform(
                    self.df[self.feature_columns]
                )
            else:
                self.df = pd.DataFrame(
                    self.preprocessing_pipeline.fit_transform(self.df), columns=self.df.columns
                )
        self.X = torch.from_numpy(self.df[self.feature_columns].to_numpy()).type(torch.float32)

        self.y = self.df[self.target_columns]
        if self.encode_categorical_target:
            self.y = pd.get_dummies(self.y.astype("category"))
        self.y = torch.from_numpy(self.y.to_numpy()).type(torch.float32)

    def __getitem__(self, idx: list[int]) -> tuple[Tensor, Tensor]:
        X = self.X[idx]
        y = self.y[idx]

        return X, y

    def __len__(self) -> int:
        return self.df.shape[0]


class RandomFeaturesPandasDataset(Dataset):
    """
    Torch wrapper to pandas DataFrame which makes it usable
    as torch.Dataset. Every persist_features_iter iteration
    object selects random attributes and responses columns
    which will be returned as Tensor.
    """

    def __init__(
        self,
        data_path: Path,
        persist_features_iter: int = 2,
        preprocess_data: bool = True,
    ):
        """
        Args:
            data_path (Path): Path to data to be loaded
            persist_features_iter (int, optional): For how many
                iterations persist current selection of features.
                Defaults to 2.
            preprocess_data(bool, optional): If true, then imputes data
                using mean strategy and standardizes using StandardScaler.
                Defaults to True.
        """
        self.data_path = data_path
        self.persist_features_iter = persist_features_iter

        self.df = pd.read_csv(data_path)
        self.columns = self.df.columns.values
        self.n_columns = len(self.columns)

        if preprocess_data:
            self.preprocessing_pipeline = get_preprocessing_pipeline()
            self.df = pd.DataFrame(
                self.preprocessing_pipeline.fit_transform(self.df), columns=self.df.columns
            )

        self.persist_features_counter = 0
        self.features = None
        self.target = None

    def __getitem__(self, idx: list[int]) -> tuple[Tensor, Tensor]:
        if self.persist_features_counter == 0:
            self.persist_features_counter = self.persist_features_iter
            col_idx = np.arange(self.n_columns)
            features_size = np.random.randint(low=1, high=self.n_columns)
            features_idx = np.random.choice(col_idx, features_size)
            remaining_idx = list(set(col_idx) - set(features_idx))
            target_idx = np.random.choice(remaining_idx, 1)
            self.features, self.target = self.columns[features_idx], self.columns[target_idx]
        self.persist_features_counter -= 1

        X = torch.from_numpy(self.df[self.features].to_numpy()).type(torch.float32)
        y = torch.from_numpy(self.df[self.target].to_numpy()).type(torch.float32)

        return X[idx], y[idx]

    def __len__(self) -> int:
        return self.df.shape[0]

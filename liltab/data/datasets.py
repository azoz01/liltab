import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import torch

from abc import ABC, abstractmethod
from pathlib import PosixPath
from sklearn.preprocessing import OneHotEncoder
from torch import Tensor
from typing import Union

from .preprocessing import get_preprocessing_pipeline


class Dataset(ABC):
    """
    Abstract class for Datasets. It reads and stores data as Pandas
    DataFrame. __getitem__ method is to be implemented with custom
    indexing strategy.
    """

    def __init__(
        self,
        data: Union[PosixPath, str, pd.DataFrame],
        attribute_columns: list[str],
        response_columns: list[str],
        preprocess_data: bool,
        encode_categorical_response: bool,
    ):
        if (
            response_columns is not None
            and len(response_columns) > 1
            and encode_categorical_response
        ):
            raise ValueError("One-hot encoding is supported only for single target")

        self.data = data
        if type(data) in [str, PosixPath]:
            self.df = pd.read_csv(data)
        elif type(data) == pd.DataFrame:
            self.df = data
        else:
            raise ValueError(
                f"Data should be PosixPath, " f"str or pandas.DataFrame but is {type(data)}"
            )

        self.attribute_columns = np.array(
            attribute_columns if attribute_columns is not None else self.df.columns.tolist()[:-1]
        )
        self.response_columns = np.array(
            response_columns if response_columns is not None else [self.df.columns.tolist()[-1]]
        )
        self.n_attributes = len(self.attribute_columns)
        self.n_responses = len(self.response_columns)

        self.encode_categorical_response = encode_categorical_response
        self.preprocess_data = preprocess_data

        if self.preprocess_data:
            self._preprocess_data()
        if self.encode_categorical_response:
            self._encode_categorical_response()
        else:
            self.y = self.df[self.response_columns].values

    def _preprocess_data(self):
        """
        Performs following preprocessing:
            * data imputation
            * z-score scaling of numerical columns
            * one-hot encoding categorical columns
        If encode_categorical_response = True, then omits response column.
        """
        self.preprocessing_pipeline = get_preprocessing_pipeline()
        df_preproc = self.preprocessing_pipeline.fit_transform(self.df[self.attribute_columns])
        self.df = self.df.drop(columns=self.attribute_columns)
        self.df = pd.concat([df_preproc, self.df], axis=1)
        self.attribute_columns = df_preproc.columns.values

        if not self.encode_categorical_response:
            self.df[self.response_columns] = StandardScaler().fit_transform(
                self.df[self.response_columns]
            )

    def _encode_categorical_response(self):
        """
        Encodes categorical response using one-hot encoding.
        """
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.raw_y = self.df[self.response_columns]
        self.y = self.one_hot_encoder.fit_transform((self.df[self.response_columns]))

    @abstractmethod
    def __getitem__(self):
        pass

    def __len__(self) -> int:
        return self.df.shape[0]


class PandasDataset(Dataset):
    """
    Torch wrapper to pandas DataFrame which makes it usable
    as torch.Dataset. Data from this class is returned
    as Tensor.
    """

    def __init__(
        self,
        data: Union[PosixPath, str, pd.DataFrame],
        attribute_columns: list[str] = None,
        response_columns: list[str] = None,
        preprocess_data: bool = True,
        encode_categorical_response: bool = False,
    ):
        """
        Args:
            data (Union[PosixPath, str, pd.DataFrame]): Frame with data or path to .csv file.
            attribute_columns (list[str], optional): Columns from frame
                which will be used as attributes.
                Defaults to all columns without last.
            response_columns (list[str], optional): Columns from frame
                to be used as responses. Defaults to last column from frame.
            preprocess_data (bool, optional): If true, then imputes data
                using mean strategy and standardizes using StandardScaler.
                Defaults to True.
            encode_categorical_response(bool, optional): if True, then target column
                will be encoded using one-hot. Works only with single target variable.
                Default to False.
        """
        super().__init__(
            data=data,
            attribute_columns=attribute_columns,
            response_columns=response_columns,
            encode_categorical_response=encode_categorical_response,
            preprocess_data=preprocess_data,
        )

        self.X = torch.from_numpy(self.df[self.attribute_columns].to_numpy()).type(torch.float32)
        self.y = torch.from_numpy(self.y).type(torch.float32)

    def __getitem__(self, idx: list[int]) -> tuple[Tensor, Tensor]:
        X = self.X[idx]
        y = self.y[idx]

        return X, y


class RandomFeaturesPandasDataset(Dataset):
    """
    Torch wrapper to pandas DataFrame which makes it usable
    as torch.Dataset. Every persist_features_iter iteration
    object selects random attributes and responses columns
    which will be returned as Tensor.
    """

    def __init__(
        self,
        data: Union[PosixPath, str, pd.DataFrame],
        attribute_columns: list[str] = None,
        response_columns: list[str] = None,
        total_random_feature_sampling: bool = False,
        preprocess_data: bool = True,
        encode_categorical_response: bool = False,
        persist_features_iter: int = 2,
    ):
        """
        Args:
            data (Union[PosixPath, str, pd.DataFrame]): Frame with data or path to .csv file.
            attribute_columns (list[str], optional): Columns from frame
                which will be attributes sampled from.
                Ignored when total_random_feature_sampling = True.
                Defaults to all columns without last.
            response_columns (list[str], optional): Columns from frame
                to be responses sampled from.
                Ignored when total_random_feature_sampling = True.
                Defaults to last column from frame.
            total_random_feature_sampling (list[bool], optional): If True then attributes
                and responses are sampled from all datat columns and ignores
                attribute_columns and response_columns. Defaults to False.
            preprocess_data(bool, optional): If true, then imputes data
                using mean strategy and standardizes using StandardScaler.
                Defaults to True.
            encode_categorical_response(bool, optional): if True, then target column
                will be encoded using one-hot.
                When total_random_feature_sampling=True it should be False.
                Works only with single target variable.
                Default to False.
            persist_features_iter (int, optional): For how many
                iterations persist current selection of features.
                Defaults to 2.
        """
        super().__init__(
            data=data,
            attribute_columns=attribute_columns,
            response_columns=response_columns,
            encode_categorical_response=encode_categorical_response,
            preprocess_data=preprocess_data,
        )
        if total_random_feature_sampling and (
            attribute_columns is not None or response_columns or encode_categorical_response
        ):
            raise ValueError(
                "total_random_feature_sampling doesn't support feature or encoding specification"
            )

        self.total_random_feature_sampling = total_random_feature_sampling
        self.persist_features_iter = persist_features_iter
        self.persist_features_counter = 0
        self.n_columns = self.df.shape[1]
        self.columns = self.df.columns.values
        self.attributes = None
        self.responses = None

    def __getitem__(self, idx: list[int]) -> tuple[Tensor, Tensor]:
        if self.persist_features_counter == 0:
            self.persist_features_counter = self.persist_features_iter

            if self.total_random_feature_sampling:
                attributes_idx, responses_idx = self._get_features_from_all_columns()
                self.attributes, self.responses = (
                    self.columns[attributes_idx],
                    self.columns[responses_idx],
                )
            else:
                (
                    attributes_idx,
                    responses_idx,
                ) = self._get_features_from_selected_columns()
                self.attributes, self.responses = (
                    self.attribute_columns[attributes_idx],
                    self.response_columns[responses_idx],
                )
        self.persist_features_counter -= 1

        X = torch.from_numpy(self.df[self.attributes].to_numpy()).type(torch.float32)
        if self.encode_categorical_response:
            y = torch.from_numpy(self.y).type(torch.float32)
        else:
            y = torch.from_numpy(self.df[self.responses].to_numpy()).type(torch.float32)

        return X[idx], y[idx]

    def _get_features_from_selected_columns(self) -> tuple[int, int]:
        attributes_size = np.random.randint(low=1, high=self.n_attributes + 1)
        responses_size = np.random.randint(low=1, high=self.n_responses + 1)
        attributes_idx = np.random.choice(len(self.attribute_columns), attributes_size).tolist()
        responses_idx = np.random.choice(len(self.response_columns), responses_size).tolist()

        return attributes_idx, responses_idx

    def _get_features_from_all_columns(self) -> tuple[int, int]:
        col_idx = np.arange(self.n_columns)
        features_size = np.random.randint(low=1, high=self.n_columns)
        attributes_idx = np.random.choice(col_idx, features_size)
        remaining_idx = list(set(col_idx) - set(attributes_idx))
        responses_idx = np.random.choice(remaining_idx, 1)
        return attributes_idx, responses_idx

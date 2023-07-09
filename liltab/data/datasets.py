import numpy as np
import pandas as pd
import torch

from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset


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
    ):
        """
        Args:
            data_path (Path): Path to data to be loaded
            feature_columns (list[str], optional): Columns from frame
                which will be used as features.
                Defaults to all columns without last.
            target_columns (list[str], optional): Columns from frame
                to be used as features. Defaults to last column from frame.
        """
        self.data_path = data_path
        self.df = pd.read_csv(data_path)

        self.feature_columns = (
            self.df.columns.tolist()
            if not feature_columns
            else feature_columns.copy()
        )
        self.target_columns = (
            [self.feature_columns.pop(-1)]
            if not target_columns
            else target_columns
        )

        self.X = torch.from_numpy(
            self.df[self.feature_columns].to_numpy()
        ).type(torch.float32)
        self.y = torch.from_numpy(
            self.df[self.target_columns].to_numpy()
        ).type(torch.float32)

    def __getitem__(self, idx: list[str]) -> tuple[Tensor, Tensor]:
        X = self.X[idx]
        y = self.y[idx]

        return X, y

    def __len__(self) -> int:
        return self.df.shape[0]


class FewShotDataset(IterableDataset):
    """
    Wrapper around Dataset, which returns data in few-shot learning manner
    i.e. when called next(dataset) it returns properly sampled support and query sets.
    """

    def __init__(
        self,
        dataset: Dataset,
        support_size: int,
        query_size: int,
        n_episodes: int = None,
    ):
        """
        Args:
            dataset (Dataset): dataset to sample data from.
            support_size (int): size of support set in each episode.
            query_size (int): size of query set in each episode.
            n_episodes (int, optional): number of episodes.
                If none, then iterator without end. Defaults to None.
        """
        self.dataset = dataset
        self.support_size = support_size
        self.query_size = query_size
        self.n_episodes = n_episodes

        self.curr_episode = 0

        self.n_rows = len(self.dataset)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns support and query sets with sized specified in constructors.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]:
                (X_support, y_support, X_query, y_query)
        """
        if self.n_episodes:
            self.curr_episode += 1
            if self.curr_episode == self.n_episodes:
                raise StopIteration()

        all_drawn_indices = np.random.choice(
            self.n_rows, self.support_size + self.query_size, replace=False
        )
        support_indices = np.random.choice(
            all_drawn_indices, self.support_size, replace=False
        )
        query_indices = np.array(
            list(set(all_drawn_indices) - set(support_indices))
        )
        return *self.dataset[support_indices], *self.dataset[query_indices]

    def has_next(self) -> bool:
        return self.curr_episode != self.n_episodes


class ComposedFewShotDataset(IterableDataset):
    """
    Dataset which wraps list of FewShotDataset objects and
    when next(dataset) is called, then returns episode from
    randomly chosen one of passed datasets.
    """

    def __init__(self, datasets: list[FewShotDataset], n_episodes: int = None):
        """
        Args:
            datasets (list[FewShotDataset]): list of datasets to sample from
            n_episodes (int, optional): number of episodes.
                If none, then iterator without end. Defaults to None.
        """
        self.datasets = datasets
        self.n_episodes = n_episodes

        self.curr_episode = 0
        self.n_datasets = len(datasets)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns support and query sets from one Dataset from
        randomly chosen from passed datasets.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]:
                (X_support, y_support, X_query, y_query)
        """
        if self.n_episodes:
            self.curr_episode += 1
            if self.curr_episode == self.n_episodes:
                raise StopIteration()

        dataset_hasnt_next = True
        while dataset_hasnt_next:
            dataset_idx = np.random.choice(self.n_datasets, 1)[0]
            dataset = self.datasets[dataset_idx]
            dataset_hasnt_next = not dataset.has_next()
        return next(dataset)

    def has_next(self) -> bool:
        return self.curr_episode != self.n_episodes

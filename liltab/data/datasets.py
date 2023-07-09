import numpy as np
import pandas as pd
import torch

from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset

class PandasDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        feature_columns: list[str] = None,
        target_columns: list[str] = None,
    ):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)

        self.feature_columns = (
            self.df.columns.tolist() if not feature_columns else feature_columns.copy()
        )
        self.target_columns = (
            self.feature_columns.pop(-1) if not target_columns else target_columns
        )

        self.X = torch.from_numpy(self.df[self.feature_columns].to_numpy())
        self.y = torch.from_numpy(self.df[self.target_columns].to_numpy())

    def __getitem__(self, idx: list[str]) -> tuple[Tensor, Tensor]:
        X = self.X[idx]
        y = self.y[idx]

        return X, y

    def __len__(self) -> int:
        return self.df.shape[0]
    

class FewShotDataset(IterableDataset):
    def __init__(self, dataset: Dataset, support_size: int, query_size: int, n_episodes: int = None):
        self.dataset = dataset
        self.support_size = support_size
        self.query_size = query_size
        self.n_episodes = n_episodes
        
        self.curr_episode = 0

        self.n_rows = len(self.dataset)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
        query_indices = np.array(list(set(all_drawn_indices) - set(support_indices)))
        return *self.dataset[support_indices], *self.dataset[query_indices]
    
    def has_next(self) -> bool:
        return self.curr_episode != self.n_episodes


class ComposedFewShotDataset(IterableDataset):
    def __init__(self, datasets: list[FewShotDataset], n_episodes: int = None):
        self.datasets = datasets
        self.n_episodes = n_episodes

        self.curr_episode = 0
        self.n_datasets = len(datasets)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
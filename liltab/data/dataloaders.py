import numpy as np

from torch import Tensor
from torch.utils.data import Dataset
from typing import Iterable


class FewShotDataLoader:
    """
    DataLoader, which iterates dataset in few-shot learning manner
    i.e. when called next(loader) it returns episode with
    properly sampled support and query sets.
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
            dataset (Dataset): dataset to load data from.
            support_size (int): size of support set in each episode.
            query_size (int): size of query set in each episode.
            n_episodes (int, optional): number of episodes.
                If none, then iterator is without end. Defaults to None.
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
        Returns episode i.e. support and query sets
        with sized specified in constructors.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]:
                (X_support, y_support, X_query, y_query)
        """
        if self.n_episodes:
            if self.curr_episode == self.n_episodes:
                raise StopIteration()
            self.curr_episode += 1

        all_drawn_indices = np.random.choice(
            self.n_rows, self.support_size + self.query_size, replace=False
        )
        support_indices = np.random.choice(all_drawn_indices, self.support_size, replace=False)
        query_indices = np.array(list(set(all_drawn_indices) - set(support_indices)))
        return *self.dataset[support_indices], *self.dataset[query_indices]

    def has_next(self) -> bool:
        return self.curr_episode != self.n_episodes


class ComposedDataLoader:
    """
    DataLoader which wraps list of FewShotDataLoader objects and
    when next(dataloader) is called, then returns episode from
    randomly chosen one of passed dataloaders.
    """

    def __init__(self, dataloaders: list[Iterable], n_episodes: int = None):
        """
        Args:
            dataloaders (list[Iterable]): list of
                dataloaders to sample from
            n_episodes (int, optional): number of episodes.
                If none, then iterator is without end. Defaults to None.
        """
        self.dataloaders = dataloaders
        self.n_episodes = n_episodes

        self.curr_episode = 0
        self.n_dataloaders = len(dataloaders)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns support and query sets from one DataLoaders from
        randomly chosen from passed dataloaders.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]:
                (X_support, y_support, X_query, y_query)
        """
        if self.n_episodes:
            if self.curr_episode == self.n_episodes:
                raise StopIteration()
            self.curr_episode += 1

        dataloader_has_next = False
        while not dataloader_has_next:
            dataloader_idx = np.random.choice(self.n_dataloaders, 1)[0]
            dataloader = self.dataloaders[dataloader_idx]
            dataloader_has_next = dataloader.has_next()
        return next(dataloader)

    def has_next(self) -> bool:
        return self.curr_episode != self.n_episodes

import numpy as np

from copy import deepcopy
from torch import Tensor
from torch.utils.data import Dataset
from typing import Iterable, OrderedDict


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
        return deepcopy(self)

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

        replace = True if self.support_size + self.query_size >= self.n_rows else False
        all_drawn_indices = np.random.choice(
            self.n_rows, self.support_size + self.query_size, replace=replace
        )
        support_indices = np.random.choice(all_drawn_indices, self.support_size, replace=False)
        query_indices = np.array(list(set(all_drawn_indices) - set(support_indices)))
        return *self.dataset[support_indices], *self.dataset[query_indices]

    def has_next(self) -> bool:
        return self.curr_episode != self.n_episodes


class ComposedDataLoader:
    """
    DataLoader which wraps list of data loaders objects and
    when next(dataloader) is called, then returns episode from
    randomly chosen one of passed dataloaders.
    """

    def __init__(
        self,
        dataloaders: list[Iterable],
        batch_size: int = 32,
        num_batches: int = 1,
    ):
        """
        Args:
            dataloaders (list[Iterable]): list of
                dataloaders to sample from
            batch_size (int, optional): size of batch.
                Defaults to 32.
            batch_size (int, optional): number of returned batches.
                Defaults to 1.
        """
        self.dataloaders = dataloaders
        self.batch_size = batch_size

        self.counter = 0
        self.num_batches = num_batches
        self.n_dataloaders = len(dataloaders)

    def __iter__(self):
        return deepcopy(self)

    def __next__(self):
        if self.counter == self.num_batches:
            raise StopIteration()
        self.counter += 1
        return [self._get_single_example() for _ in range(self.batch_size)]

    def _get_single_example(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns support and query sets from one DataLoaders from
        randomly chosen from passed dataloaders.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]:
                (X_support, y_support, X_query, y_query)
        """
        dataloader_has_next = False
        while not dataloader_has_next:
            dataloader_idx = np.random.choice(self.n_dataloaders, 1)[0]
            dataloader = self.dataloaders[dataloader_idx]
            dataloader_has_next = dataloader.has_next()
        return next(dataloader)


class RepeatableOutputComposedDataLoader:
    """
    DataLoader which wraps list of data loaders objects
    and takes one observation from each of them and when
    next(dataloader) is called it returns always the same batch
    of data. Useful with test/validation datasets.
    """

    def __init__(self, dataloaders: list[Iterable], *args, **kwargs):
        """
        Args:
            dataloaders (list[Iterable]): list of
                dataloaders to sample from.
        """
        self.dataloaders = dataloaders

        self.batch_counter = 0
        self.n_dataloaders = len(dataloaders)

        self.loaded = False
        self.cache = OrderedDict()
        for i, dataloader in enumerate(self.dataloaders):
            self.cache[i] = next(dataloader)

    def __iter__(self):
        return deepcopy(self)

    def __next__(self):
        if self.loaded:
            raise StopIteration()
        self.loaded = True
        return [sample for _, sample in self.cache.items()]

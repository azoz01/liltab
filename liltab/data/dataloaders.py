import numpy as np

from copy import deepcopy
from torch import Tensor
from typing import Iterable, OrderedDict, Dict, Union

from liltab.data.datasets import PandasDataset, RandomFeaturesPandasDataset


class FewShotDataLoader:
    """
    DataLoader, which iterates dataset in few-shot learning manner
    i.e. when called next(loader) it returns episode with
    properly sampled support and query sets.
    """

    def __init__(
        self,
        dataset: Union[PandasDataset, RandomFeaturesPandasDataset],
        support_size: int,
        query_size: int,
        n_episodes: int = None,
        sample_classes_equally: bool = False,
        sample_classes_stratified: bool = False,
    ):
        """
        Args:
            dataset (Union[PandasDataset, RandomFeaturesPandasDataset]): dataset to load data from.
            support_size (int): size of support set in each episode.
            query_size (int): size of query set in each episode.
            n_episodes (int, optional): number of episodes.
                If none, then iterator is without end. Defaults to None.
            sample_classes_equally (bool, optional): If True, then in each iteration gives
                in task equal number of observations per class.
                Apply only to classification.
            sample_classes_stratified (bool, optional): If True, then in each iteration gives
                in task stratified number of observations per class.
                Apply only to classification.
        """
        self.dataset = dataset
        self.support_size = support_size
        self.query_size = query_size
        self.n_episodes = n_episodes
        self.sample_classes_equally = sample_classes_equally
        self.sample_classes_stratified = sample_classes_stratified
        if self.sample_classes_equally and self.sample_classes_stratified:
            raise ValueError("Only one of equal or stratified sampling can be used.")

        self.curr_episode = 0

        self.n_rows = len(self.dataset)

        if self.sample_classes_equally or self.sample_classes_stratified:
            self.y = dataset.raw_y
            self.class_values = np.unique(self.y)
            if len(self.class_values) > self.support_size:
                raise ValueError(
                    "When sampling equally the support size should "
                    "be higher than number of distinct values"
                )
            if len(self.class_values) > self.query_size:
                raise ValueError(
                    "When sampling equally the query size should "
                    "be higher than number of distinct values"
                )
            self.class_values_idx = dict()
            for val in self.class_values:
                self.class_values_idx[val] = np.where(self.y == val)[0]

        if sample_classes_equally:
            self.samples_per_class_support = {
                class_value: self.support_size // len(self.class_values)
                for class_value in self.class_values
            }
            self.samples_per_class_query = {
                class_value: self.query_size // len(self.class_values)
                for class_value in self.class_values
            }
        if self.sample_classes_stratified:
            self.samples_per_class_support = {
                class_value: int(self.support_size * (self.y == class_value).sum() / len(self.y))
                for class_value in self.class_values
            }
            self.samples_per_class_query = {
                class_value: int(self.query_size * (self.y == class_value).sum() / len(self.y))
                for class_value in self.class_values
            }

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

        if self.sample_classes_equally or self.sample_classes_stratified:
            return self._sample_with_custom_proportion_classes()
        else:
            return self._sample_without_stratified_classes()

    def _sample_with_custom_proportion_classes(self):
        support_indices = self._generate_stratified_sampling_idx(
            self.samples_per_class_support, self.support_size
        )
        query_indices = self._generate_stratified_sampling_idx(
            self.samples_per_class_query, self.query_size
        )
        support_indices = np.random.permutation(support_indices)
        query_indices = np.random.permutation(query_indices)
        return *self.dataset[support_indices], *self.dataset[query_indices]

    def _generate_stratified_sampling_idx(
        self, samples_per_class_dict: Dict[int, np.ndarray], set_size: int
    ) -> list[int]:
        sampled_indices = []
        for val, idx in self.class_values_idx.items():
            replace = samples_per_class_dict[val] > len(idx)
            sampled_indices.extend(
                np.random.choice(idx, samples_per_class_dict[val], replace=replace)
            )
        remaining_to_sample = set_size - len(sampled_indices)
        if remaining_to_sample > 0:
            available_idx_for_sampling = list(set(range(self.n_rows)) - set(sampled_indices))
            replace = len(available_idx_for_sampling) > remaining_to_sample
            sampled_indices.extend(
                np.random.choice(available_idx_for_sampling, remaining_to_sample, replace=replace)
            )

        return sampled_indices

    def _sample_without_stratified_classes(
        self,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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

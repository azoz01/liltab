from .datasets import PandasDataset
from .dataloaders import FewShotDataLoader, ComposedDataLoader, RepeatableOutputComposedDataLoader
from pathlib import Path
from typing import Any, Type


class ComposedDataLoaderFactory:
    """
    Factory which simplifies creation of ComposedDataLoader.
    """

    @staticmethod
    def create_composed_dataloader_from_path(
        path: Path,
        dataset_cls: Type = PandasDataset,
        dataset_creation_args: dict[str, Any] = None,
        loader_cls: Type = FewShotDataLoader,
        dataloader_creation_args: dict[str, Any] = None,
        composed_dataloader_cls: Type = ComposedDataLoader,
        batch_size: int = 32,
    ) -> ComposedDataLoader | RepeatableOutputComposedDataLoader:
        """
        Creates composed data loader using directory with csv files. Each file is wrapped in
        dataset selected dataset_cls. Next these datasets are wrapped in dataloader
        specified in loader_cls and finally it is composed to dataset specified in
        composed_dataloader_cls.

        Args:
            path (Path): path to data stored in csv format. Should contain only csv.
            dataset_cls (Type, optional): Class which will encapsulate csv data to torch dataset.
                Defaults to PandasDataset.
            dataset_creation_args (dict[str, Any], optional): Arguments passed to dataset
            constructors. Defaults to None.
            loader_cls (Type, optional): Class which will be used to encapsulate datasets.
                Defaults to FewShotDataLoader.
            dataloader_creation_args (dict[str, Any], optional): Arguments passed to dalaoader
                constructors. Defaults to None.
            composed_dataloader_cls (Type, optional): Class encapsulating all created dataloaders.
                Defaults to ComposedDataLoader.
            batch_size (int, optional): size of batch which created dataloader will return.
                Defaults to 32.

        Returns:
            ComposedDataLoader | RepeatableOutputComposedDataLoader:
                Dataloader created using aforementioned algorithm and parameters.
        """
        if not dataset_creation_args:
            dataset_creation_args = {}
        if not dataloader_creation_args:
            dataloader_creation_args = {}

        pandas_datasets = [dataset_cls(file, **dataset_creation_args) for file in path.iterdir()]
        loaders = [loader_cls(dataset, **dataloader_creation_args) for dataset in pandas_datasets]
        return composed_dataloader_cls(loaders, batch_size)

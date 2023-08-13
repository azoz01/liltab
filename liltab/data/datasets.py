import pandas as pd
import torch

from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset


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
            feature_columns if feature_columns is not None else self.df.columns.tolist()[:-1]
        )
        self.target_columns = (
            target_columns if target_columns is not None else [self.df.columns.tolist()[-1]]
        )

        self.X = torch.from_numpy(self.df[self.feature_columns].to_numpy()).type(torch.float32)
        self.y = torch.from_numpy(self.df[self.target_columns].to_numpy()).type(torch.float32)

    def __getitem__(self, idx: list[str]) -> tuple[Tensor, Tensor]:
        X = self.X[idx]
        y = self.y[idx]

        return X, y

    def __len__(self) -> int:
        return self.df.shape[0]

import pytorch_lightning as pl

from liltab.model.heterogenous_attributes_network import HeterogenousAttributesNetwork
from liltab.data.dataloaders import (
    ComposedDataLoader,
    RepeatableOutputComposedDataLoader,
)
from .utils import LightningWrapper


class HeterogenousAttributesNetworkTrainer:
    """
    Class used for traning HeterogenousAttributesNetwork.
    """

    def __init__(
        self, n_epochs: int, gradient_clipping: bool, learning_rate: float, weight_decay: float
    ):
        """
        Args:
            n_epochs (int): number of epochs to train
            gradient_clipping (bool): If true, then gradient clipping is applied
            learning_rate (float): learning rate during training.
            weight_decay (float): weight decay during training.
        """
        self.trainer = pl.Trainer(
            max_epochs=n_epochs, gradient_clip_val=1 if gradient_clipping else 0
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def train_and_test(
        self,
        model: HeterogenousAttributesNetwork,
        train_loader: ComposedDataLoader | RepeatableOutputComposedDataLoader,
        val_loader: ComposedDataLoader | RepeatableOutputComposedDataLoader,
        test_loader: ComposedDataLoader | RepeatableOutputComposedDataLoader,
    ) -> tuple[HeterogenousAttributesNetwork, list[dict[str, float]]]:
        """
        Method used to train and test model.

        Args:
            model (HeterogenousAttributesNetwork): model to train
            train_loader (ComposedDataLoader | RepeatableOutputComposedDataLoader):
                loader withTrainingData
            val_loader (ComposedDataLoader | RepeatableOutputComposedDataLoader):
                loader with validation data
            test_loader (ComposedDataLoader | RepeatableOutputComposedDataLoader):
                loader with test data

        Returns:
            tuple[HeterogenousAttributesNetwork, list[dict[str, float]]]:
                trained network with metrics on test set.
        """
        model_wrapper = LightningWrapper(
            model, learning_rate=self.learning_rate, weight_decay=self.weight_decay
        )
        self.trainer.fit(model_wrapper, train_loader, val_loader)
        test_results = self.trainer.test(model_wrapper, test_loader)
        return model_wrapper.model, test_results

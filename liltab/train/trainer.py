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

    def __init__(self, n_epochs: int, gradient_clipping: bool):
        self.trainer = pl.Trainer(
            max_epochs=n_epochs, gradient_clip_val=1 if gradient_clipping else 0
        )

    def train(
        self,
        model: HeterogenousAttributesNetwork,
        train_loader: ComposedDataLoader | RepeatableOutputComposedDataLoader,
        val_loader: ComposedDataLoader | RepeatableOutputComposedDataLoader,
    ) -> HeterogenousAttributesNetwork:
        """
        Method used to train model

        Args:
            model (HeterogenousAttributesNetwork): model to train
            train_loader (ComposedDataLoader | RepeatableOutputComposedDataLoader):
                loader withTrainingData
            val_loader (ComposedDataLoader | RepeatableOutputComposedDataLoader):
                loader with validation data

        Returns:
            HeterogenousAttributesNetwork: trained network
        """
        model_wrapper = LightningWrapper(model)
        self.trainer.fit(model_wrapper, train_loader, val_loader)
        return model_wrapper.model

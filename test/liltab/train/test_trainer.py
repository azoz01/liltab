from liltab.model.heterogenous_attributes_network import HeterogenousAttributesNetwork
from liltab.data.dataloaders import ComposedDataLoader, FewShotDataLoader
from liltab.data.datasets import PandasDataset
from liltab.train.trainer import HeterogenousAttributesNetworkTrainer

from copy import deepcopy


def test_lighting_wrapper(resources_path):
    dataloader_1 = ComposedDataLoader(
        [
            FewShotDataLoader(
                PandasDataset(resources_path / "random_df_1.csv"),
                4,
                6,
                n_episodes=10,
            ),
            FewShotDataLoader(
                PandasDataset(resources_path / "random_df_2.csv"),
                4,
                6,
                n_episodes=10,
            ),
        ],
        batch_size=10,
    )

    dataloader_2 = deepcopy(dataloader_1)
    dataloader_3 = deepcopy(dataloader_2)

    model = HeterogenousAttributesNetwork()

    trainer = HeterogenousAttributesNetworkTrainer(
        n_epochs=1,
        gradient_clipping=True,
        learning_rate=1e-3,
        weight_decay=0.1,
        file_logger=False,
        tb_logger=False,
        model_checkpoints=False,
    )
    trainer.train_and_test(model, dataloader_1, dataloader_2, dataloader_3)

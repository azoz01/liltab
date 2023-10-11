import numpy as np

from liltab.data.datasets import PandasDataset
from liltab.data.dataloaders import (
    FewShotDataLoader,
    ComposedDataLoader,
    RepeatableOutputComposedDataLoader,
)


def test_few_shot_data_loader_returns_proper_episodes_count(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    n_episodes = 100
    dataset = PandasDataset(frame_path)
    dataloader = FewShotDataLoader(dataset, 2, 8, n_episodes)

    episodes = list(dataloader)
    assert len(episodes) == n_episodes


def test_few_shot_data_loader_returns_proper_shape_tensors(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    dataset = PandasDataset(frame_path)
    dataloader = FewShotDataLoader(dataset, 4, 6)
    n_features = len(dataset.feature_columns)

    X_support, y_support, X_query, y_query = next(dataloader)

    assert X_support.shape == (4, n_features)
    assert y_support.shape == (4, 1)
    assert X_query.shape == (6, n_features)
    assert y_query.shape == (6, 1)


def test_few_shot_data_loader_returns_disjoint_tensors(resources_path, utils):
    frame_path = resources_path / "random_df_1.csv"
    dataset = PandasDataset(frame_path)
    dataloader = FewShotDataLoader(dataset, 4, 6, n_episodes=10)

    for episode in dataloader:
        X_support, y_support, X_query, y_query = episode

        assert not utils.tensors_have_common_rows(X_support, X_query)
        assert not utils.tensors_have_common_rows(y_support, y_query)


def test_few_shot_data_loader_has_next(resources_path):
    frame_path = resources_path / "random_df_1.csv"
    dataset = PandasDataset(frame_path)
    dataloader = FewShotDataLoader(dataset, 4, 6, n_episodes=10)

    assert dataloader.has_next()
    for _ in range(10):
        next(dataloader)
    assert not dataloader.has_next()


def test_composed_data_loader_returns_batch_with_proper_size(resources_path):
    dataloader = ComposedDataLoader(
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
    loaded_dataset = list(dataloader)[0]
    assert len(loaded_dataset) == 10


def test_composed_data_loader_returns_proper_batch_count(resources_path):
    dataloader = ComposedDataLoader(
        [
            FewShotDataLoader(
                PandasDataset(resources_path / "random_df_1.csv"),
                4,
                6,
                n_episodes=100,
            ),
            FewShotDataLoader(
                PandasDataset(resources_path / "random_df_2.csv"),
                4,
                6,
                n_episodes=100,
            ),
        ],
        batch_size=10,
        num_batches=5,
    )
    loaded_dataset = list(dataloader)
    assert len(loaded_dataset) == 5


def test_composed_data_loader_returns_from_all_loaders_properly(
    resources_path,
):
    n_episodes = 11
    dataloader = ComposedDataLoader(
        [
            FewShotDataLoader(
                PandasDataset(resources_path / "random_df_1.csv"),
                4,
                6,
                n_episodes=4,
            ),
            FewShotDataLoader(
                PandasDataset(resources_path / "random_df_2.csv"),
                4,
                6,
                n_episodes=7,
            ),
        ],
        batch_size=n_episodes,
    )
    loaded_dataset = list(dataloader)
    batches_lens = np.array(list(map(lambda t: t[0][1].shape[0], loaded_dataset[0])))

    assert (batches_lens == 4).sum() == 4
    assert (batches_lens == 2).sum() == 7


def test_composed_data_loader_returns_from_all_loaders_almost_equally(
    resources_path,
):
    def run_experiment():
        batch_size = 32
        dataloader = ComposedDataLoader(
            [
                FewShotDataLoader(
                    PandasDataset(resources_path / "random_df_1.csv"),
                    4,
                    6,
                    n_episodes=100,
                ),
                FewShotDataLoader(
                    PandasDataset(resources_path / "random_df_2.csv"),
                    4,
                    6,
                    n_episodes=100,
                ),
            ],
            batch_size=batch_size,
        )
        batch = list(dataloader)[0]
        batches_lens = np.array(list(map(lambda t: t[0][1].shape[0], batch)))
        return [(batches_lens == 4).sum(), (batches_lens == 2).sum()]

    experiments_results = np.array([run_experiment() for _ in range(20)]).mean(axis=0)
    assert (experiments_results >= 14).all()


def test_repeatable_output_composed_data_loader_repeat_samples(resources_path, utils):
    dataloader = RepeatableOutputComposedDataLoader(
        [
            FewShotDataLoader(
                PandasDataset(resources_path / "random_df_1.csv"),
                4,
                6,
                n_episodes=100,
            ),
            FewShotDataLoader(
                PandasDataset(resources_path / "random_df_2.csv"),
                4,
                6,
                n_episodes=100,
            ),
        ]
    )

    dataloader_1 = list(dataloader)[0]
    dataloader_2 = list(dataloader)[0]

    for sample_1, sample_2 in zip(dataloader_1, dataloader_2):
        X_support_1, y_support_1, X_query_1, y_query_1 = sample_1
        X_support_2, y_support_2, X_query_2, y_query_2 = sample_2
        assert utils.tensors_have_common_rows(X_support_1, X_support_2)
        assert utils.tensors_have_common_rows(y_support_1, y_support_2)
        assert utils.tensors_have_common_rows(X_query_1, X_query_2)
        assert utils.tensors_have_common_rows(y_query_1, y_query_2)

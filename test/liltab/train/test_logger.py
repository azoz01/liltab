import torch

from liltab.train.logger import FileLogger, TensorBoardLogger


def test_file_logger_init_folders(results_path):
    logger_dir = results_path / "csv_logger"
    version = "test_1"
    _ = FileLogger(save_dir=logger_dir, version=version)

    test_file = logger_dir / version / "test.csv"
    train_file = logger_dir / version / "train.csv"
    validate_file = logger_dir / version / "validate.csv"

    assert test_file.exists()
    assert train_file.exists()
    assert validate_file.exists()


def test_tb_logger_init_folders(results_path):
    logger_dir = results_path / "tb_logger"
    version = "test"
    logger = TensorBoardLogger(save_dir=logger_dir, version=version)
    logger.log_train_value(torch.Tensor([1]))

    # dictionary created after first call
    logger_full_dir = logger_dir / version

    assert logger_full_dir.exists()


def test_file_logger_insert_to_file(results_path):
    logger_dir = results_path / "csv_logger"
    version = "test_2"
    logger = FileLogger(save_dir=logger_dir, version=version)
    train_file = logger_dir / version / "train.csv"

    values_inserted = 99

    for _ in range(values_inserted):
        logger.log_train_value(torch.Tensor([1]))

    with open(train_file, "r") as f:
        assert len(f.readlines()) == values_inserted

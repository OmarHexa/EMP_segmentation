from pathlib import Path

import pytest
import torch

from src.dataset.EMP_datamodule import EmpDataModule


@pytest.mark.parametrize("batch_size", [32, 64])
def test_emp_datamodule(batch_size: int) -> None:
    """Tests `EMPDataModule` to verify that the necessary attributes were created (e.g., the
    dataloader objects), and that dtypes and batch sizes correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "/home/omar/code/pytorch/EMP_data/"
    assert Path(data_dir).exists()

    dm = EmpDataModule(data_dir=data_dir, batch_size=batch_size, train_split=0.8)

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "images").exists()
    assert Path(data_dir, "segmaps").exists()

    dm.setup()
    assert dm.data_train and dm.data_val
    assert dm.train_dataloader() and dm.val_dataloader()

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int32

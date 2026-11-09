from typing import Optional
import os
import numpy as np
import lightning as L
from torchvision.datasets import CIFAR100
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset import CIFAR100Dataset, get_default_transforms, build_dataloader


class CIFAR100DataModule(L.LightningDataModule):
    """
    LightningDataModule encapsulating CIFAR-100 data download, train/val split,
    preprocessing, and DataLoader creation.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        val_split: float = 0.2,
        batch_size: int = 64,
        num_workers: int = 2,
        pin_memory: bool = True,
        augment: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augment = augment
        self.seed = seed

        self.train_dataset: Optional[CIFAR100Dataset] = None
        self.val_dataset: Optional[CIFAR100Dataset] = None
        self.test_dataset: Optional[CIFAR100Dataset] = None

    def prepare_data(self) -> None:
        """Download CIFAR-100 dataset if not already present."""
        os.makedirs(self.data_dir, exist_ok=True)
        CIFAR100(root=self.data_dir, train=True, download=True)
        CIFAR100(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up train, validation, and test datasets."""
        train_transform, eval_transform = get_default_transforms(augment=self.augment)

        if stage in ("fit", None):
            raw_train = CIFAR100(root=self.data_dir, train=True, download=False)
            data = raw_train.data
            targets = np.array(raw_train.targets)

            # Stratified split to ensure equal class distribution across train and validation
            train_idx, val_idx = train_test_split(
                np.arange(len(targets)),
                test_size=self.val_split,
                random_state=self.seed,
                stratify=targets,
            )

            self.train_dataset = CIFAR100Dataset(
                data=data[train_idx],
                targets=targets[train_idx].tolist(),
                transform=train_transform,
            )

            self.val_dataset = CIFAR100Dataset(
                data=data[val_idx],
                targets=targets[val_idx].tolist(),
                transform=eval_transform,
            )

        if stage in ("test", None):
            raw_test = CIFAR100(root=self.data_dir, train=False, download=False)
            self.test_dataset = CIFAR100Dataset(
                data=raw_test.data,
                targets=raw_test.targets,
                transform=eval_transform,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule setup('fit') must be called before requesting train_dataloader.")
        return build_dataloader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("DataModule setup('fit') must be called before requesting val_dataloader.")
        return build_dataloader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("DataModule setup('test') must be called before requesting test_dataloader.")
        return build_dataloader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

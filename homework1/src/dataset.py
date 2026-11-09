from typing import Optional, Callable, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as T
from PIL import Image


class CIFAR100Dataset(Dataset):
    """
    Custom PyTorch Dataset for CIFAR-100 image classification.
    Wraps raw images and integer class labels with custom transformations.
    """

    def __init__(
        self,
        data: np.ndarray,
        targets: list[int] | np.ndarray | torch.Tensor,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Args:
            data: ndarray of shape (N, 32, 32, 3) representing RGB images.
            targets: array-like of integer class labels (0..99).
            transform: Optional callable transformation applied to PIL Image / Tensor.
            target_transform: Optional callable transformation applied to label.
        """
        self.data = np.asarray(data)
        self.targets = list(targets)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Returns a single sample (image, target).
        """
        img_arr = self.data[idx]
        target = self.targets[idx]

        # Convert to PIL Image for torchvision transforms consistency
        img = Image.fromarray(img_arr)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_default_transforms(augment: bool = True) -> Tuple[Callable, Callable]:
    """
    Returns train and evaluation transforms with CIFAR-100 normalization stats.
    """
    # CIFAR-100 mean and std channel stats
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    if augment:
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return train_transform, eval_transform


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: bool = True,
) -> DataLoader:
    """
    Constructs a PyTorch DataLoader for a given Dataset.
    """
    use_persistent = persistent_workers if num_workers > 0 else False
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=use_persistent,
    )

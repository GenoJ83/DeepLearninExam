"""CIFAR-10 loading, normalization, train/val/test splits, and augmentation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def get_transforms(
    augment_train: bool = True,
    use_rotation: bool = False,
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Training: optional random crop (with padding), horizontal flip, optional rotation.
    Val/test: only ToTensor + normalize (no leakage of augmentation statistics).
    """
    aug_list: list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if use_rotation:
        aug_list.append(transforms.RandomRotation(degrees=10))
    aug_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    train_tf = transforms.Compose(aug_list) if augment_train else transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_tf, eval_tf


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_fraction: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
    augment_train: bool = True,
    use_rotation: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Official CIFAR-10: 50k train images, 10k test.
    Split the 50k into train (1 - val_fraction) and validation (val_fraction).
    Test loader uses the official test set only for final evaluation.
    """
    train_tf, eval_tf = get_transforms(augment_train=augment_train, use_rotation=use_rotation)

    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_tf)

    n_total = len(full_train)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_train,
        [n_train, n_val],
        generator=generator,
    )

    # Validation must use eval transforms (no augmentation)
    val_indices = val_subset.indices
    base_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=eval_tf)
    val_eval = Subset(base_train, val_indices)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader, test_loader

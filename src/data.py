import torch
from functools import lru_cache
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


@lru_cache(maxsize=None)
def _get_cifar10_datasets(root: str = "E:/ml_datasets", eval_mode=False):
    if eval_mode:
        train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_test)
    else:
        train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)


    # Load raw test dataset without transforms, then apply once
    raw_test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=None)
    test_imgs = []
    test_labels = []
    for img, label in raw_test_ds:
        test_imgs.append(transform_test(img))  # apply deterministic transform once
        test_labels.append(label)
    test_tensor = torch.stack(test_imgs)
    label_tensor = torch.tensor(test_labels)
    test_ds = TensorDataset(test_tensor, label_tensor)

    return train_ds, test_ds


def get_cifar10_loaders(
    batch_size: int,
    eval_batch_size: int = 10000,
    root: str = "E:/ml_datasets",
    num_workers: int = 1,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    eval_mode=False
):
    if num_workers == 0 and persistent_workers:
        persistent_workers = False

    train_ds, test_ds = _get_cifar10_datasets(root, eval_mode=eval_mode)

    shuffle_train = not eval_mode

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,  # no need for workers here
    )
    return train_loader, test_loader

# ---------- CIFAR-100 transforms ----------
transform_train_100 = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # CIFAR images are 3-channel; normalize to roughly [-1, 1] like your CIFAR-10 code
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

transform_test_100 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


@lru_cache(maxsize=None)
def _get_cifar100_datasets(root: str = "E:/ml_datasets", eval_mode: bool = False):
    """
    Returns:
        (train_ds, test_ds) where:
          - train_ds is a torchvision.datasets.CIFAR100 with online transforms.
            If eval_mode=True, it uses transform_test_100 (no augmentation).
          - test_ds is a TensorDataset with test transforms applied once (deterministic).
    """
    # Train set: switch transform based on eval_mode
    train_tf = transform_test_100 if eval_mode else transform_train_100
    train_ds = datasets.CIFAR100(root=root, train=True, download=True, transform=train_tf)

    # Test set: load raw, apply deterministic transform once, cache as tensors
    raw_test_ds = datasets.CIFAR100(root=root, train=False, download=True, transform=None)
    test_imgs, test_labels = [], []
    for img, label in raw_test_ds:
        test_imgs.append(transform_test_100(img))  # deterministic
        test_labels.append(label)

    test_tensor = torch.stack(test_imgs)                # [10000, 3, 32, 32]
    label_tensor = torch.tensor(test_labels, dtype=torch.long)  # [10000]
    test_ds = TensorDataset(test_tensor, label_tensor)

    return train_ds, test_ds


def get_cifar100_loaders(
    batch_size: int,
    eval_batch_size: int = 1024,   # default matches docstring
    root: str = "E:/ml_datasets",
    num_workers: int = 1,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    eval_mode: bool = False,
):
    """
    Create DataLoaders for CIFAR-100 mirroring your CIFAR-10 loaders.

    Args:
        batch_size: Training batch size.
        eval_batch_size: Test/eval batch size (default 1024).
        root: Dataset root directory.
        num_workers: Dataloader workers for training.
        pin_memory: Pin CPU memory for faster H2D transfer.
        persistent_workers: Keep workers alive across epochs (ignored if num_workers == 0).
        eval_mode: If True, use test transforms for train_ds and disable train shuffling.
    """
    # Guard: persistent_workers requires num_workers > 0
    persistent_workers = persistent_workers and (num_workers > 0)

    train_ds, test_ds = _get_cifar100_datasets(root, eval_mode=eval_mode)

    shuffle_train = not eval_mode
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # Test loader: tensor-backed dataset; keep workers at 0 to avoid extra copies
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, test_loader

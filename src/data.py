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
def _get_cifar10_datasets(root: str = "E:/ml_datasets"):
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
):
    if num_workers == 0 and persistent_workers:
        persistent_workers = False

    train_ds, test_ds = _get_cifar10_datasets(root)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
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

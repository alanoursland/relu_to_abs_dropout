import torch
import torch.nn as nn
import torch.optim as optim

from config import experiment, get_experiment_config, ExperimentConfig
from resnet18 import ReLU2AbsDropout, ReLUDropout, ReLUMixedAbsDropout, ResNet18_CIFAR10
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from data import get_cifar10_loaders


def fn_resnet18_cifar10(activation):
    return lambda: ResNet18_CIFAR10(num_classes=10, activation=activation)


@experiment()
def cifar10_baseline() -> ExperimentConfig:
    """Factory function for absolute value XOR experiment."""
    batch_size = 128
    dropout_rate = 0.0
    epochs = 150
    stop_delta_loss = 1e-3
    stop_delta_patience = 20
    learning_rate = 0.001

    # CIFAR-10 dataset
    # Data transforms for CIFAR-10
    train_loader, test_loader = get_cifar10_loaders(batch_size, num_workers=1)

    config = ExperimentConfig()
    config.description = "Baseline resnet18 for CIFAR-10"

    config.num_runs = 15
    config.random_seeds = [3553]
    config.train_loader = train_loader
    config.test_loader = test_loader
    config.epochs = epochs
    config.stop_delta_loss = stop_delta_loss
    config.stop_delta_patience = stop_delta_patience
    config.model_fn = fn_resnet18_cifar10(activation=ReLUDropout(dropout_rate=dropout_rate))
    config.optimizer_fn = lambda model: optim.Adam(model.parameters(), lr=learning_rate)
    config.criterion = nn.CrossEntropyLoss()
    return config

@experiment()
def cifar10_std_dropout_5em3() -> ExperimentConfig:
    dropout_rate = 5e-3
    config = get_experiment_config("cifar10_baseline")
    config.description = "Resnet18 with standard dropout 5e-3 for CIFAR-10"
    config.model_fn = fn_resnet18_cifar10(activation=ReLUDropout(dropout_rate=dropout_rate))
    return config

@experiment()
def cifar10_std_dropout_1em2() -> ExperimentConfig:
    dropout_rate = 1e-2
    config = get_experiment_config("cifar10_baseline")
    config.description = "Resnet18 with standard dropout 1e-2 for CIFAR-10"
    config.model_fn = fn_resnet18_cifar10(activation=ReLUDropout(dropout_rate=dropout_rate))
    return config

@experiment()
def cifar10_std_dropout_2em2() -> ExperimentConfig:
    dropout_rate = 2e-2
    config = get_experiment_config("cifar10_baseline")
    config.description = "Resnet18 with standard dropout 2e-2 for CIFAR-10"
    config.model_fn = fn_resnet18_cifar10(activation=ReLUDropout(dropout_rate=dropout_rate))
    return config

@experiment()
def cifar10_std_dropout_3em2() -> ExperimentConfig:
    dropout_rate = 3e-2
    config = get_experiment_config("cifar10_baseline")
    config.description = "Resnet18 with standard dropout 3e-2 for CIFAR-10"
    config.model_fn = fn_resnet18_cifar10(activation=ReLUDropout(dropout_rate=dropout_rate))
    return config

@experiment()
def cifar10_abs_dropout_5em3() -> ExperimentConfig:
    dropout_rate = 5e-3
    config = get_experiment_config("cifar10_baseline")
    config.description = "Resnet18 with abs dropout 5e-3 for CIFAR-10"
    config.model_fn = fn_resnet18_cifar10(activation=ReLU2AbsDropout(dropout_rate=dropout_rate))
    return config

@experiment()
def cifar10_abs_dropout_1em2() -> ExperimentConfig:
    dropout_rate = 1e-2
    config = get_experiment_config("cifar10_baseline")
    config.description = "Resnet18 with abs dropout 1e-2 for CIFAR-10"
    config.model_fn = fn_resnet18_cifar10(activation=ReLU2AbsDropout(dropout_rate=dropout_rate))
    return config

@experiment()
def cifar10_abs_dropout_2em2() -> ExperimentConfig:
    dropout_rate = 2e-2
    config = get_experiment_config("cifar10_baseline")
    config.description = "Resnet18 with abs dropout 2e-2 for CIFAR-10"
    config.model_fn = fn_resnet18_cifar10(activation=ReLU2AbsDropout(dropout_rate=dropout_rate))
    return config

@experiment()
def cifar10_abs_dropout_3em2() -> ExperimentConfig:
    dropout_rate = 3e-2
    config = get_experiment_config("cifar10_baseline")
    config.description = "Resnet18 with abs dropout 3e-2 for CIFAR-10"
    config.model_fn = fn_resnet18_cifar10(activation=ReLU2AbsDropout(dropout_rate=dropout_rate))
    return config

@experiment()
def cifar10_mixed_dropout_2em2() -> ExperimentConfig:
    dropout_rate = 2e-2
    config = get_experiment_config("cifar10_baseline")
    config.description = "Resnet18 with standard dropout 2e-2 and abs dropout 2e-2 for CIFAR-10"
    config.model_fn = fn_resnet18_cifar10(activation=ReLUMixedAbsDropout(dropout_rate=dropout_rate))
    return config


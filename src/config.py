import torch

from dataclasses import dataclass, field
from typing import Dict, Callable, Optional, List
from torch.utils.data import DataLoader


@dataclass
class ExperimentConfig:
    name: str = "<ERROR: unnamed experiment>"
    description: str = ""
    output_dir: str = "results"  # f"{output_dir}/{name}/{run_label}"

    num_runs: int = 1
    random_seeds: Optional[List[int]] = None
    continue_from: Optional[str] = None  # continues training from another experiment config
    load_optimizer_state: bool = False  # should continue_from load the optimizer state
    device: str = "auto"

    train_loader: DataLoader = None  # loader contain the batch size
    test_loader: DataLoader = None

    epochs: int = 0
    stop_delta_loss: float = 1e-3
    stop_delta_patience: int = 0
    model_fn: Callable[[], torch.nn.Module] = field(default=None)
    optimizer_fn: Callable[[torch.nn.Module], torch.optim.Optimizer] = None
    criterion: torch.nn.Module = None  # loss function


experiments: Dict[str, Callable[[], ExperimentConfig]] = {}


def get_experiment_config(name: str) -> ExperimentConfig:
    """
    Retrieve experiment configuration by name.

    Args:
        name: Name of experiment configuration

    Returns:
        Complete experiment configuration
    """
    print(f"Retrieving experiment {name}")

    if name not in experiments:
        available = list(experiments.keys())
        raise KeyError(f"Unknown experiment '{name}'. Available experiments: {available}")

    # Call the factory function to create the config
    config_factory = experiments[name]
    config = config_factory()

    return config


def experiment(name: str = None):
    """Decorator to register and label experiment configuration functions."""

    def decorator(func: Callable[[], ExperimentConfig]):
        exp_name = name or func.__name__  # fall back to function name
        print(f"Registering experiment {exp_name}")

        def wrapped_func():
            config = func()
            config.name = exp_name
            return config

        experiments[exp_name] = wrapped_func
        return wrapped_func

    return decorator

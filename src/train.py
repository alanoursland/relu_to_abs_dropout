import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import os
import traceback
from config import ExperimentConfig, get_experiment_config
from results import get_experimental_results, ExperimentalResults
from typing import Dict, Callable, Optional, List, Any
import inspect
import gc

from resnet18 import ReLU2AbsDropout, ResNet18_CIFAR10


def _resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _set_seed(seed: int):
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # keep cuDNN defaults (fast) unless you want strict determinism


def _callable_name(fn: Optional[Callable]) -> Optional[str]:
    if fn is None:
        return None
    try:
        if hasattr(fn, "__name__"):
            # named function or lambda
            nm = fn.__name__
            if nm == "<lambda>":
                # include source if short; else just say "<lambda>"
                try:
                    src = inspect.getsource(fn).strip()
                    return f"<lambda>: {src}"[:200]
                except OSError:
                    return "<lambda>"
            return nm
        # partials or callables with __class__.__name__
        return getattr(fn, "__qualname__", fn.__class__.__name__)
    except Exception:
        return repr(fn)[:200]


def _extract_loader_info(train_loader) -> Dict[str, Any]:
    if train_loader is None:
        return {}
    info = {
        "dataset": getattr(getattr(train_loader, "dataset", None), "__class__", type("X", (object,), {})).__name__,
        "train_batch_size": getattr(train_loader, "batch_size", None),
        "num_workers": getattr(train_loader, "num_workers", None),
        "pin_memory": getattr(train_loader, "pin_memory", None),
        "persistent_workers": getattr(train_loader, "persistent_workers", None),
    }
    return info


class SimpleTimer:
    def __init__(self):
        self.start_time = time.time()
        self.last_tick = self.start_time

    def tick(self, log=False):
        now = time.time()
        total_elapsed = now - self.start_time
        delta_elapsed = now - self.last_tick
        self.last_tick = now
        if log:
            print(f"Total elapsed: {total_elapsed:.2f} s | Since last tick: {delta_elapsed:.2f} s")
        return delta_elapsed

    def elapsed(self):
        now = time.time()
        total_elapsed = now - self.start_time
        return total_elapsed


def load_standard_resnet18():
    # Load ResNet-18 and adjust for CIFAR-10 (10 classes, smaller images)
    model = models.resnet18(weights=None)

    # This is a standard adaptation for CIFAR-10, widely used in literature (e.g., original ResNet paper’s CIFAR experiments). It maintains the 64 output channels, ensuring compatibility with the rest of the architecture.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # This is a common practice for CIFAR-10 with ResNet. The original ResNet paper (He et al., 2016) and many implementations (e.g., PyTorch community models) skip max pooling for small inputs to avoid excessive downsampling.
    model.maxpool = nn.Identity()

    # This is standard and correct for adapting ResNet-18 to CIFAR-10.
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def eval_and_save_predictions(model, loader, device, pred_results_obj):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            preds.append(model(xb).detach().cpu())
    full_preds = torch.cat(preds, dim=0)
    pred_results_obj.set_from_tensor(full_preds)
    pred_results_obj.write()


def save_init_predictions(run_results, model, train_loader, device):
    run_results.load_init_model(model)  # from model_init.pth
    eval_and_save_predictions(model, train_loader, device, run_results.predictions_init)


def save_train_predictions(run_results, model, train_loader, device):
    run_results.load_final_model(model)
    eval_and_save_predictions(model, train_loader, device, run_results.predictions_train)


def save_test_predictions(run_results, outputs, device):
    run_results.predictions_test.set_from_tensor(outputs.detach().cpu())
    run_results.predictions_test.write()


def save_best_predictions(run_results, model, train_loader, device):
    run_results.load_best_model(model)
    eval_and_save_predictions(model, train_loader, device, run_results.predictions_best)

def evaluate_test_batched(model, test_inputs, test_labels, criterion, batch_size=2000, device='cuda'):
    """
    Evaluate test set in batches to avoid memory explosion.
    Returns: test_loss, test_outputs, test_acc
    """
    model.eval()
    total_loss = 0.0
    all_outputs = []
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for i in range(0, test_inputs.size(0), batch_size):
            # Get batch slice
            batch_inputs = test_inputs[i:i+batch_size]
            batch_labels = test_labels[i:i+batch_size]
            
            # Forward pass
            batch_outputs = model(batch_inputs)
            batch_loss = criterion(batch_outputs, batch_labels)
            
            # Accumulate results
            total_loss += batch_loss.item() * batch_inputs.size(0)
            all_outputs.append(batch_outputs)
            
            # Accuracy calculation
            _, predicted = batch_outputs.max(1)
            total_correct += predicted.eq(batch_labels).sum().item()
            total_samples += batch_labels.size(0)
    
    # Combine all outputs
    test_outputs = torch.cat(all_outputs, dim=0)
    test_loss = total_loss / total_samples
    test_acc = 100.0 * total_correct / total_samples
    
    return test_loss, test_outputs, test_acc

def run_training_loop(
    run_idx,
    epochs,
    stop_delta_loss,
    stop_delta_patience,
    model,
    optimizer,
    criterion,
    train_loader,
    test_inputs,
    test_labels,
    device,
    timer,
    run_results,
):
    """
    Early stop on TRAIN loss:
      - Track best train_loss.
      - If best_train_loss - train_loss < stop_delta_loss for `stop_delta_patience` consecutive epochs, stop.
      - When stop_delta_patience == 0, early stopping is disabled.
    """
    timer.tick()
    print("Starting training loop")

    best_train_loss = float("inf")
    no_improve = 0

    run_results.save_init_model(model, optimizer)

    best_train_acc = 0.0

    epoch_timer = SimpleTimer()
    data_timer = SimpleTimer()
    compute_timer = SimpleTimer()
    eval_timer = SimpleTimer()

    # print_mem(f"run{run_idx}-epoch{0}-start")
    # torch.cuda.empty_cache()

    # --- Epoch 0 (pre-training) evaluation ---
    print("Evaluating initial state")
    model.eval()
    with torch.inference_mode():
        # Test (using the preloaded test_inputs/test_labels if you already have them)
        test_loss0, _, test_acc0 = evaluate_test_batched(
            model, 
            test_inputs, 
            test_labels, 
            criterion, 
            batch_size=2000, 
            device=device)

        # Train eval (fast pass, same transforms as train loader)
        running_loss0, correct0, total0 = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(xb)
            running_loss0 += criterion(out, yb).item() * xb.size(0)
            _, pr = out.max(1)
            total0 += yb.size(0)
            correct0 += pr.eq(yb).sum().item()
        train_loss0 = running_loss0 / total0
        train_acc0 = 100.0 * correct0 / total0

        print(
            f"Epoch [{0}/{epochs}] | Train Loss: {train_loss0:.4f} | "
            f"Train Acc: {train_acc0:.2f}% | Test Acc: {test_acc0:.2f}%"
        )
        # Write this as the first stats row so analysis picks it up as "init"
        run_results.stats.add_epoch(train_loss0, test_loss0, train_acc0, test_acc0)
    # print_mem(f"run{run_idx}-epoch{0}-init")
    # torch.cuda.empty_cache()

    print("Starting epoch loop")
    for epoch in range(epochs):
        epoch_timer.tick()
        data_sec = 0.0
        compute_sec = 0.0
        eval_sec = 0.0

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # print_mem(f"run{run_idx}-epoch{epoch+1}-train")
        torch.cuda.empty_cache()
        for inputs, labels in train_loader:
            data_timer.tick()
            inputs, labels = inputs.to(device), labels.to(device)
            data_sec += data_timer.tick()

            compute_timer.tick()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            compute_sec += compute_timer.tick()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # print_mem(f"run{run_idx}-epoch{epoch+1}-test")
        torch.cuda.empty_cache()
        # if device.type == "cuda":
        #     torch.cuda.synchronize()
        timer.tick()
        eval_timer.tick()
        print("Evaluating on test")
        test_loss, _, test_acc = evaluate_test_batched(
            model, 
            test_inputs, 
            test_labels, 
            criterion, 
            batch_size=2000, 
            device=device)

        # model.eval()
        # with torch.no_grad():
        #     test_outputs = model(test_inputs)
        #     test_loss = criterion(test_outputs, test_labels).item()
        #     _, predicted = test_outputs.max(1)
        #     test_correct = predicted.eq(test_labels).sum().item()
        #     test_total = test_labels.size(0)
        # test_acc = 100.0 * test_correct / test_total
        # if device.type == "cuda":
        #     torch.cuda.synchronize()
        eval_sec += eval_timer.tick(log=True)

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            run_results.save_best_model(model, optimizer)
            run_results.stats.set_best(train_loss, test_loss, train_acc, test_acc)

        run_results.stats.add_epoch(train_loss, test_loss, train_acc, test_acc)

        timer.tick()
        print(
            f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%"
        )
        run_results.timing.add_epoch_timing(epoch + 1, epoch_timer.tick(), data_sec, compute_sec, eval_sec)

        # ----- Early stopping on train loss -----
        if stop_delta_patience > 0:
            if best_train_loss - train_loss >= stop_delta_loss:
                best_train_loss = train_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= stop_delta_patience:
                    print(
                        f"Early stopping: no train-loss improvement ≥ {stop_delta_loss} "
                        f"for {stop_delta_patience} epochs (best={best_train_loss:.4f})."
                    )
                    break
        # print_mem(f"run{run_idx}-epoch{epoch+1}-end")
        torch.cuda.empty_cache()


    run_results.stats.set_final(train_loss, test_loss, train_acc, test_acc)
    run_results.save_final_model(model, optimizer)

    # save_init_predictions(run_results, model, train_loader, device)
    # save_train_predictions(run_results, model, train_loader, device)
    # save_test_predictions(run_results, test_outputs, device)
    # save_best_predictions(run_results, model, train_loader, device)

    run_results.stats.write()
    run_results.timing.write()


def validate_experiment(config: ExperimentConfig):
    # basic validations (fail fast, as discussed)
    if config.epochs < 0:
        raise ValueError("epochs must be > 0")
    if config.model_fn is None:
        raise ValueError("model_fn must be provided")
    if config.optimizer_fn is None:
        raise ValueError("optimizer_fn must be provided")
    if config.train_loader is None or config.test_loader is None:
        raise ValueError("train_loader and test_loader must be provided")
    if config.continue_from is not None:
        raise NotImplementedError("continue_from is not implemented here; results.py will handle it.")


def resolve_random_seeds(config: ExperimentConfig):
    # resolve seeds
    if config.random_seeds is None:
        import os

        seeds = [int.from_bytes(os.urandom(4), "little") for _ in range(config.num_runs)]
    else:
        seeds = list(config.random_seeds)
        if len(seeds) < config.num_runs:
            if not seeds:
                raise ValueError("random_seeds is empty and cannot be extended")
            last_seed = seeds[-1]
            while len(seeds) < config.num_runs:
                last_seed += 1
                seeds.append(last_seed)
        elif len(seeds) > config.num_runs:
            seeds = seeds[: config.num_runs]
    return seeds


def build_results_metadata(results: ExperimentalResults, config: ExperimentConfig, seeds):
    metadata = results.metadata
    metadata.name = config.name
    metadata.description = config.description or ""
    metadata.num_runs = config.num_runs
    metadata.random_seeds = seeds
    metadata.status = "in_progress"
    # lightweight environment snapshot
    import platform, torch

    metadata.framework_versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": getattr(__import__("torchvision"), "__version__", None),
    }
    metadata.device_info = {
        "requested": config.device,
        "available_cuda": torch.cuda.is_available(),
        "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
    }


def build_results_config(results: ExperimentalResults, config: "ExperimentConfig", seeds: List[int]):
    """
    Populate results.config from the live ExperimentConfig (serializable view only).
    Does NOT write by itself; call results.config.write() when ready.
    """
    cfg = results.config
    cfg.name = config.name
    cfg.description = config.description or ""
    cfg.output_dir = config.output_dir

    cfg.num_runs = config.num_runs
    cfg.random_seeds = list(seeds) if seeds is not None else None
    cfg.continue_from = config.continue_from
    cfg.load_optimizer_state = bool(getattr(config, "load_optimizer_state", False))
    cfg.device = config.device

    cfg.epochs = config.epochs
    cfg.stop_delta_loss = float(getattr(config, "stop_delta_loss", getattr(config, "stop_loss", 1e-3)))
    cfg.stop_delta_patience = int(getattr(config, "stop_delta_patience", getattr(config, "stop_patience", 0)))

    cfg.model_fn_repr = _callable_name(getattr(config, "model_fn", None))
    cfg.optimizer_fn_repr = _callable_name(getattr(config, "optimizer_fn", None))
    # criterion can be a module instance; capture its class name
    crit = getattr(config, "criterion", None)
    cfg.criterion_repr = None if crit is None else crit.__class__.__name__

    cfg.dataset_info = _extract_loader_info(getattr(config, "train_loader", None))
    # add test batch size if available
    test_loader = getattr(config, "test_loader", None)
    if test_loader is not None:
        cfg.dataset_info = cfg.dataset_info or {}
        cfg.dataset_info["test_batch_size"] = getattr(test_loader, "batch_size", None)


def print_mem(tag=""):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated()/1024**2
    r = torch.cuda.memory_reserved()/1024**2
    print(f"[{tag}] alloc={a:.1f}MB  reserved={r:.1f}MB")

def run_experiment(config: ExperimentConfig, start_run=0):
    """
    Minimal runner adapted from run_experiment_hardcoded().
    - Builds a fresh model/optimizer per run.
    - Uses the provided loaders as-is.
    - Preloads the (single-batch) test tensors to device and evaluates like your hardcoded script.
    - Fast-fails on continue_from until results.py handles it.
    """
    results = get_experimental_results(config.name, create_mode=True)
    results.start_log()

    device = _resolve_device(config.device)
    print(f"Device: {device}")

    validate_experiment(config)
    seeds = resolve_random_seeds(config)
    build_results_metadata(results, config, seeds)
    build_results_config(results, config, seeds)
    results.metadata.write()
    results.config.write()

    # build results metadata

    for run_idx, seed in enumerate(seeds):
        if run_idx < start_run:
            print(f"=== {config.name} | run {run_idx+1}/{config.num_runs} | SKIPPED ===")
            continue

        run_results = results.get_run(run_idx)

        print(f"\n=== {config.name} | run {run_idx+1}/{config.num_runs} | seed={seed} ===")
        timer = SimpleTimer()
        _set_seed(seed)

        # Build model/criterion/optimizer
        timer.tick()
        print("Creating model")
        model = config.model_fn()
        model = model.to(device)

        # (optional) log dropout rate if present; fail fast if you rely on it
        if not hasattr(model, "relu") or not hasattr(model.relu, "dropout_rate"):
            # You said: fail fast is sufficient if it's missing
            print("Warning: model.activation.dropout_rate not found (baseline ReLU without RAD?)")

        criterion = config.criterion or nn.CrossEntropyLoss()
        optimizer = config.optimizer_fn(model)

        # Preload test tensors to device (your existing pattern)
        timer.tick()
        print("Preloading test set")
        test_inputs = None
        test_labels = None
        for ti, tl in config.test_loader:
            test_inputs, test_labels = ti.to(device), tl.to(device)
            break  # loaders use a single big batch; take the first (and only)

        # Train
        run_training_loop(
            run_idx,
            config.epochs,
            config.stop_delta_loss,
            config.stop_delta_patience,
            model,
            optimizer,
            criterion,
            config.train_loader,
            test_inputs,
            test_labels,
            device,
            timer,
            run_results,
        )

        timer.tick()
        print(f"Run {run_idx} complete")

    results.metadata.status = "completed"
    results.metadata.write()
    results.end_log()

    # print_mem("Cleaning memory")
    # del model, optimizer, criterion, test_inputs, test_labels
    # gc.collect()
    # torch.cuda.empty_cache()
    # print_mem("Memory cleaned")
    print(f"Total time elapsed: {timer.elapsed()}s")


def main():
    # run_experiment(get_experiment_config("cifar10_baseline"))
    
    # run_experiment(get_experiment_config("cifar10_std_dropout_5em3"), start_run=11)
    # run_experiment(get_experiment_config("cifar10_abs_dropout_5em3"))    

    # run_experiment(get_experiment_config("cifar10_std_dropout_1em2"), start_run=6)
    run_experiment(get_experiment_config("cifar10_abs_dropout_1em2"), start_run=10)

    run_experiment(get_experiment_config("cifar10_std_dropout_2em2"))
    run_experiment(get_experiment_config("cifar10_abs_dropout_2em2"))

    run_experiment(get_experiment_config("cifar10_std_dropout_3em2"))
    run_experiment(get_experiment_config("cifar10_std_dropout_3em2"))



if __name__ == "__main__":
    import experiments as _  # only load/execute decorators in the main process

    try:
        main()
    except BaseException as e:
        print(e)
        traceback.print_exc()

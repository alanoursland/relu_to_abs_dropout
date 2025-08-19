# eval.py
"""
Evaluation for trained CIFAR experiments using experiment-provided DataLoaders.

Highlights
- Uses each experiment's config-provided train/test DataLoaders (no giant tensor staging).
- Clear separation of model restoration, BN conditioning (optional), and split evaluation.
- Minimal, readable prints; optional JSON outputs with backward-compatible filenames.
- CLI via argparse; programmatic API preserved (evaluate_experiments).

Assumptions:
- Experiments are registered via `experiments` module side-effects.
- `config.get_experiment_config(exp_name)` returns an object with:
    - `.model_fn()` (callable building a fresh model)
    - `.criterion` (optional; defaults to CrossEntropyLoss if absent)
    - `.train_loader` and `.test_loader` (PyTorch DataLoaders)
- `results.get_experimental_results(exp_name, create_mode=False)` exposes:
    - `.metadata.num_runs`
    - `.get_run(i)` -> RunResults with `.id` and `.load_final_model(...)`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from config import ExperimentConfig

import torch
import torch.nn as nn

# project imports
from config import get_experiment_config
from results import get_experimental_results


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def check_mem():
    # print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    pass

def resolve_device(spec: str) -> torch.device:
    """Resolve a device string ('auto' | 'cpu' | 'cuda' | 'cuda:0' | etc.) to torch.device."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def to_device_batch(batch, device: torch.device):
    """Move a (inputs, targets) batch to device with non_blocking when CUDA is used."""
    non_blocking = device.type == "cuda"
    x, y = batch
    return x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)


def get_criterion(exp_cfg: ExperimentConfig) -> nn.Module:
    return exp_cfg.criterion if getattr(exp_cfg, "criterion", None) is not None else nn.CrossEntropyLoss()


def rebuild_model(exp_cfg: ExperimentConfig) -> nn.Module:
    if exp_cfg.model_fn is None:
        raise ValueError("config.model_fn is None; cannot rebuild model")
    return exp_cfg.model_fn()


def get_loaders(exp_cfg: ExperimentConfig):
    if not hasattr(exp_cfg, "train_loader") or not hasattr(exp_cfg, "test_loader"):
        raise ValueError(
            f"Experiment '{exp_cfg.name}' does not provide 'train_loader' and 'test_loader' on its config."
        )
    return exp_cfg.train_loader, exp_cfg.test_loader

def create_eval_loaders(exp_cfg: ExperimentConfig):
    import data

    if exp_cfg.name.startswith("cifar10_"):      
        return data.get_cifar10_loaders(
            batch_size=4096, 
            eval_batch_size=4096,
            num_workers=0,
            eval_mode=True
            )
    elif exp_cfg.name.startswith("cifar100_"):
        return data.get_cifar100_loaders(
            batch_size=2048, 
            eval_batch_size=2048,
            num_workers=0,
            eval_mode=True
            )
    raise Exception(f"Unsupported data set for {exp_cfg.name}")

def disable_all_dropout(model: nn.Module) -> None:
    """Disable standard and common custom dropout fields in-place."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.p = 0.0
            m.eval()
        elif hasattr(m, "set_dropout"):
            m.set_dropout(0)
            m.eval()


@torch.no_grad()
def condition_batchnorm_with_loader(
    model: nn.Module,
    train_loader,
    device: torch.device,
    *,
    passes: int = 1,
) -> None:
    """
    Recompute BatchNorm running statistics using passes over the training inputs.
    Keeps dropout disabled. No labels needed. Uses the config's train_loader.
    """
    has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in model.modules())
    if not has_bn:
        return

    was_training = model.training
    model.train(True)

    for _ in range(max(1, int(passes))):
        for xb, _ in train_loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            _ = model(xb)

    if not was_training:
        model.eval()


@torch.no_grad()
def eval_split_with_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """
    Evaluate mean loss and accuracy (in %) over a split using its DataLoader.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        # print(f"\tEval batch {total_examples}")
        xb, yb = to_device_batch(batch, device)
        logits = model(xb)
        loss = criterion(logits, yb)

        bsz = yb.size(0)
        total_examples += bsz
        total_loss += float(loss.item()) * bsz
        preds = logits.argmax(dim=1)
        total_correct += int((preds == yb).sum().item())

    mean_loss = total_loss / max(1, total_examples)
    acc = 100.0 * total_correct / max(1, total_examples)
    return mean_loss, acc


def summarize_runs(per_run: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Compute mean/variance for final metrics across runs."""
    keys = ["train_loss", "test_loss", "train_acc", "test_acc"]
    values = {k: [] for k in keys}
    for r in per_run:
        for k in keys:
            values[k].append(r["final"][k])

    def mean_var(arr: List[float]) -> Tuple[float, float]:
        n = len(arr)
        if n == 0:
            return 0.0, 0.0
        mean = sum(arr) / n
        if n < 2:
            return mean, 0.0
        var = sum((a - mean) ** 2 for a in arr) / (n - 1)
        return mean, var

    out = {}
    for k, arr in values.items():
        m, v = mean_var(arr)
        out[k] = {"mean": m, "variance": v}
    return out


# ---------------------------------------------------------------------
# Per-experiment evaluator
# ---------------------------------------------------------------------
@dataclass
class EvalConfig:
    exp_name: str
    device: torch.device
    bn_passes: int = 1
    save_json: bool = True


class LoaderEvaluator:
    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg
        self.exp_cfg = get_experiment_config(self.cfg.exp_name)
        self.results = get_experimental_results(cfg.exp_name, create_mode=False)
        self.meta = self.results.metadata
        self.criterion = get_criterion(self.exp_cfg)
        self.train_loader, self.test_loader = create_eval_loaders(self.exp_cfg)

    def evaluate(self) -> Dict:
        if not hasattr(self.meta, "num_runs") or self.meta.num_runs <= 0:
            raise FileNotFoundError(f"No runs recorded for experiment '{self.cfg.exp_name}'")

        print(f"[eval] Experiment: {self.cfg.exp_name} | runs: {self.meta.num_runs}")
        check_mem()
        per_run: List[Dict] = []

        for run_idx in range(self.meta.num_runs):
            run = self.results.get_run(run_idx)  # RunResults
            print(f"  • Run {run_idx} (id={run.id})")
            check_mem()

            # Rebuild & load
            model = rebuild_model(self.exp_cfg).to(self.cfg.device)
            loaded = run.load_final_model(model, optimizer=None, strict=True, map_location=self.cfg.device)
            if not loaded:
                print("    - No final checkpoint found; skipping")
                continue

            # Prepare model for deterministic eval
            disable_all_dropout(model)
            print("Conditioning Batch Norm")
            check_mem()
            condition_batchnorm_with_loader(
                model,
                self.train_loader,
                self.cfg.device,
                passes=self.cfg.bn_passes,
            )
            check_mem()

            # Evaluate
            with torch.inference_mode():
                print("Evaluating train")
                check_mem()
                tr_loss, tr_acc = eval_split_with_loader(
                    model, self.train_loader, self.cfg.device, self.criterion
                )
                check_mem()
                print("Evaluating test")
                te_loss, te_acc = eval_split_with_loader(
                    model, self.test_loader, self.cfg.device, self.criterion
                )
                check_mem()

            per_run.append(
                {
                    "id": run.id,
                    "index": run_idx,
                    "final": {
                        "train_loss": float(tr_loss),
                        "test_loss": float(te_loss),
                        "train_acc": float(tr_acc),
                        "test_acc": float(te_acc),
                    },
                }
            )

        # Aggregate
        summary = summarize_runs(per_run)
        out_aggregate = {
            "experiment": self.cfg.exp_name,
            "num_runs": len(per_run),
            "metrics": {"final": summary},
        }
        out_full = {**out_aggregate, "runs": per_run}

        # Save (keep original filenames for compatibility)
        if self.cfg.save_json:
            results_dir: Path = self.results.results_dir
            agg_path = results_dir / "fast_eval.json"
            runs_path = results_dir / "fast_eval_runs.json"

            agg_path.write_text(json.dumps(out_aggregate, indent=2))
            runs_payload = {
                "experiment": self.cfg.exp_name,
                "num_runs": len(per_run),
                "runs": per_run,
            }
            runs_path.write_text(json.dumps(runs_payload, indent=2))
            print(f"[eval] Wrote {agg_path.name} and {runs_path.name} in {results_dir}")

        return out_full


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def evaluate_experiments(
    exp_names: Iterable[str],
    *,
    device: str = "auto",
    bn_passes: int = 1,
    save_json: bool = True,
) -> Dict[str, Dict]:
    dev = resolve_device(device)

    results_by_exp: Dict[str, Dict] = {}
    for name in exp_names:
        evaluator = LoaderEvaluator(
            EvalConfig(
                exp_name=name,
                device=dev,
                bn_passes=bn_passes,
                save_json=save_json,
            )
        )
        results_by_exp[name] = evaluator.evaluate()
    return results_by_exp

def main():
    # Put whatever you want to evaluate here (order doesn’t matter)
    exp_names = [
        "cifar10_baseline",

        "cifar10_abs_dropout_1em2",
        "cifar10_abs_dropout_2em2",
        "cifar10_abs_dropout_3em2",
        "cifar10_abs_dropout_5em3",

        "cifar10_std_dropout_1em2",
        "cifar10_std_dropout_2em2",
        "cifar10_std_dropout_3em2",
        "cifar10_std_dropout_5em3",

        # "cifar100_baseline",
        # "cifar100_std_dropout_2em2",
        # "cifar100_std_dropout_1em1",
        # "cifar100_std_dropout_2em1"
    ]

    out = evaluate_experiments(
        exp_names=exp_names,
        device="auto",
        bn_passes=1,
        save_json=True,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import experiments as _  # ensure experiment registry is populated
    main()

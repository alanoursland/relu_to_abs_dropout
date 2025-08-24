# eval.py  (updated)
"""
Evaluation for trained CIFAR experiments using experiment-provided DataLoaders.

Highlights
- Uses each experiment's config-provided train/test DataLoaders (no giant tensor staging).
- Clear separation of model restoration, BN conditioning (optional), and split evaluation.
- Minimal, readable prints; optional JSON outputs with backward-compatible filenames.
- CLI via argparse; programmatic API preserved (evaluate_experiments).
- Expected Calibration Error (ECE) on train/test (default 15 bins).
- Optional mCE on CIFAR-10-C / CIFAR-100-C when corruption loaders are available.

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
from typing import Dict, Iterable, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# project imports
from config import ExperimentConfig, get_experiment_config
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


def get_dataset_kind(exp_cfg: ExperimentConfig) -> str:
    """Return 'cifar10' or 'cifar100' based on experiment name prefix."""
    if exp_cfg.name.startswith("cifar10_"):
        return "cifar10"
    if exp_cfg.name.startswith("cifar100_"):
        return "cifar100"
    raise Exception(f"Unsupported data set for {exp_cfg.name}")


def get_base_loaders(exp_cfg: ExperimentConfig):
    """Fresh eval-time loaders; avoids relying on training-time loader state."""
    import data
    kind = get_dataset_kind(exp_cfg)

    if kind == "cifar10":
        return data.get_cifar10_loaders(
            batch_size=4096,
            eval_batch_size=4096,
            num_workers=0,
            eval_mode=True,
        )
    elif kind == "cifar100":
        return data.get_cifar100_loaders(
            batch_size=2048,
            eval_batch_size=2048,
            num_workers=0,
            eval_mode=True,
        )
    else:
        raise Exception(f"Unsupported data set for {exp_cfg.name}")


def maybe_get_corruption_loaders(exp_cfg: ExperimentConfig, *, eval_batch_size: int = 1024, num_workers: int = 0):
    """
    Try to retrieve CIFAR-C loaders from your `data` module.

    Expected shapes (any one of these is fine; the function picks what exists):
      1) data.get_cifar10c_loaders(eval_batch_size=..., num_workers=..., eval_mode=True)
         -> Dict[str, List[DataLoader]]  (per corruption: list of 5 severities)
      2) data.get_cifar10c_loaders() -> Dict[str, List[DataLoader]]
      (and similarly for cifar100c_*)
    Returns: (dict_or_none, dataset_kind)
    """
    import data
    kind = get_dataset_kind(exp_cfg)

    fn_names = []
    if kind == "cifar10":
        fn_names = ["get_cifar10c_loaders", "get_cifar10_c_loaders"]
    elif kind == "cifar100":
        fn_names = ["get_cifar100c_loaders", "get_cifar100_c_loaders"]

    for fn in fn_names:
        if hasattr(data, fn):
            get_fn = getattr(data, fn)
            try:
                # Try with kwargs first
                loaders = get_fn(eval_batch_size=eval_batch_size, num_workers=num_workers, eval_mode=True)
            except TypeError:
                try:
                    loaders = get_fn(eval_batch_size=eval_batch_size, num_workers=num_workers)
                except TypeError:
                    loaders = get_fn()
            # Expect a dict[str, list_of_5_loaders] or dict[str, dict[severity->loader]]
            return loaders, kind

    return None, kind


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

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_ece_from_accumulated(confidences: torch.Tensor, correctness: torch.Tensor, *, n_bins: int = 15) -> float:
    """
    confidences: (N,) max-softmax per example on CPU
    correctness: (N,) bool tensor indicating prediction correctness
    Returns scalar ECE in [0,1].
    """
    # Ensure CPU + float
    conf = confidences.detach().cpu()
    corr = correctness.detach().cpu().float()

    # Bin edges inclusive of 0 and 1
    bin_edges = torch.linspace(0.0, 1.0, steps=n_bins + 1)
    ece = torch.tensor(0.0)
    N = conf.numel()
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right edge on last bin to catch conf==1.0
        if i < n_bins - 1:
            in_bin = (conf >= lo) & (conf < hi)
        else:
            in_bin = (conf >= lo) & (conf <= hi)
        count = in_bin.sum().item()
        if count == 0:
            continue
        acc_bin = corr[in_bin].mean()
        conf_bin = conf[in_bin].mean()
        ece += (count / N) * (acc_bin - conf_bin).abs()
    return float(ece.item())


@torch.no_grad()
def eval_split_with_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    *,
    ece_bins: int = 15,
) -> Tuple[float, float, float]:
    """
    Evaluate mean loss, accuracy (in %), and ECE (0..1) over a split.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    # Accumulate confidences & correctness for ECE
    all_confidences: List[torch.Tensor] = []
    all_correct: List[torch.Tensor] = []

    for batch in loader:
        xb, yb = to_device_batch(batch, device)
        logits = model(xb)
        loss = criterion(logits, yb)

        probs = F.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)
        correct = preds.eq(yb)

        bsz = yb.size(0)
        total_examples += bsz
        total_loss += float(loss.item()) * bsz
        total_correct += int(correct.sum().item())

        all_confidences.append(conf.detach().cpu())
        all_correct.append(correct.detach().cpu())

    mean_loss = total_loss / max(1, total_examples)
    acc = 100.0 * total_correct / max(1, total_examples)

    # ECE
    if len(all_confidences) > 0:
        confidences = torch.cat(all_confidences, dim=0)
        correctness = torch.cat(all_correct, dim=0)
        ece = compute_ece_from_accumulated(confidences, correctness, n_bins=ece_bins)
    else:
        ece = 0.0

    return mean_loss, acc, ece


@torch.no_grad()
def eval_cifar_c_mce(
    model: nn.Module,
    corruption_loaders: Dict[str, List],
    device: torch.device,
    criterion: nn.Module,
    *,
    ece_bins: int = 15,
) -> Dict[str, float]:
    """
    Compute unnormalized corruption metrics across CIFAR-C loaders:
    - For each corruption type and severity, compute accuracy and error.
    - Report:
        * mCE_unscaled: mean error across all corruptions/severities (0..100)
        * mean_acc_c: mean accuracy across all corruptions/severities (0..100)
        * mean_ece_c: mean ECE across all corruptions/severities (0..1)

    Notes:
    - This is the common "average over corruptions" metric. If you want the
      *normalized* mCE (relative to a baseline), you can post-process using a
      baseline JSON later; this function keeps runtime simple and dependency-free.
    """
    model.eval()
    accs = []
    errs = []
    eces = []

    for corr_name, sev_loaders in corruption_loaders.items():
        # Accept either list[5] or dict{severity->loader}
        if isinstance(sev_loaders, dict):
            items = sorted(sev_loaders.items(), key=lambda kv: int(kv[0]))
            loaders_iter = [ld for _, ld in items]
        else:
            loaders_iter = list(sev_loaders)

        for ld in loaders_iter:
            total_loss = 0.0
            total_correct = 0
            total_examples = 0
            all_conf, all_corr = [], []

            for batch in ld:
                xb, yb = to_device_batch(batch, device)
                logits = model(xb)
                loss = criterion(logits, yb)

                probs = F.softmax(logits, dim=1)
                conf, preds = probs.max(dim=1)
                correct = preds.eq(yb)

                bsz = yb.size(0)
                total_examples += bsz
                total_loss += float(loss.item()) * bsz
                total_correct += int(correct.sum().item())

                all_conf.append(conf.detach().cpu())
                all_corr.append(correct.detach().cpu())

            if total_examples == 0:
                continue

            acc = 100.0 * total_correct / total_examples
            err = 100.0 - acc
            ece = compute_ece_from_accumulated(torch.cat(all_conf), torch.cat(all_corr), n_bins=ece_bins)

            accs.append(acc)
            errs.append(err)
            eces.append(ece)

    if len(errs) == 0:
        return {
            "mCE_unscaled": float("nan"),
            "mean_acc_c": float("nan"),
            "mean_ece_c": float("nan"),
        }

    return {
        "mCE_unscaled": float(sum(errs) / len(errs)),
        "mean_acc_c": float(sum(accs) / len(accs)),
        "mean_ece_c": float(sum(eces) / len(eces)),
    }


def summarize_runs(per_run: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Compute mean/variance for final metrics across runs."""
    keys = [
        "train_loss",
        "test_loss",
        "train_acc",
        "test_acc",
        "train_ece",
        "test_ece",
    ]
    # mCE is optional; if present in any run, we’ll aggregate its fields too
    mce_keys = ["mCE_unscaled", "mean_acc_c", "mean_ece_c"]

    values = {k: [] for k in keys + mce_keys}
    for r in per_run:
        for k in keys:
            values[k].append(r["final"][k])
        if "cifar_c" in r["final"]:
            for mk in mce_keys:
                if mk in r["final"]["cifar_c"]:
                    values[mk].append(r["final"]["cifar_c"][mk])

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
        if len(arr) == 0:
            continue
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
    ece_bins: int = 15
    eval_cifar_c: bool = False  # set True to attempt mCE on CIFAR-C


class LoaderEvaluator:
    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg
        self.exp_cfg = get_experiment_config(self.cfg.exp_name)
        self.results = get_experimental_results(cfg.exp_name, create_mode=False)
        self.meta = self.results.metadata
        self.criterion = get_criterion(self.exp_cfg)
        self.train_loader, self.test_loader = get_base_loaders(self.exp_cfg)

        # Optional corruption loaders (if available)
        self.corruption_loaders = None
        if self.cfg.eval_cifar_c:
            loaders, _ = maybe_get_corruption_loaders(self.exp_cfg, eval_batch_size=1024, num_workers=0)
            if loaders is None:
                print("  • CIFAR-C loaders not found in `data` module; skipping mCE.")
            else:
                self.corruption_loaders = loaders

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
                tr_loss, tr_acc, tr_ece = eval_split_with_loader(
                    model, self.train_loader, self.cfg.device, self.criterion, ece_bins=self.cfg.ece_bins
                )
                check_mem()
                print("Evaluating test")
                te_loss, te_acc, te_ece = eval_split_with_loader(
                    model, self.test_loader, self.cfg.device, self.criterion, ece_bins=self.cfg.ece_bins
                )
                check_mem()

                c_metrics = None
                if self.corruption_loaders is not None:
                    print("Evaluating CIFAR-C (mCE)")
                    c_metrics = eval_cifar_c_mce(
                        model,
                        self.corruption_loaders,
                        self.cfg.device,
                        self.criterion,
                        ece_bins=self.cfg.ece_bins,
                    )
                    check_mem()

            final_dict = {
                "train_loss": float(tr_loss),
                "test_loss": float(te_loss),
                "train_acc": float(tr_acc),
                "test_acc": float(te_acc),
                "train_ece": float(tr_ece),
                "test_ece": float(te_ece),
            }
            if c_metrics is not None:
                final_dict["cifar_c"] = c_metrics

            per_run.append(
                {
                    "id": run.id,
                    "index": run_idx,
                    "final": final_dict,
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
    ece_bins: int = 15,
    eval_cifar_c: bool = False,
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
                ece_bins=ece_bins,
                eval_cifar_c=eval_cifar_c,
            )
        )
        results_by_exp[name] = evaluator.evaluate()
    return results_by_exp


def main():
    # Put whatever you want to evaluate here (order doesn’t matter)
    exp_names = [
        "cifar10_baseline",

        "cifar10_abs_dropout_5em3",
        "cifar10_abs_dropout_1em2",
        "cifar10_abs_dropout_2em2",
        "cifar10_abs_dropout_3em2",
        "cifar10_abs_dropout_5em2",
        "cifar10_abs_dropout_1em1",
        "cifar10_abs_dropout_2em1",

        "cifar10_std_dropout_5em3",
        "cifar10_std_dropout_1em2",
        "cifar10_std_dropout_2em2",
        "cifar10_std_dropout_3em2",
        "cifar10_std_dropout_5em2",
        "cifar10_std_dropout_1em1",
        "cifar10_std_dropout_2em1",
        "cifar10_std_dropout_3em1",


        # "cifar100_baseline",
        # "cifar100_std_dropout_2em2",
        # "cifar100_std_dropout_1em1",
        # "cifar100_std_dropout_2em1",
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

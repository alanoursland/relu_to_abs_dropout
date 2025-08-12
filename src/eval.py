# eval.py
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# project imports
from config import get_experiment_config
from results import get_experimental_results
from data import _get_cifar10_datasets  # cached deterministic test TensorDataset


# ------------------------------
# Shared data (one load on GPU)
# ------------------------------
class DataBundle:
    def __init__(self, device: torch.device, chunk_size: int = 8192):
        self.device = device
        self.chunk_size = int(chunk_size)
        self.train_x, self.train_y, self.test_x, self.test_y = self._load_cifar10_tensors()
        self._move_to_device_()

    def _load_cifar10_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        - Train: apply train transform *once* per sample (no DataLoader), stack into one big tensor
        - Test: data.py already builds a TensorDataset with deterministic transforms
        """
        train_ds, test_ds = _get_cifar10_datasets()

        train_imgs, train_labels = [], []
        for img, label in train_ds:
            train_imgs.append(img)
            train_labels.append(label)
        train_x = torch.stack(train_imgs, dim=0)  # [50000,3,32,32]
        train_y = torch.tensor(train_labels, dtype=torch.long)  # [50000]

        test_x, test_y = test_ds.tensors  # ([10000,3,32,32], [10000])
        return train_x, train_y, test_x, test_y

    def _move_to_device_(self):
        self.train_x = self.train_x.to(self.device, non_blocking=True)
        self.train_y = self.train_y.to(self.device, non_blocking=True)
        self.test_x = self.test_x.to(self.device, non_blocking=True)
        self.test_y = self.test_y.to(self.device, non_blocking=True)


# ------------------------------
# Single-experiment evaluator
# ------------------------------
class FastEvaluator:
    def __init__(
        self,
        exp_name: str,
        data: DataBundle,
        device: torch.device,
        bn_passes: int = 1,
        save_json: bool = True,
    ):
        self.exp_name = exp_name
        self.device = device
        self.data = data
        self.bn_passes = int(bn_passes)
        self.save_json = save_json

        # Load results handle (no creation) and metadata
        self.results = get_experimental_results(exp_name, create_mode=False)
        self.meta = self.results.metadata

        # Loss for evaluation (per-experiment)
        cfg = get_experiment_config(exp_name)
        self.criterion = cfg.criterion if cfg.criterion is not None else nn.CrossEntropyLoss()

    def evaluate(self) -> Dict:
        if not hasattr(self.meta, "num_runs") or self.meta.num_runs <= 0:
            raise FileNotFoundError(f"No runs recorded for experiment '{self.exp_name}'")

        print(f"Evaluating experiment {self.exp_name}")
        per_run = []
        for run_idx in range(self.meta.num_runs):
            print(f"    Starting run {run_idx}")
            run = self.results.get_run(run_idx)  # RunResults

            # Rebuild model via experiment registry (guarantees arch parity)
            print(f"        Rebuilding model")
            model = self._rebuild_model().to(self.device, non_blocking=True)

            # Load final checkpoint via RunResults API (device-aware)
            print(f"        Loading model checkpoint")
            loaded = run.load_final_model(model, optimizer=None, strict=True, map_location=self.device)
            if not loaded:
                # Skip runs without a final checkpoint
                continue

            # Turn off *all* dropout variants (standard + custom)
            print(f"        Disabling dropout")
            self._disable_dropout_everywhere(model)

            # Condition BatchNorm on the full training set (dropout stays off)
            print(f"        Conditioning Batchnorm")
            self._condition_batchnorm(model, self.data.train_x, chunk_size=self.data.chunk_size, passes=self.bn_passes)

            # Eval on train/test
            model.eval()
            with torch.inference_mode():
                print(f"        Evaluating training set")
                tr_loss, tr_acc = self._eval_split(
                    model, self.data.train_x, self.data.train_y, self.criterion, self.data.chunk_size
                )
                print(f"        Evaluating test set")
                te_loss, te_acc = self._eval_split(
                    model, self.data.test_x, self.data.test_y, self.criterion, self.data.chunk_size
                )

            per_run.append(
                {
                    "id": run.id,
                    "final": {
                        "train_loss": float(tr_loss),
                        "test_loss": float(te_loss),
                        "train_acc": float(tr_acc),
                        "test_acc": float(te_acc),
                    },
                }
            )

        summary = self._summarize(per_run)
        out = {
            "experiment": self.exp_name,
            "num_runs": len(per_run),
            "metrics": {"final": summary},
            # "runs": per_run,
        }

        if self.save_json:
            out_path = self.results.results_dir / "fast_eval.json"
            out_path.write_text(json.dumps(out, indent=2))
            print(f"[fast-eval] wrote {out_path}")

        return out

    # --- internals ---
    def _rebuild_model(self) -> nn.Module:
        cfg = get_experiment_config(self.exp_name)
        if cfg.model_fn is None:
            raise ValueError("config.model_fn is None; cannot rebuild model")
        return cfg.model_fn()

    @staticmethod
    def _disable_dropout_everywhere(model: nn.Module) -> None:
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
                m.p = 0.0
                m.eval()
            if hasattr(m, "dropout") and isinstance(m.dropout, nn.Dropout):
                m.dropout.p = 0.0
            if hasattr(m, "dropout_rate"):
                try:
                    m.dropout_rate = 0.0
                except Exception:
                    pass

    @torch.no_grad()
    def _condition_batchnorm(self, model: nn.Module, train_x: torch.Tensor, chunk_size: int = 8192, passes: int = 1):
        has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in model.modules())
        if not has_bn:
            return

        was_training = model.training
        model.train(True)

        N = train_x.size(0)
        for _ in range(max(1, passes)):
            for i in range(0, N, chunk_size):
                xb = train_x[i : i + chunk_size]
                _ = model(xb)

        if not was_training:
            model.eval()

    @torch.no_grad()
    def _eval_split(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
        chunk_size: int = 8192,
    ) -> Tuple[float, float]:
        N = x.size(0)
        total_loss = 0.0
        total_correct = 0
        for i in range(0, N, chunk_size):
            xb = x[i : i + chunk_size]
            yb = y[i : i + chunk_size]
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += int((preds == yb).sum().item())
        return total_loss / N, 100.0 * total_correct / N

    @staticmethod
    def _summarize(per_run: List[Dict]) -> Dict[str, Dict[str, float]]:
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
            var = sum((a - mean) ** 2 for a in arr) / (n - 1)  # sample variance
            return mean, var

        out = {}
        for k, arr in values.items():
            m, v = mean_var(arr)
            out[k] = {"mean": m, "variance": v}
        return out


# ------------------------------
# Multi-experiment driver
# ------------------------------
def _resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def evaluate_experiments(
    exp_names: List[str],
    device: str = "auto",
    chunk_size: int = 16384,
    bn_passes: int = 1,
    save_json: bool = True,
) -> Dict[str, Dict]:
    dev = _resolve_device(device)
    # One CIFAR-10 load, shared across all experiments
    data = DataBundle(device=dev, chunk_size=chunk_size)

    results_by_exp: Dict[str, Dict] = {}
    for name in exp_names:
        evaluator = FastEvaluator(
            exp_name=name,
            data=data,
            device=dev,
            bn_passes=bn_passes,
            save_json=save_json,
        )
        results_by_exp[name] = evaluator.evaluate()

    return results_by_exp


def main():
    # Put whatever you want to evaluate here (order doesn’t matter)
    exp_names = [
        "cifar10_baseline",
        "cifar10_std_dropout_2em2",
        "cifar10_abs_dropout_2em2",
        # "cifar10_std_dropout_5em3",
        # "cifar10_std_dropout_1em2",
        # "cifar10_std_dropout_2em3",
    ]

    out = evaluate_experiments(
        exp_names=exp_names,
        device="auto",
        chunk_size=4096,
        bn_passes=1,
        save_json=True,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import experiments as _  # ensure experiment registry is populated
    main()

import json
import platform
import torch
import sys, io, contextlib

from dataclasses import dataclass, field, asdict
from typing import Dict, Callable, Optional, List, Any
from pathlib import Path
from datetime import datetime, timezone
from torch.optim import Optimizer
import csv
import pandas as pd


@dataclass
class MetadataResults:
    date: str
    seed: int
    hardware: str
    git_commit: str
    env: str


@dataclass
class ConfigResults:
    date: str
    seed: int
    hardware: str
    git_commit: str
    env: str


@dataclass
class LogResults:
    msg: List[str]


@dataclass
class CurveResult:
    y: List[float]


@dataclass
class Metadata:
    name: str = ""
    created_at: str = datetime.now(timezone.utc).isoformat()
    created_by: str = platform.node()
    description: str = ""
    framework_versions: dict = None
    device_info: dict = None
    num_runs: int = 0
    random_seeds: list = None
    notes: str = ""
    status: str = "in_progress"
    _path: Path = None  # not serialized

    def write(self):
        if self._path is None:
            raise ValueError("Metadata path is not set.")
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(asdict(self, dict_factory=lambda x: {k: v for k, v in x if not k.startswith("_")}), f, indent=2)

    def load(self):
        if self._path is None:
            raise ValueError("Metadata path is not set.")
        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            setattr(self, k, v)


@dataclass
class Config:
    # Core, all JSON-serializable
    name: str = ""
    description: str = ""
    output_dir: str = "results"

    num_runs: int = 1
    random_seeds: Optional[List[int]] = None
    continue_from: Optional[str] = None
    load_optimizer_state: bool = False
    device: str = "auto"

    epochs: int = 0
    stop_delta_loss: float = 1e-3
    stop_delta_patience: int = 0

    # Human-readable representations for non-serializable callables/objects
    model_fn_repr: Optional[str] = None
    optimizer_fn_repr: Optional[str] = None
    criterion_repr: Optional[str] = None

    # Optional dataset snapshot (kept minimal and safe to serialize)
    dataset_info: Optional[Dict[str, Any]] = None

    # Internal (not serialized)
    _path: Optional[Path] = None

    def write(self):
        if self._path is None:
            raise ValueError("Config path is not set.")
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in asdict(self).items() if not k.startswith("_")}, f, indent=2)

    def load(self):
        if self._path is None:
            raise ValueError("Config path is not set.")
        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            setattr(self, k, v)


@dataclass
class TimingRow:
    epoch: int
    epoch_sec: float
    data_sec: float
    compute_sec: float
    eval_sec: float


@dataclass
class PredictionResults:
    path: Path
    create_mode: bool = False
    _tensor: Optional[torch.Tensor] = None  # cached in-memory (N, C)

    def set_from_tensor(self, logits: torch.Tensor):
        """Store logits (N, C) in-memory; no device constraints."""
        if logits.dim() != 2:
            raise ValueError(f"Predictions must be 2D (N, C), got shape {tuple(logits.shape)}")
        # Copy to CPU float32 to ensure stable serialization
        self._tensor = logits.detach().to("cpu", dtype=torch.float32).contiguous()

    def write(self):
        """Write the cached tensor to CSV at self.path."""
        if self._tensor is None:
            raise ValueError("No predictions set. Call set_from_tensor() first.")
        self.path.parent.mkdir(parents=True, exist_ok=True)

        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # optional header with column indices
            n_classes = self._tensor.shape[1]
            header = [f"c{i}" for i in range(n_classes)]
            w.writerow(header)
            for row in self._tensor.tolist():
                w.writerow(row)
        tmp.replace(self.path)

    def load(self) -> torch.Tensor:
        """Load predictions from CSV into memory and return as torch.Tensor (float32)."""
        if not self.path.exists():
            raise FileNotFoundError(f"Predictions file not found: {self.path}")
        rows = []
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            # skip header if present (assume first row is header of 'c0','c1',...)
            first = next(r)
            try:
                # try to parse as floats; if fails, it's a header
                [float(x) for x in first]
                rows.append([float(x) for x in first])
            except ValueError:
                # header detected; continue
                pass
            for line in r:
                rows.append([float(x) for x in line])
        self._tensor = torch.tensor(rows, dtype=torch.float32)
        return self._tensor

    @property
    def tensor(self) -> torch.Tensor:
        """Return cached tensor if available, else load from disk."""
        if self._tensor is None:
            return self.load()
        return self._tensor


class TimingResults:
    """
    Manages per-epoch timing for a single run.
    Storage:
      - runs/<NNN>/timings.csv   (append-friendly, human readable)
      - runs/<NNN>/timings.json  (optional snapshot of all rows)
    """

    def __init__(self, run_dir: Path, create_mode: bool = False):
        self.run_dir = run_dir
        self.create_mode = create_mode
        self.csv_path = self.run_dir / "timings.csv"
        self.json_path = self.run_dir / "timings.json"
        self._rows: List[TimingRow] = []

    def add_epoch_timing(self, epoch: int, epoch_sec: float, data_sec: float, compute_sec: float, eval_sec: float):
        self._rows.append(TimingRow(epoch, epoch_sec, data_sec, compute_sec, eval_sec))

    def write(self):
        # Write CSV (append-safe: rewrites full file with header for simplicity)
        header = ["epoch", "epoch_sec", "data_sec", "compute_sec", "eval_sec"]
        tmp = self.csv_path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in self._rows:
                w.writerow([r.epoch, r.epoch_sec, r.data_sec, r.compute_sec, r.eval_sec])
        tmp.replace(self.csv_path)

        # Optional JSON snapshot (handy for quick programmatic reads)
        tmpj = self.json_path.with_suffix(".tmp")
        with open(tmpj, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self._rows], f, indent=2)
        tmpj.replace(self.json_path)

    def load(self):
        self._rows.clear()
        if not self.csv_path.exists():
            return
        with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                self._rows.append(
                    TimingRow(
                        epoch=int(row["epoch"]),
                        epoch_sec=float(row["epoch_sec"]),
                        data_sec=float(row["data_sec"]),
                        compute_sec=float(row["compute_sec"]),
                        eval_sec=float(row["eval_sec"]),
                    )
                )

    @property
    def rows(self) -> List[TimingRow]:
        return self._rows[:]


class StatsResults:
    def __init__(self, path: Path, create_mode: bool):
        self.path = path
        self.create_mode = create_mode
        self._train_loss = []
        self._test_loss = []
        self._train_acc = []
        self._test_acc = []
        self._best_metrics = None
        self._final_metrics = None

    def add_epoch(self, train_loss, test_loss, train_acc, test_acc):
        """Append epoch stats to internal lists."""
        self._train_loss.append(train_loss)
        self._test_loss.append(test_loss)
        self._train_acc.append(train_acc)
        self._test_acc.append(test_acc)

    def set_final(self, train_loss, test_loss, train_acc, test_acc):
        """Record the final epoch metrics."""
        self._final_metrics = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
        }

    def set_best(self, train_loss, test_loss, train_acc, test_acc):
        """Record the best epoch metrics (your criteria)."""
        self._best_metrics = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
        }

    def load(self):
        """Load stats.json and CSVs into memory."""
        stats_file = self.path / "stats.json"
        if stats_file.exists():
            data = json.loads(stats_file.read_text())
            self._final_metrics = data.get("final")
            self._best_metrics = data.get("best")

        def read_csv(fname):
            f = self.path / fname
            return pd.read_csv(f, header=None).iloc[:, 0].tolist() if f.exists() else []

        self._train_loss = read_csv("loss_train.csv")
        self._test_loss = read_csv("loss_test.csv")
        self._train_acc = read_csv("acc_train.csv")
        self._test_acc = read_csv("acc_test.csv")

    def write(self):
        """Write stats.json and CSVs."""
        # stats.json
        summary = {
            "final": self._final_metrics,
            "best": self._best_metrics,
        }
        (self.path / "stats.json").write_text(json.dumps(summary, indent=2))

        # CSVs
        pd.Series(self._train_loss).to_csv(self.path / "loss_train.csv", index=False, header=False)
        pd.Series(self._test_loss).to_csv(self.path / "loss_test.csv", index=False, header=False)
        pd.Series(self._train_acc).to_csv(self.path / "acc_train.csv", index=False, header=False)
        pd.Series(self._test_acc).to_csv(self.path / "acc_test.csv", index=False, header=False)


class RunResults:
    def __init__(self, run_dir: Path, index: int, create_mode: bool = False):
        self.run_dir = run_dir  # e.g., results/<exp>/runs/001
        self.index = index  # 0-based index used by your loop
        self.create_mode = create_mode
        if create_mode:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self._timing = None
        self._pred_init = None
        self._pred_train = None
        self._pred_test = None
        self._pred_best = None

    def __repr__(self):
        return f"RunResults(run_dir={self.run_dir}, index={self.index}, create_mode={self.create_mode})"

    def _checkpoint_path(self, tag: str) -> Path:
        return self.run_dir / f"model_{tag}.pth"

    @property
    def id(self) -> str:
        # zero-padded, human-friendly id ("001", "002", ...)
        return f"{self.index + 1:03d}"

    @property
    def timing(self) -> TimingResults:
        if self._timing is None:
            t = TimingResults(self.run_dir, create_mode=self.create_mode)
            if not self.create_mode:
                t.load()
            self._timing = t
        return self._timing

    @property
    def predictions_init(self) -> PredictionResults:
        if self._pred_init is None:
            self._pred_init = PredictionResults(self.run_dir / "predictions_init.csv", self.create_mode)
            if not self.create_mode and self._pred_init.path.exists():
                self._pred_init.load()
        return self._pred_init

    @property
    def predictions_train(self) -> PredictionResults:
        if self._pred_train is None:
            self._pred_train = PredictionResults(self.run_dir / "predictions_train.csv", self.create_mode)
            if not self.create_mode and self._pred_train.path.exists():
                self._pred_train.load()
        return self._pred_train

    @property
    def predictions_test(self) -> PredictionResults:
        if self._pred_test is None:
            self._pred_test = PredictionResults(self.run_dir / "predictions_test.csv", self.create_mode)
            if not self.create_mode and self._pred_test.path.exists():
                self._pred_test.load()
        return self._pred_test

    @property
    def predictions_best(self) -> PredictionResults:
        if self._pred_best is None:
            self._pred_best = PredictionResults(self.run_dir / "predictions_best.csv", self.create_mode)
            if not self.create_mode and self._pred_best.path.exists():
                self._pred_best.load()
        return self._pred_best

    @property
    def stats(self) -> StatsResults:
        if not hasattr(self, "_stats"):
            self._stats = StatsResults(self.run_dir, self.create_mode)
            if not self.create_mode:
                self._stats.load()  # optional if we want reading
        return self._stats

    def save_checkpoint(self, tag: str, model, optimizer: Optional[Optimizer] = None):
        path = self._checkpoint_path(tag)
        state = {"model": model.state_dict()}
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()

        tmp_path = path.with_suffix(".tmp")
        torch.save(state, tmp_path)
        tmp_path.replace(path)  # atomic-ish
        print(f"[RunResults] Saved checkpoint '{tag}' at {path}")

    def load_checkpoint(
        self,
        tag: str,
        model,
        optimizer: Optional[Optimizer] = None,
        strict: bool = True,
        map_location: str = "cpu",
    ) -> bool:
        path = self._checkpoint_path(tag)
        if not path.exists():
            print(f"[RunResults] Checkpoint '{tag}' not found at {path}")
            return False

        checkpoint = torch.load(path, map_location=map_location)
        model.load_state_dict(checkpoint["model"], strict=strict)

        if optimizer is not None and "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"[RunResults] Loaded checkpoint '{tag}' from {path}")
        return True

    # Convenience wrappers
    def save_init_model(self, model, optimizer: Optional[Optimizer] = None):
        self.save_checkpoint("init", model, optimizer)

    def save_final_model(self, model, optimizer: Optional[Optimizer] = None):
        self.save_checkpoint("final", model, optimizer)

    def save_best_model(self, model, optimizer: Optional[Optimizer] = None):
        self.save_checkpoint("best", model, optimizer)

    def load_init_model(
        self, model, optimizer: Optional[Optimizer] = None, strict: bool = True, map_location: str = "cpu"
    ) -> bool:
        return self.load_checkpoint("init", model, optimizer, strict, map_location)

    def load_final_model(
        self, model, optimizer: Optional[Optimizer] = None, strict: bool = True, map_location: str = "cpu"
    ) -> bool:
        return self.load_checkpoint("final", model, optimizer, strict, map_location)

    def load_best_model(
        self, model, optimizer: Optional[Optimizer] = None, strict: bool = True, map_location: str = "cpu"
    ) -> bool:
        return self.load_checkpoint("best", model, optimizer, strict, map_location)


class ExperimentalResults:
    def __init__(self, results_dir: Path, create_mode=False):
        self.results_dir = results_dir  # e.g., results/<name>
        self.create_mode = create_mode
        self._orig_stdout = None
        self._log_file = None
        self._metadata = None
        self._config = None

    def __repr__(self):
        return f"ExperimentalResults(results_dir={self.results_dir})"

    def get_run(self, run_idx: int, create_mode: bool = False) -> RunResults:
        """
        Return a handle for runs/<NNN>/, where NNN = run_idx+1 zero-padded.
        - If the directory exists, return it.
        - If it doesn't exist:
            - create it when create_mode is True
            - otherwise raise FileNotFoundError
        """
        runs_root = self.results_dir / "runs"
        run_id = f"{run_idx + 1:03d}"
        run_dir = runs_root / run_id

        if run_dir.exists():
            if not run_dir.is_dir():
                raise NotADirectoryError(f"Path exists but is not a directory: {run_dir}")
            return RunResults(run_dir, run_idx, create_mode=False)

        if not create_mode:
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        run_dir.mkdir(parents=True, exist_ok=True)
        return RunResults(run_dir, run_idx, create_mode=True)

    @property
    def metadata(self) -> Metadata:
        if self._metadata is None:
            meta_path = self.results_dir / "metadata.json"
            md = Metadata(_path=meta_path)

            if self.create_mode:
                # Fill some defaults that make sense immediately
                md.framework_versions = {
                    "python": platform.python_version(),
                    "torch": torch.__version__ if torch else None,
                }
                md.device_info = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
            else:
                md.load()

            self._metadata = md
        return self._metadata

    @property
    def config(self) -> Config:
        if self._config is None:
            cfg_path = self.results_dir / "config.json"
            cfg = Config(_path=cfg_path)
            if self.create_mode:
                # caller will populate + write()
                pass
            else:
                cfg.load()
            self._config = cfg
        return self._config

    def start_log(self, mode="a"):
        """Redirect stdout to also write into logs.txt."""
        log_path = self.results_dir / "logs.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        class _Tee(io.TextIOBase):
            def __init__(self, *streams):
                self.streams = streams

            def write(self, s):
                for st in self.streams:
                    st.write(s)
                    st.flush()
                return len(s)

            def flush(self):
                for st in self.streams:
                    st.flush()

        if self._orig_stdout is not None:
            raise RuntimeError("Logging already started")

        self._log_file = open(log_path, mode, buffering=1, encoding="utf-8")
        self._orig_stdout = sys.stdout
        sys.stdout = _Tee(self._orig_stdout, self._log_file)

    def end_log(self):
        """Restore stdout and close the log file."""
        if self._orig_stdout is None:
            return
        sys.stdout = self._orig_stdout
        self._orig_stdout = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None


def get_experimental_results(name: str, create_mode: bool = False) -> ExperimentalResults:
    root = Path("results") / name

    if root.exists():
        if not root.is_dir():
            raise NotADirectoryError(f"Path exists but is not a directory: {root}")
        return ExperimentalResults(root, create_mode=create_mode)

    if not create_mode:
        raise FileNotFoundError(f"Experiment directory not found: {root}")

    # create on demand when allowed
    root.mkdir(parents=True, exist_ok=True)
    return ExperimentalResults(root, create_mode=create_mode)


"""
results/
└── <experiment_name>/
    ├── metadata.json
    ├── config.json
    ├── logs.txt
    └── runs/
        ├── 001/
        │   ├── stats.json                  # include losses, accuracies, timings, 
        │   ├── timings.csv
        │   ├── predictions_train.csv
        │   ├── predictions_test.csv
        │   ├── predictions_best.csv
        │   ├── loss_train.csv
        │   ├── loss_test.csv
        │   ├── acc_train.csv
        │   ├── acc_test.csv
        │   ├── model_init.pth              # Before training
        │   ├── model_final.pth             # After training
        │   └── model_best.pth              # Best val accuracy
        ├── 002/
        │   ├── ...
        ├── ...
"""


"""
results/
└── cifar10_std_dropout_2em3/                  # Experiment name
    ├── metadata.json                          # Date, seed, hardware, git commit, env
    ├── config.json                            # All hyperparams and dataset/model configs
    ├── logs.txt                               # Raw training logs (stdout/stderr)
    ├── curves/                                # Metric & loss curves
    │   ├── train_loss.csv
    │   ├── val_loss.csv
    │   ├── train_acc.csv
    │   ├── val_acc.csv
    │   └── lr_schedule.csv
    ├── models/                                # Model checkpoints
    │   ├── init_weights.pth                   # Before training
    │   ├── final_weights.pth                  # After training
    │   └── best_weights.pth                   # Best val accuracy
    ├── predictions/                           # Model outputs
    │   ├── test_predictions.csv               # id, true_label, predicted_label, confidence
    │   ├── misclassified/                     # Visual inspection
    │   │   ├── img_0001.png
    │   │   ├── img_0002.png
    │   │   └── ...
    ├── timings.json                           # Total time, per-epoch time, eval time
    ├── splits/                                # Data split indices for reproducibility
    │   ├── train_indices.npy
    │   ├── val_indices.npy
    │   └── test_indices.npy
    └── env/                                   # Environment snapshot
        ├── conda_env.yml                      # or requirements.txt
        └── pip_freeze.txt
"""

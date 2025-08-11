# analysis.py
import json
import math
import traceback
from pathlib import Path
from statistics import mean, variance
from typing import Dict, List, Any, Optional

import pandas as pd

from config import ExperimentConfig, get_experiment_config
from results import get_experimental_results


def _first_scalar_from_csv(path: Path) -> Optional[float]:
    """Return the first scalar value from a 1-col CSV written without header, or None if missing/empty."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, header=None)
        if df.empty:
            return None
        return float(df.iat[0, 0])
    except Exception:
        return None


def _sample_variance(values: List[float]) -> float:
    """Sample variance (n-1). Return 0.0 if fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    return float(variance(values))


def _collect_metric_arrays(per_run: List[Dict[str, Any]], section: str) -> Dict[str, List[float]]:
    """
    Build {metric_key: [values...]} for a given section ('init'/'final'/'best').
    Skips non-numeric / NaN entries.
    """
    # Find a template of keys from the first run containing the section
    keys = None
    for r in per_run:
        sec = r.get(section)
        if isinstance(sec, dict) and sec:
            keys = list(sec.keys())
            break
    if not keys:
        return {}

    arrays: Dict[str, List[float]] = {k: [] for k in keys}
    for r in per_run:
        sec = r.get(section)
        if not isinstance(sec, dict):
            continue
        for k in keys:
            v = sec.get(k)
            if isinstance(v, (int, float)) and math.isfinite(v):
                arrays[k].append(float(v))
    return arrays


def _summarize(arrays: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    return {
        k: {"mean": float(mean(vals)), "variance": _sample_variance(vals)}
        for k, vals in arrays.items() if vals
    }


def _read_init_from_epoch_csvs(run_dir: Path) -> Dict[str, float]:
    """
    Init metrics are the *first* values of the per-epoch CSVs.
    loss_train.csv, loss_test.csv, acc_train.csv, acc_test.csv  -> first row
    Missing files are ignored.
    """
    init: Dict[str, float] = {}

    v = _first_scalar_from_csv(run_dir / "loss_train.csv")
    if v is not None:
        init["train_loss"] = v

    v = _first_scalar_from_csv(run_dir / "loss_test.csv")
    if v is not None:
        init["test_loss"] = v

    v = _first_scalar_from_csv(run_dir / "acc_train.csv")
    if v is not None:
        init["train_acc"] = v

    v = _first_scalar_from_csv(run_dir / "acc_test.csv")
    if v is not None:
        init["test_acc"] = v

    return init


def analyze_experiment(config: ExperimentConfig) -> Path:
    """
    Aggregate INIT, FINAL, BEST metrics across runs into:
      <results>/<exp>/aggregate.json
    """
    exp = get_experimental_results(config.name, create_mode=False)
    runs_root = exp.results_dir / "runs"
    if not runs_root.exists():
        raise FileNotFoundError(f"No runs directory found at: {runs_root}")

    run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run subdirectories in: {runs_root}")

    per_run: List[Dict[str, Any]] = []
    for rd in run_dirs:
        entry: Dict[str, Any] = {"id": rd.name}

        # INIT from first entries of epoch CSVs
        init_metrics = _read_init_from_epoch_csvs(rd)
        if init_metrics:
            entry["init"] = init_metrics

        # FINAL/BEST from stats.json (if present)
        stats_path = rd / "stats.json"
        if stats_path.exists():
            try:
                stats = json.loads(stats_path.read_text())
                if isinstance(stats.get("final"), dict):
                    entry["final"] = stats["final"]
                if isinstance(stats.get("best"), dict):
                    entry["best"] = stats["best"]
            except Exception:
                pass  # keep whatever we have

        # Keep runs that contributed any section
        if any(k in entry for k in ("init", "final", "best")):
            per_run.append(entry)

    if not per_run:
        raise RuntimeError(f"No usable metrics found under {runs_root}")

    # Aggregate sections
    init_arrays = _collect_metric_arrays(per_run, "init")
    final_arrays = _collect_metric_arrays(per_run, "final")
    best_arrays = _collect_metric_arrays(per_run, "best")

    out = {
        "experiment": config.name,
        "num_runs": len(per_run),
        "metrics": {
            "init": _summarize(init_arrays),
            "final": _summarize(final_arrays),
            "best": _summarize(best_arrays),
        },
        "runs": per_run,
    }

    out_path = exp.results_dir / "aggregate.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote aggregate to {out_path}")
    return out_path


def main():
    analyze_experiment(get_experiment_config("cifar10_baseline"))


if __name__ == "__main__":
    import experiments as _  # register configs

    try:
        main()
    except BaseException as e:
        print(e)
        traceback.print_exc()

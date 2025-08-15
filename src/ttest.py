#!/usr/bin/env python3
"""
ttest.py

Paired t-tests on test accuracy between two experiments stored under:
    results/<experiment_name>/runs/<NNN>/

The script expects each run directory to contain either:
  - stats.json with a "final.test_acc" field, OR
  - acc_test.csv (1-column, no header) from which the last value is used as final test accuracy.

Usage:
    python ttest.py <experiment_A> <experiment_B>

Example:
    python ttest.py cifar10_abs_dropout_2em2 cifar10_baseline

The function calculate_paired_ttests(exp_a, exp_b) can be imported and used directly.
"""

from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

# Import project helpers
# We rely on results.py (ExperimentalResults, RunResults, StatsResults) being in PYTHONPATH
from results import get_experimental_results, RunResults

def _final_test_acc_from_run(run: RunResults) -> Optional[float]:
    """
    Extract the final test accuracy for a single run.
    Priority:
      1) stats.json -> final.test_acc
      2) acc_test.csv -> last value
    Returns None if neither is available or value is not a finite float.
    """
    stats = run.stats  # ensures load() on first access
    acc = None

    # 1) stats.json "final.test_acc"
    if getattr(stats, "_final_metrics", None):
        acc = stats._final_metrics.get("test_acc")  # may be None

    # 2) fallback to last value in acc_test.csv
    if acc is None:
        series = getattr(stats, "_test_acc", None) or []
        if len(series) > 0:
            acc = series[-1]

    # sanitize
    try:
        acc = float(acc) if acc is not None else None
    except Exception:
        acc = None
    if acc is None or not math.isfinite(acc):
        return None
    return acc


def _collect_final_test_accs(exp_name: str) -> List[Optional[float]]:
    """
    Read final test accuracies from results/<experiment_name>/fast_eval_runs.json.
    Returns a list ordered by run index (if present) or run id; entries are None
    when a value is missing or non-finite.
    """
    accs: List[Optional[float]] = []
    # Locate experiment directory via results helper (ensures consistent root)
    exp = get_experimental_results(exp_name, create_mode=False)
    fast_path = exp.results_dir / "fast_eval_runs.json"

    if not fast_path.exists():
        raise FileNotFoundError(f"fast_eval_runs.json not found for experiment '{exp_name}' at {fast_path}")

    with open(fast_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = data.get("runs", [])
    # Sort runs by explicit 'index' (0-based) when available; otherwise by numeric id if possible.
    def _sort_key(r):
        if isinstance(r.get("index"), int):
            return r["index"]
        rid = r.get("id")
        try:
            return int(rid)
        except Exception:
            return 10**9  # push unparseable ids to the end

    runs_sorted = sorted(runs, key=_sort_key)

    for r in runs_sorted:
        final = r.get("final") or {}
        acc = final.get("test_acc", None)
        try:
            val = float(acc) if acc is not None else None
        except Exception:
            val = None
        if val is not None and not math.isfinite(val):
            val = None
        accs.append(val)

    return accs


def _paired_indices(a: List[Optional[float]], b: List[Optional[float]]) -> List[Tuple[int, float, float]]:
    """
    Return list of tuples (i, a_i, b_i) for indices where both experiments
    have a finite accuracy value.
    """
    n = min(len(a), len(b))
    out = []
    for i in range(n):
        ai, bi = a[i], b[i]
        if ai is None or bi is None:
            continue
        if not (math.isfinite(ai) and math.isfinite(bi)):
            continue
        out.append((i, ai, bi))
    return out


def _paired_t_statistic(diffs: List[float]) -> Tuple[float, int]:
    """
    Compute the paired-sample t statistic and degrees of freedom.
    t = dbar / (sd_d / sqrt(n)), df = n-1
    where sd_d is the sample standard deviation of differences (denominator n-1).
    """
    n = len(diffs)
    if n < 2:
        return float("nan"), 0
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    sd_d = math.sqrt(var_d)
    if sd_d == 0.0:
        t = float("inf") if mean_d > 0 else (-float("inf") if mean_d < 0 else float("nan"))
    else:
        t = mean_d / (sd_d / math.sqrt(n))
    return t, n - 1


def _p_value_from_t(t: float, df: int) -> Optional[float]:
    """
    Two-sided p-value for a t statistic with df degrees of freedom.
    Tries SciPy if available; otherwise returns None.
    """
    try:
        from scipy.stats import t as tdist  # type: ignore
    except Exception:
        return None
    # two-sided p-value
    # sf gives survival function (1-CDF) for |t|
    return 2.0 * tdist.sf(abs(t), df)


def calculate_paired_ttests(experiment_a: str, experiment_b: str) -> None:
    """
    Calculate and print paired t-tests comparing final test accuracy
    of experiment A against experiment B (A - B).
    """
    accs_a = _collect_final_test_accs(experiment_a)
    accs_b = _collect_final_test_accs(experiment_b)

    pairs = _paired_indices(accs_a, accs_b)
    diffs = [a - b for (_, a, b) in pairs]

    print("=" * 72)
    print(f"Paired t-test on final test accuracy: '{experiment_a}' vs '{experiment_b}' (A - B)")
    print("- Experiments dir: results/<experiment>/runs/<NNN>/")
    print(f"- Runs discovered: A={len(accs_a)}, B={len(accs_b)}")
    print(f"- Paired usable runs: {len(diffs)}")
    skipped = max(len(accs_a), len(accs_b)) - len(diffs)
    if skipped > 0:
        print(f"- Note: skipped {skipped} run(s) due to missing/invalid accuracy values.")

    if len(diffs) < 2:
        print("Not enough paired data (need at least 2 pairs) to compute a t-test.")
        return

    # Descriptive stats
    mean_a = sum(a for (_, a, _) in pairs) / len(pairs)
    mean_b = sum(b for (_, _, b) in pairs) / len(pairs)
    mean_diff = sum(diffs) / len(diffs)

    t_stat, df = _paired_t_statistic(diffs)
    pval = _p_value_from_t(t_stat, df)

    print("\nDescriptive statistics on paired runs:")
    print(f"- Mean test acc (A): {mean_a:.4f}")
    print(f"- Mean test acc (B): {mean_b:.4f}")
    print(f"- Mean difference (A - B): {mean_diff:.4f}")

    print("\nPaired t-test:")
    print(f"- t statistic: {t_stat:.6g}")
    print(f"- degrees of freedom: {df}")
    if pval is None:
        print("- p-value: (SciPy not found) install scipy for exact p-values, e.g., `pip install scipy`")
    else:
        print(f"- two-sided p-value: {pval:.6g}")

    # Optional: show per-run diffs
    print("\nPer-run summary (index is 1-based run id):")
    for (i, a, b), d in zip(pairs, diffs):
        print(f"  run {i+1:03d}: A={a:.4f}  B={b:.4f}  (A-B)={d:.4f}")
    print("=" * 72)


def main():
    # ap = argparse.ArgumentParser(description="Paired t-tests on test accuracy between two experiments.")
    # ap.add_argument("experiment_a", type=str, help="First experiment name (A) under results/")
    # ap.add_argument("experiment_b", type=str, help="Second experiment name (B) under results/")
    # args = ap.parse_args()
    # calculate_paired_ttests("cifar10_abs_dropout_3em2", "cifar10_baseline")
    # calculate_paired_ttests("cifar10_abs_dropout_3em2", "cifar10_std_dropout_3em2")
    calculate_paired_ttests("cifar10_std_dropout_3em2", "cifar10_baseline")


if __name__ == "__main__":
    main()

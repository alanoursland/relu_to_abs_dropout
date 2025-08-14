#!/usr/bin/env python3
# container_main.py
"""
Container entrypoint for a single ML training run + results upload.

Overview
--------
This program is designed to be the default command of a Docker image that you
run on Salad's GPU nodes (or any Linux host). It orchestrates exactly ONE run:

1) Parse CLI args (experiment name, run index, seed, hyperparams).
2) Call your project's per-run training API (train.run_single(...)).
3) Zip the run directory (results/<experiment>/runs/<NNN>.zip).
4) Upload the archive to a short-lived, pre-signed GCS URL (RESULT_URL).
5) Print a concise machine-readable "RUN_STATUS ..." line and exit.

Why one-run-per-process?
------------------------
Short-lived, embarrassingly-parallel jobs (your CIFAR sweeps) map cleanly to
"one process = one run". This improves isolation, retries, metrics, and costs.

Requirements
------------
- Python 3.9+ inside the container.
- The image must include the 'requests' package for HTTP PUT uploads.
  (Add `requests` to requirements.txt)
- Your codebase must expose:
    train.run_single(
        experiment_name: str,
        run_index: int,          # 0-based, maps to runs/<NNN> with NNN = run_index+1 zero-padded
        dataset: str,
        model: str,
        seed: int,
        lr: float,
        batch_size: int,
        epochs: int,
        tag: str = "",
    ) -> dict

  The dict MUST include at least:
    { "run_dir": "results/<experiment>/runs/<NNN>", ... }

- The training function is expected to create (via your results.py utilities)
  files such as:
    model_init.pth, model_best.pth, model_final.pth,
    loss_*.csv, acc_*.csv, predictions_*.csv, etc.

Environment Variables
---------------------
- RESULT_URL  (required): GCS V4 signed URL for HTTP PUT of the zip file.
                          Must have Content-Type=application/zip in the signature.
- RUN_TIMEOUT_SEC        (optional): hard cap for total runtime (integer).
- LOG_LEVEL              (optional): INFO (default) or DEBUG.

Exit Codes
----------
0   success (train finished and upload succeeded)
1   training error (exception in run_single or downstream code)
2   configuration error (missing args/env or invalid values)
3   upload error (non-2xx response, timeout, or network failure)
130 interrupted (SIGINT/SIGTERM surfaced)

Stdout Contract (final line)
----------------------------
On success:
  RUN_STATUS ok run_index=NNN seed=S dataset=cifar10 model=... elapsed_sec=... uploaded=runs/NNN.zip
On failure:
  RUN_STATUS fail run_index=NNN reason=<short_code> details=<brief>

Usage Examples
--------------
# Local smoke test (skip real upload)
RESULT_URL="https://example.invalid/unused" \
python3 container_main.py \
  --experiment cifar10_smoke --run_index 0 --seed 123 \
  --dataset cifar10 --model resnet18_mod --lr 0.1 --batch_size 128 --epochs 1 \
  --allow-missing-upload

# Normal run (Salad will inject a real RESULT_URL per job)
python3 container_main.py \
  --experiment cifar10_absdrop_v1 --run_index 17 --seed 8675309 \
  --dataset cifar10 --model resnet18_mod --lr 0.05 --batch_size 128 --epochs 90
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import shutil
import signal
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import requests


# ------------------------- Helpers ------------------------- #

def _debug(msg: str):
    if os.environ.get("LOG_LEVEL", "INFO").upper() == "DEBUG":
        print(f"[DEBUG] {msg}", flush=True)


def _zip_run_dir(run_dir: Path) -> Path:
    """Create a .zip alongside the run directory (results/<exp>/runs/NNN.zip)."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    base = run_dir.with_suffix("")  # .../runs/NNN  (shutil will append .zip)
    zip_path = shutil.make_archive(base.as_posix(), "zip", run_dir.as_posix())
    return Path(zip_path)


def _put_signed(url: str, file_path: Path, content_type: str = "application/zip", timeout_sec: int = 900):
    """HTTP PUT upload to a pre-signed URL; raises for non-2xx responses."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Archive not found: {file_path}")
    headers = {"Content-Type": content_type}
    with file_path.open("rb") as f:
        resp = requests.put(url, data=f, headers=headers, timeout=timeout_sec)
    if not (200 <= resp.status_code < 300):
        # include limited response text to help debug signatures/mime/type
        snippet = (resp.text or "")[:200].replace("\n", " ")
        raise requests.HTTPError(f"Upload failed: HTTP {resp.status_code} {snippet}")


_TERMINATING = False
def _on_term(signum, frame):
    # Just mark and allow main to exit soon; container runtimes send SIGTERM for preemption.
    global _TERMINATING
    _TERMINATING = True
    print("RUN_STATUS fail reason=terminated details=received_signal", flush=True)


# ------------------------- Argument Parsing ------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one ML experiment and upload results archive.")
    p.add_argument("--experiment", required=True, help="Experiment name (results/<experiment> root).")
    p.add_argument("--run_index", type=int, required=True, help="0-based run index; maps to runs/<NNN>.")
    p.add_argument("--seed", type=int, required=True, help="Random seed for this run.")

    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--model", default="resnet18_mod")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--tag", default="", help="Freeform label to include in results metadata.")

    p.add_argument("--upload-timeout-sec", type=int, default=900, help="HTTP PUT timeout.")
    p.add_argument("--archive", choices=["zip", "none"], default="zip", help="Whether to zip the run dir.")
    p.add_argument("--results-root", default="results", help="Root results directory (default: results).")
    p.add_argument("--allow-missing-upload", action="store_true",
                   help="If set, treat upload errors as warnings (exit 0).")

    return p


# ------------------------- Main Orchestration ------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    args = _build_arg_parser().parse_args(argv)

    # Validate environment
    signed_url = os.environ.get("RESULT_URL", "").strip()
    if not signed_url and not args.allow_missing_upload:
        print("RUN_STATUS fail reason=missing_env details=RESULT_URL_required", flush=True)
        return 2

    # Optional global timeout
    time_cap = os.environ.get("RUN_TIMEOUT_SEC")
    deadline = None
    if time_cap:
        try:
            deadline = time.time() + int(time_cap)
        except ValueError:
            print("RUN_STATUS fail reason=bad_timeout details=RUN_TIMEOUT_SEC_not_int", flush=True)
            return 2

    start = time.time()

    # 1) Run exactly one training job via your codebase
    try:
        # Import here so container can start even if train deps are heavy.
        from train import run_single  # type: ignore
    except Exception as e:
        print("RUN_STATUS fail reason=import_error details=train.run_single_not_found", flush=True)
        traceback.print_exc()
        return 2

    if deadline and time.time() > deadline:
        print("RUN_STATUS fail reason=timeout details=before_training", flush=True)
        return 130

    try:
        info: Dict[str, Any] = run_single(
            experiment_name=args.experiment,
            run_index=args.run_index,
            dataset=args.dataset,
            model=args.model,
            seed=args.seed,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            tag=args.tag,
        )
    except BaseException as e:
        print(f"RUN_STATUS fail run_index={args.run_index} reason=train_error details={type(e).__name__}", flush=True)
        traceback.print_exc()
        return 1

    if _TERMINATING:
        return 130

    # 2) Resolve run_dir and optionally create archive
    run_dir = Path(info.get("run_dir", ""))
    if not run_dir:
        print("RUN_STATUS fail reason=missing_run_dir details=run_single_return", flush=True)
        return 1
    if not run_dir.exists():
        print(f"RUN_STATUS fail reason=run_dir_not_found details={run_dir}", flush=True)
        return 1

    if deadline and time.time() > deadline:
        print("RUN_STATUS fail reason=timeout details=after_training", flush=True)
        return 130

    archive_path: Optional[Path] = None
    if args.archive == "zip":
        try:
            archive_path = _zip_run_dir(run_dir)
            _debug(f"Zipped archive at {archive_path}")
        except BaseException as e:
            print(f"RUN_STATUS fail reason=zip_error details={type(e).__name__}", flush=True)
            traceback.print_exc()
            return 1

    # 3) Upload (if URL provided)
    if signed_url:
        if not archive_path:
            print("RUN_STATUS fail reason=no_archive details=archive=none_but_upload_required", flush=True)
            return 2
        try:
            _put_signed(signed_url, archive_path, content_type="application/zip", timeout_sec=args.upload_timeout_sec)
        except requests.HTTPError as e:
            if args.allow_missing_upload:
                print(f"RUN_STATUS ok-with-warning run_index={args.run_index} reason=upload_http_error", flush=True)
                _debug(str(e))
            else:
                print(f"RUN_STATUS fail run_index={args.run_index} reason=upload_http_error", flush=True)
                _debug(str(e))
                return 3
        except requests.RequestException as e:
            if args.allow_missing_upload:
                print(f"RUN_STATUS ok-with-warning run_index={args.run_index} reason=upload_network_error", flush=True)
                _debug(str(e))
            else:
                print(f"RUN_STATUS fail run_index={args.run_index} reason=upload_network_error", flush=True)
                _debug(str(e))
                return 3

    elapsed = time.time() - start
    uploaded_name = archive_path.name if archive_path else "none"
    print(
        "RUN_STATUS ok "
        f"run_index={args.run_index} seed={args.seed} dataset={args.dataset} model={args.model} "
        f"elapsed_sec={elapsed:.2f} uploaded={uploaded_name}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        import experiments as _  # only load/execute decorators in the main process
        rc = main()
        # Surface SIGTERM/SIGINT consistently as 130 if flagged late
        if _TERMINATING and rc == 0:
            rc = 130
        sys.exit(rc)
    except SystemExit as e:
        raise
    except BaseException as e:
        print(f"RUN_STATUS fail reason=unexpected details={type(e).__name__}", flush=True)
        traceback.print_exc()
        sys.exit(1)

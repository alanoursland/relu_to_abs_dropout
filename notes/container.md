Here’s a rewritten `container.md` that matches the new `container_main.py` design and keeps it concise but complete for someone picking up the project:

---

# Containerized Training Jobs

This document explains how to run a **single training run** of your experiments inside a container, using `container_main.py`.

---

## Overview

The container workflow is designed for **batch-processing environments** (e.g., Salad, Kubernetes, cloud batch systems).
Each container run executes:

1. **Load Experiment Config** from `experiments.py` via `--experiment`.
2. **Resolve Run Parameters** from the config and optional CLI overrides.
3. **Train a Single Run** using `train.run_single(...)`.
4. **Package Results** (e.g., into a `.zip` file).
5. **Upload to Cloud** via a pre-signed URL from `RESULT_URL`.

---

## Requirements

### Environment Variables

* **`RESULT_URL`** *(required unless `--allow-missing-upload`)*
  Pre-signed HTTP PUT URL to receive the zipped results directory.
* **`RUN_TIMEOUT_SEC`** *(optional)*
  Max runtime in seconds before the container exits.
* **`LOG_LEVEL`** *(optional)*
  Defaults to `INFO`; can be `DEBUG`.

### Code Requirements

* `experiments.py` must register all experiments via `@experiment()`.
* `train.py` must provide:

  ```python
  def run_single(experiment_name: str, run_index: int, **kwargs) -> dict:
      # returns at least {"run_dir": "<path-to-results>"}
  ```
* `config.py` should expose `apply_overrides` and `materialize_run` to merge CLI overrides with experiment defaults.

---

## CLI Usage

```bash
python container_main.py \
    --experiment EXPERIMENT_NAME \
    --run_index N \
    [--seed 123] \
    [--no-prefer-config-seed] \
    [--override-lr 0.01] \
    [--override-batch-size 256] \
    [--override-epochs 200] \
    [--override-dataset cifar100] \
    [--override-model resnet18] \
    [--tag mytag] \
    [--results-root results] \
    [--archive zip] \
    [--allow-missing-upload]
```

---

## Parameter Resolution

1. **Experiment defaults** come from `ExperimentConfig` in `experiments.py`.
2. **Overrides** via `--override-*` flags replace config values for this run only.
3. **Seed selection**:

   * Default: `cfg.random_seeds[run_index]` if available.
   * `--seed` flag overrides.
   * `--no-prefer-config-seed` forces use of `--seed` or a deterministic fallback.

---

## Output Layout

Each run produces a directory:

```
results/<experiment_name>/runs/<NNN>/
    checkpoints/
        init.pt
        best.pt
        final.pt
    metrics.json
    logs.txt
```

If `--archive zip` is used, the directory is zipped and uploaded.

---

## Exit Codes

| Code | Meaning                       |
| ---- | ----------------------------- |
| 0    | Success                       |
| 1    | Training error                |
| 2    | Configuration or import error |
| 3    | Upload error                  |
| 130  | Interrupted (SIGINT/SIGTERM)  |

---

## Example: Salad Batch Job

**Job setup:**

1. Build a Docker image with this repo.
2. Launch N containers in parallel, each with:

   * `--experiment EXPERIMENT_NAME`
   * A unique `--run_index`
   * A unique `RESULT_URL`

Example command inside container:

```bash
python container_main.py \
    --experiment resnet_cifar10 \
    --run_index 5 \
    --archive zip \
    --tag batch1
```

---

Do you want me to also include a **"Developer Notes"** section in `container.md` that explains how to extend configs and overrides for future datasets like CIFAR-100? That could make this even more future-proof.

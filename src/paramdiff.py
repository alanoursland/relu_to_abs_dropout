# paramdiff.py
import json
from math import acos, degrees
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Project imports
from results import get_experimental_results  # exp & run discovery (results/<exp>/runs/<NNN>/)  :contentReference[oaicite:3]{index=3}
from results import RunResults                                                     # load_*_model wrappers             :contentReference[oaicite:4]{index=4}
from config import get_experiment_config                                          # rebuild model from registry       :contentReference[oaicite:5]{index=5}


def _rebuild_model(exp_name: str) -> nn.Module:
    cfg = get_experiment_config(exp_name)
    if getattr(cfg, "model_fn", None) is None:
        raise ValueError(f"[{exp_name}] config.model_fn is None; cannot rebuild model.")
    return cfg.model_fn()


def _vectorize_state_dict(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten all floating-point tensors in a state_dict into one 1-D vector (float64 on CPU)."""
    parts: List[torch.Tensor] = []
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        if not v.is_floating_point():
            continue
        parts.append(v.detach().to("cpu", dtype=torch.float64).reshape(-1))
    if not parts:
        return torch.zeros(0, dtype=torch.float64)
    return torch.cat(parts, dim=0)


def _l1(v: torch.Tensor, w: torch.Tensor) -> float:
    return float(torch.linalg.norm(v - w, ord=1).item())


def _l2(v: torch.Tensor, w: torch.Tensor) -> float:
    return float(torch.linalg.norm(v - w).item())


def _cos_angle_deg(v: torch.Tensor, w: torch.Tensor) -> Optional[float]:
    """Return the angle (in degrees) between v and w; None if a vector is all zeros."""
    v_norm = torch.linalg.norm(v)
    w_norm = torch.linalg.norm(w)
    if v_norm == 0 or w_norm == 0:
        return None
    cos_sim = torch.clamp((v @ w) / (v_norm * w_norm), -1.0, 1.0)
    return float(degrees(acos(cos_sim.item())))


def _load_tag_into_vector(run: RunResults, tag: str, model: nn.Module) -> Optional[torch.Tensor]:
    """
    Load a checkpoint tag ('init'|'final'|'best') into model; return vectorized params.
    Uses RunResults.load_*_model helpers under the hood.
    """
    if tag == "init":
        ok = run.load_init_model(model, optimizer=None, strict=True, map_location="cpu")  # :contentReference[oaicite:6]{index=6}
    elif tag == "final":
        ok = run.load_final_model(model, optimizer=None, strict=True, map_location="cpu") # :contentReference[oaicite:7]{index=7}
    elif tag == "best":
        ok = run.load_best_model(model, optimizer=None, strict=True, map_location="cpu")  # :contentReference[oaicite:8]{index=8}
    else:
        raise ValueError(f"Unknown tag: {tag}")

    if not ok:
        return None

    # Move tensors safely to CPU & vectorize
    state = {k: t.detach().to("cpu") for k, t in model.state_dict().items()}
    return _vectorize_state_dict(state)


def _pairwise_compare(exp_a: str, exp_b: str) -> Dict:
    """
    For each run index i shared by both experiments, compare init/final/best parameter vectors.
    Returns a JSON-serializable dict with per-run and per-tag metrics (L1, L2, cosine angle in degrees).
    """
    # Discover experiments & how many runs to attempt
    A = get_experimental_results(exp_a, create_mode=False)  # results/<A>
    B = get_experimental_results(exp_b, create_mode=False)  # results/<B>
    num_runs_a = int(getattr(A.metadata, "num_runs", 0))
    num_runs_b = int(getattr(B.metadata, "num_runs", 0))
    n = min(num_runs_a, num_runs_b)

    out_runs: List[Dict] = []
    tags = ["init", "final", "best"]

    print(f"Comparing params for '{exp_a}' vs '{exp_b}' over {n} paired run(s) [A has {num_runs_a}, B has {num_runs_b}].")

    for idx in range(n):
        run_a = A.get_run(idx)  # results/<A>/runs/<NNN>                          :contentReference[oaicite:9]{index=9}
        run_b = B.get_run(idx)  # results/<B>/runs/<NNN>

        # Rebuild models (ensures same architecture as originally trained)      :contentReference[oaicite:10]{index=10}
        model_a = _rebuild_model(exp_a)
        model_b = _rebuild_model(exp_b)

        run_entry = {"index": idx, "id_a": run_a.id, "id_b": run_b.id, "tags": {}}

        for tag in tags:
            va = _load_tag_into_vector(run_a, tag, model_a)
            vb = _load_tag_into_vector(run_b, tag, model_b)

            if va is None or vb is None or va.numel() == 0 or vb.numel() == 0:
                run_entry["tags"][tag] = {
                    "available": False,
                    "note": f"Missing or empty checkpoint for tag='{tag}' in one or both runs.",
                }
                continue

            # Align length (defensive; architectures should match but safeguard anyway)
            m = min(va.numel(), vb.numel())
            va2, vb2 = va[:m], vb[:m]

            run_entry["tags"][tag] = {
                "available": True,
                "l1": _l1(va2, vb2),
                "l2": _l2(va2, vb2),
                "cos_angle_deg": _cos_angle_deg(va2, vb2),
                "num_params_compared": int(m),
            }

        out_runs.append(run_entry)

    return {
        "experiment_a": exp_a,
        "experiment_b": exp_b,
        "num_paired_runs": len(out_runs),
        "runs": out_runs,
        "explanations": {
            "l1": "Sum of absolute differences of flattened parameters",
            "l2": "Euclidean distance between flattened parameter vectors",
            "cos_angle_deg": "Angle in degrees between the two parameter vectors",
        },
    }


def compare_params(experiment_a: str, experiment_b: str, save_json: bool = True) -> Dict:
    """
    Pair runs across two experiments and compute L1/L2/cos-angle for init/final/best checkpoints.
    Prints a compact summary and optionally saves a JSON report under results/<A>_vs_<B>/param_distances.json
    """
    report = _pairwise_compare(experiment_a, experiment_b)

    # Pretty print a brief summary to stdout
    print("=" * 72)
    print(f"Parameter distances: '{experiment_a}' vs '{experiment_b}'")
    for r in report["runs"]:
        idx = r["index"]
        print(f"run {idx+1:03d}:")
        for tag, d in r["tags"].items():
            if not d.get("available", False):
                print(f"  {tag:>5}: (missing)")
                continue
            l1 = d["l1"]
            l2 = d["l2"]
            ang = d["cos_angle_deg"]
            ncmp = d["num_params_compared"]
            ang_str = f"{ang:.6f}°" if ang is not None else "None"
            print(f"  {tag:>5}: L1={l1:.6g}  L2={l2:.6g}  Angle={ang_str}  (N={ncmp})")
    print("=" * 72)

    # Save JSON next to results
    if save_json:
        safe_dir = Path("results") / f"{experiment_a}_vs_{experiment_b}"
        safe_dir.mkdir(parents=True, exist_ok=True)
        out_path = safe_dir / "param_distances.json"
        out_path.write_text(json.dumps(report, indent=2))
        print(f"[paramdiff] wrote {out_path}")

    return report


def main():
    # Example from your prompt; adjust as needed
    compare_params("cifar10_std_dropout_3em2", "cifar10_baseline")


if __name__ == "__main__":
    # Ensure your experiments registry is imported somewhere if required,
    # e.g., `import experiments as _`
    try:
        import experiments as _  # noqa: F401
    except Exception:
        pass
    main()

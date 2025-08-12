That plan makes sense. Two key points:

1. **Turn off dropout, but refresh BatchNorm stats.**
   If you just set `dropout_rate=0` and keep BN’s running means/vars from the dropout phase, you’ll carry a small mismatch. Do a short **BN recalibration pass** with dropout off: run forward over the train loader in `model.train()` with `no_grad()` and **no optimizer step** so BN updates its running stats to the new regime.

2. **Then fine-tune a bit with a small LR.**
   A few epochs (e.g., 1–5) at 10× smaller LR than before is usually enough. Keep weight decay modest; early-stop on val/train as you prefer.

Here’s a drop-in utility for your codebase:

```python
# utils_bn.py
import torch

def set_abs_dropout(model, p: float):
    # Your activation is on model.relu in ResNet18_CIFAR10
    if hasattr(model, "relu") and hasattr(model.relu, "dropout_rate"):
        model.relu.dropout_rate = float(p)

@torch.no_grad()
def recalibrate_bn(model, loader, device, max_batches=None):
    """
    Recompute BN running mean/var with dropout OFF.
    Use training transforms & model.train() so BN updates, but do not update weights.
    """
    was_training = model.training
    model.train()  # so BN updates running stats
    cnt = 0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        model(xb)
        cnt += 1
        if max_batches is not None and cnt >= max_batches:
            break
    if not was_training:
        model.eval()
```

Example flow inside your training script after the main training finishes:

```python
# after run_training_loop(...) has saved final/best
# 1) reload the final (or best) weights
run_results.load_final_model(model)  # or load_best_model

# 2) turn off abs-dropout
from utils_bn import set_abs_dropout, recalibrate_bn
set_abs_dropout(model, 0.0)

# 3) recalibrate BN stats (no weight updates)
recalibrate_bn(model, config.train_loader, device, max_batches=200)  # a few hundred batches is plenty

# 4) optional: short fine-tune with small LR
for g in optimizer.param_groups:
    g["lr"] *= 0.1
epochs_finetune = 1 if config.epochs < 10 else 3  # tweak as desired

model.train()
for epoch in range(epochs_finetune):
    for xb, yb in config.train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

# 5) eval and save new “no-dropout” metrics/checkpoint if you want
run_results.save_checkpoint("finetuned_nodrop", model, optimizer)
```

Notes & gotchas:

* **State dicts:** flipping `dropout_rate` won’t affect `state_dict`; you can safely load checkpoints trained with dropout, then zero it out.
* **BN momentum:** if training is very short, you can lower BN momentum (e.g., 0.01) during recalibration to average more aggressively.
* **Transforms:** use **train-time transforms** for recalibration (match the distribution BN saw during training).
* **Logging:** write a new tag (e.g., `model_finetuned_nodrop.pth`) to keep provenance clean.

This gives you the regularization benefits of training with abs-dropout, then the clean determinism and slightly sharper decision surfaces of dropout-off at the end — plus well-calibrated BN so test/inference matches what the network now expects.

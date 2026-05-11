from verification.registry import register

register(
    name="loss_convergence",
    description="Verify training loss decreases and converges",
    priority=85,
    applicability_hint="Apply when a model is trained iteratively (neural networks, gradient descent, any optimization loop with a loss function).",
    prompt_snippet="""\
Loss convergence check:
- Training loss should generally decrease over epochs/steps
- Final loss should be significantly lower than initial loss
- Check for loss plateau (last 10% of training has < 1% improvement)
- Loss should not be NaN, Inf, or negative (unless the loss function allows it)
- Sudden spikes may indicate learning rate issues or data problems""",
    code_example="""\
import numpy as np
losses = np.array(train_losses)
if not np.all(np.isfinite(losses)):
    print("FAIL: Loss contains NaN or Inf values")
elif losses[-1] >= losses[0]:
    print(f"FAIL: Loss did not decrease (start={losses[0]:.4f}, end={losses[-1]:.4f})")
else:
    improvement = (losses[0] - losses[-1]) / abs(losses[0]) * 100
    print(f"PASS: Loss decreased by {improvement:.1f}% (start={losses[0]:.4f}, end={losses[-1]:.4f})")
# Check for spikes
diffs = np.diff(losses)
spike_threshold = 3 * np.std(diffs)
spikes = np.where(diffs > spike_threshold)[0]
if len(spikes) > 0:
    print(f"WARNING: {len(spikes)} loss spikes detected at steps {spikes.tolist()}")""",
)

register(
    name="overfitting_detection",
    description="Detect overfitting by comparing train vs validation metrics",
    priority=80,
    applicability_hint="Apply when both training and validation metrics are tracked (any supervised learning task with a held-out set).",
    prompt_snippet="""\
Overfitting detection — check for signs that the model memorizes training data:
- Primary signal: validation loss INCREASES or plateaus while training loss decreases.
  This is the strongest overfitting indicator.
- Secondary signal: large val/train loss ratio. But note that SOME gap is normal
  and expected in deep learning, especially with regularization. A ratio of val_loss /
  train_loss up to ~10x can be normal if validation loss is still decreasing.
- Only flag overfitting if validation loss is clearly worsening or the ratio is extreme (>20x).
- Also check if validation accuracy stopped improving in the last 30% of training.""",
    code_example="""\
import numpy as np
val_losses_arr = np.array(val_losses)
train_losses_arr = np.array(train_losses)
# Check if val loss increased in last 30% of training
n = len(val_losses_arr)
late_start = max(1, int(n * 0.7))
val_late = val_losses_arr[late_start:]
val_increasing = len(val_late) > 1 and val_late[-1] > val_late[0]
ratio = val_losses_arr[-1] / train_losses_arr[-1] if train_losses_arr[-1] > 0 else 0
if val_increasing and ratio > 10:
    print(f"FAIL: Overfitting detected. Val loss increasing (late: {val_late[0]:.4f} -> {val_late[-1]:.4f}), ratio={ratio:.1f}x")
elif ratio > 20:
    print(f"FAIL: Extreme train/val gap (ratio={ratio:.1f}x)")
elif val_increasing:
    print(f"WARNING: Val loss slightly increasing in late training, monitor for overfitting")
else:
    print(f"PASS: No overfitting detected (val/train ratio={ratio:.1f}x, val loss still improving)")""",
)

register(
    name="metric_bounds",
    description="Verify metrics are within valid mathematical bounds",
    priority=75,
    applicability_hint="Apply when computing bounded metrics like accuracy, precision, recall, F1, AUC, R-squared, etc.",
    prompt_snippet="""\
Metric bounds validation:
- Accuracy, precision, recall, F1, AUC should be in [0, 1]
- R-squared can be negative (poor fit) but values < -1 are suspicious
- Loss should typically be non-negative (cross-entropy, MSE, MAE)
- Perplexity should be >= 1
- Any metric outside its mathematical bounds indicates a bug""",
    code_example="""\
import numpy as np
bounded_metrics = {"accuracy": (0, 1), "precision": (0, 1), "recall": (0, 1), "f1": (0, 1)}
for name, (low, high) in bounded_metrics.items():
    if name in results:
        val = results[name]
        if val < low or val > high:
            print(f"FAIL: {name}={val:.4f} outside valid range [{low}, {high}]")
        else:
            print(f"PASS: {name}={val:.4f} within [{low}, {high}]")""",
)

register(
    name="learning_rate_schedule",
    description="Validate learning rate behavior during training",
    priority=50,
    applicability_hint="Apply when a learning rate schedule is used (warmup, cosine decay, step decay, etc.) and LR values are logged.",
    prompt_snippet="""\
Learning rate schedule validation:
- LR should follow the expected schedule pattern (warmup then decay, step drops, etc.)
- LR should never be negative or zero (unless intentionally paused)
- If warmup is used, initial LR should be small and ramp up
- Final LR should be smaller than peak LR for decay schedules""",
    code_example="""\
import numpy as np
lrs = np.array(learning_rates)
if np.any(lrs <= 0):
    print(f"FAIL: Learning rate <= 0 at steps {np.where(lrs <= 0)[0].tolist()}")
elif lrs[-1] > lrs.max() * 0.5 and len(lrs) > 10:
    print("WARNING: LR has not decayed significantly -- check if schedule is applied")
else:
    print(f"PASS: LR schedule looks reasonable (peak={lrs.max():.2e}, final={lrs[-1]:.2e})")""",
)

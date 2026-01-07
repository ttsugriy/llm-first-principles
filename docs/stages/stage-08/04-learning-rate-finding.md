# Section 8.4: Learning Rate Finding

*Reading time: 12 minutes*

## The Learning Rate Problem

The learning rate is the most important hyperparameter. Too high and training explodes. Too low and training takes forever (or never converges).

**The typical approach**: Try 1e-3, if that fails, try 1e-4, then 1e-2...

**The systematic approach**: LR Range Test.

## The LR Range Test

Invented by Leslie Smith (2017), this technique finds the optimal learning rate in a single run.

### Algorithm

1. Start with a very small LR ($10^{-7}$)
2. Train for one batch
3. Increase LR exponentially
4. Record loss at each LR
5. Stop when loss explodes

### The Loss-LR Curve

```
Loss
  │
  │                           ╱ Explosion
  │                          ╱
  │────────────────────____╱
  │  Too slow        ↑   ↑
  │               Best   Too fast
  └──────────────────────────── log(LR)
       10⁻⁷    10⁻⁵   10⁻³   10⁻¹
```

### Finding Optimal LR

Look for where loss decreases most steeply—that's your optimal LR.

**Rule of thumb**: Choose a LR about 10x smaller than where explosion begins.

## Implementation

```python
class LearningRateFinder:
    """Find optimal learning rate using range test."""

    def __init__(
        self,
        min_lr: float = 1e-7,
        max_lr: float = 10.0,
        num_steps: int = 100,
        smooth_factor: float = 0.05,
    ):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.smooth_factor = smooth_factor

    def range_test(self, train_fn):
        """
        Run LR range test.

        Args:
            train_fn: Takes LR, does one step, returns loss
        """
        # Exponential LR schedule
        lr_schedule = np.exp(np.linspace(
            np.log(self.min_lr),
            np.log(self.max_lr),
            self.num_steps
        ))

        lrs, losses = [], []
        smoothed_loss = None
        best_loss = float('inf')

        for lr in lr_schedule:
            loss = train_fn(lr)

            # Stop if exploding
            if np.isnan(loss) or loss > 10 * best_loss:
                break

            lrs.append(lr)
            losses.append(loss)
            best_loss = min(best_loss, loss)

            # Smooth for cleaner curve
            if smoothed_loss is None:
                smoothed_loss = loss
            else:
                smoothed_loss = 0.05 * loss + 0.95 * smoothed_loss

        return self._find_suggested_lr(lrs, losses)

    def _find_suggested_lr(self, lrs, losses):
        """Find LR with steepest negative slope."""
        # Compute gradient in log space
        log_lrs = np.log(lrs)
        gradients = np.gradient(losses, log_lrs)

        # Steepest descent point
        min_grad_idx = np.argmin(gradients)

        # Go back 10% for safety margin
        suggest_idx = max(0, min_grad_idx - len(losses) // 10)
        return lrs[suggest_idx]
```

## Using the LR Range Test

### Step 1: Prepare Your Model

```python
# Initialize fresh model and optimizer
model = create_model()
initial_weights = model.get_weights()  # Save for reset

# Single batch for testing
test_batch = next(iter(dataloader))
```

### Step 2: Define Training Function

```python
def train_step(lr):
    """One training step at given LR."""
    # Update optimizer LR
    optimizer.lr = lr

    # Forward + backward
    loss = model.train_on_batch(test_batch)
    return loss
```

### Step 3: Run Test

```python
lr_finder = LearningRateFinder(min_lr=1e-7, max_lr=10.0)
result = lr_finder.range_test(train_step)
print(f"Suggested LR: {result['suggested_lr']:.2e}")

# Reset model to initial state
model.set_weights(initial_weights)
```

## Interpreting Results

### Good Result

```
LR: 1e-7  Loss: 2.30
LR: 1e-6  Loss: 2.30
LR: 1e-5  Loss: 2.28
LR: 1e-4  Loss: 2.15  ← Starting to learn
LR: 1e-3  Loss: 1.85  ← Good progress
LR: 1e-2  Loss: 1.50  ← Best zone
LR: 1e-1  Loss: 2.80  ← Too fast
LR: 1.00  Loss: NaN   ← Explosion
```

**Suggested LR**: 1e-3 to 1e-2

### Problem: Flat Everywhere

```
LR: 1e-7  Loss: 2.30
LR: 1e-3  Loss: 2.30
LR: 1e-1  Loss: 2.30
```

**Meaning**: Model isn't learning at all. Check for bugs.

### Problem: Immediate Explosion

```
LR: 1e-7  Loss: 2.30
LR: 1e-6  Loss: 5.00
LR: 1e-5  Loss: NaN
```

**Meaning**: Something very wrong. Check initialization, data normalization.

## Common LR Ranges

Different architectures need different learning rates:

| Architecture | Typical LR Range |
|--------------|------------------|
| MLP | 1e-4 to 1e-2 |
| CNN | 1e-4 to 1e-2 |
| Transformer | 1e-5 to 1e-3 |
| Fine-tuning | 1e-6 to 1e-4 |
| Large batch | 1e-3 to 1e-1 |

## LR Schedules

Once you find the base LR, add a schedule:

### Warmup + Decay

```python
def warmup_cosine_schedule(step, warmup_steps, total_steps, base_lr):
    """Warmup then cosine decay."""
    if step < warmup_steps:
        # Linear warmup
        return base_lr * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
```

### One-Cycle Policy

```python
def one_cycle_schedule(step, total_steps, base_lr, max_lr):
    """One-cycle: warmup to max, then decay to near-zero."""
    pct = step / total_steps

    if pct < 0.3:  # Warmup phase
        return base_lr + (max_lr - base_lr) * (pct / 0.3)
    else:  # Decay phase
        return max_lr * (1 - (pct - 0.3) / 0.7) ** 2
```

## The Connection to Optimization

Why does LR matter so much?

$$\theta_{t+1} = \theta_t - \eta \nabla L$$

- LR ($\eta$) scales the step size
- Too large: Overshoot minimum, oscillate or explode
- Too small: Tiny steps, slow progress

**Loss surface analogy**: LR is your stride length walking down a mountain.

- Too long: You might leap past the valley into another mountain
- Too short: You'll take forever to reach the bottom

## Adaptive Methods

Adam, RMSprop, etc. adapt per-parameter:

$$\theta_i^{(t+1)} = \theta_i^{(t)} - \frac{\eta}{\sqrt{v_i^{(t)}} + \epsilon} m_i^{(t)}$$

They still need a base LR! The range test works for these too.

## Best Practices

1. **Always run LR range test** on new architectures
2. **Use warmup** for transformers
3. **Start conservative** (lower LR) when unsure
4. **Monitor grad norms** as LR sanity check
5. **Re-run test** after major architecture changes

## Summary

| Metric | Meaning |
|--------|---------|
| Flat loss everywhere | Model not learning (bug) |
| Steady decrease | Learning, LR might be higher |
| Sharp decrease | Optimal LR zone |
| Upturn/explosion | LR too high |

**Key insight**: The LR range test replaces guesswork with systematic experimentation. Always use it.

**Next**: We'll monitor activations to detect dead neurons and saturation.

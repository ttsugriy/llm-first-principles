# Section 4.6: Learning Rate Schedules — The Art of Annealing

*Reading time: 16 minutes | Difficulty: ★★★☆☆*

The learning rate is the most important hyperparameter. But a single fixed value is rarely optimal. This section covers how to vary the learning rate during training for faster convergence and better final performance.

## Why Schedules Matter

**Early training**: We want large steps to make rapid progress
**Late training**: We want small steps to fine-tune and converge

A fixed learning rate forces a compromise. Schedules let us have both.

```
        Learning Rate
              ↑
              │╲
              │ ╲
              │  ╲
              │   ╲─────────────
              │
              └─────────────────→ Steps
                   Decay schedule
```

## Common Schedules

### Step Decay

Reduce learning rate by a factor at fixed intervals:

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}$$

Where:
- η₀ is the initial learning rate
- γ is the decay factor (e.g., 0.1)
- s is the step size (e.g., every 30 epochs)

```python
def step_decay(step, init_lr, decay_rate=0.1, decay_every=30):
    """Step decay: reduce by factor every N epochs."""
    return init_lr * (decay_rate ** (step // decay_every))
```

**Pros**: Simple, interpretable, works well
**Cons**: Discontinuous, requires choosing when to decay

### Exponential Decay

Smooth continuous decay:

$$\eta_t = \eta_0 \cdot \gamma^t$$

```python
def exponential_decay(step, init_lr, decay_rate=0.99):
    """Exponential decay: multiply by rate each step."""
    return init_lr * (decay_rate ** step)
```

**Pros**: Smooth, no discontinuities
**Cons**: Decays too fast early, too slow late

### Inverse Square Root Decay

Popular for transformers:

$$\eta_t = \eta_0 \cdot \frac{1}{\sqrt{t}}$$

Or with warmup:

$$\eta_t = \eta_0 \cdot \min\left(t^{-0.5}, t \cdot \text{warmup}^{-1.5}\right)$$

```python
def inverse_sqrt_decay(step, init_lr, warmup_steps=4000):
    """Inverse square root with linear warmup (Transformer schedule)."""
    if step == 0:
        step = 1
    warmup_factor = min(1.0, step / warmup_steps)
    decay_factor = 1.0 / np.sqrt(max(step, warmup_steps))
    return init_lr * warmup_factor * decay_factor * np.sqrt(warmup_steps)
```

This is the original Transformer schedule from "Attention Is All You Need" (2017).

### Cosine Annealing

Smooth decay following a cosine curve:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

```python
def cosine_annealing(step, total_steps, init_lr, min_lr=0):
    """Cosine annealing from init_lr to min_lr."""
    return min_lr + 0.5 * (init_lr - min_lr) * (1 + np.cos(np.pi * step / total_steps))
```

```
        Learning Rate
              ↑
        η_max │─╲
              │  ╲
              │   ╲
              │    ╲
              │     ╲
              │      ╲_
        η_min │────────╲─
              └───────────→ Steps
                  Cosine shape
```

**Pros**: Smooth, no hyperparameters except endpoints
**Cons**: Need to know total training steps in advance

!!! info "Connection to Modern LLMs"

    **Cosine annealing is the standard for LLM training.**

    Typical settings:
    - Warmup: 1-2% of total steps
    - Peak learning rate: 1e-4 to 3e-4
    - Decay to: 0.1 × peak (or 0)
    - Total steps: 100K to 1M

    Example from LLaMA training:
    ```
    warmup_steps = 2000
    total_steps = 100000
    peak_lr = 3e-4
    min_lr = 3e-5  # 10% of peak
    ```

## Warmup: Starting Slow

### Why Warmup?

At initialization:
- Weights are random
- Gradients are unreliable
- Large steps could be catastrophic

Warmup starts with tiny learning rate and gradually increases:

```python
def linear_warmup(step, warmup_steps, target_lr):
    """Linear warmup from 0 to target_lr."""
    if step < warmup_steps:
        return target_lr * step / warmup_steps
    return target_lr


def warmup_cosine_decay(step, warmup_steps, total_steps, init_lr, min_lr=0):
    """Linear warmup followed by cosine decay."""
    if step < warmup_steps:
        # Linear warmup
        return init_lr * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (init_lr - min_lr) * (1 + np.cos(np.pi * progress))
```

```
        Learning Rate
              ↑
              │   ╱─╲
              │  ╱   ╲
              │ ╱     ╲
              │╱       ╲
              │         ╲_
              └───────────╲─→ Steps
              │←─→│
             warmup  decay
```

### How Much Warmup?

| Model Size | Typical Warmup |
|------------|----------------|
| Small (< 100M) | 100-1000 steps |
| Medium (100M-1B) | 1000-5000 steps |
| Large (> 1B) | 2000-10000 steps |

**Rule of thumb**: 1-5% of total training steps.

## Warmup for Large Batches

Large batch training requires careful warmup:

**The linear scaling rule** (Goyal et al., 2017):
- If batch size increases by k, multiply learning rate by k
- But warmup for k× longer

```python
def large_batch_schedule(step, base_lr, base_batch, actual_batch, warmup_steps):
    """Learning rate schedule for large batch training."""
    # Scale learning rate with batch size
    scale = actual_batch / base_batch
    target_lr = base_lr * scale

    # Extended warmup
    scaled_warmup = warmup_steps * scale

    if step < scaled_warmup:
        return target_lr * step / scaled_warmup
    return target_lr
```

## Cosine Annealing with Restarts

**Loshchilov & Hutter (2016)**: SGDR (SGD with Warm Restarts)

Instead of decaying once, reset to high learning rate periodically:

```
        Learning Rate
              ↑
              │╱╲    ╱╲   ╱╲
              │  ╲  ╱  ╲ ╱  ╲
              │   ╲╱    ╳    ╲
              │              ╲
              └───────────────→ Steps
                   Restarts
```

```python
def cosine_with_restarts(step, init_lr, restart_period, restart_mult=2):
    """Cosine annealing with warm restarts."""
    # Find which cycle we're in
    cycle = 0
    cycle_start = 0
    current_period = restart_period

    while step >= cycle_start + current_period:
        cycle_start += current_period
        current_period *= restart_mult
        cycle += 1

    # Position within current cycle
    progress = (step - cycle_start) / current_period

    return init_lr * 0.5 * (1 + np.cos(np.pi * progress))
```

**Why restarts help**:
- Escape local minima
- Explore different regions of loss landscape
- Create "snapshots" at each minimum for ensembling

## One-Cycle Policy

**Smith (2018)**: Train with one big cycle

1. Warmup from low to high learning rate
2. Anneal from high back to very low

```python
def one_cycle(step, total_steps, max_lr, div_factor=25, final_div=1e4):
    """One-cycle learning rate policy."""
    initial_lr = max_lr / div_factor
    final_lr = max_lr / final_div

    if step < total_steps * 0.3:
        # Warmup phase: 30% of training
        progress = step / (total_steps * 0.3)
        return initial_lr + (max_lr - initial_lr) * progress
    else:
        # Annealing phase: 70% of training
        progress = (step - total_steps * 0.3) / (total_steps * 0.7)
        return max_lr - (max_lr - final_lr) * progress
```

The one-cycle policy often allows training with much higher learning rates!

## Learning Rate Finding

**Smith (2015)**: Learning Rate Range Test

Before training, find the optimal learning rate:

1. Start with very small η (e.g., 1e-7)
2. Train for a few iterations, gradually increasing η
3. Plot loss vs learning rate
4. Choose η where loss decreases fastest (not the minimum!)

```python
def lr_finder(model, train_fn, init_lr=1e-7, final_lr=10, num_steps=100):
    """Find optimal learning rate by exponential sweep."""
    mult = (final_lr / init_lr) ** (1 / num_steps)
    lr = init_lr
    lrs, losses = [], []

    for step in range(num_steps):
        loss = train_fn(model, lr)

        lrs.append(lr)
        losses.append(loss)

        # Exponentially increase
        lr *= mult

        # Stop if loss explodes
        if loss > 4 * min(losses):
            break

    return lrs, losses
```

```
        Loss
          ↑
          │╲
          │ ╲
          │  ╲___      ╱
          │      ╲____╱
          │       ↑
          │    optimal
          └──────────────→ log(lr)
```

Choose learning rate at the steepest descent (before the minimum), typically 1/10 of the minimum loss learning rate.

## Practical Schedule for LLM Training

Here's a complete schedule matching modern practice:

```python
class LLMSchedule:
    """Learning rate schedule for LLM training."""

    def __init__(
        self,
        peak_lr=3e-4,
        min_lr=3e-5,
        warmup_steps=2000,
        total_steps=100000,
    ):
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step):
        # Phase 1: Linear warmup
        if step < self.warmup_steps:
            return self.peak_lr * step / self.warmup_steps

        # Phase 2: Cosine decay
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)  # Clamp to [0, 1]

        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay


# Usage
schedule = LLMSchedule(
    peak_lr=3e-4,
    min_lr=3e-5,  # 10% of peak
    warmup_steps=2000,
    total_steps=100000,
)

for step in range(100000):
    current_lr = schedule.get_lr(step)
    optimizer.lr = current_lr
    train_step(...)
```

## Schedule vs Optimizer Choice

The schedule and optimizer interact:

| Optimizer | Recommended Schedule |
|-----------|---------------------|
| SGD | Step decay or cosine |
| Momentum | Cosine with longer warmup |
| Adam | Cosine, less sensitive to schedule |
| AdamW | Cosine with warmup (standard) |

Adam's adaptive rates provide some "built-in" schedule, making it less sensitive to the explicit schedule. But even Adam benefits from warmup and decay.

## Historical Note

Learning rate schedules evolved with deep learning:

- **1990s**: Fixed learning rate, manually tuned
- **2012**: Step decay became standard (AlexNet)
- **2015**: Learning rate range test (Smith)
- **2016**: Cosine annealing, warm restarts (Loshchilov & Hutter)
- **2017**: Transformer schedule (Vaswani et al.)
- **2018**: One-cycle policy (Smith)
- **2020s**: Cosine with warmup is the default for LLMs

## Common Mistakes

!!! warning "Pitfalls"

    1. **No warmup**: Especially with Adam, can cause early instability

    2. **Too short warmup**: May not be enough for large models

    3. **Decaying too fast**: Loss plateaus before convergence

    4. **Decaying too slow**: Final loss higher than necessary

    5. **Not matching batch size**: When scaling batch size, scale warmup too

## Exercises

1. **Compare schedules**: Train the same model with constant, step decay, and cosine. Plot training curves.

2. **Warmup ablation**: Train with warmup = 0, 100, 1000, 10000 steps. What changes?

3. **LR finder**: Implement the learning rate range test. Find optimal learning rate for a model.

4. **Restarts**: Implement cosine with restarts. Compare to single cosine decay.

5. **One-cycle**: Implement the one-cycle policy. Can you use higher learning rates?

## Summary

| Schedule | Formula | Best For |
|----------|---------|----------|
| Step decay | η₀ × γ^⌊t/s⌋ | Simple, interpretable |
| Exponential | η₀ × γᵗ | Smooth decay |
| Inverse sqrt | η₀ / √t | Transformers (original) |
| Cosine | ½(1 + cos(πt/T)) | LLM training (standard) |
| Warmup + cosine | Linear → cosine | Modern best practice |

**Key takeaway**: The learning rate schedule is as important as the optimizer choice. Modern LLM training uses linear warmup followed by cosine decay to minimum. This simple schedule, combined with AdamW, forms the backbone of successful large-scale training.

→ **Next**: [Section 4.7: Implementation](07-implementation.md)

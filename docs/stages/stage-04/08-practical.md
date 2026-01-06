# Section 4.8: Practical Considerations — Making Training Work

*Reading time: 18 minutes | Difficulty: ★★★☆☆*

Theory is beautiful, but training real models involves countless practical details. This section covers the tricks, debugging strategies, and hard-won wisdom that make the difference between training runs that work and those that don't.

## Initialization Matters

### Why Initialization is Critical

At initialization, the network must:
1. Produce reasonable output magnitudes
2. Have gradients that can flow backward
3. Not saturate activations

**Bad initialization leads to**:
- Vanishing gradients (weights too small)
- Exploding gradients (weights too large)
- Dead neurons (ReLUs stuck at 0)

### Xavier/Glorot Initialization

For linear layers with tanh or sigmoid:

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

Or with normal distribution:
$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

### Kaiming/He Initialization

For ReLU networks:

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$$

The factor of 2 accounts for ReLU killing half the activations.

```python
def xavier_init(shape):
    """Xavier/Glorot initialization."""
    fan_in, fan_out = shape
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)


def kaiming_init(shape):
    """Kaiming/He initialization for ReLU."""
    fan_in = shape[0]
    std = np.sqrt(2 / fan_in)
    return np.random.randn(*shape) * std
```

!!! info "Connection to Modern LLMs"

    Transformers typically use:
    - **Normal initialization** with std = 0.02 for embeddings and projections
    - **Scaled initialization** for residual layers: std = 0.02 / √(2N) where N is number of layers
    - This prevents the residual sum from growing with depth

## Batch Size Selection

### The Compute-Quality Trade-off

| Batch Size | Pros | Cons |
|------------|------|------|
| Small (32-256) | Better generalization, less memory | Noisy, underutilizes GPU |
| Medium (256-2K) | Good balance | — |
| Large (2K+) | Efficient, stable | May need LR tuning, can hurt generalization |

### The Linear Scaling Rule

When increasing batch size by k:
1. Multiply learning rate by k
2. Increase warmup steps by k

```python
# Example: scaling from batch 256 to batch 1024
base_lr = 1e-4
base_batch = 256
new_batch = 1024
scale = new_batch / base_batch  # 4x

new_lr = base_lr * scale  # 4e-4
new_warmup = base_warmup * scale  # 4x warmup steps
```

### Gradient Accumulation

When batch size exceeds GPU memory:

```python
def train_with_accumulation(model, data, micro_batch_size, accumulation_steps):
    """Simulate large batch with gradient accumulation."""
    effective_batch_size = micro_batch_size * accumulation_steps

    accumulated_grads = [np.zeros_like(p) for p in model.params]

    for i, micro_batch in enumerate(data):
        # Forward/backward on micro-batch
        loss, grads = model.forward_backward(micro_batch)

        # Accumulate gradients
        for acc_g, g in zip(accumulated_grads, grads):
            acc_g += g / accumulation_steps

        # Update after accumulation_steps micro-batches
        if (i + 1) % accumulation_steps == 0:
            optimizer.step(accumulated_grads)
            accumulated_grads = [np.zeros_like(p) for p in model.params]
```

## Debugging Training

### Signs of Problems

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss = NaN | Exploding gradients | Reduce LR, add clipping |
| Loss stuck | Vanishing gradients | Check initialization, use residuals |
| Loss oscillates | LR too high | Reduce learning rate |
| Loss decreases then rises | Overfitting | Add regularization |
| Very slow progress | LR too low | Increase learning rate |

### The Sanity Check Protocol

Before full training, verify:

1. **Overfit one batch**
   ```python
   # Training loss should go to ~0
   for _ in range(1000):
       loss = train_step(single_batch)
   assert loss < 0.01, "Can't overfit single batch!"
   ```

2. **Loss starts at expected value**
   ```
   Random init cross-entropy ≈ log(vocab_size)
   For vocab=50000: expect loss ≈ 10.8
   ```

3. **Gradients are reasonable**
   ```python
   # Not too small, not too large
   assert 1e-7 < grad_norm < 1e3
   ```

4. **Parameters are updating**
   ```python
   # Params should change each step
   old_params = [p.copy() for p in model.params]
   train_step(batch)
   for old, new in zip(old_params, model.params):
       assert not np.allclose(old, new)
   ```

### Gradient Checking

Verify backprop with numerical gradients:

```python
def check_gradients(model, batch, eps=1e-5):
    """Verify analytical gradients match numerical."""
    # Analytical gradient
    loss, grads = model.forward_backward(batch)

    for i, (param, grad) in enumerate(zip(model.params, grads)):
        # Sample random positions
        for _ in range(10):
            idx = tuple(np.random.randint(0, s) for s in param.shape)

            # Numerical gradient
            param[idx] += eps
            loss_plus = model.forward(batch)
            param[idx] -= 2 * eps
            loss_minus = model.forward(batch)
            param[idx] += eps  # Restore

            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytical = grad[idx]

            rel_error = abs(numerical - analytical) / (abs(numerical) + abs(analytical) + 1e-8)
            assert rel_error < 1e-4, f"Gradient check failed: {rel_error}"
```

## Gradient Clipping in Practice

### When to Clip

Always! Gradient clipping is cheap insurance against instability.

```python
# Typical settings
max_grad_norm = 1.0  # For LLMs
max_grad_norm = 5.0  # For smaller models
```

### Monitoring Gradient Norms

Track gradient norms during training:

```python
def log_gradient_stats(grads, step):
    """Log gradient statistics for debugging."""
    norms = [np.linalg.norm(g) for g in grads]
    total_norm = np.sqrt(sum(n**2 for n in norms))

    stats = {
        'grad_norm': total_norm,
        'grad_max': max(np.max(np.abs(g)) for g in grads),
        'grad_min': min(np.min(np.abs(g)) for g in grads),
    }

    # Warning signs
    if total_norm > 100:
        print(f"WARNING: Large gradient norm at step {step}: {total_norm}")
    if total_norm < 1e-7:
        print(f"WARNING: Vanishing gradients at step {step}: {total_norm}")

    return stats
```

## Numerical Stability

### Mixed Precision Training

Modern GPUs support float16, which is faster but less precise:

```python
# Typical mixed precision strategy:
# - Forward pass: float16
# - Loss computation: float32
# - Gradients: float16
# - Optimizer state: float32
# - Weight update: float32
```

### Loss Scaling

Float16 has limited range. Scale loss to prevent underflow:

```python
def train_step_mixed_precision(model, batch, loss_scale=1024):
    # Forward in float16
    logits = model.forward_fp16(batch)

    # Loss in float32
    loss = compute_loss_fp32(logits, batch.targets)

    # Scale loss for backward pass
    scaled_loss = loss * loss_scale

    # Backward (gradients are scaled)
    grads = model.backward(scaled_loss)

    # Unscale gradients
    grads = [g / loss_scale for g in grads]

    # Update in float32
    optimizer.step(grads)
```

### Avoiding Overflow in Softmax

Never compute softmax naively:

```python
# BAD: overflow for large logits
def softmax_bad(x):
    return np.exp(x) / np.sum(np.exp(x))

# GOOD: subtract max for stability
def softmax_good(x):
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

## Regularization Strategies

### Weight Decay

Already covered with AdamW. Typical values:
- LLMs: 0.1
- Vision models: 1e-4 to 1e-2
- Small models: 1e-4

### Dropout

Randomly zero activations during training:

```python
def dropout(x, p=0.1, training=True):
    """Apply dropout."""
    if not training or p == 0:
        return x
    mask = np.random.binomial(1, 1-p, x.shape) / (1-p)
    return x * mask
```

### Label Smoothing

Don't train to 100% confidence:

```python
def smooth_labels(targets, num_classes, smoothing=0.1):
    """Apply label smoothing."""
    confidence = 1.0 - smoothing
    smooth_value = smoothing / num_classes

    # One-hot with smoothing
    one_hot = np.eye(num_classes)[targets]
    return one_hot * confidence + smooth_value
```

## Monitoring and Logging

### Essential Metrics

Track these during training:

```python
metrics = {
    'train_loss': [],      # Training loss
    'val_loss': [],        # Validation loss
    'learning_rate': [],   # Current LR
    'grad_norm': [],       # Gradient magnitude
    'step_time': [],       # Wall clock per step
    'throughput': [],      # Tokens/second
}
```

### Early Stopping

Stop when validation loss stops improving:

```python
class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def should_stop(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
```

## Common Mistakes Checklist

!!! warning "Things That Break Training"

    1. **Forgetting to zero gradients**
       ```python
       optimizer.zero_grad()  # Don't forget!
       ```

    2. **Using same random seed for train/val split**
       - Validation will contain training examples

    3. **Not shuffling data each epoch**
       - Creates artificial patterns

    4. **Training on validation data**
       - Hyperparameter tuning counts!

    5. **Incorrect tensor shapes**
       - Off-by-one in dimensions

    6. **Division by zero in loss**
       - Empty batches, all padding

    7. **Learning rate too high after loading checkpoint**
       - Schedule state must also be restored

    8. **Gradient accumulation without averaging**
       - Must divide by accumulation steps

## Hyperparameter Search

### Grid Search

Try all combinations:

```python
def grid_search():
    for lr in [1e-4, 3e-4, 1e-3]:
        for batch_size in [32, 128, 512]:
            for weight_decay in [0.01, 0.1]:
                train(lr=lr, batch_size=batch_size, wd=weight_decay)
```

### Random Search

Often more efficient than grid:

```python
def random_search(n_trials=50):
    for _ in range(n_trials):
        lr = 10 ** np.random.uniform(-5, -2)
        batch_size = np.random.choice([32, 64, 128, 256, 512])
        weight_decay = 10 ** np.random.uniform(-3, 0)
        train(lr=lr, batch_size=batch_size, wd=weight_decay)
```

### Bayesian Optimization

Use previous results to guide search. Libraries like Optuna automate this.

## Exercises

1. **Initialization experiment**: Train same model with zeros, random uniform, Xavier, Kaiming. Compare.

2. **Batch size scaling**: Verify the linear scaling rule empirically.

3. **Debug a broken training**: Given intentionally buggy code, find and fix all issues.

4. **Implement early stopping**: Add early stopping to the training loop.

5. **Gradient histogram**: Plot histogram of gradient values during training. How does it change?

## Summary

| Practice | Purpose |
|----------|---------|
| Proper initialization | Healthy gradient flow from start |
| Gradient clipping | Prevent explosions |
| Learning rate warmup | Stabilize early training |
| Gradient accumulation | Larger effective batch size |
| Mixed precision | Faster training |
| Monitoring | Catch problems early |
| Checkpointing | Recover from failures |

**Key takeaway**: Successful training requires attention to many practical details beyond the algorithm itself. Proper initialization, gradient clipping, careful monitoring, and systematic debugging are as important as choosing the right optimizer. The difference between training that works and training that fails is often in these details.

→ **Back to**: [Stage 4 Overview](index.md)

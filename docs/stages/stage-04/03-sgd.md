# Section 4.3: Stochastic Gradient Descent

*Reading time: 16 minutes | Difficulty: ★★★☆☆*

Computing the gradient over millions of examples is expensive. Stochastic Gradient Descent (SGD) uses a simple but powerful idea: estimate the gradient from a small random sample.

## The Key Insight

The full gradient is an average over all training examples:

$$\nabla L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \nabla \ell_i(\theta)$$

**Observation**: A random sample gives an unbiased estimate of this average!

If we pick a random subset B (a "mini-batch"):

$$\nabla \tilde{L}(\theta) = \frac{1}{|B|} \sum_{i \in B} \nabla \ell_i(\theta)$$

Then: $\mathbb{E}[\nabla \tilde{L}(\theta)] = \nabla L(\theta)$

The expected value of our estimate equals the true gradient.

## The SGD Algorithm

```python
def sgd(loss_fn, grad_fn, data, theta_init, lr, batch_size, epochs):
    """
    Stochastic Gradient Descent.

    Args:
        loss_fn: Loss function
        grad_fn: Gradient function for a batch
        data: Training data (list of examples)
        theta_init: Initial parameters
        lr: Learning rate
        batch_size: Mini-batch size
        epochs: Number of passes through data

    Returns:
        Final parameters
    """
    theta = theta_init.copy()
    n = len(data)

    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            # Get mini-batch
            batch_idx = indices[start:start + batch_size]
            batch = [data[i] for i in batch_idx]

            # Compute gradient estimate
            grad = grad_fn(theta, batch)

            # Update
            theta = theta - lr * grad

    return theta
```

## Why Stochastic?

### Computational Efficiency

| Method | Cost per Update | Updates per Epoch |
|--------|-----------------|-------------------|
| Full Batch | O(N × n) | 1 |
| Mini-batch B | O(B × n) | N/B |
| Pure Stochastic (B=1) | O(n) | N |

For the same compute budget, SGD makes N/B more updates. Early progress is much faster.

### Implicit Regularization

The noise in SGD estimates acts as regularization:

1. **Escapes sharp minima**: Sharp minima are sensitive to noise; SGD bounces out
2. **Finds flat minima**: Flat minima are stable under noise; SGD settles there
3. **Flat minima generalize better**: They're robust to distribution shift

!!! info "Connection to Modern LLMs"

    Modern LLM training uses mini-batches of millions of tokens. The batch size affects:

    - **Noise level**: Larger batches = less noise = more stable but potentially worse generalization
    - **Parallelization**: Larger batches utilize hardware better
    - **Learning rate scaling**: Larger batches allow larger learning rates

    The "linear scaling rule" says: if you double batch size, double learning rate.

## Variance and the Noise Trade-off

The variance of our gradient estimate is:

$$\text{Var}[\nabla \tilde{L}] = \frac{\sigma^2}{B}$$

where σ² is the variance of individual gradients and B is batch size.

**Trade-off**:
- Small B: High variance, more exploration, less stable
- Large B: Low variance, less exploration, more stable

```
           Noise Level
               ↑
               │
      High     │    B=1 (pure SGD)
               │         •
               │
               │              B=32
               │                 •
               │
               │                     B=256
               │                        •
      Low      │                            B=∞ (full batch)
               │                               •
               └────────────────────────────────→
                   Compute per Update
```

## Convergence of SGD

### For Convex Functions

**Theorem**: For a convex function with bounded gradients $\|\nabla \ell_i\| \leq G$ and learning rate $\eta_t = \frac{1}{\sqrt{t}}$:

$$\mathbb{E}[L(\bar{\theta}_T)] - L(\theta^*) \leq O\left(\frac{1}{\sqrt{T}}\right)$$

where $\bar{\theta}_T$ is the average of all iterates.

**Compared to full GD**: O(1/√T) vs O(1/T). SGD is slower per iteration but cheaper per iteration.

### For Non-Convex Functions

For neural networks (non-convex), we can show SGD finds approximate stationary points:

$$\mathbb{E}[\|\nabla L(\theta)\|^2] \leq O\left(\frac{1}{\sqrt{T}}\right)$$

This means the gradient magnitude goes to zero — we reach a critical point.

## Learning Rate Decay

For SGD to converge, we often need to decrease the learning rate over time:

**Robbins-Monro conditions**:
$$\sum_{t=1}^{\infty} \eta_t = \infty \quad \text{and} \quad \sum_{t=1}^{\infty} \eta_t^2 < \infty$$

**Common schedules**:
- $\eta_t = \eta_0 / \sqrt{t}$ (1/√t decay)
- $\eta_t = \eta_0 / (1 + \alpha t)$ (inverse decay)
- Step decay: halve every k epochs

```python
def lr_schedule(t, eta_0, schedule='sqrt'):
    if schedule == 'sqrt':
        return eta_0 / np.sqrt(t + 1)
    elif schedule == 'inverse':
        return eta_0 / (1 + 0.01 * t)
    elif schedule == 'constant':
        return eta_0
```

## Mini-Batch Size Selection

### The Compute-Quality Trade-off

| Batch Size | Pros | Cons |
|------------|------|------|
| Small (32-64) | More updates, better generalization | High variance, poor GPU utilization |
| Medium (128-512) | Good balance | — |
| Large (1024+) | Stable, efficient | May need LR tuning, can hurt generalization |

### The Critical Batch Size

There's often a "critical batch size" beyond which increasing batch size doesn't help:

- Below critical: Noise-limited (more compute → better)
- Above critical: Curvature-limited (more compute → same result)

For language models, this is typically 10⁵ - 10⁶ tokens.

## Shuffling Matters

**Always shuffle data each epoch!**

Without shuffling:
- The model sees patterns in a fixed order
- Can lead to poor generalization
- Convergence guarantees may not hold

```python
# Good: shuffle each epoch
for epoch in range(num_epochs):
    np.random.shuffle(data)
    for batch in get_batches(data, batch_size):
        update(batch)

# Bad: same order every epoch
for epoch in range(num_epochs):
    for batch in get_batches(data, batch_size):  # Same order!
        update(batch)
```

## SGD Variants

### Pure SGD (B = 1)
- Maximum noise
- Rarely used in practice (too unstable)

### Mini-batch SGD (B = 32-512)
- The standard approach
- Balances noise and stability

### Large-batch SGD (B = 1000+)
- Requires learning rate scaling
- Used for distributed training
- May need warmup (Section 4.6)

## Practical Implementation Details

### Gradient Accumulation

When batch size is too large for memory:

```python
def train_with_accumulation(model, data, target_batch_size, micro_batch_size):
    """Accumulate gradients over multiple micro-batches."""
    accumulation_steps = target_batch_size // micro_batch_size

    optimizer.zero_grad()

    for i, micro_batch in enumerate(split_batch(data, micro_batch_size)):
        loss = model(micro_batch) / accumulation_steps
        loss.backward()  # Accumulates gradients

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

### Gradient Clipping

Prevent exploding gradients by limiting gradient magnitude:

```python
def clip_gradients(grads, max_norm):
    """Clip gradients to maximum norm."""
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        return [g * clip_coef for g in grads]
    return grads
```

!!! warning "Common Mistake: Not Averaging Over Batch"

    When computing loss, make sure to average (not sum) over the batch:

    ❌ `loss = sum(losses)` — loss scales with batch size
    ✓ `loss = mean(losses)` — loss is independent of batch size

    If you sum, you need to adjust learning rate when changing batch size.

## Historical Note

**Herbert Robbins and Sutton Monro** (1951) proved convergence of stochastic approximation, laying the theoretical foundation for SGD.

**Léon Bottou** championed SGD for machine learning in the 1990s-2000s, showing it could outperform batch methods despite higher variance.

The deep learning revolution (2012+) relied heavily on SGD with momentum, proving that stochastic methods scale to massive datasets.

## Complexity Comparison

For N examples, n parameters, B batch size, T total updates:

| Method | Cost per Update | Updates for ε Error |
|--------|-----------------|---------------------|
| Full GD | O(Nn) | O(1/ε) |
| SGD | O(Bn) | O(1/ε²) |

Total cost to reach ε error:
- Full GD: O(Nn/ε)
- SGD: O(Bn/ε²)

For small ε, SGD wins when Bn/ε² < Nn/ε, i.e., when B < Nε.

## Exercises

1. **Variance experiment**: Plot gradient variance vs batch size on a simple problem.

2. **Convergence comparison**: Compare full GD vs SGD on logistic regression. Plot loss vs wall-clock time.

3. **Batch size ablation**: Train the same model with B = 1, 16, 64, 256. Compare final loss and training curves.

4. **Learning rate scaling**: Verify the linear scaling rule: doubling B requires doubling η.

5. **Noise visualization**: For a 2D problem, plot multiple SGD trajectories to visualize the noise.

## Summary

| Concept | Definition | Key Insight |
|---------|------------|-------------|
| Stochastic gradient | Gradient estimated from random subset | Unbiased estimate of true gradient |
| Mini-batch | Small random subset of data | Balances compute and variance |
| Variance | Var ∝ 1/B | Larger batches = less noise |
| Learning rate decay | η → 0 as t → ∞ | Required for convergence |

**Key takeaway**: SGD trades gradient accuracy for computational efficiency. The noise isn't just a necessary evil — it provides regularization and helps find better minima. This is why SGD (with enhancements) remains the dominant approach for training neural networks.

→ **Next**: [Section 4.4: Momentum](04-momentum.md)

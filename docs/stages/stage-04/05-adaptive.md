# Section 4.5: Adaptive Learning Rates — Per-Parameter Tuning

*Reading time: 22 minutes | Difficulty: ★★★★☆*

Different parameters may need different learning rates. Adaptive methods automatically adjust the learning rate for each parameter based on the history of gradients it has seen.

## The Motivation

Consider a neural network with:

- **Embedding weights**: Sparse updates (most gradients are zero)
- **Output layer**: Dense updates (every example affects it)

Using the same learning rate for both is suboptimal:

- Embeddings need larger steps (rare updates)
- Output layer needs smaller steps (frequent updates)

**Idea**: Scale the learning rate inversely with how much each parameter has been updated.

## AdaGrad: The Foundation

**Duchi, Hazan, Singer (2011)**

AdaGrad tracks the sum of squared gradients for each parameter:

$$G_t = G_{t-1} + g_t^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t$$

Where:

- $g_t = \nabla L(\theta_t)$ is the gradient
- $G_t$ accumulates squared gradients (element-wise)
- $\epsilon \approx 10^{-8}$ prevents division by zero

### Why AdaGrad Works

1. **Frequently updated parameters**: Large $G_t$ → small effective learning rate
2. **Rarely updated parameters**: Small $G_t$ → large effective learning rate

This is perfect for sparse problems like NLP where some words appear rarely!

### The Problem with AdaGrad

$G_t$ only grows. Eventually, the denominator becomes so large that learning stops:

$$\lim_{t \to \infty} \frac{\eta}{\sqrt{G_t}} = 0$$

This is called **learning rate decay to zero**. Good for convex problems, bad for neural networks.

```python
def adagrad(loss_fn, grad_fn, theta_init, lr, num_steps, eps=1e-8):
    """AdaGrad optimizer."""
    theta = theta_init.copy()
    G = np.zeros_like(theta)  # Accumulated squared gradients

    for t in range(num_steps):
        grad = grad_fn(theta)

        # Accumulate squared gradients
        G += grad ** 2

        # Update with adaptive learning rate
        theta = theta - lr * grad / (np.sqrt(G) + eps)

    return theta
```

## RMSprop: Exponential Decay

**Hinton (2012)** — From Coursera lecture, never formally published!

RMSprop fixes AdaGrad's decay problem by using an exponential moving average:

$$v_t = \beta v_{t-1} + (1-\beta) g_t^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} g_t$$

The key change: $(1-\beta)$ weight on new gradient prevents unbounded growth.

Typical: β = 0.9 (average over ~10 recent gradients)

```python
def rmsprop(loss_fn, grad_fn, theta_init, lr, beta=0.9, num_steps=1000, eps=1e-8):
    """RMSprop optimizer."""
    theta = theta_init.copy()
    v = np.zeros_like(theta)  # Running average of squared gradients

    for t in range(num_steps):
        grad = grad_fn(theta)

        # Update running average of squared gradients
        v = beta * v + (1 - beta) * grad ** 2

        # Update parameters
        theta = theta - lr * grad / (np.sqrt(v) + eps)

    return theta
```

## Adam: Combining Momentum + RMSprop

**Kingma & Ba (2014)**

Adam combines the best of both worlds:

- **Momentum** (first moment): Smooths gradient direction
- **RMSprop** (second moment): Adapts learning rate per parameter

### The Adam Algorithm

**First moment** (momentum):
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

**Second moment** (RMSprop):
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

**Bias correction** (critical!):
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

**Update**:
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### Default Hyperparameters

The original paper recommends:

- η = 0.001 (learning rate)
- β₁ = 0.9 (momentum decay)
- β₂ = 0.999 (RMSprop decay)
- ε = 10⁻⁸ (numerical stability)

These work well for most problems!

### Why Bias Correction?

At t = 1:

- $m_1 = (1-\beta_1) g_1$ ← biased toward 0
- $v_1 = (1-\beta_2) g_1^2$ ← biased toward 0

The correction $\frac{1}{1-\beta^t}$ accounts for the zero initialization:

| Step | 1-β₁ᵗ (β₁=0.9) | Correction factor |
|------|----------------|-------------------|
| 1 | 0.1 | 10× |
| 5 | 0.41 | 2.4× |
| 10 | 0.65 | 1.5× |
| 100 | ~1.0 | ~1× |

Early steps get boosted; later steps are unaffected.

### Implementation

```python
class Adam:
    """Adam optimizer from scratch."""

    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = None  # First moment
        self.v = None  # Second moment

    def step(self, theta, grad):
        """Perform one optimization step."""
        # Initialize moments on first call
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)

        self.t += 1

        # Update biased moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update parameters
        theta = theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return theta
```

## AdamW: Weight Decay Done Right

**Loshchilov & Hutter (2017)**

L2 regularization is typically implemented as:
$$L_{reg}(\theta) = L(\theta) + \frac{\lambda}{2}\|\theta\|^2$$

This adds λθ to the gradient. But in Adam, this gets scaled by the adaptive learning rate!

**AdamW** fixes this by applying weight decay directly to parameters:

$$\theta_{t+1} = (1 - \eta\lambda)\theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

The weight decay term $(1 - \eta\lambda)$ is applied **before** the adaptive update.

```python
class AdamW:
    """AdamW: Adam with decoupled weight decay."""

    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = None
        self.v = None

    def step(self, theta, grad):
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)

        self.t += 1

        # Weight decay (decoupled from gradient)
        theta = theta * (1 - self.lr * self.weight_decay)

        # Standard Adam update
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        theta = theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return theta
```

!!! info "Connection to Modern LLMs"

    **AdamW is the standard optimizer for LLM training.**

    Typical settings for models like LLaMA:
    - Learning rate: 1e-4 to 3e-4
    - β₁ = 0.9, β₂ = 0.95 (slightly lower β₂ than default)
    - Weight decay: 0.1
    - ε = 1e-8

    The lower β₂ = 0.95 allows faster adaptation to changing gradient magnitudes during training.

## Comparison of Methods

| Method | First Moment | Second Moment | Key Feature |
|--------|--------------|---------------|-------------|
| SGD | ✗ | ✗ | Baseline |
| Momentum | ✓ (EMA) | ✗ | Velocity |
| AdaGrad | ✗ | ✓ (sum) | Sparse updates |
| RMSprop | ✗ | ✓ (EMA) | Non-decaying adaptive |
| Adam | ✓ (EMA) | ✓ (EMA) | Best of both |
| AdamW | ✓ (EMA) | ✓ (EMA) | Proper weight decay |

## When to Use What

### Use SGD + Momentum when:
- Training is stable
- You want maximum control
- Generalization is critical (SGD often generalizes better)
- You have time to tune learning rate schedule

### Use Adam/AdamW when:
- You want fast, reliable convergence
- Training large models
- Limited time for hyperparameter tuning
- Learning rate is sensitive

### The SGD vs Adam Debate

Research shows:

- **Adam converges faster** in terms of training loss
- **SGD often generalizes better** on held-out data
- **For LLMs, Adam wins** due to scale and complexity

```
        Training Loss
              ↑
              │    SGD
              │      ╲
              │       ╲
              │        ╲────────
              │    Adam ╲
              │          ╲
              │           ╲─────
              └──────────────────→ Steps
                  Adam reaches low loss faster,
                  but SGD may find flatter minima
```

## Understanding Adaptive Methods Geometrically

Consider gradient descent in 2D with different scales:

```
Without adaptation:          With adaptation:
                             (rescaled space)
    y (large scale)              y' = y/σy
    ↑                            ↑
    │  ↗↙↗↙                      │  ↘
    │   ↗↙                       │    ↘
    │    ↗↙                      │      ↘
    └────────→ x                 └────────→ x' = x/σx
    (small scale)

    Zigzag path                  Direct path
```

Adaptive methods effectively rescale the parameter space so all directions have similar curvature.

## The Warmup Mystery

Adam often benefits from **learning rate warmup** (starting with small η, gradually increasing).

**Why?** The second moment estimate v is unreliable early in training:
- Few samples seen
- Bias correction helps but isn't perfect
- Large initial steps can be destabilizing

Warmup gives time for v to stabilize before taking large steps.

```python
def warmup_lr(step, warmup_steps, base_lr):
    """Linear warmup schedule."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr
```

## Numerical Stability

### The ε Parameter

Without ε, we'd divide by zero when v ≈ 0. But ε also affects the effective learning rate:

$$\text{effective } \eta \approx \frac{\eta}{\sqrt{v} + \epsilon}$$

- If √v >> ε: ε doesn't matter
- If √v ≈ ε: ε significantly affects the update

For parameters with tiny gradients, ε can dominate!

### Gradient Explosion

Even with adaptive rates, gradients can explode. Always use gradient clipping:

```python
def clip_grad_norm(grads, max_norm):
    """Clip gradient norm."""
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        return [g * clip_coef for g in grads]
    return grads
```

## Complete Training Loop

```python
def train_with_adam(model, data, epochs, lr=1e-3, weight_decay=0.01):
    """Complete training loop with AdamW."""
    optimizer = AdamW(lr=lr, weight_decay=weight_decay)
    warmup_steps = 1000
    total_steps = 0

    for epoch in range(epochs):
        for batch in data:
            # Forward pass
            loss, grads = model.forward_backward(batch)

            # Warmup learning rate
            current_lr = warmup_lr(total_steps, warmup_steps, lr)
            optimizer.lr = current_lr

            # Clip gradients
            grads = clip_grad_norm(grads, max_norm=1.0)

            # Update parameters
            for param, grad in zip(model.params, grads):
                param = optimizer.step(param, grad)

            total_steps += 1

        print(f"Epoch {epoch}: loss = {loss:.4f}")
```

## Historical Note

The development of adaptive methods:

- **2011**: AdaGrad (Duchi et al.) — First per-parameter adaptation
- **2012**: RMSprop (Hinton) — Unpublished Coursera lecture!
- **2013**: Adadelta (Zeiler) — No learning rate needed
- **2014**: Adam (Kingma & Ba) — Momentum + RMSprop
- **2017**: AdamW (Loshchilov & Hutter) — Fixed weight decay

Adam became the default optimizer, but research continues:

- **RAdam** (2019): Rectified Adam with variance reduction
- **LAMB** (2019): Layer-wise adaptive rates for large batches
- **AdaFactor** (2018): Memory-efficient Adam for huge models
- **Lion** (2023): Discovered by neural architecture search

## Exercises

1. **Implement AdaGrad**: Train on a sparse problem and observe learning rate decay.

2. **Compare optimizers**: Train the same model with SGD, momentum, RMSprop, Adam. Plot training curves.

3. **Ablate Adam**: What happens if you remove bias correction? Momentum? Adaptive rates?

4. **β₂ sensitivity**: Train with β₂ = 0.9, 0.99, 0.999, 0.9999. How does it affect stability?

5. **Weight decay ablation**: Compare Adam with L2 vs AdamW. Is there a difference?

## Summary

| Concept | Definition | Key Insight |
|---------|------------|-------------|
| AdaGrad | G += g² | Adapts to gradient history |
| RMSprop | v = βv + (1-β)g² | Exponential decay prevents stalling |
| Adam | m, v moments + bias correction | Momentum + adaptation |
| AdamW | Decoupled weight decay | Proper regularization |

**Key takeaway**: Adaptive methods automatically tune per-parameter learning rates based on gradient history. Adam combines the benefits of momentum (smooth direction) and RMSprop (adaptive magnitude), making it the go-to optimizer for modern deep learning. AdamW's proper treatment of weight decay makes it the standard for LLM training.

→ **Next**: [Section 4.6: Learning Rate Schedules](06-schedules.md)

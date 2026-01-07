# Stage 4 Exercises

## Conceptual Questions

### Exercise 4.1: Loss Landscape
Consider minimizing f(x) = x⁴ - 2x² + 1.

**a)** Find all critical points (where f'(x) = 0)
**b)** Classify each as local min, local max, or saddle point
**c)** If gradient descent starts at x=0.1, where will it converge?

### Exercise 4.2: Learning Rate Effects
For f(x) = x², gradient descent update is: x ← x - η * 2x

**a)** Starting at x=10 with η=0.1, compute x after 5 steps
**b)** What happens with η=0.6?
**c)** What happens with η=1.0?
**d)** What is the maximum stable learning rate?

### Exercise 4.3: Momentum Intuition
A ball rolling down a hill with momentum:

**a)** How does momentum help with narrow valleys?
**b)** How does momentum help escape shallow local minima?
**c)** What's the downside of high momentum?

### Exercise 4.4: Adam's Components
Adam combines momentum and adaptive learning rates.

**a)** What does the first moment (m) track?
**b)** What does the second moment (v) track?
**c)** Why does Adam divide by √v + ε?

---

## Implementation Exercises

### Exercise 4.5: Gradient Descent Variants
Implement and compare:

```python
def sgd(params, grads, lr):
    """Basic SGD: θ ← θ - lr * g"""
    # TODO
    pass

def sgd_momentum(params, grads, velocity, lr, beta=0.9):
    """SGD with momentum: v ← βv + g; θ ← θ - lr * v"""
    # TODO
    pass

def sgd_nesterov(params, grads, velocity, lr, beta=0.9):
    """Nesterov momentum: look ahead before computing gradient"""
    # TODO
    pass
```

### Exercise 4.6: Implement Adam
Implement Adam optimizer:

```python
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}  # First moment
        self.v = {}  # Second moment

    def step(self, params, grads):
        """
        m = β₁m + (1-β₁)g
        v = β₂v + (1-β₂)g²
        m̂ = m / (1 - β₁ᵗ)   # Bias correction
        v̂ = v / (1 - β₂ᵗ)
        θ = θ - lr * m̂ / (√v̂ + ε)
        """
        # TODO
        pass
```

### Exercise 4.7: Learning Rate Schedules
Implement common schedules:

```python
def constant_lr(step, base_lr):
    return base_lr

def step_decay(step, base_lr, decay_rate=0.1, decay_every=1000):
    """Reduce LR by decay_rate every decay_every steps"""
    # TODO
    pass

def cosine_annealing(step, base_lr, total_steps):
    """Cosine decay from base_lr to 0"""
    # TODO
    pass

def warmup_then_decay(step, base_lr, warmup_steps, total_steps):
    """Linear warmup, then cosine decay"""
    # TODO
    pass
```

### Exercise 4.8: Gradient Clipping
Implement gradient clipping:

```python
def clip_grad_norm(grads, max_norm):
    """
    Clip gradients so that global norm ≤ max_norm.

    global_norm = sqrt(sum(g² for all g in grads))
    if global_norm > max_norm:
        scale all gradients by max_norm / global_norm
    """
    # TODO
    pass

def clip_grad_value(grads, max_value):
    """Clip each gradient element to [-max_value, max_value]"""
    # TODO
    pass
```

---

## Challenge Exercises

### Exercise 4.9: Optimizer Comparison
Train the same model with different optimizers:

**a)** SGD with different learning rates (0.001, 0.01, 0.1, 1.0)
**b)** SGD + Momentum (β=0.9)
**c)** Adam (default hyperparameters)

Plot training loss curves. Which converges fastest? Most stably?

### Exercise 4.10: Learning Rate Finder
Implement the LR range test:

```python
def find_lr(model, data, min_lr=1e-7, max_lr=10, num_steps=100):
    """
    1. Start with very small LR
    2. Train one step, record loss
    3. Increase LR exponentially
    4. Stop when loss explodes
    5. Return LR where loss decreased fastest
    """
    # TODO
    pass
```

### Exercise 4.11: Batch Size and Learning Rate
Train models with different batch sizes: 16, 32, 64, 128, 256.

**a)** Use the same learning rate for all. Compare convergence.
**b)** Scale learning rate linearly with batch size. Does this help?
**c)** The "linear scaling rule" says lr ∝ batch_size. Test this.

---

## Solutions

Solutions are available in `code/stage-04/solutions/`.

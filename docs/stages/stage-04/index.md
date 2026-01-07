# Stage 4: Optimization — Making Learning Work

*Estimated reading time: 2-3 hours | Prerequisites: Stages 1-3*

## Overview

We've built neural language models that can theoretically learn from data. But *how* do they learn? This stage derives optimization algorithms from first principles, explaining why gradient descent works and how to make it work better.

**The central question**: Given a loss function L(θ) and its gradients ∇L(θ), how do we update parameters θ to minimize loss?

## What You'll Learn

By the end of this stage, you'll understand:

1. **Why gradient descent works** — The mathematical justification
2. **Learning rate selection** — Too high = divergence, too low = slow convergence
3. **Momentum** — Using velocity to escape local minima
4. **Adaptive methods** — Adam, AdaGrad, RMSprop and why they're default choices
5. **Learning rate schedules** — Warmup, decay, cosine annealing
6. **Batch size effects** — The compute-convergence trade-off

## Sections

| Section | Topic | Key Concepts |
|---------|-------|--------------|
| 4.1 | [The Optimization Problem](01-optimization-problem.md) | Loss landscapes, local vs global minima, saddle points |
| 4.2 | [Gradient Descent](02-gradient-descent.md) | Derivation, convergence proof, learning rate |
| 4.3 | [Stochastic Gradient Descent](03-sgd.md) | Mini-batches, noise as regularization, variance |
| 4.4 | [Momentum](04-momentum.md) | Physics intuition, Nesterov momentum, derivation |
| 4.5 | [Adaptive Learning Rates](05-adaptive.md) | AdaGrad, RMSprop, Adam, when to use each |
| 4.6 | [Learning Rate Schedules](06-schedules.md) | Warmup, decay, cosine annealing, restarts |
| 4.7 | [Implementation](07-implementation.md) | Building optimizers from scratch |
| 4.8 | [Practical Considerations](08-practical.md) | Batch size, gradient clipping, debugging |

## Key Mathematical Results

### Gradient Descent Convergence

For a convex function with L-Lipschitz gradients:

$$\|x_{t+1} - x^*\| \leq \left(1 - \frac{\mu}{L}\right)^t \|x_0 - x^*\|$$

where μ is the strong convexity parameter. This gives the optimal learning rate η = 1/L.

### Adam Update Rule

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

We'll derive this step by step and explain why each component is necessary.

## Connection to Modern LLMs

Modern LLM training uses:

- **Adam or AdamW** as the base optimizer
- **Linear warmup** for the first 1-10% of training
- **Cosine decay** to final learning rate
- **Gradient clipping** to prevent exploding gradients
- **Large batch sizes** (millions of tokens) with gradient accumulation

Understanding these choices requires understanding the fundamentals we'll cover here.

## Code Preview

```python
class Adam:
    """Adam optimizer from scratch."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            # Momentum
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # RMSprop
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            # Update
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

## Prerequisites

Before starting this stage, ensure you understand:

- [ ] Gradients and the chain rule (Stage 2)
- [ ] Loss functions for language models (Stage 3)
- [ ] Basic calculus (derivatives, Taylor series)
- [ ] Matrix/vector operations

## Exercises Preview

1. **Implement vanilla SGD** and train a small model
2. **Compare optimizers** on the same problem: SGD, Momentum, Adam
3. **Learning rate finder**: Implement the learning rate range test
4. **Visualize optimization**: Plot loss landscapes and optimizer trajectories
5. **Ablate Adam**: What happens if you remove momentum? Adaptive rates?

## Historical Context

- **1847**: Cauchy introduces gradient descent
- **1951**: Robbins-Monro prove stochastic approximation convergence
- **1964**: Polyak introduces momentum
- **1983**: Nesterov's accelerated gradient
- **2011**: Duchi et al. introduce AdaGrad
- **2014**: Kingma & Ba introduce Adam
- **2017**: Loshchilov & Hutter introduce AdamW (weight decay fix)

## Begin

→ [Start with Section 4.1: The Optimization Problem](01-optimization-problem.md)

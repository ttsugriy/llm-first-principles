# Section 4.4: Momentum — Learning with Velocity

*Reading time: 18 minutes | Difficulty: ★★★☆☆*

Gradient descent can be painfully slow on ill-conditioned problems. Momentum fixes this by adding "velocity" to our updates, allowing the optimizer to build up speed in consistent directions.

## The Problem with Vanilla Gradient Descent

Recall the update rule: θ ← θ - η∇L(θ)

Consider minimizing L(x, y) = x² + 100y²:

```
     y
     ↑
     │    ←───────────→
     │   ╱  Oscillates  ╲
     │  ╱                ╲
     │ ↙                  ↘
     │↗                    ↖
     └──────────────────────→ x
           Slow progress
```

The gradient in y is 100× larger than in x, causing:

1. **Oscillation** in the y direction (overshooting)
2. **Slow progress** in the x direction (undershooting)

This is the **condition number problem**: κ = λ_max/λ_min = 100.

## The Physics Analogy

Imagine a ball rolling down a hill:

- Without friction, it **builds up speed** going downhill
- It **carries momentum** through flat regions
- It **overshoots** valleys but eventually settles

This is exactly what we want for optimization!

## The Momentum Update Rule

**Classical Momentum** (Polyak, 1964):

$$v_t = \beta v_{t-1} + \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

Or equivalently:

$$v_t = \beta v_{t-1} + \eta \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_t$$

Where:

- **v** is the velocity (accumulated gradient)
- **β** is the momentum coefficient (typically 0.9)
- **η** is the learning rate

## Why Momentum Works

### Intuition 1: Averaging Gradients

Expand the velocity term:

$$v_t = \eta \nabla L_t + \beta \eta \nabla L_{t-1} + \beta^2 \eta \nabla L_{t-2} + ...$$

This is an **exponentially weighted moving average** of gradients!

- Recent gradients have weight ~1
- Old gradients have weight ~β^t → 0

### Intuition 2: Dampening Oscillations

In the oscillating y-direction:

- Gradients alternate signs: +, -, +, -, ...
- They cancel out: v_y ≈ 0

In the consistent x-direction:

- Gradients have the same sign: +, +, +, +, ...
- They accumulate: v_x grows large

```
Without momentum:        With momentum:
     ↗↘↗↘↗↘              ────────→
     oscillates          smooth progress
```

### Intuition 3: Effective Learning Rate

After many steps with consistent gradient g:

$$v_\infty = g + \beta g + \beta^2 g + ... = \frac{g}{1-\beta}$$

The effective learning rate becomes $\frac{\eta}{1-\beta}$.

For β = 0.9: effective rate is 10× the base rate!

This is why we often reduce η when adding momentum.

## Mathematical Analysis

### Convergence for Quadratics

For a quadratic L(θ) = ½θᵀHθ with eigenvalues λ₁ ≤ ... ≤ λₙ:

**Without momentum**: Convergence rate = (κ - 1)/(κ + 1)

**With optimal momentum**: Convergence rate = (√κ - 1)/(√κ + 1)

For κ = 100:

- Without: rate = 0.98 (slow)
- With: rate = 0.82 (much faster!)

**Key insight**: Momentum reduces the effective condition number from κ to √κ.

### Optimal Momentum Coefficient

For a quadratic with condition number κ:

$$\beta_{opt} = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^2$$

In practice, β = 0.9 works well for most problems.

## Nesterov Accelerated Gradient (NAG)

Nesterov (1983) proposed a subtle but important modification:

**Standard momentum**: Look at gradient where we ARE
$$v_t = \beta v_{t-1} + \nabla L(\theta_t)$$

**Nesterov momentum**: Look at gradient where we're GOING
$$v_t = \beta v_{t-1} + \nabla L(\theta_t - \eta \beta v_{t-1})$$

Or equivalently, with the "lookahead" θ̃:
$$\tilde{\theta} = \theta_t - \eta \beta v_{t-1}$$
$$v_t = \beta v_{t-1} + \nabla L(\tilde{\theta})$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

### Why Nesterov is Better

```
Standard:                    Nesterov:
     θ                            θ
     │                            │
     ↓ compute gradient           ↓ take momentum step first
     │                            θ̃
     ↓ then momentum step         ↓ compute gradient at θ̃
     θ'                           ↓ correct trajectory
                                  θ'
```

Nesterov "looks ahead" to where momentum will take us, then corrects. This provides:

- **Faster convergence** on convex problems
- **Better handling of overshooting**
- **Provably optimal** for smooth convex optimization

### Convergence Guarantee

**Theorem** (Nesterov, 1983): For smooth convex functions with optimal β:

$$L(\theta_T) - L(\theta^*) \leq O\left(\frac{1}{T^2}\right)$$

This is **optimal** for first-order methods! (Compared to O(1/T) for vanilla GD)

## Implementation

```python
def momentum_sgd(loss_fn, grad_fn, theta_init, lr, momentum, num_steps):
    """
    SGD with momentum.

    Args:
        loss_fn: Loss function
        grad_fn: Gradient function
        theta_init: Initial parameters
        lr: Learning rate
        momentum: Momentum coefficient (typically 0.9)
        num_steps: Number of iterations

    Returns:
        Final parameters and history
    """
    theta = theta_init.copy()
    velocity = np.zeros_like(theta)
    history = []

    for t in range(num_steps):
        loss = loss_fn(theta)
        grad = grad_fn(theta)
        history.append(loss)

        # Update velocity
        velocity = momentum * velocity + grad

        # Update parameters
        theta = theta - lr * velocity

    return theta, history


def nesterov_sgd(loss_fn, grad_fn, theta_init, lr, momentum, num_steps):
    """
    SGD with Nesterov momentum.

    The key difference: compute gradient at the "lookahead" position.
    """
    theta = theta_init.copy()
    velocity = np.zeros_like(theta)
    history = []

    for t in range(num_steps):
        # Lookahead position
        lookahead = theta - lr * momentum * velocity

        loss = loss_fn(theta)
        grad = grad_fn(lookahead)  # Gradient at lookahead!
        history.append(loss)

        # Update velocity
        velocity = momentum * velocity + grad

        # Update parameters
        theta = theta - lr * velocity

    return theta, history
```

## Choosing the Momentum Coefficient

| β Value | Behavior | Use Case |
|---------|----------|----------|
| 0.0 | No momentum (vanilla SGD) | Debugging, baselines |
| 0.5 | Light momentum | Noisy gradients |
| 0.9 | Standard | Most applications |
| 0.99 | Heavy momentum | Very smooth landscapes |

**Rule of thumb**: Start with β = 0.9. If training is unstable, reduce it.

!!! warning "Common Mistake: Not Adjusting Learning Rate"

    When adding momentum, the effective learning rate increases by ~1/(1-β).

    For β = 0.9, effective rate is 10× higher!

    **Fix**: Reduce η by a factor of (1-β) when adding momentum, or tune from scratch.

## Momentum and SGD Noise

An interesting interaction: momentum smooths out SGD noise!

Without momentum, each step is based on a single noisy gradient estimate.

With momentum, each step is based on an exponential average of many gradients.

**Effect**: Lower variance updates, but also less exploration.

```
                    Noise Level
                        ↑
                        │   SGD alone
                        │      •
                        │
                        │         SGD + momentum
                        │              •
                        │
                        │                  Large batch
                        │                      •
                        └──────────────────────────→
                              Stability
```

!!! info "Connection to Modern LLMs"

    Modern LLM training typically uses:

    - **β = 0.9** for the first moment (momentum) in Adam
    - This provides the noise-smoothing benefits of momentum
    - Combined with adaptive learning rates (next section)

    The momentum component of Adam is critical for stable training on noisy gradients from massive datasets.

## Visualizing Momentum

Consider the Rosenbrock function: L(x,y) = (1-x)² + 100(y-x²)²

This has a curved valley that's challenging for gradient descent:

```
Without momentum:           With momentum:
    Start                       Start
      ╲                           ╲
       ╲↗                          ╲
        ╲↙                          ╲
         ╲↗                          ╲
          ╲↙                          ╲
           ╲ (slow zigzag)             ╲ (smooth curve)
            ↓                           ↓
          Optimum                     Optimum
```

Momentum allows following the valley smoothly rather than bouncing off its walls.

## Historical Note

**Boris Polyak** introduced the momentum method in 1964, drawing on classical mechanics.

**Yurii Nesterov** developed his accelerated method in 1983 as part of his PhD thesis, proving it's optimal for smooth convex optimization.

The connection between momentum and acceleration in physics was made explicit by **Sutskever et al. (2013)**, who showed that Nesterov momentum works well for deep learning.

## Heavy Ball vs. Nesterov

The original momentum (Polyak) is sometimes called the "Heavy Ball" method:

| Method | Update | Convergence (convex) |
|--------|--------|---------------------|
| Heavy Ball | v = βv + ∇L(θ) | O(1/T) |
| Nesterov | v = βv + ∇L(θ - ηβv) | O(1/T²) |

Nesterov's lookahead makes a theoretical difference, though in practice both work similarly for neural networks.

## Exercises

1. **Visualize momentum**: On a 2D quadratic, plot trajectories with β = 0, 0.5, 0.9, 0.99.

2. **Optimal β**: For L(x,y) = x² + 100y², derive the optimal momentum coefficient.

3. **Compare methods**: Implement both classical and Nesterov momentum. Compare convergence on a test problem.

4. **Effective learning rate**: Verify empirically that momentum β = 0.9 gives ~10× effective learning rate.

5. **Noise smoothing**: Compare gradient variance with and without momentum on a stochastic problem.

## Summary

| Concept | Definition | Key Insight |
|---------|------------|-------------|
| Momentum | v = βv + ∇L | Exponential moving average of gradients |
| β coefficient | Decay rate for velocity | Higher = more "memory" |
| Nesterov | Gradient at lookahead position | Provably optimal acceleration |
| Effective rate | η/(1-β) | Momentum amplifies learning rate |

**Key takeaway**: Momentum transforms gradient descent from a memoryless random walk into a smooth trajectory with inertia. This simple addition—tracking velocity—provides dramatic speedups on ill-conditioned problems and remains a core component of modern optimizers like Adam.

→ **Next**: [Section 4.5: Adaptive Learning Rates](05-adaptive.md)

# Section 4.2: Gradient Descent — The Foundation

*Reading time: 18 minutes | Difficulty: ★★★☆☆*

Gradient descent is the workhorse of machine learning optimization. This section derives the algorithm from first principles, proves when it converges, and builds intuition for why it works.

## The Core Idea

We want to minimize L(θ). We can't solve ∇L(θ) = 0 analytically for neural networks. Instead, we iterate:

1. Compute the gradient ∇L(θ) at current position
2. Take a small step in the opposite direction
3. Repeat until convergence

**The update rule**:

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

where η (eta) is the **learning rate** — how big a step we take.

## Why Negative Gradient?

The gradient ∇L(θ) points in the direction of steepest **increase**. We want to decrease, so we go the opposite way.

**Formal justification**: Consider a Taylor expansion around θ:

$$L(\theta + \Delta\theta) \approx L(\theta) + \nabla L(\theta)^T \Delta\theta + O(\|\Delta\theta\|^2)$$

To decrease L, we want: $\nabla L(\theta)^T \Delta\theta < 0$

The most negative value for a fixed step size ‖Δθ‖ = η occurs when:

$$\Delta\theta = -\eta \frac{\nabla L(\theta)}{\|\nabla L(\theta)\|}$$

This is exactly the negative gradient direction (normalized). Taking Δθ = -η∇L gives the steepest descent.

## Derivation from Taylor Series

Let's be more rigorous. The second-order Taylor expansion is:

$$L(\theta + \Delta\theta) = L(\theta) + \nabla L(\theta)^T \Delta\theta + \frac{1}{2} \Delta\$theta^T$ H \Delta\theta + O(\|\Delta\theta\|^3)$$

where H is the Hessian matrix.

Setting Δθ = -η∇L:

$$L(\theta - \eta\nabla L) = L(\theta) - \eta \|\nabla L\|^2 + \frac{\$eta^2$}{2} \nabla $L^T$ H \nabla L + O(\$eta^3$)$$

For small η, the dominant term is $-\eta \|\nabla L\|^2 < 0$.

**Key insight**: As long as ∇L ≠ 0 and η is small enough, we're guaranteed to decrease the loss!

## Choosing the Learning Rate

The learning rate η is crucial:

| η too small | η too large |
|-------------|-------------|
| Slow convergence | Overshooting |
| May get stuck | Divergence |
| Wastes computation | Oscillation |

### The Goldilocks Zone

For a quadratic function $L(\theta) = \frac{1}{2}\theta^T H \theta$ with eigenvalues λ₁ ≤ ... ≤ λₙ:

- **Convergence requires**: η < 2/λₘₐₓ
- **Optimal for quadratic**: η = 2/(λₘᵢₙ + λₘₐₓ)

!!! warning "Common Mistake: Fixed Learning Rate"

    Using the same learning rate throughout training is usually suboptimal. Early in training, you can take larger steps. Later, you need smaller steps for fine-tuning. This motivates learning rate schedules (Section 4.6).

### Practical Heuristics

Since we can't compute eigenvalues for neural networks:

1. **Start small**: η = 0.001 is a common default
2. **Learning rate finder**: Sweep η from 10⁻⁷ to 10, plot loss vs η
3. **If loss explodes**: Reduce η by 10x
4. **If loss plateaus**: Try increasing η

## Convergence Analysis

When does gradient descent converge? Under what conditions?

### Lipschitz Continuous Gradients

A function has **L-Lipschitz continuous gradients** if:

$$\|\nabla L(\theta_1) - \nabla L(\theta_2)\| \leq L \|\theta_1 - \theta_2\|$$

This means the gradient doesn't change too rapidly. L is called the Lipschitz constant.

### Convergence Theorem

**Theorem**: For a function with L-Lipschitz gradients and learning rate η ≤ 1/L:

$$L(\theta_T) - L(\theta^*) \leq \frac{\|\theta_0 - \theta^*\|^2}{2\eta T}$$

**Proof sketch**:

1. From Lipschitz property: $L(\theta_{t+1}) \leq L(\theta_t) - \frac{\eta}{2}\|\nabla L(\theta_t)\|^2$
2. Sum over t = 0 to T-1
3. Use telescoping and rearrange

**Interpretation**: After T iterations, we're within O(1/T) of the optimum. To halve the error, we need to double the iterations.

### For Strongly Convex Functions

If L is also **μ-strongly convex** (curves upward everywhere):

$$L(\theta_T) - L(\theta^*) \leq \left(1 - \frac{\mu}{L}\right)^T (L(\theta_0) - L(\theta^*))$$

This is **linear convergence** — exponentially fast! The rate depends on the condition number κ = L/μ.

## The Algorithm

```python
def gradient_descent(loss_fn, grad_fn, theta_init, lr, num_steps):
    """
    Vanilla gradient descent.

    Args:
        loss_fn: Function computing L(θ)
        grad_fn: Function computing ∇L(θ)
        theta_init: Initial parameters
        lr: Learning rate η
        num_steps: Number of iterations

    Returns:
        Final parameters and loss history
    """
    theta = theta_init.copy()
    history = []

    for t in range(num_steps):
        loss = loss_fn(theta)
        grad = grad_fn(theta)

        history.append(loss)

        # The core update
        theta = theta - lr * grad

    return theta, history
```

## Visualizing Gradient Descent

Consider minimizing $L(x, y) = x^2 + 10y^2$ (an elliptical bowl):

```
Step 0: (5.0, 5.0), Loss = 275.0
Step 1: (4.0, 3.0), Loss = 106.0
Step 2: (3.2, 1.8), Loss = 42.64
...
Step 20: (0.11, 0.00), Loss = 0.012
```

The path zigzags because:

1. The gradient in y is 10x larger than in x
2. We overshoot in y, undershoot in x
3. This is the "condition number problem"

```
     y
     ↑     Start
     │      ╲
     │       ╲
     │        ↘
     │      ↙  ↘
     │    ↙     ↘
     │  ↙        →→→ Optimum
     └──────────────→ x
```

## Limitations of Vanilla Gradient Descent

### 1. Sensitive to Condition Number

For ill-conditioned problems (elongated loss landscapes), convergence is slow due to zigzagging.

**Solution**: Momentum (Section 4.4) or adaptive learning rates (Section 4.5).

### 2. Same Learning Rate for All Parameters

Some parameters may need larger updates, others smaller.

**Solution**: Per-parameter adaptive rates (AdaGrad, Adam).

### 3. Stuck in Saddle Points

At saddle points, the gradient is zero, so we stop moving.

**Solution**: Noise from stochastic gradients helps escape.

### 4. Can't Escape Local Minima

Pure gradient descent always goes downhill. It can't climb over barriers to find better minima.

**Solution**: Stochastic noise, learning rate schedules, or random restarts.

## Batch Gradient Descent vs. Full Gradient

**Full (batch) gradient descent**: Compute gradient over ALL training data.

$$\nabla L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \nabla \ell_i(\theta)$$

**Problems**:

- For N = 1 billion examples, this is very expensive
- Must load all data into memory
- One update per full pass through data

**Solution**: Stochastic Gradient Descent (next section).

## Historical Note

**Augustin-Louis Cauchy** first described gradient descent in 1847 for solving systems of equations. He called it "méthode de descente" (descent method).

The method was rediscovered and popularized in the 1950s with the advent of computers. **Haskell Curry** (1944) and **Kenneth Arrow & Leonid Hurwicz** (1958) provided convergence analyses.

For neural networks, gradient descent combined with backpropagation was popularized by **Rumelhart, Hinton, and Williams** (1986).

!!! info "Connection to Modern LLMs"

    Modern LLMs don't use vanilla gradient descent. They use:

    - **Stochastic** gradients (mini-batches of 1-4 million tokens)
    - **Momentum** (usually β = 0.9)
    - **Adaptive learning rates** (Adam or AdamW)
    - **Learning rate warmup** (prevents early instability)
    - **Gradient clipping** (prevents explosions)

    But all of these are built on the foundation we derived here.

## Complexity Analysis

| Operation | Cost |
|-----------|------|
| Compute full gradient | O(N × n) |
| Update parameters | O(n) |
| Total per iteration | O(N × n) |

Where N = dataset size, n = number of parameters.

For GPT-3 (175B parameters) on 300B tokens: each full gradient would require ~10²³ FLOPs. This is why we use stochastic methods.

## Exercises

1. **Implement gradient descent**: Write gradient descent for $L(x) = x^4 - 3x^2 + 2$. Find all critical points.

2. **Learning rate exploration**: For $L(x,y) = x^2 + 100y^2$, run gradient descent with η = 0.001, 0.01, 0.1. What happens?

3. **Convergence rate**: Empirically verify the O(1/T) convergence rate on a convex quadratic.

4. **Visualize trajectories**: Plot the optimization path on a contour plot of the loss surface.

5. **Compare to closed-form**: For linear regression, compare gradient descent solution to the closed-form solution $(X^TX)^{-1}X^Ty$.

## Summary

| Concept | Definition | Key Insight |
|---------|------------|-------------|
| Gradient descent | θ ← θ - η∇L | Move opposite to gradient |
| Learning rate | Step size η | Too big = diverge, too small = slow |
| Convergence rate | O(1/T) general, O(ρᵀ) strongly convex | Depends on problem conditioning |
| Lipschitz constant | Upper bound on gradient change | Determines max safe learning rate |

**Key takeaway**: Gradient descent is simple but powerful. Its limitations (sensitivity to conditioning, same rate for all parameters) motivate the more sophisticated methods we'll study next.

→ **Next**: [Section 4.3: Stochastic Gradient Descent](03-sgd.md)

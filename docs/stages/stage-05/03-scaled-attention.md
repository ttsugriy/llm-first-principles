# Section 5.3: Scaled Attention — Why √d Matters

*Reading time: 12 minutes | Difficulty: ★★★☆☆*

A subtle but crucial detail: we divide attention scores by √d_k before applying softmax. This section derives why this scaling is necessary for stable training.

## The Problem: Exploding Dot Products

Consider the dot product between two random vectors:

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

If q and k have components drawn independently from a standard normal distribution (mean 0, variance 1), what's the distribution of q·k?

### Variance Analysis

Each term q_i k_i is a product of two independent N(0,1) variables.

**Fact**: If X, Y ~ N(0,1) independently, then Var(XY) = 1.

**Proof sketch**:

- E[XY] = E[X]E[Y] = 0
- E[(XY)²] = E[X²]E[Y²] = 1 × 1 = 1
- Var(XY) = E[(XY)²] - E[XY]² = 1 - 0 = 1

Since q·k is a sum of d_k such terms:

$$\text{Var}(q \cdot k) = d_k$$

**The dot product's variance grows linearly with dimension!**

### Why This Matters

For d_k = 64, the standard deviation of scores is √64 = 8.

With scores on this scale, softmax becomes extremely peaked:

```
Before softmax with large scores:
scores = [8.5, -2.1, 7.9, ...]

After softmax:
weights ≈ [0.99, 0.00, 0.01, ...]
           ↑
    Almost all attention on one position!
```

When softmax saturates:

1. **Gradients vanish**: ∂softmax/∂input → 0 at extremes
2. **No learning**: The model can't adjust attention patterns
3. **Information loss**: Only one position contributes

## The Solution: Scale by √d_k

Divide scores by √d_k before softmax:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{$QK^T$}{\sqrt{d_k}}\right) V$$

### Why √d_k Works

If we scale by √d_k:

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{\text{Var}(q \cdot k)}{d_k} = \frac{d_k}{d_k} = 1$$

The scaled scores have unit variance, regardless of dimension!

```
d_k = 64

Unscaled:  Var(q·k) = 64, Std = 8   → softmax saturates
Scaled:    Var(q·k/8) = 1, Std = 1  → softmax works well
```

## Visualizing the Effect

Consider scores before softmax for a 5-position sequence:

**Without scaling** (d_k = 64, scores have std ≈ 8):
```
Scores:  [9.2, -3.1, 8.8, -5.4, 1.2]

Softmax: [0.599, 0.000, 0.401, 0.000, 0.000]
         ↑             ↑
    Attention concentrated on just 2 positions
```

**With scaling** (divide by 8):
```
Scores:  [1.15, -0.39, 1.10, -0.68, 0.15]

Softmax: [0.31, 0.07, 0.30, 0.05, 0.12]
         ↑           ↑
    Attention distributed more evenly
```

The scaled version allows attention to spread across multiple positions, enabling the model to combine information from several sources.

## Mathematical Derivation

### Setup

Let q, k ∈ $ℝ^{d_k}$ with components:

- q_i ~ N(0, 1) independently
- k_i ~ N(0, 1) independently

### Mean of Dot Product

$$E[q \cdot k] = E\left[\sum_{i=1}^{d_k} q_i k_i\right] = \sum_{i=1}^{d_k} E[q_i k_i] = \sum_{i=1}^{d_k} E[q_i]E[k_i] = 0$$

### Variance of Dot Product

$$\text{Var}(q \cdot k) = \text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k \cdot 1 = d_k$$

### After Scaling

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{1}{d_k} \text{Var}(q \cdot k) = \frac{d_k}{d_k} = 1$$

## When Scaling Matters Most

The impact of scaling depends on:

| Factor | Effect | Scaling Importance |
|--------|--------|-------------------|
| High d_k | Larger variance | Critical |
| Low d_k | Smaller variance | Less critical |
| Random init | Scores near N(0, d_k) | Critical |
| Trained model | Scores may be controlled | Less critical |

In practice, always scale. The cost is negligible and prevents instability.

## Alternative Scaling Factors

### Temperature Scaling

Some implementations use a learnable temperature:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{$QK^T$}{\tau}\right) V$$

where τ is learned or set as a hyperparameter.

- τ < √d_k: Sharper attention (more peaked)
- τ > √d_k: Softer attention (more uniform)

### Query-Dependent Scaling

Some architectures learn position-dependent scaling:

$$\text{score}_{ij} = \frac{q_i \cdot k_j}{g(q_i)}$$

This allows different queries to have different "sharpness."

!!! info "Connection to Modern LLMs"

    Modern LLMs universally use √d_k scaling. Some variations:

    - **GPT**: Standard √d_k scaling
    - **LLaMA**: Standard scaling
    - **Some efficient variants**: Learn the temperature

    The scaling factor is so fundamental it's rarely mentioned in papers—it's just assumed.

## Implementation

```python
import math

def scaled_dot_product_attention(Q, K, V):
    """
    Scaled dot-product attention.

    Args:
        Q: Queries [n, d_k]
        K: Keys [n, d_k]
        V: Values [n, d_v]

    Returns:
        Output [n, d_v], attention weights [n, n]
    """
    d_k = Q.shape[-1]

    # Compute scaled scores
    scores = Q @ K.T / math.sqrt(d_k)  # [n, n]

    # Apply softmax
    attention_weights = softmax(scores, axis=-1)

    # Weighted sum
    output = attention_weights @ V

    return output, attention_weights
```

## Numerical Stability

Softmax can overflow with large inputs. The standard trick:

$$\text{softmax}(x)_i = \frac{$e^{x_i}$}{\sum_j $e^{x_j}$} = \frac{$e^{x_i - \max(x)}$}{\sum_j $e^{x_j - \max(x)}$}$$

Subtracting the maximum prevents overflow while giving the same result:

```python
def stable_softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)
```

## Gradient Flow

The scaling also helps gradient flow during backpropagation.

**Without scaling**: Softmax outputs are near 0 or 1, gradients are tiny.

**With scaling**: Softmax outputs are moderate, gradients are healthy.

```
∂softmax(x)_i/∂x_j = softmax(x)_i (δ_ij - softmax(x)_j)

When softmax ≈ [1, 0, 0, ...]:
  gradients ≈ [0, 0, 0, ...]  ← vanishing!

When softmax ≈ [0.3, 0.3, 0.2, 0.2]:
  gradients ≈ [0.21, 0.21, 0.16, 0.16]  ← healthy!
```

## Exercises

1. **Variance calculation**: For d_k = 512, what's the standard deviation of unscaled scores?

2. **Softmax saturation**: Plot softmax([x, 0, 0, 0]) as x varies from 0 to 10. Where does it saturate?

3. **Scaling ablation**: Train attention with and without scaling. Compare learning curves.

4. **Temperature sweep**: Try τ = 0.5√d_k, √d_k, 2√d_k. How do attention patterns differ?

5. **Gradient analysis**: Compute ∂softmax/∂input for peaked vs uniform distributions.

## Summary

| Concept | Definition | Purpose |
|---------|------------|---------|
| Score variance | Var(q·k) = d_k | Grows with dimension |
| Scaling factor | 1/√d_k | Normalize variance to 1 |
| Softmax saturation | Peaked outputs | Vanishing gradients |
| Scaled attention | softmax($QK^T$/√d_k)V | Stable training |

**Key takeaway**: Dot product variance scales with dimension, causing softmax to saturate and gradients to vanish. Dividing by √d_k normalizes variance to 1, ensuring stable training regardless of dimension. This simple fix is essential for attention to work in practice.

→ **Next**: [Section 5.4: Self-Attention](04-self-attention.md)

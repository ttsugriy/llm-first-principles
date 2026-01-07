# Section 8.3: Gradient Statistics

*Reading time: 10 minutes*

## Gradients Tell the Story

The loss curve tells you *that* something is wrong. Gradients tell you *what*.

## Key Gradient Metrics

### 1. Gradient Norm

The L2 norm of all gradients concatenated:

$$\|\nabla L\| = \sqrt{\sum_i \left(\frac{\partial L}{\partial \theta_i}\right)^2}$$

**Healthy range**: 0.1 - 10 (varies by architecture)

**Warning signs**:

- $> 100$: Likely to explode
- $< 10^{-7}$: Vanishing, no learning

### 2. Per-Layer Gradient Norms

Track gradients layer by layer:

```python
Layer 1:  grad_norm = 0.5
Layer 2:  grad_norm = 0.4
Layer 3:  grad_norm = 0.3
Layer 10: grad_norm = 0.001  ← vanishing!
```

Gradients should be similar magnitude across layers. Large differences indicate flow problems.

### 3. Gradient Statistics

```python
@dataclass
class GradientStats:
    norm: float        # L2 norm
    max_val: float     # Largest absolute value
    min_val: float     # Smallest nonzero absolute value
    mean: float        # Average (should be ~0)
    std: float         # Standard deviation
    num_zeros: int     # Dead gradients
    num_nans: int      # Catastrophic failure
    num_infs: int      # Explosion

    def is_healthy(self) -> Tuple[bool, List[str]]:
        issues = []
        if self.num_nans > 0:
            issues.append(f'NaN gradients: {self.num_nans}')
        if self.num_infs > 0:
            issues.append(f'Inf gradients: {self.num_infs}')
        if self.norm > 1000:
            issues.append(f'Norm too large: {self.norm:.2f}')
        if self.norm < 1e-8:
            issues.append(f'Norm too small: {self.norm:.2e}')
        return len(issues) == 0, issues
```

## Gradient Clipping

When gradients explode, clip them:

```python
def clip_gradients(gradients, max_norm):
    """Clip gradients to maximum norm."""
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))

    if total_norm > max_norm:
        scale = max_norm / total_norm
        gradients = [g * scale for g in gradients]

    return gradients, total_norm
```

**Typical max_norm**: 1.0 for transformers, 5.0 for RNNs

## The Gradient Ratio Test

Compare gradient magnitude to parameter magnitude:

$$\text{ratio} = \frac{\|\nabla_\theta L\|}{\|\theta\|}$$

**Interpretation**:

- $\text{ratio} \approx 10^{-3}$: Normal for most layers
- $\text{ratio} > 1$: Gradients too large (reduce LR)
- $\text{ratio} < 10^{-6}$: Gradients too small (increase LR or fix architecture)

## Layer-wise Analysis

```python
def analyze_gradients_by_layer(model):
    """Compute gradient stats per layer."""
    results = {}
    for name, param in model.parameters():
        if param.grad is not None:
            grad = param.grad.numpy()
            results[name] = {
                'grad_norm': np.linalg.norm(grad),
                'param_norm': np.linalg.norm(param.numpy()),
                'ratio': np.linalg.norm(grad) / (np.linalg.norm(param.numpy()) + 1e-8),
            }
    return results
```

Example output:

```
embed.weight:     grad_norm=0.45, param_norm=12.3, ratio=0.037
layer1.weight:    grad_norm=0.32, param_norm=8.7,  ratio=0.037
layer10.weight:   grad_norm=0.0001, param_norm=8.9, ratio=0.00001  ← Problem!
```

## Detecting Dead Neurons

Neurons with zero gradients are "dead":

```python
def count_dead_neurons(gradients, threshold=1e-10):
    """Count neurons that never receive gradient signal."""
    dead = 0
    total = 0
    for grad in gradients:
        dead += np.sum(np.abs(grad) < threshold)
        total += grad.size
    return dead, total, dead / total
```

**Healthy**: < 5% dead neurons
**Warning**: > 20% dead neurons
**Critical**: > 50% dead neurons (model not learning)

## Gradient Flow Visualization

Track how gradients flow through the network:

```
Input → [grad: 0.5] → Layer1 → [grad: 0.3] → Layer2 → [grad: 0.001] → Output
                                                        ↑
                                                   Problem here!
```

When gradients decrease dramatically through layers, you have vanishing gradients.

## Common Gradient Patterns

| Pattern | Symptom | Cause |
|---------|---------|-------|
| All NaN | Loss = NaN | Learning rate explosion |
| Decreasing with depth | Deep layers don't learn | Vanishing gradients |
| Spiking | Occasional huge values | Outliers in data |
| Many zeros | Dead neurons | ReLU with bad init |

## Best Practices

1. **Log gradient norm every step** - Early warning system
2. **Use gradient clipping by default** - Prevents explosions
3. **Check per-layer norms weekly** - Catch vanishing early
4. **Set alerts for NaN/Inf** - Immediate notification of catastrophe

## Summary

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| Gradient norm | 0.1 - 10 | 10 - 100 | > 100 or NaN |
| Zero ratio | < 5% | 5-20% | > 20% |
| Layer variance | < 10x | 10-100x | > 100x |

**Next**: We'll learn to find the optimal learning rate systematically.

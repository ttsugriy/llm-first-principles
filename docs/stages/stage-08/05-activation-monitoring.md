# Section 8.5: Activation Monitoring

*Reading time: 10 minutes*

## Why Monitor Activations?

Gradients tell you about learning. Activations tell you about what the model has learned—and what's broken.

**Key insight**: A healthy network has activations that flow, transform, and differentiate. Unhealthy activations are stuck, dead, or explosive.

## Dead Neurons

The most common activation pathology.

### What Are Dead Neurons?

Neurons that always output zero (or effectively zero). They contribute nothing to the model.

```
Layer 3: [0.0, 0.5, 0.0, 0.3, 0.0, 0.0, 0.1, 0.0]
                ↑         ↑     ↑
              alive     alive  alive

Dead ratio: 5/8 = 62.5%  ← Critical!
```

### Why Neurons Die

**ReLU with bad initialization**:

$$\text{ReLU}(x) = \max(0, x)$$

If weights are initialized such that the pre-activation is always negative:

$$\text{ReLU}(-5) = 0$$
$$\text{ReLU}(-3) = 0$$
$$\text{ReLU}(-10) = 0$$

Once a neuron dies, it stays dead—gradient is zero for negative inputs.

**Too-high learning rate**: Large update pushes weights into "always negative" territory.

### Detecting Dead Neurons

```python
def detect_dead_neurons(activations, threshold=0.01):
    """
    Find neurons that consistently output near-zero.

    Args:
        activations: Shape [batch, ..., neurons]
        threshold: Below this is "dead"
    """
    # Average absolute activation per neuron
    neuron_means = np.mean(np.abs(activations), axis=0)
    num_dead = np.sum(neuron_means < threshold)
    ratio_dead = num_dead / len(neuron_means.flatten())
    return num_dead, ratio_dead
```

### Health Thresholds

| Dead Ratio | Status | Action |
|------------|--------|--------|
| < 5% | Healthy | None |
| 5-20% | Warning | Monitor |
| 20-50% | Serious | Fix init or LR |
| > 50% | Critical | Model barely learning |

### Fixes

1. **Use Leaky ReLU**: Small negative slope prevents death

   $$\text{LeakyReLU}(x) = \max(0.01x, x)$$

2. **Better initialization**: He initialization for ReLU

   $$W \sim \mathcal{N}(0, \sqrt{2/n_{in}})$$

3. **Lower learning rate**: Prevents catastrophic updates

4. **Batch normalization**: Keeps activations centered

## Saturation

The opposite of dead neurons: stuck at maximum output.

### What Is Saturation?

Activations stuck near the extremes of sigmoid/tanh:

```
tanh output:  [0.99, -0.99, 0.99, -0.99, 0.99]
              ↑ All saturated!
```

### Why Saturation Is Bad

At saturation, gradients vanish:

$$\frac{d \tanh(x)}{dx} = 1 - \tanh^2(x)$$

When $\tanh(x) \approx 1$:

$$\frac{d \tanh(1000)}{dx} \approx 0$$

The gradient is effectively zero. No learning happens.

### Detecting Saturation

```python
def detect_saturation(activations, threshold=0.99):
    """
    Find neurons with extreme activations.

    Works for tanh (|output| > threshold)
    or sigmoid (output > threshold or < 1-threshold).
    """
    saturated = np.sum(np.abs(activations) > threshold)
    total = activations.size
    return saturated, saturated / total
```

### Fixes

1. **Use ReLU**: No saturation (but watch for dead neurons)
2. **Normalize inputs**: Keep pre-activations in reasonable range
3. **Better initialization**: Xavier for tanh/sigmoid

   $$W \sim \mathcal{N}(0, \sqrt{1/n_{in}})$$

4. **Batch normalization**: Prevents drift toward saturation

## Exploding Activations

When activations grow without bound.

### Symptoms

```
Step 1:   max_activation = 10
Step 10:  max_activation = 100
Step 50:  max_activation = 10000
Step 51:  max_activation = Inf
```

### Causes

- Learning rate too high
- No normalization in deep network
- Residual connections without proper scaling

### Fixes

1. **Layer normalization**: Normalize each layer's output
2. **Gradient clipping**: Prevents updates that cause explosion
3. **Proper residual scaling**: For very deep networks

## Vanishing Activations

The opposite of explosion: everything shrinks to zero.

### Symptoms

```
Layer 1:  std = 1.0
Layer 5:  std = 0.1
Layer 10: std = 0.001
Layer 20: std = 0.00001  ← Information gone
```

### Causes

- Sigmoid/tanh without proper initialization
- Missing residual connections in deep networks
- Aggressive dropout

### Fixes

1. **Residual connections**: $h_{l+1} = h_l + f(h_l)$
2. **ReLU family**: Doesn't squash magnitudes
3. **Proper initialization**: Match activation function

## The Activation Monitor

Track activations systematically:

```python
@dataclass
class ActivationStats:
    mean: float
    std: float
    min_val: float
    max_val: float
    num_zeros: int
    num_saturated: int
    total_elements: int

    @property
    def zero_ratio(self) -> float:
        return self.num_zeros / self.total_elements

    @property
    def saturation_ratio(self) -> float:
        return self.num_saturated / self.total_elements


class ActivationMonitor:
    """Monitor activations over training."""

    def __init__(self, window_size=100):
        self.history = {}  # layer_name -> deque of stats

    def record(self, layer_name, activations):
        stats = compute_activation_stats(activations)
        self.history[layer_name].append(stats)

    def diagnose(self):
        issues = {}
        for layer, stats_history in self.history.items():
            layer_issues = []

            # Check recent history
            recent = list(stats_history)[-10:]

            # Dead neurons
            avg_zero_ratio = np.mean([s.zero_ratio for s in recent])
            if avg_zero_ratio > 0.5:
                layer_issues.append(f'Dead: {avg_zero_ratio:.0%}')

            # Saturation
            avg_sat = np.mean([s.saturation_ratio for s in recent])
            if avg_sat > 0.1:
                layer_issues.append(f'Saturated: {avg_sat:.0%}')

            if layer_issues:
                issues[layer] = layer_issues

        return issues
```

## Layer-by-Layer Health

A healthy network looks like this:

```
Layer 1:  mean=0.01, std=0.45, dead=0%, saturated=0%
Layer 2:  mean=0.02, std=0.42, dead=2%, saturated=0%
Layer 3:  mean=0.01, std=0.40, dead=3%, saturated=0%
...
Layer 10: mean=0.01, std=0.38, dead=5%, saturated=0%
```

An unhealthy network might show:

```
Layer 1:  mean=0.01, std=0.45, dead=0%, saturated=0%
Layer 2:  mean=0.00, std=0.30, dead=10%, saturated=0%
Layer 3:  mean=0.00, std=0.15, dead=25%, saturated=0%  ← Problem
...
Layer 10: mean=0.00, std=0.01, dead=80%, saturated=0%  ← Critical
```

## What To Log

Every N steps:

| Metric | Purpose |
|--------|---------|
| Activation mean | Detect bias |
| Activation std | Detect vanishing/exploding |
| Zero ratio | Detect dead neurons |
| Saturation ratio | Detect stuck neurons |
| Max/min values | Detect outliers |

## Summary

| Problem | Symptom | Cause | Fix |
|---------|---------|-------|-----|
| Dead neurons | Zero outputs | ReLU + bad init/LR | LeakyReLU, He init |
| Saturation | Values at ±1 | tanh/sigmoid + bad init | Xavier init, BatchNorm |
| Explosion | Values → ∞ | High LR, no norm | LayerNorm, clip grads |
| Vanishing | Values → 0 | Deep network, no skip | ResNet, proper init |

**Key insight**: Activations are windows into your model's behavior. Monitor them.

**Next**: We'll put everything together into systematic debugging strategies.

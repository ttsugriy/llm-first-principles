# Section 8.2: Loss Curve Analysis

*Reading time: 12 minutes*

## The Loss Curve as Diagnostic Tool

Your loss curve is a medical chart for your model. Learn to read it.

## Healthy Training

A healthy loss curve shows:

```
Loss
  │
  │╲
  │ ╲
  │  ╲___
  │      ╲____
  │           ╲_______
  └──────────────────── Steps
```

Characteristics:

- **Rapid initial descent**: Model learning easy patterns
- **Gradual slowdown**: Diminishing returns on harder patterns
- **Eventual plateau**: Model capacity reached or learning rate too low

## Pathological Patterns

### Pattern 1: Explosion

```
Loss
  │           ╱
  │          ╱
  │         ╱
  │   _____╱
  │  ╱
  │_╱
  └──────────────────── Steps
```

**Diagnosis**: Gradients exploding

**Action**:

- Reduce LR by 10x immediately
- Add gradient clipping
- Check for NaN in data

### Pattern 2: Flat Line

```
Loss
  │
  │────────────────────
  │
  │
  │
  └──────────────────── Steps
```

**Diagnosis**: Learning rate too low OR vanishing gradients OR bug

**Action**:

1. Check gradient norms (are they zero?)
2. Run one-batch overfit test
3. Increase LR by 10x

### Pattern 3: Oscillation

```
Loss
  │    ╱╲    ╱╲    ╱╲
  │   ╱  ╲  ╱  ╲  ╱  ╲
  │  ╱    ╲╱    ╲╱    ╲
  │ ╱
  │╱
  └──────────────────── Steps
```

**Diagnosis**: Learning rate too high

**Action**: Reduce LR by 2-5x

### Pattern 4: Diverging Train/Val

```
Loss
  │         _____  ← Val
  │   _____/
  │  ╱
  │ ╱ ╲_________  ← Train
  │╱
  └──────────────────── Steps
```

**Diagnosis**: Overfitting

**Action**:

- Add regularization (dropout, weight decay)
- Early stopping at minimum val loss
- Data augmentation

### Pattern 5: Slow Descent

```
Loss
  │
  │╲
  │ ╲
  │  ╲
  │   ╲
  │    ╲ (still going after 10x expected steps)
  └──────────────────── Steps
```

**Diagnosis**: LR too low OR model too small

**Action**:

- Try higher LR
- Increase model capacity
- Use LR warmup

## Implementing Loss Analysis

```python
@dataclass
class TrainingHistory:
    """Records and analyzes training metrics."""

    loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)

    def record(self, loss, grad_norm, val_loss=None):
        self.loss.append(loss)
        self.grad_norm.append(grad_norm)
        if val_loss is not None:
            self.val_loss.append(val_loss)

    def diagnose(self) -> Dict:
        """Analyze training history and diagnose issues."""
        issues = []
        recommendations = []

        if self._detect_explosion():
            issues.append('LOSS_EXPLOSION')
            recommendations.append('Reduce learning rate by 10x')
            recommendations.append('Add gradient clipping')

        if self._detect_plateau():
            issues.append('LOSS_PLATEAU')
            recommendations.append('Increase learning rate')
            recommendations.append('Check model capacity')

        if self._detect_overfitting():
            issues.append('OVERFITTING')
            recommendations.append('Add dropout')
            recommendations.append('Use early stopping')

        return {
            'status': 'critical' if 'EXPLOSION' in str(issues) else
                     'warning' if issues else 'healthy',
            'issues': issues,
            'recommendations': recommendations,
        }

    def _detect_explosion(self):
        """Check for loss explosion."""
        if len(self.loss) < 5:
            return False
        recent = self.loss[-5:]
        return any(np.isnan(l) or np.isinf(l) for l in recent)

    def _detect_plateau(self, window=50, threshold=0.001):
        """Check if loss has plateaued."""
        if len(self.loss) < window * 2:
            return False
        recent = self.loss[-window:]
        relative_change = (max(recent) - min(recent)) / np.mean(recent)
        return relative_change < threshold

    def _detect_overfitting(self):
        """Check for overfitting."""
        if len(self.val_loss) < 20:
            return False
        n = len(self.val_loss)
        early_val = np.mean(self.val_loss[:n//4])
        late_val = np.mean(self.val_loss[-n//4:])
        early_train = np.mean(self.loss[:n//4])
        late_train = np.mean(self.loss[-n//4:])
        return late_train < early_train and late_val > early_val
```

## Smoothing for Clarity

Raw loss curves are noisy. Use exponential smoothing:

```python
def smooth_loss(losses, factor=0.9):
    """Exponential moving average for cleaner visualization."""
    smoothed = []
    current = losses[0]
    for loss in losses:
        current = factor * current + (1 - factor) * loss
        smoothed.append(current)
    return smoothed
```

## What to Log

At minimum, log every step:

| Metric | Why |
|--------|-----|
| Loss | Primary signal |
| Grad norm | Gradient health |
| Learning rate | Track schedule |

Log less frequently:

| Metric | Why |
|--------|-----|
| Val loss | Overfitting detection |
| Param norm | Weight growth |
| Per-layer grad norms | Identify problem layers |

## Summary

| Pattern | Meaning | Fix |
|---------|---------|-----|
| Explosion | Gradients too large | Reduce LR, clip |
| Flat | No learning | Check bugs, increase LR |
| Oscillation | LR too high | Reduce LR |
| Train/Val diverge | Overfitting | Regularize |
| Very slow | LR too low | Increase LR |

**Next**: We'll dive deeper into gradient statistics as health indicators.

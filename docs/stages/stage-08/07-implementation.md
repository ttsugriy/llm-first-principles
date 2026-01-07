# Section 8.7: Implementation

*Reading time: 15 minutes*

## Overview

In this section, we implement the complete training diagnostics toolkit:

1. **TrainingHistory**: Records and analyzes training metrics
2. **GradientStats**: Monitors gradient health
3. **LearningRateFinder**: Systematic LR search
4. **ActivationMonitor**: Detects dead neurons and saturation
5. **TrainingDebugger**: All-in-one diagnostic tool

All code is available in `code/stage-08/diagnostics.py`.

## TrainingHistory Class

The foundation of all diagnostics—record everything:

```python
@dataclass
class TrainingHistory:
    """Records and analyzes training metrics over time."""

    loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    step: List[int] = field(default_factory=list)

    def record(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        lr: float,
        val_loss: Optional[float] = None,
    ) -> None:
        """Record metrics for a training step."""
        self.step.append(step)
        self.loss.append(loss)
        self.grad_norm.append(grad_norm)
        self.learning_rate.append(lr)
        if val_loss is not None:
            self.val_loss.append(val_loss)

    def diagnose(self) -> Dict[str, Any]:
        """Analyze history and diagnose issues."""
        issues = []
        recommendations = []

        if self._detect_explosion():
            issues.append('LOSS_EXPLOSION')
            recommendations.append('Reduce learning rate by 10x')
            recommendations.append('Add gradient clipping')

        if self._detect_plateau():
            issues.append('LOSS_PLATEAU')
            recommendations.append('Increase learning rate')

        if self._detect_overfitting():
            issues.append('OVERFITTING')
            recommendations.append('Add dropout or weight decay')

        status = 'critical' if 'EXPLOSION' in str(issues) else \
                 'warning' if issues else 'healthy'

        return {
            'status': status,
            'issues': issues,
            'recommendations': recommendations,
        }
```

### Detection Methods

```python
def _detect_explosion(self) -> bool:
    """Check for loss explosion."""
    if len(self.loss) < 5:
        return False
    recent = self.loss[-5:]
    return any(np.isnan(l) or np.isinf(l) for l in recent)

def _detect_plateau(self, window=50, threshold=0.001) -> bool:
    """Check if loss has plateaued."""
    if len(self.loss) < window * 2:
        return False
    recent = self.loss[-window:]
    relative_change = (max(recent) - min(recent)) / np.mean(recent)
    return relative_change < threshold

def _detect_overfitting(self) -> bool:
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

## GradientStats Class

Monitor gradient health at every step:

```python
@dataclass
class GradientStats:
    """Compute and track gradient statistics."""

    norm: float = 0.0
    max_val: float = 0.0
    min_val: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    num_zeros: int = 0
    num_nans: int = 0
    num_infs: int = 0
    total_elements: int = 0

    @classmethod
    def from_gradients(cls, gradients: List[np.ndarray]) -> 'GradientStats':
        """Compute statistics from gradient arrays."""
        all_grads = np.concatenate([g.flatten() for g in gradients])

        return cls(
            norm=float(np.sqrt(np.sum(all_grads ** 2))),
            max_val=float(np.max(np.abs(all_grads))),
            min_val=float(np.min(np.abs(all_grads[all_grads != 0]))),
            mean=float(np.mean(all_grads)),
            std=float(np.std(all_grads)),
            num_zeros=int(np.sum(all_grads == 0)),
            num_nans=int(np.sum(np.isnan(all_grads))),
            num_infs=int(np.sum(np.isinf(all_grads))),
            total_elements=len(all_grads),
        )

    def is_healthy(self) -> Tuple[bool, List[str]]:
        """Check if gradients are healthy."""
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

## LearningRateFinder Class

Systematic LR search:

```python
class LearningRateFinder:
    """Find optimal LR using range test."""

    def __init__(
        self,
        min_lr: float = 1e-7,
        max_lr: float = 10.0,
        num_steps: int = 100,
    ):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.lrs: List[float] = []
        self.losses: List[float] = []

    def range_test(self, train_fn: Callable[[float], float]) -> Dict:
        """Run LR range test."""
        # Exponential schedule
        lr_schedule = np.exp(np.linspace(
            np.log(self.min_lr),
            np.log(self.max_lr),
            self.num_steps
        ))

        best_loss = float('inf')

        for lr in lr_schedule:
            loss = train_fn(lr)

            # Stop if exploding
            if np.isnan(loss) or loss > 10 * best_loss:
                break

            self.lrs.append(lr)
            self.losses.append(loss)
            best_loss = min(best_loss, loss)

        return {
            'suggested_lr': self._find_suggested_lr(),
            'lrs': self.lrs,
            'losses': self.losses,
        }

    def _find_suggested_lr(self) -> float:
        """Find LR with steepest negative slope."""
        if len(self.losses) < 10:
            return self.min_lr * 10

        log_lrs = np.log(self.lrs)
        gradients = np.gradient(self.losses, log_lrs)
        min_grad_idx = np.argmin(gradients)

        # Safety margin
        suggest_idx = max(0, min_grad_idx - len(self.losses) // 10)
        return self.lrs[suggest_idx]
```

## ActivationMonitor Class

Track activation health:

```python
@dataclass
class ActivationStats:
    """Statistics for layer activations."""
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
    """Monitor activations during training."""

    def __init__(self, window_size: int = 100):
        self.history: Dict[str, deque] = {}
        self.window_size = window_size

    def record(self, layer_name: str, activations: np.ndarray) -> None:
        """Record activation statistics."""
        if layer_name not in self.history:
            self.history[layer_name] = deque(maxlen=self.window_size)

        flat = activations.flatten()
        stats = ActivationStats(
            mean=float(np.mean(flat)),
            std=float(np.std(flat)),
            min_val=float(np.min(flat)),
            max_val=float(np.max(flat)),
            num_zeros=int(np.sum(flat == 0)),
            num_saturated=int(np.sum(np.abs(flat) > 0.99)),
            total_elements=len(flat),
        )
        self.history[layer_name].append(stats)

    def diagnose(self) -> Dict[str, List[str]]:
        """Diagnose activation issues."""
        issues = {}

        for layer, stats_history in self.history.items():
            layer_issues = []
            recent = list(stats_history)[-10:]

            # Dead neurons
            avg_zero = np.mean([s.zero_ratio for s in recent])
            if avg_zero > 0.5:
                layer_issues.append(f'Dead: {avg_zero:.0%}')

            # Saturation
            avg_sat = np.mean([s.saturation_ratio for s in recent])
            if avg_sat > 0.1:
                layer_issues.append(f'Saturated: {avg_sat:.0%}')

            if layer_issues:
                issues[layer] = layer_issues

        return issues
```

## TrainingDebugger Class

All-in-one diagnostic tool:

```python
class TrainingDebugger:
    """Comprehensive training debugger."""

    def __init__(self):
        self.history = TrainingHistory()
        self.activation_monitor = ActivationMonitor()
        self.gradient_history: deque = deque(maxlen=100)

    def step(
        self,
        step: int,
        loss: float,
        gradients: List[np.ndarray],
        learning_rate: float,
        activations: Optional[Dict[str, np.ndarray]] = None,
        val_loss: Optional[float] = None,
    ) -> None:
        """Record one training step."""
        # Gradient stats
        grad_stats = GradientStats.from_gradients(gradients)
        self.gradient_history.append(grad_stats)

        # Record history
        self.history.record(
            step=step,
            loss=loss,
            grad_norm=grad_stats.norm,
            lr=learning_rate,
            val_loss=val_loss,
        )

        # Record activations
        if activations:
            for name, act in activations.items():
                self.activation_monitor.record(name, act)

    def report(self) -> Dict[str, Any]:
        """Generate diagnostic report."""
        diagnosis = self.history.diagnose()

        # Gradient health
        if self.gradient_history:
            recent = list(self.gradient_history)[-10:]
            grad_healthy = all(g.is_healthy()[0] for g in recent)
        else:
            grad_healthy = True

        # Activation issues
        activation_issues = self.activation_monitor.diagnose()

        return {
            'status': diagnosis['status'],
            'issues': diagnosis['issues'],
            'recommendations': diagnosis['recommendations'],
            'gradient_healthy': grad_healthy,
            'activation_issues': activation_issues,
        }

    def quick_check(self) -> bool:
        """Quick health check."""
        if len(self.history.loss) < 5:
            return True

        # NaN check
        if any(np.isnan(l) for l in self.history.loss[-5:]):
            return False

        # Gradient check
        if self.gradient_history:
            latest = self.gradient_history[-1]
            if latest.num_nans > 0 or latest.norm > 1000:
                return False

        return True
```

## Utility Functions

### Gradient Clipping

```python
def clip_gradients(
    gradients: List[np.ndarray],
    max_norm: float = 1.0,
) -> Tuple[List[np.ndarray], float]:
    """Clip gradients by global norm."""
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients))
    clip_coef = min(1.0, max_norm / (total_norm + 1e-6))
    clipped = [g * clip_coef for g in gradients]
    return clipped, total_norm
```

### Dead Neuron Detection

```python
def detect_dead_neurons(
    activations: np.ndarray,
    threshold: float = 0.01,
) -> Tuple[int, float]:
    """Detect neurons that always output near-zero."""
    neuron_means = np.mean(np.abs(activations), axis=0)
    num_dead = np.sum(neuron_means < threshold)
    return num_dead, num_dead / len(neuron_means.flatten())
```

### Initialization Check

```python
def check_initialization(weights: List[np.ndarray]) -> Dict:
    """Check if weight initialization is reasonable."""
    results = []

    for i, w in enumerate(weights):
        fan_in = w.shape[0] if w.ndim >= 2 else 1
        expected_std = np.sqrt(2.0 / fan_in)  # He init
        actual_std = np.std(w)

        status = 'ok'
        if actual_std < expected_std * 0.1:
            status = 'too_small'
        elif actual_std > expected_std * 10:
            status = 'too_large'

        results.append({
            'layer': i,
            'expected_std': expected_std,
            'actual_std': actual_std,
            'status': status,
        })

    return {
        'layers': results,
        'all_ok': all(r['status'] == 'ok' for r in results),
    }
```

## Usage Example

```python
# Initialize debugger
debugger = TrainingDebugger()

# Training loop
for step, (x, y) in enumerate(dataloader):
    # Forward + backward
    loss, gradients = model.train_step(x, y)

    # Record step
    debugger.step(
        step=step,
        loss=loss,
        gradients=gradients,
        learning_rate=optimizer.lr,
    )

    # Check health periodically
    if step % 100 == 0:
        if not debugger.quick_check():
            report = debugger.report()
            print(f"Issues detected: {report['issues']}")
            print(f"Recommendations: {report['recommendations']}")
```

## Running the Demo

```bash
cd code/stage-08
python diagnostics.py
```

Output:

```
============================================================
Stage 8: Training Dynamics & Debugging Demo
============================================================

1. Training History Analysis
----------------------------------------
Status: healthy
Issues: []
Metrics: {'loss_start': 3.05, 'loss_end': 0.58, ...}

2. Gradient Statistics
----------------------------------------
Norm: 44.7234
Mean: 0.000012
Std: 0.100023
Zeros: 0 / 50000
Healthy: True

3. Learning Rate Finder
----------------------------------------
Suggested LR: 1.23e-03
Tested 45 learning rates

4. Activation Monitoring
----------------------------------------
layer2: ['Dead neurons: 72% zeros']

5. Full Training Debugger
----------------------------------------
Status: healthy
Quick check: True
```

## Summary

| Component | Purpose | Key Methods |
|-----------|---------|-------------|
| TrainingHistory | Track loss/metrics | `record()`, `diagnose()` |
| GradientStats | Monitor gradient health | `from_gradients()`, `is_healthy()` |
| LearningRateFinder | Find optimal LR | `range_test()` |
| ActivationMonitor | Detect dead/saturated neurons | `record()`, `diagnose()` |
| TrainingDebugger | All-in-one | `step()`, `report()`, `quick_check()` |

These tools transform debugging from guesswork into systematic engineering.

## Exercises

1. **Add early stopping**: Modify TrainingHistory to automatically detect when to stop
2. **Layer-wise LR**: Implement per-layer learning rate finder
3. **Visualization**: Add plotting functions for loss curves and gradient distributions
4. **Checkpointing**: Save model when validation loss improves
5. **Anomaly detection**: Implement automatic detection of unusual training patterns

## What's Next

With debugging tools in hand, we're ready to explore parameter-efficient fine-tuning in Stage 9—how to adapt massive pretrained models with minimal computation.

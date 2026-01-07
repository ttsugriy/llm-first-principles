"""
Stage 8: Training Dynamics & Debugging

This module provides tools for understanding and debugging neural network training.
Most ML education shows the happy path - this teaches you what to do when things go wrong.

Key concepts:
- Loss curve analysis: What different patterns mean
- Gradient statistics: Health indicators during training
- Learning rate finding: Systematic approach to hyperparameter selection
- Activation analysis: Understanding what the network learns
- Debugging strategies: Systematic approaches to common problems

"Debugging neural networks is 80% of the job. This module teaches that 80%."
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import json


# =============================================================================
# Training History & Loss Analysis
# =============================================================================

@dataclass
class TrainingHistory:
    """
    Records and analyzes training metrics over time.

    This is your primary diagnostic tool. A well-maintained history
    tells you almost everything about what's happening during training.
    """

    loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    step: List[int] = field(default_factory=list)

    # Extended metrics
    param_norm: List[float] = field(default_factory=list)
    grad_max: List[float] = field(default_factory=list)
    grad_min: List[float] = field(default_factory=list)

    def record(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        lr: float,
        val_loss: Optional[float] = None,
        param_norm: Optional[float] = None,
        grad_max: Optional[float] = None,
        grad_min: Optional[float] = None,
    ) -> None:
        """Record metrics for a training step."""
        self.step.append(step)
        self.loss.append(loss)
        self.grad_norm.append(grad_norm)
        self.learning_rate.append(lr)

        if val_loss is not None:
            self.val_loss.append(val_loss)
        if param_norm is not None:
            self.param_norm.append(param_norm)
        if grad_max is not None:
            self.grad_max.append(grad_max)
        if grad_min is not None:
            self.grad_min.append(grad_min)

    def diagnose(self) -> Dict[str, Any]:
        """
        Analyze training history and diagnose potential issues.

        Returns a dictionary with:
        - status: 'healthy', 'warning', or 'critical'
        - issues: List of detected problems
        - recommendations: Suggested fixes
        """
        issues = []
        recommendations = []

        if len(self.loss) < 10:
            return {
                'status': 'insufficient_data',
                'issues': ['Not enough data for diagnosis'],
                'recommendations': ['Train for more steps'],
            }

        # Check for loss explosion
        if self._detect_explosion():
            issues.append('LOSS_EXPLOSION')
            recommendations.append('Reduce learning rate by 10x')
            recommendations.append('Check for NaN in data')
            recommendations.append('Add gradient clipping')

        # Check for loss plateau
        if self._detect_plateau():
            issues.append('LOSS_PLATEAU')
            recommendations.append('Increase learning rate')
            recommendations.append('Check if model has enough capacity')
            recommendations.append('Verify data is being shuffled')

        # Check for gradient vanishing
        if self._detect_vanishing_gradients():
            issues.append('VANISHING_GRADIENTS')
            recommendations.append('Use residual connections')
            recommendations.append('Try different activation function')
            recommendations.append('Check initialization')

        # Check for gradient explosion
        if self._detect_exploding_gradients():
            issues.append('EXPLODING_GRADIENTS')
            recommendations.append('Add gradient clipping')
            recommendations.append('Reduce learning rate')
            recommendations.append('Use gradient norm monitoring')

        # Check for overfitting
        if len(self.val_loss) > 10 and self._detect_overfitting():
            issues.append('OVERFITTING')
            recommendations.append('Add dropout or weight decay')
            recommendations.append('Use data augmentation')
            recommendations.append('Reduce model capacity')

        # Determine status
        if any(i in ['LOSS_EXPLOSION', 'EXPLODING_GRADIENTS'] for i in issues):
            status = 'critical'
        elif issues:
            status = 'warning'
        else:
            status = 'healthy'

        return {
            'status': status,
            'issues': issues,
            'recommendations': recommendations,
            'metrics': self._compute_summary_metrics(),
        }

    def _detect_explosion(self, threshold: float = 100.0) -> bool:
        """Detect if loss has exploded."""
        if len(self.loss) < 5:
            return False

        # Check for NaN or Inf
        if any(np.isnan(l) or np.isinf(l) for l in self.loss[-5:]):
            return True

        # Check for sudden large increase
        recent = self.loss[-5:]
        if max(recent) > threshold * min(self.loss[:10]):
            return True

        return False

    def _detect_plateau(self, window: int = 50, threshold: float = 0.001) -> bool:
        """Detect if loss has plateaued."""
        if len(self.loss) < window * 2:
            return False

        recent = self.loss[-window:]
        relative_change = (max(recent) - min(recent)) / (np.mean(recent) + 1e-8)
        return relative_change < threshold

    def _detect_vanishing_gradients(self, threshold: float = 1e-7) -> bool:
        """Detect vanishing gradients."""
        if len(self.grad_norm) < 10:
            return False
        return np.mean(self.grad_norm[-10:]) < threshold

    def _detect_exploding_gradients(self, threshold: float = 100.0) -> bool:
        """Detect exploding gradients."""
        if len(self.grad_norm) < 10:
            return False
        return np.max(self.grad_norm[-10:]) > threshold

    def _detect_overfitting(self) -> bool:
        """Detect overfitting (val_loss increasing while train_loss decreasing)."""
        if len(self.val_loss) < 20:
            return False

        # Compare first and last quarters
        n = len(self.val_loss)
        early_val = np.mean(self.val_loss[:n//4])
        late_val = np.mean(self.val_loss[-n//4:])

        early_train = np.mean(self.loss[:n//4])
        late_train = np.mean(self.loss[-n//4:])

        # Overfitting: train improves but val worsens
        return late_train < early_train and late_val > early_val

    def _compute_summary_metrics(self) -> Dict[str, float]:
        """Compute summary statistics."""
        metrics = {}

        if self.loss:
            metrics['loss_start'] = self.loss[0]
            metrics['loss_end'] = self.loss[-1]
            metrics['loss_min'] = min(self.loss)
            metrics['loss_improvement'] = self.loss[0] - min(self.loss)

        if self.grad_norm:
            metrics['grad_norm_mean'] = np.mean(self.grad_norm)
            metrics['grad_norm_std'] = np.std(self.grad_norm)
            metrics['grad_norm_max'] = max(self.grad_norm)

        return metrics

    def save(self, path: str) -> None:
        """Save history to JSON file."""
        data = {
            'loss': self.loss,
            'val_loss': self.val_loss,
            'grad_norm': self.grad_norm,
            'learning_rate': self.learning_rate,
            'step': self.step,
            'param_norm': self.param_norm,
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'TrainingHistory':
        """Load history from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        history = cls()
        history.loss = data.get('loss', [])
        history.val_loss = data.get('val_loss', [])
        history.grad_norm = data.get('grad_norm', [])
        history.learning_rate = data.get('learning_rate', [])
        history.step = data.get('step', [])
        history.param_norm = data.get('param_norm', [])
        return history


# =============================================================================
# Gradient Statistics
# =============================================================================

@dataclass
class GradientStats:
    """
    Compute and track gradient statistics.

    Healthy gradients:
    - Norm in reasonable range (not too small, not too large)
    - No NaN or Inf values
    - Relatively stable across steps
    """

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
        """Compute statistics from a list of gradient arrays."""
        all_grads = np.concatenate([g.flatten() for g in gradients if g is not None])

        return cls(
            norm=float(np.sqrt(np.sum(all_grads ** 2))),
            max_val=float(np.max(np.abs(all_grads))),
            min_val=float(np.min(np.abs(all_grads[all_grads != 0]))) if np.any(all_grads != 0) else 0.0,
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
            issues.append(f'NaN gradients detected: {self.num_nans}')

        if self.num_infs > 0:
            issues.append(f'Inf gradients detected: {self.num_infs}')

        if self.norm > 1000:
            issues.append(f'Gradient norm very large: {self.norm:.2f}')

        if self.norm < 1e-8:
            issues.append(f'Gradient norm very small: {self.norm:.2e}')

        zero_ratio = self.num_zeros / self.total_elements if self.total_elements > 0 else 0
        if zero_ratio > 0.5:
            issues.append(f'Many zero gradients: {zero_ratio:.1%}')

        return len(issues) == 0, issues


def compute_layer_gradient_stats(
    gradients: List[np.ndarray],
    layer_names: Optional[List[str]] = None
) -> Dict[str, GradientStats]:
    """Compute gradient statistics per layer."""
    if layer_names is None:
        layer_names = [f'layer_{i}' for i in range(len(gradients))]

    return {
        name: GradientStats.from_gradients([grad])
        for name, grad in zip(layer_names, gradients)
        if grad is not None
    }


# =============================================================================
# Learning Rate Finder
# =============================================================================

class LearningRateFinder:
    """
    Find optimal learning rate using the LR range test.

    Algorithm (Smith, 2017):
    1. Start with a very small learning rate
    2. Increase LR exponentially each batch
    3. Record loss at each LR
    4. Plot loss vs LR
    5. Choose LR where loss starts decreasing rapidly

    The optimal LR is typically 1-2 orders of magnitude below the point
    where loss starts exploding.
    """

    def __init__(
        self,
        min_lr: float = 1e-7,
        max_lr: float = 10.0,
        num_steps: int = 100,
        smooth_factor: float = 0.05,
    ):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.smooth_factor = smooth_factor

        self.lrs: List[float] = []
        self.losses: List[float] = []
        self.smoothed_losses: List[float] = []

    def range_test(
        self,
        model: Any,
        train_fn: Callable[[float], float],
        reset_fn: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run LR range test.

        Args:
            model: The model to test
            train_fn: Function that takes LR, does one step, returns loss
            reset_fn: Optional function to reset model to initial state

        Returns:
            Dictionary with results and suggested learning rate
        """
        # Compute LR schedule (exponential)
        lr_schedule = np.exp(np.linspace(
            np.log(self.min_lr),
            np.log(self.max_lr),
            self.num_steps
        ))

        self.lrs = []
        self.losses = []
        self.smoothed_losses = []

        best_loss = float('inf')
        smoothed_loss = None

        for i, lr in enumerate(lr_schedule):
            # Train one step
            loss = train_fn(lr)

            # Check for explosion
            if np.isnan(loss) or np.isinf(loss) or loss > 10 * best_loss:
                break

            # Record
            self.lrs.append(lr)
            self.losses.append(loss)

            # Exponential smoothing
            if smoothed_loss is None:
                smoothed_loss = loss
            else:
                smoothed_loss = self.smooth_factor * loss + (1 - self.smooth_factor) * smoothed_loss
            self.smoothed_losses.append(smoothed_loss)

            best_loss = min(best_loss, loss)

        # Find suggested LR
        suggested_lr = self._find_suggested_lr()

        if reset_fn:
            reset_fn()

        return {
            'suggested_lr': suggested_lr,
            'lrs': self.lrs,
            'losses': self.losses,
            'smoothed_losses': self.smoothed_losses,
        }

    def _find_suggested_lr(self) -> float:
        """Find the learning rate with steepest negative slope."""
        if len(self.smoothed_losses) < 10:
            return self.min_lr * 10

        # Find point with maximum negative gradient
        losses = np.array(self.smoothed_losses)
        lrs = np.array(self.lrs)

        # Compute gradients in log space
        log_lrs = np.log(lrs)
        gradients = np.gradient(losses, log_lrs)

        # Find steepest descent point
        min_grad_idx = np.argmin(gradients)

        # Suggest LR a bit before the minimum
        # (conservative choice to avoid instability)
        suggest_idx = max(0, min_grad_idx - len(losses) // 10)

        return float(lrs[suggest_idx])


# =============================================================================
# Activation Statistics
# =============================================================================

@dataclass
class ActivationStats:
    """Statistics for layer activations."""

    mean: float
    std: float
    min_val: float
    max_val: float
    num_zeros: int
    num_saturated: int  # Very large or very small values
    total_elements: int

    @property
    def zero_ratio(self) -> float:
        return self.num_zeros / self.total_elements if self.total_elements > 0 else 0

    @property
    def saturation_ratio(self) -> float:
        return self.num_saturated / self.total_elements if self.total_elements > 0 else 0


def compute_activation_stats(
    activations: np.ndarray,
    saturation_threshold: float = 0.99,
) -> ActivationStats:
    """Compute statistics for a layer's activations."""
    flat = activations.flatten()

    return ActivationStats(
        mean=float(np.mean(flat)),
        std=float(np.std(flat)),
        min_val=float(np.min(flat)),
        max_val=float(np.max(flat)),
        num_zeros=int(np.sum(flat == 0)),
        num_saturated=int(np.sum(np.abs(flat) > saturation_threshold)),
        total_elements=len(flat),
    )


class ActivationMonitor:
    """
    Monitor activations during training to detect issues.

    Common problems detected:
    - Dead ReLU: Neurons that always output zero
    - Saturation: Neurons stuck at extreme values
    - Exploding activations: Values growing without bound
    - Vanishing activations: Values shrinking to zero
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: Dict[str, deque] = {}

    def record(self, layer_name: str, activations: np.ndarray) -> None:
        """Record activation statistics for a layer."""
        if layer_name not in self.history:
            self.history[layer_name] = deque(maxlen=self.window_size)

        stats = compute_activation_stats(activations)
        self.history[layer_name].append(stats)

    def diagnose(self) -> Dict[str, List[str]]:
        """Diagnose activation-related issues."""
        issues = {}

        for layer_name, stats_history in self.history.items():
            layer_issues = []

            if not stats_history:
                continue

            recent = list(stats_history)[-10:]

            # Check for dead neurons (consistently high zero ratio)
            avg_zero_ratio = np.mean([s.zero_ratio for s in recent])
            if avg_zero_ratio > 0.5:
                layer_issues.append(f'Dead neurons: {avg_zero_ratio:.1%} zeros')

            # Check for saturation
            avg_saturation = np.mean([s.saturation_ratio for s in recent])
            if avg_saturation > 0.1:
                layer_issues.append(f'Saturation: {avg_saturation:.1%} saturated')

            # Check for vanishing activations
            avg_std = np.mean([s.std for s in recent])
            if avg_std < 1e-6:
                layer_issues.append(f'Vanishing activations: std={avg_std:.2e}')

            # Check for exploding activations
            max_val = max(s.max_val for s in recent)
            if max_val > 1000:
                layer_issues.append(f'Exploding activations: max={max_val:.2e}')

            if layer_issues:
                issues[layer_name] = layer_issues

        return issues


# =============================================================================
# Common Training Fixes
# =============================================================================

def clip_gradients(
    gradients: List[np.ndarray],
    max_norm: float = 1.0,
) -> Tuple[List[np.ndarray], float]:
    """
    Clip gradients by global norm.

    This is the #1 fix for exploding gradients.
    """
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients if g is not None))

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = min(1.0, clip_coef)

    clipped = [g * clip_coef if g is not None else None for g in gradients]

    return clipped, float(total_norm)


def detect_dead_neurons(
    activations: np.ndarray,
    threshold: float = 0.01,
) -> Tuple[int, float]:
    """
    Detect dead neurons (always output zero or near-zero).

    Returns:
        (num_dead, ratio_dead)
    """
    # For ReLU, check which neurons are always <= 0
    # Assumes activations shape is [batch, ..., neurons]
    neuron_means = np.mean(np.abs(activations), axis=tuple(range(activations.ndim - 1)))
    num_dead = int(np.sum(neuron_means < threshold))
    ratio_dead = num_dead / len(neuron_means)

    return num_dead, ratio_dead


def check_initialization(
    weights: List[np.ndarray],
    expected_std: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Check if weight initialization is reasonable.

    Good initialization:
    - Variance should be approximately 2/fan_in (He) or 1/fan_in (Xavier)
    - No extreme values
    - Mean close to zero
    """
    results = []

    for i, w in enumerate(weights):
        if w.ndim < 2:
            continue

        fan_in = w.shape[0] if w.ndim == 2 else np.prod(w.shape[:-1])
        expected = np.sqrt(2.0 / fan_in) if expected_std is None else expected_std

        actual_std = np.std(w)
        actual_mean = np.mean(w)

        status = 'ok'
        if actual_std < expected * 0.1:
            status = 'too_small'
        elif actual_std > expected * 10:
            status = 'too_large'
        if abs(actual_mean) > 0.1:
            status = 'biased'

        results.append({
            'layer': i,
            'shape': w.shape,
            'fan_in': fan_in,
            'expected_std': expected,
            'actual_std': actual_std,
            'actual_mean': actual_mean,
            'status': status,
        })

    return {
        'layers': results,
        'all_ok': all(r['status'] == 'ok' for r in results),
    }


# =============================================================================
# Training Debugger
# =============================================================================

class TrainingDebugger:
    """
    All-in-one training debugger.

    Usage:
        debugger = TrainingDebugger()

        for step, (x, y) in enumerate(dataloader):
            loss, grads = model.train_step(x, y)

            debugger.step(
                step=step,
                loss=loss,
                gradients=grads,
                learning_rate=optimizer.lr,
            )

            # Check for issues
            if step % 100 == 0:
                report = debugger.report()
                if report['status'] != 'healthy':
                    print(report)
    """

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

        # Record in history
        self.history.record(
            step=step,
            loss=loss,
            grad_norm=grad_stats.norm,
            lr=learning_rate,
            val_loss=val_loss,
            grad_max=grad_stats.max_val,
        )

        # Record activations
        if activations:
            for name, act in activations.items():
                self.activation_monitor.record(name, act)

    def report(self) -> Dict[str, Any]:
        """Generate diagnostic report."""
        # Get history diagnosis
        history_diagnosis = self.history.diagnose()

        # Get gradient health
        if self.gradient_history:
            recent_grads = list(self.gradient_history)[-10:]
            grad_healthy = all(g.is_healthy()[0] for g in recent_grads)
            grad_issues = []
            for g in recent_grads:
                _, issues = g.is_healthy()
                grad_issues.extend(issues)
            grad_issues = list(set(grad_issues))  # Deduplicate
        else:
            grad_healthy = True
            grad_issues = []

        # Get activation issues
        activation_issues = self.activation_monitor.diagnose()

        # Combine
        all_issues = history_diagnosis['issues'] + grad_issues
        for layer, issues in activation_issues.items():
            all_issues.extend([f'{layer}: {i}' for i in issues])

        status = history_diagnosis['status']
        if not grad_healthy:
            status = 'warning' if status == 'healthy' else status

        return {
            'status': status,
            'issues': all_issues,
            'recommendations': history_diagnosis['recommendations'],
            'metrics': history_diagnosis.get('metrics', {}),
            'gradient_healthy': grad_healthy,
            'activation_issues': activation_issues,
        }

    def quick_check(self) -> bool:
        """Quick health check - returns True if training looks healthy."""
        if len(self.history.loss) < 5:
            return True

        # Check for NaN
        if any(np.isnan(l) for l in self.history.loss[-5:]):
            return False

        # Check for gradient issues
        if self.gradient_history:
            latest = self.gradient_history[-1]
            if latest.num_nans > 0 or latest.num_infs > 0:
                return False
            if latest.norm > 1000 or latest.norm < 1e-10:
                return False

        return True


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate training diagnostics."""
    print("=" * 60)
    print("Stage 8: Training Dynamics & Debugging Demo")
    print("=" * 60)

    # Simulate training history
    print("\n1. Training History Analysis")
    print("-" * 40)

    history = TrainingHistory()

    # Simulate healthy training
    for i in range(100):
        loss = 3.0 * np.exp(-i / 30) + 0.5 + np.random.normal(0, 0.1)
        grad_norm = 1.0 + np.random.normal(0, 0.1)
        history.record(step=i, loss=loss, grad_norm=grad_norm, lr=1e-3)

    diagnosis = history.diagnose()
    print(f"Status: {diagnosis['status']}")
    print(f"Issues: {diagnosis['issues']}")
    print(f"Metrics: {diagnosis['metrics']}")

    # Gradient statistics
    print("\n2. Gradient Statistics")
    print("-" * 40)

    grads = [np.random.randn(100, 100) * 0.1 for _ in range(5)]
    stats = GradientStats.from_gradients(grads)
    print(f"Norm: {stats.norm:.4f}")
    print(f"Mean: {stats.mean:.6f}")
    print(f"Std: {stats.std:.6f}")
    print(f"Zeros: {stats.num_zeros} / {stats.total_elements}")

    healthy, issues = stats.is_healthy()
    print(f"Healthy: {healthy}")
    if issues:
        print(f"Issues: {issues}")

    # Learning rate finder (simulated)
    print("\n3. Learning Rate Finder")
    print("-" * 40)

    lr_finder = LearningRateFinder(min_lr=1e-6, max_lr=1.0, num_steps=50)

    # Simulate training function
    def fake_train(lr: float) -> float:
        # Simulated loss that increases with very high LR
        base_loss = 2.0
        if lr > 0.1:
            return base_loss + (lr - 0.1) * 100
        else:
            return base_loss - lr * 10 + np.random.normal(0, 0.1)

    result = lr_finder.range_test(None, fake_train)
    print(f"Suggested LR: {result['suggested_lr']:.2e}")
    print(f"Tested {len(result['lrs'])} learning rates")

    # Activation monitoring
    print("\n4. Activation Monitoring")
    print("-" * 40)

    monitor = ActivationMonitor()

    # Simulate some activations
    for _ in range(20):
        # Normal activations
        monitor.record('layer1', np.random.randn(32, 64) * 0.5)
        # Dead neurons (many zeros)
        dead_act = np.maximum(0, np.random.randn(32, 64) - 2)  # Mostly zeros
        monitor.record('layer2', dead_act)

    issues = monitor.diagnose()
    for layer, layer_issues in issues.items():
        print(f"{layer}: {layer_issues}")

    print("\n5. Full Training Debugger")
    print("-" * 40)

    debugger = TrainingDebugger()

    for i in range(50):
        loss = 2.0 - i * 0.02 + np.random.normal(0, 0.05)
        grads = [np.random.randn(100) * 0.1 for _ in range(3)]
        debugger.step(step=i, loss=loss, gradients=grads, learning_rate=1e-3)

    report = debugger.report()
    print(f"Status: {report['status']}")
    print(f"Quick check: {debugger.quick_check()}")


if __name__ == '__main__':
    demo()

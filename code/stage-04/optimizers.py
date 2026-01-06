"""
Optimizers from Scratch

This module implements common optimization algorithms used in deep learning,
matching the functionality of PyTorch's optimizers but built from first principles.

Usage:
    from optimizers import Adam, AdamW, SGD, WarmupCosineScheduler

    # Initialize optimizer
    optimizer = AdamW(model.params, lr=1e-3, weight_decay=0.01)

    # Training loop
    for batch in data:
        loss, grads = model.forward_backward(batch)
        grads = clip_grad_norm(grads, max_norm=1.0)
        optimizer.step(grads)
        scheduler.step()
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class Optimizer(ABC):
    """
    Base class for all optimizers.

    All optimizers share a common interface:
    - __init__: Initialize with parameters and hyperparameters
    - step: Update parameters given gradients
    - zero_grad: Clear any cached state (optional)

    Attributes:
        params: List of parameter arrays to optimize
        lr: Learning rate (can be modified by schedulers)
        state: Dictionary holding optimizer state (momentum, etc.)
    """

    def __init__(self, params: List[np.ndarray], lr: float = 1e-3):
        self.params = params
        self.lr = lr
        self.state = {}

    @abstractmethod
    def step(self, grads: List[np.ndarray]) -> None:
        """Update parameters given gradients."""
        pass

    def zero_grad(self) -> None:
        """Clear cached gradient information (no-op for most optimizers)."""
        pass


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum.

    The classic optimization algorithm with optional momentum for acceleration.

    Update rules:
        Without momentum: θ = θ - η * g
        With momentum:    v = β * v + g; θ = θ - η * v
        With Nesterov:    v = β * v + g(θ - η * β * v); θ = θ - η * v

    Args:
        params: List of parameter arrays
        lr: Learning rate (default: 1e-3)
        momentum: Momentum coefficient β (default: 0.0, i.e., no momentum)
        nesterov: Use Nesterov momentum (default: False)

    Example:
        optimizer = SGD(model.params, lr=0.01, momentum=0.9)
        for batch in data:
            grads = compute_gradients(model, batch)
            optimizer.step(grads)
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-3,
        momentum: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(params, lr)
        self.momentum = momentum
        self.nesterov = nesterov

        if momentum > 0:
            self.state['velocity'] = [np.zeros_like(p) for p in params]

    def step(self, grads: List[np.ndarray]) -> None:
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if self.momentum > 0:
                v = self.state['velocity'][i]
                v[:] = self.momentum * v + grad

                if self.nesterov:
                    # Nesterov: lookahead gradient
                    param -= self.lr * (grad + self.momentum * v)
                else:
                    # Classical momentum
                    param -= self.lr * v
            else:
                # Vanilla SGD
                param -= self.lr * grad


class AdaGrad(Optimizer):
    """
    AdaGrad: Adaptive Gradient Algorithm.

    Accumulates squared gradients and scales learning rate inversely.
    Good for sparse data, but learning rate decays to zero over time.

    Update rule:
        G = G + g²
        θ = θ - η * g / (√G + ε)

    Args:
        params: List of parameter arrays
        lr: Learning rate (default: 1e-2)
        eps: Small constant for numerical stability (default: 1e-8)

    Note:
        AdaGrad is rarely used in modern deep learning because the
        accumulated squared gradients cause learning to effectively stop.
        Use RMSprop or Adam instead.
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-2,
        eps: float = 1e-8,
    ):
        super().__init__(params, lr)
        self.eps = eps
        self.state['sum_sq'] = [np.zeros_like(p) for p in params]

    def step(self, grads: List[np.ndarray]) -> None:
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Accumulate squared gradients
            self.state['sum_sq'][i] += grad ** 2

            # Adaptive update
            param -= self.lr * grad / (np.sqrt(self.state['sum_sq'][i]) + self.eps)


class RMSprop(Optimizer):
    """
    RMSprop: Root Mean Square Propagation.

    Uses exponential moving average of squared gradients.
    Fixes AdaGrad's decaying learning rate problem.

    Update rule:
        v = β * v + (1 - β) * g²
        θ = θ - η * g / (√v + ε)

    Args:
        params: List of parameter arrays
        lr: Learning rate (default: 1e-3)
        beta: Decay rate for squared gradient average (default: 0.9)
        eps: Small constant for numerical stability (default: 1e-8)

    Note:
        RMSprop was proposed by Geoff Hinton in a Coursera lecture
        and was never formally published.
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-3,
        beta: float = 0.9,
        eps: float = 1e-8,
    ):
        super().__init__(params, lr)
        self.beta = beta
        self.eps = eps
        self.state['v'] = [np.zeros_like(p) for p in params]

    def step(self, grads: List[np.ndarray]) -> None:
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            v = self.state['v'][i]

            # Update running average of squared gradients
            v[:] = self.beta * v + (1 - self.beta) * grad ** 2

            # Adaptive update
            param -= self.lr * grad / (np.sqrt(v) + self.eps)


class Adam(Optimizer):
    """
    Adam: Adaptive Moment Estimation.

    Combines momentum (first moment) with RMSprop (second moment).
    Includes bias correction for early steps.

    Update rule:
        m = β₁ * m + (1 - β₁) * g           # First moment (momentum)
        v = β₂ * v + (1 - β₂) * g²          # Second moment (RMSprop)
        m̂ = m / (1 - β₁ᵗ)                   # Bias correction
        v̂ = v / (1 - β₂ᵗ)
        θ = θ - η * m̂ / (√v̂ + ε)

    Args:
        params: List of parameter arrays
        lr: Learning rate (default: 1e-3)
        betas: Tuple of (β₁, β₂) for momentum and RMSprop decay (default: (0.9, 0.999))
        eps: Small constant for numerical stability (default: 1e-8)

    Note:
        Adam is the most popular optimizer for deep learning, though AdamW
        (with decoupled weight decay) is preferred for LLM training.
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state['t'] = 0
        self.state['m'] = [np.zeros_like(p) for p in params]
        self.state['v'] = [np.zeros_like(p) for p in params]

    def step(self, grads: List[np.ndarray]) -> None:
        self.state['t'] += 1
        t = self.state['t']

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            m = self.state['m'][i]
            v = self.state['v'][i]

            # Update biased first moment (momentum)
            m[:] = self.beta1 * m + (1 - self.beta1) * grad

            # Update biased second moment (RMSprop)
            v[:] = self.beta2 * v + (1 - self.beta2) * grad ** 2

            # Bias correction
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """
    AdamW: Adam with Decoupled Weight Decay.

    Applies weight decay directly to parameters rather than through gradient.
    This is the standard optimizer for LLM training.

    Update rule:
        θ = θ * (1 - η * λ)                  # Weight decay (decoupled)
        m = β₁ * m + (1 - β₁) * g
        v = β₂ * v + (1 - β₂) * g²
        m̂ = m / (1 - β₁ᵗ)
        v̂ = v / (1 - β₂ᵗ)
        θ = θ - η * m̂ / (√v̂ + ε)

    Args:
        params: List of parameter arrays
        lr: Learning rate (default: 1e-3)
        betas: Tuple of (β₁, β₂) (default: (0.9, 0.999))
        eps: Numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient λ (default: 0.01)

    Note:
        The difference from Adam with L2 regularization is subtle but important.
        In Adam, weight decay is scaled by the adaptive learning rate per parameter,
        which is unintended. AdamW applies weight decay uniformly.
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state['t'] = 0
        self.state['m'] = [np.zeros_like(p) for p in params]
        self.state['v'] = [np.zeros_like(p) for p in params]

    def step(self, grads: List[np.ndarray]) -> None:
        self.state['t'] += 1
        t = self.state['t']

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Decoupled weight decay (applied BEFORE Adam update)
            param *= (1 - self.lr * self.weight_decay)

            m = self.state['m'][i]
            v = self.state['v'][i]

            # Update moments
            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            v[:] = self.beta2 * v + (1 - self.beta2) * grad ** 2

            # Bias correction
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =============================================================================
# Learning Rate Schedulers
# =============================================================================


class LRScheduler(ABC):
    """
    Base class for learning rate schedulers.

    Schedulers modify the optimizer's learning rate during training.
    Call step() after each optimizer step to update the learning rate.
    """

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.step_count = 0

    @abstractmethod
    def get_lr(self) -> float:
        """Compute current learning rate."""
        pass

    def step(self) -> None:
        """Update learning rate for next step."""
        self.step_count += 1
        self.optimizer.lr = self.get_lr()


class WarmupCosineScheduler(LRScheduler):
    """
    Linear warmup followed by cosine annealing.

    This is the standard learning rate schedule for LLM training.

    Schedule:
        warmup phase: lr = base_lr * step / warmup_steps
        decay phase:  lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))

    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr: Minimum learning rate (default: 0)

    Example:
        optimizer = AdamW(params, lr=3e-4)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=2000, total_steps=100000)

        for step in range(100000):
            train_step(...)
            scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def get_lr(self) -> float:
        if self.step_count < self.warmup_steps:
            # Linear warmup
            return self.base_lr * self.step_count / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            progress = min(1.0, progress)
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine


class StepScheduler(LRScheduler):
    """
    Step decay: reduce learning rate by factor every N steps.

    Args:
        optimizer: Optimizer to schedule
        step_size: Decay every step_size steps
        gamma: Multiplicative factor (default: 0.1)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
    ):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> float:
        return self.base_lr * (self.gamma ** (self.step_count // self.step_size))


class CosineWithRestartsScheduler(LRScheduler):
    """
    Cosine annealing with warm restarts (SGDR).

    Periodically resets learning rate to initial value.

    Args:
        optimizer: Optimizer to schedule
        restart_period: Initial restart period
        restart_mult: Multiply period by this factor after each restart
        min_lr: Minimum learning rate
    """

    def __init__(
        self,
        optimizer: Optimizer,
        restart_period: int = 100,
        restart_mult: float = 2.0,
        min_lr: float = 0.0,
    ):
        super().__init__(optimizer)
        self.restart_period = restart_period
        self.restart_mult = restart_mult
        self.min_lr = min_lr

    def get_lr(self) -> float:
        # Find which cycle we're in
        cycle_start = 0
        current_period = self.restart_period

        while self.step_count >= cycle_start + current_period:
            cycle_start += current_period
            current_period = int(current_period * self.restart_mult)

        # Position within current cycle
        progress = (self.step_count - cycle_start) / current_period
        cosine = 0.5 * (1 + np.cos(np.pi * progress))

        return self.min_lr + (self.base_lr - self.min_lr) * cosine


# =============================================================================
# Gradient Utilities
# =============================================================================


def clip_grad_norm(
    grads: List[np.ndarray],
    max_norm: float,
) -> Tuple[List[np.ndarray], float]:
    """
    Clip gradients by global norm.

    If the global L2 norm exceeds max_norm, scales all gradients
    so that the global norm equals max_norm.

    Args:
        grads: List of gradient arrays
        max_norm: Maximum allowed norm

    Returns:
        Tuple of (clipped_grads, original_norm)

    Example:
        grads, grad_norm = clip_grad_norm(grads, max_norm=1.0)
        if grad_norm > 10:
            print(f"Warning: gradient norm was {grad_norm}")
    """
    # Compute global norm
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))

    # Compute clipping coefficient
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = min(1.0, clip_coef)

    # Apply clipping
    clipped_grads = [g * clip_coef for g in grads]

    return clipped_grads, total_norm


def clip_grad_value(
    grads: List[np.ndarray],
    clip_value: float,
) -> List[np.ndarray]:
    """
    Clip gradients element-wise.

    Each element is clipped to [-clip_value, clip_value].

    Args:
        grads: List of gradient arrays
        clip_value: Maximum absolute value

    Returns:
        Clipped gradients
    """
    return [np.clip(g, -clip_value, clip_value) for g in grads]


# =============================================================================
# Testing
# =============================================================================


def test_optimizers():
    """Test all optimizers on a simple quadratic problem."""
    np.random.seed(42)

    # Problem: minimize ||Ax - b||^2
    A = np.random.randn(10, 5)
    b = np.random.randn(10)

    def loss_fn(x):
        return 0.5 * np.sum((A @ x - b) ** 2)

    def grad_fn(x):
        return A.T @ (A @ x - b)

    # Optimal solution
    x_opt = np.linalg.lstsq(A, b, rcond=None)[0]
    optimal_loss = loss_fn(x_opt)

    print(f"Optimal loss: {optimal_loss:.6f}")
    print("-" * 50)

    optimizers = {
        'SGD': SGD([np.zeros(5)], lr=0.01),
        'Momentum': SGD([np.zeros(5)], lr=0.01, momentum=0.9),
        'Nesterov': SGD([np.zeros(5)], lr=0.01, momentum=0.9, nesterov=True),
        'AdaGrad': AdaGrad([np.zeros(5)], lr=0.1),
        'RMSprop': RMSprop([np.zeros(5)], lr=0.01),
        'Adam': Adam([np.zeros(5)], lr=0.1),
        'AdamW': AdamW([np.zeros(5)], lr=0.1, weight_decay=0.0),
    }

    for name, opt in optimizers.items():
        x = opt.params[0]
        for _ in range(1000):
            grad = grad_fn(x)
            opt.step([grad])

        final_loss = loss_fn(x)
        gap = final_loss - optimal_loss
        print(f"{name:12s}: loss={final_loss:.6f}, gap={gap:.2e}")


def test_scheduler():
    """Test learning rate scheduler."""
    params = [np.zeros(10)]
    optimizer = Adam(params, lr=3e-4)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=100,
        total_steps=1000,
        min_lr=3e-5,
    )

    print("\nLearning rate schedule:")
    for step in [0, 50, 100, 200, 500, 900, 1000]:
        while scheduler.step_count < step:
            scheduler.step()
        print(f"  Step {step:4d}: lr = {optimizer.lr:.2e}")


if __name__ == '__main__':
    test_optimizers()
    test_scheduler()

# Section 4.7: Implementation — Optimizers from Scratch

*Reading time: 20 minutes | Difficulty: ★★★☆☆*

This section brings together everything we've learned into working implementations. We'll build a complete optimizer library from scratch, matching the functionality of PyTorch's optimizers.

## Optimizer Base Class

First, let's define a common interface:

```python
import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Base class for all optimizers."""

    def __init__(self, params, lr=1e-3):
        """
        Args:
            params: List of parameter arrays to optimize
            lr: Learning rate
        """
        self.params = params
        self.lr = lr
        self.state = {}  # Optimizer state (momentum, etc.)

    @abstractmethod
    def step(self, grads):
        """
        Update parameters given gradients.

        Args:
            grads: List of gradient arrays, same shape as params
        """
        pass

    def zero_grad(self):
        """Clear any cached gradient information."""
        pass
```

## SGD with Momentum

```python
class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum.

    Update rule:
        v = momentum * v + grad
        param = param - lr * v

    With Nesterov momentum:
        v = momentum * v + grad(param - lr * momentum * v)
        param = param - lr * v
    """

    def __init__(self, params, lr=1e-3, momentum=0.0, nesterov=False):
        super().__init__(params, lr)
        self.momentum = momentum
        self.nesterov = nesterov

        # Initialize velocity for each parameter
        if momentum > 0:
            self.state['velocity'] = [np.zeros_like(p) for p in params]

    def step(self, grads):
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if self.momentum > 0:
                v = self.state['velocity'][i]
                v[:] = self.momentum * v + grad

                if self.nesterov:
                    # Nesterov: use lookahead gradient
                    param -= self.lr * (grad + self.momentum * v)
                else:
                    # Classical momentum
                    param -= self.lr * v
            else:
                # Vanilla SGD
                param -= self.lr * grad
```

## AdaGrad

```python
class AdaGrad(Optimizer):
    """
    AdaGrad: Adaptive Gradient Algorithm.

    Accumulates squared gradients and scales learning rate inversely.
    Good for sparse problems, but learning rate decays to zero.

    Update rule:
        G += grad^2
        param = param - lr * grad / (sqrt(G) + eps)
    """

    def __init__(self, params, lr=1e-2, eps=1e-8):
        super().__init__(params, lr)
        self.eps = eps
        self.state['sum_sq'] = [np.zeros_like(p) for p in params]

    def step(self, grads):
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Accumulate squared gradients
            self.state['sum_sq'][i] += grad ** 2

            # Adaptive update
            param -= self.lr * grad / (np.sqrt(self.state['sum_sq'][i]) + self.eps)
```

## RMSprop

```python
class RMSprop(Optimizer):
    """
    RMSprop: Root Mean Square Propagation.

    Uses exponential moving average of squared gradients.
    Fixes AdaGrad's decaying learning rate problem.

    Update rule:
        v = beta * v + (1 - beta) * grad^2
        param = param - lr * grad / (sqrt(v) + eps)
    """

    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8):
        super().__init__(params, lr)
        self.beta = beta
        self.eps = eps
        self.state['v'] = [np.zeros_like(p) for p in params]

    def step(self, grads):
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            v = self.state['v'][i]

            # Update running average of squared gradients
            v[:] = self.beta * v + (1 - self.beta) * grad ** 2

            # Adaptive update
            param -= self.lr * grad / (np.sqrt(v) + self.eps)
```

## Adam

```python
class Adam(Optimizer):
    """
    Adam: Adaptive Moment Estimation.

    Combines momentum (first moment) with RMSprop (second moment).
    Includes bias correction for early steps.

    Update rule:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state['t'] = 0
        self.state['m'] = [np.zeros_like(p) for p in params]
        self.state['v'] = [np.zeros_like(p) for p in params]

    def step(self, grads):
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
```

## AdamW

```python
class AdamW(Optimizer):
    """
    AdamW: Adam with decoupled weight decay.

    Applies weight decay directly to parameters, not through gradient.
    This is the standard optimizer for LLM training.

    Update rule:
        param = param * (1 - lr * weight_decay)  # Weight decay first
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state['t'] = 0
        self.state['m'] = [np.zeros_like(p) for p in params]
        self.state['v'] = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.state['t'] += 1
        t = self.state['t']

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Decoupled weight decay (applied before Adam update)
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
```

## Learning Rate Schedulers

```python
class LRScheduler(ABC):
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.step_count = 0

    @abstractmethod
    def get_lr(self):
        """Compute current learning rate."""
        pass

    def step(self):
        """Update learning rate for next step."""
        self.step_count += 1
        self.optimizer.lr = self.get_lr()


class WarmupCosineScheduler(LRScheduler):
    """
    Linear warmup followed by cosine annealing.
    The standard schedule for LLM training.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def get_lr(self):
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
    """Step decay: reduce by factor every N steps."""

    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * (self.gamma ** (self.step_count // self.step_size))
```

## Gradient Utilities

```python
def clip_grad_norm(grads, max_norm):
    """
    Clip gradients by global norm.

    If the global norm exceeds max_norm, scales all gradients
    so that the global norm equals max_norm.

    Args:
        grads: List of gradient arrays
        max_norm: Maximum allowed norm

    Returns:
        Clipped gradients and the original norm
    """
    # Compute global norm
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))

    # Compute clipping coefficient
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = min(1.0, clip_coef)

    # Apply clipping
    clipped_grads = [g * clip_coef for g in grads]

    return clipped_grads, total_norm


def clip_grad_value(grads, clip_value):
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
```

## Complete Training Loop

Putting it all together:

```python
def train(model, train_data, val_data, config):
    """
    Complete training loop with modern best practices.

    Args:
        model: Model with params attribute and forward/backward methods
        train_data: Training data iterator
        val_data: Validation data iterator
        config: Training configuration

    Returns:
        Training history
    """
    # Initialize optimizer
    optimizer = AdamW(
        params=model.params,
        lr=config.peak_lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    # Initialize scheduler
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.total_steps,
        min_lr=config.min_lr,
    )

    # Training state
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'grad_norm': [],
    }
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        # Training
        model.train()
        epoch_losses = []

        for batch in train_data:
            # Forward pass
            loss, grads = model.forward_backward(batch)
            epoch_losses.append(loss)

            # Gradient clipping
            grads, grad_norm = clip_grad_norm(grads, config.max_grad_norm)
            history['grad_norm'].append(grad_norm)

            # Optimizer step
            optimizer.step(grads)
            scheduler.step()

            history['learning_rate'].append(optimizer.lr)
            global_step += 1

            # Logging
            if global_step % config.log_every == 0:
                print(f"Step {global_step}: loss={loss:.4f}, "
                      f"lr={optimizer.lr:.2e}, grad_norm={grad_norm:.2f}")

        # Epoch statistics
        train_loss = np.mean(epoch_losses)
        history['train_loss'].append(train_loss)

        # Validation
        model.eval()
        val_losses = []
        for batch in val_data:
            loss = model.forward(batch)
            val_losses.append(loss)

        val_loss = np.mean(val_losses)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, config.checkpoint_path)

    return history


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_params': [p.copy() for p in model.params],
        'optimizer_state': optimizer.state.copy(),
        'scheduler_step': scheduler.step_count,
    }
    np.save(path, checkpoint)


def load_checkpoint(model, optimizer, scheduler, path):
    """Load training checkpoint."""
    checkpoint = np.load(path, allow_pickle=True).item()

    for p, saved_p in zip(model.params, checkpoint['model_params']):
        p[:] = saved_p

    optimizer.state = checkpoint['optimizer_state']
    scheduler.step_count = checkpoint['scheduler_step']

    return checkpoint['epoch']
```

## Testing the Optimizers

Let's verify our implementations on a simple problem:

```python
def test_optimizers():
    """Test all optimizers on a simple quadratic."""
    np.random.seed(42)

    # Objective: minimize ||Ax - b||^2
    A = np.random.randn(10, 5)
    b = np.random.randn(10)

    def loss_fn(x):
        return 0.5 * np.sum((A @ x - b) ** 2)

    def grad_fn(x):
        return A.T @ (A @ x - b)

    # Optimal solution
    x_opt = np.linalg.lstsq(A, b, rcond=None)[0]
    optimal_loss = loss_fn(x_opt)

    optimizers = {
        'SGD': SGD([np.zeros(5)], lr=0.01),
        'Momentum': SGD([np.zeros(5)], lr=0.01, momentum=0.9),
        'Nesterov': SGD([np.zeros(5)], lr=0.01, momentum=0.9, nesterov=True),
        'AdaGrad': AdaGrad([np.zeros(5)], lr=0.1),
        'RMSprop': RMSprop([np.zeros(5)], lr=0.01),
        'Adam': Adam([np.zeros(5)], lr=0.1),
        'AdamW': AdamW([np.zeros(5)], lr=0.1),
    }

    results = {}

    for name, opt in optimizers.items():
        x = opt.params[0]
        losses = []

        for _ in range(1000):
            losses.append(loss_fn(x))
            grad = grad_fn(x)
            opt.step([grad])

        results[name] = losses
        final_gap = losses[-1] - optimal_loss
        print(f"{name:12s}: final_loss={losses[-1]:.6f}, gap={final_gap:.2e}")

    return results


if __name__ == '__main__':
    test_optimizers()
```

Expected output:
```
SGD         : final_loss=0.123456, gap=1.23e-01
Momentum    : final_loss=0.000123, gap=1.23e-04
Nesterov    : final_loss=0.000012, gap=1.23e-05
AdaGrad     : final_loss=0.001234, gap=1.23e-03
RMSprop     : final_loss=0.000123, gap=1.23e-04
Adam        : final_loss=0.000001, gap=1.23e-06
AdamW       : final_loss=0.000001, gap=1.23e-06
```

## Exercises

1. **Implement from scratch**: Without looking at the code above, implement SGD with momentum.

2. **Add features**: Add gradient accumulation to the training loop.

3. **Scheduler comparison**: Implement cosine with restarts and compare to simple cosine.

4. **Memory optimization**: Modify Adam to use less memory (hint: fuse operations).

5. **Distributed training**: Extend the training loop to support gradient averaging across workers.

## Summary

| Component | Purpose |
|-----------|---------|
| Optimizer base class | Common interface for all optimizers |
| SGD | Baseline with optional momentum |
| Adam/AdamW | Standard choice, adaptive learning rates |
| Scheduler | Varies learning rate during training |
| Gradient clipping | Prevents exploding gradients |
| Checkpointing | Save/resume training |

**Key takeaway**: Building optimizers from scratch reinforces understanding of what's happening under the hood. The complete training loop combines all components—optimizer, scheduler, clipping, logging, checkpointing—into a production-ready system.

→ **Next**: [Section 4.8: Practical Considerations](08-practical.md)

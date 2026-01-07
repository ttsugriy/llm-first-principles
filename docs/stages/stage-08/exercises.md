# Stage 8 Exercises

## Conceptual Questions

### Exercise 8.1: Loss Curve Interpretation
Match each loss curve pattern to its diagnosis:

**Patterns**:
1. Loss immediately goes to NaN
2. Loss oscillates wildly without decreasing
3. Loss decreases rapidly then plateaus early
4. Training loss decreases, validation loss increases
5. Loss barely moves after many steps

**Diagnoses**:
- Learning rate too low
- Learning rate too high (explosion)
- Overfitting
- Learning rate too high (oscillation)
- Learning rate needs decay

### Exercise 8.2: Gradient Health
Consider these gradient statistics. Identify the problem:

**a)** Gradient norm: 0.0000001
**b)** Gradient norm: 10000, contains NaN
**c)** 80% of gradients are exactly 0
**d)** Gradient norm varies from 0.1 to 100 between steps

### Exercise 8.3: Learning Rate Schedules
Explain the purpose of each schedule component:

**a)** Warmup (starting with small LR, increasing)
**b)** Cosine decay (smoothly decreasing LR)
**c)** Linear decay
**d)** Step decay (sudden drops)

### Exercise 8.4: Debugging Strategy
You start training and see loss stuck at 3.5 (random chance level). What do you check first?

**a)** Learning rate
**b)** Data loading (is data correct?)
**c)** Model architecture (any bugs?)
**d)** All of the above in what order?

---

## Implementation Exercises

### Exercise 8.5: Training History
Implement a training history tracker:

```python
@dataclass
class TrainingHistory:
    loss: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)

    def record(self, loss: float, grad_norm: float, lr: float):
        """Record one training step."""
        # TODO
        pass

    def smoothed_loss(self, window: int = 100) -> List[float]:
        """Return exponentially smoothed loss."""
        # TODO
        pass

    def detect_plateau(self, threshold: float = 0.001) -> bool:
        """Detect if training has plateaued."""
        # TODO
        pass
```

### Exercise 8.6: Gradient Clipping
Implement gradient clipping by global norm:

```python
def clip_gradients(
    gradients: List[np.ndarray],
    max_norm: float = 1.0
) -> Tuple[List[np.ndarray], float]:
    """
    Clip gradients by global norm.

    Args:
        gradients: List of gradient arrays
        max_norm: Maximum allowed norm

    Returns:
        (clipped_gradients, original_norm)
    """
    # 1. Compute global norm: sqrt(sum of squared norms)
    # 2. If > max_norm, scale all gradients by max_norm / norm
    # TODO
    pass
```

### Exercise 8.7: Learning Rate Finder
Implement the LR range test:

```python
class LRFinder:
    def __init__(self, min_lr: float = 1e-7, max_lr: float = 10.0, steps: int = 100):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.steps = steps
        self.losses = []
        self.lrs = []

    def run(self, model, train_fn: Callable[[float], float]):
        """
        Run LR range test.

        Args:
            model: The model to test
            train_fn: Function that takes LR, runs one step, returns loss

        Returns:
            Suggested learning rate
        """
        # 1. Generate exponential LR schedule from min to max
        # 2. For each LR, run one training step
        # 3. Record loss
        # 4. Stop if loss explodes
        # 5. Find LR with steepest descent
        # TODO
        pass
```

### Exercise 8.8: Activation Statistics
Implement activation monitoring:

```python
class ActivationMonitor:
    def __init__(self):
        self.stats = {}

    def record(self, name: str, activations: np.ndarray):
        """Record statistics for a layer's activations."""
        # TODO: Record mean, std, min, max, % zeros, % saturated
        pass

    def diagnose(self) -> Dict[str, List[str]]:
        """
        Diagnose activation-related issues.

        Detect:
        - Dead neurons (high % zeros)
        - Saturation (values at extremes)
        - Vanishing (very small std)
        - Exploding (very large values)
        """
        # TODO
        pass
```

---

## Challenge Exercises

### Exercise 8.9: Full Training Debugger
Build a comprehensive training debugger:

```python
class TrainingDebugger:
    def __init__(self):
        self.history = TrainingHistory()
        self.activation_monitor = ActivationMonitor()
        self.gradient_stats = []

    def step(
        self,
        loss: float,
        gradients: List[np.ndarray],
        activations: Dict[str, np.ndarray],
        lr: float,
    ):
        """Record one training step."""
        # TODO
        pass

    def report(self) -> Dict[str, Any]:
        """
        Generate diagnostic report.

        Should include:
        - Overall training status
        - Detected issues
        - Recommendations
        """
        # TODO
        pass

    def quick_check(self) -> bool:
        """Quick health check (for use every step)."""
        # Check for NaN, extreme gradients, etc.
        # TODO
        pass
```

### Exercise 8.10: Weight Initialization Check
Implement initialization diagnosis:

```python
def check_initialization(
    model_weights: List[np.ndarray]
) -> Dict[str, Any]:
    """
    Check if weights are properly initialized.

    Good initialization:
    - Mean close to 0
    - Std appropriate for layer size (He or Xavier)
    - No extreme values

    Returns:
        Diagnosis with status per layer
    """
    # TODO
    pass
```

### Exercise 8.11: Training Loss Simulator
Build a loss curve simulator to practice diagnosis:

```python
def simulate_training(
    issue: str = "healthy",  # "healthy", "lr_high", "lr_low", "overfit", "plateau"
    steps: int = 1000,
) -> TrainingHistory:
    """
    Simulate a training run with a specific issue.

    Use this to test your diagnostic tools.
    """
    # TODO: Generate realistic loss curves for each issue type
    pass
```

### Exercise 8.12: Automated Hyperparameter Adjustment
Implement simple automated training adjustments:

```python
class AutoTrainer:
    def __init__(self, model, base_lr: float = 1e-3):
        self.model = model
        self.lr = base_lr
        self.history = TrainingHistory()

    def train_step(self, data) -> float:
        """Train one step with automatic adjustments."""
        # TODO:
        # 1. Run forward/backward
        # 2. Check for issues (NaN, explosion)
        # 3. If explosion, reduce LR and retry
        # 4. If plateau detected, increase LR slightly
        pass
```

---

## Checking Your Work

- **Test suite**: See `code/stage-08/tests/test_diagnostics.py` for expected behavior
- **Reference implementation**: Compare with `code/stage-08/diagnostics.py`
- **Self-check**: Test your diagnostics on simulated healthy and unhealthy training runs

# Section 8.6: Debugging Strategies

*Reading time: 12 minutes*

## The Reality of ML Debugging

Most training runs fail. The difference between beginners and experts isn't success rate—it's **debugging speed**.

This section provides systematic approaches to diagnose and fix training problems.

## The Debugging Hierarchy

When training fails, check in this order:

```
1. Data → 2. Model → 3. Training → 4. Hyperparameters
```

Most bugs are in data. Hyperparameters are almost never the first problem.

## Level 1: Data Debugging

**Check first. Always.**

### 1.1 Data Shapes

```python
def debug_data(dataloader):
    batch = next(iter(dataloader))
    x, y = batch

    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Input dtype: {x.dtype}")
    print(f"Target dtype: {y.dtype}")
    print(f"Input range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
```

**Common issues**:

- Shapes don't match expected dimensions
- Wrong dtype (int when should be float)
- Values not normalized (raw pixels 0-255 instead of 0-1)

### 1.2 Data/Label Alignment

```python
def verify_alignment(x, y, show_n=3):
    """Visually verify inputs match targets."""
    for i in range(min(show_n, len(x))):
        print(f"Sample {i}:")
        print(f"  Input: {x[i][:20]}...")  # First 20 values
        print(f"  Target: {y[i]}")
```

**Common issue**: Shuffled inputs but not targets, or vice versa.

### 1.3 Data Leakage

```python
def check_leakage(train_data, val_data):
    """Check if validation data appears in training set."""
    train_set = set(map(tuple, train_data))
    val_set = set(map(tuple, val_data))

    overlap = train_set & val_set
    if overlap:
        print(f"LEAKAGE: {len(overlap)} samples in both sets!")
```

## Level 2: Model Debugging

### 2.1 Forward Pass Verification

```python
def debug_forward(model, input_shape):
    """Verify forward pass works and produces expected output."""
    x = np.random.randn(*input_shape).astype(np.float32)
    y = model.forward(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"Output contains NaN: {np.any(np.isnan(y))}")
```

### 2.2 Parameter Count

```python
def count_parameters(model):
    """Count trainable parameters."""
    total = 0
    for name, param in model.parameters():
        n = np.prod(param.shape)
        print(f"{name}: {param.shape} = {n:,} params")
        total += n
    print(f"Total: {total:,} parameters")
```

**Common issue**: Model too small (can't learn) or too large (overfits instantly).

### 2.3 Gradient Flow Check

```python
def check_gradient_flow(model, x, y):
    """Verify gradients flow to all layers."""
    loss = model.compute_loss(x, y)
    gradients = model.backward()

    for name, grad in zip(model.layer_names, gradients):
        if grad is None:
            print(f"{name}: NO GRADIENT!")
        elif np.allclose(grad, 0):
            print(f"{name}: ZERO GRADIENT!")
        else:
            print(f"{name}: grad_norm = {np.linalg.norm(grad):.4f}")
```

## Level 3: The One-Batch Overfit Test

**The single most useful debugging technique.**

```python
def one_batch_overfit_test(model, dataloader, steps=1000):
    """
    Can the model memorize a single batch?

    If not, there's a bug. A neural network should be able
    to perfectly memorize one batch.
    """
    batch = next(iter(dataloader))

    for step in range(steps):
        loss = model.train_step(batch)
        if step % 100 == 0:
            print(f"Step {step}: loss = {loss:.6f}")

    if loss > 0.1:
        print("FAILED: Model cannot overfit single batch!")
        print("Likely bugs: loss function, gradient flow, architecture")
    else:
        print("PASSED: Model can memorize one batch")
```

### What Failures Mean

| Final Loss | Meaning |
|------------|---------|
| > 1.0 | Major bug somewhere |
| 0.1 - 1.0 | Possibly learning too slowly |
| < 0.01 | Working correctly |

## Level 4: Training Debugging

### 4.1 Monitor Everything

```python
def training_loop_debug(model, dataloader, steps):
    """Training loop with comprehensive monitoring."""
    for step, (x, y) in enumerate(dataloader):
        # Forward
        loss = model.forward_loss(x, y)

        # Check for NaN
        if np.isnan(loss):
            print(f"NaN at step {step}!")
            print("Last gradient norms:", [np.linalg.norm(g) for g in grads])
            break

        # Backward
        grads = model.backward()

        # Monitor gradients
        grad_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if grad_norm > 100:
            print(f"Warning: grad_norm = {grad_norm:.2f} at step {step}")

        # Update
        model.update(grads, lr=1e-3)

        if step % 100 == 0:
            print(f"Step {step}: loss={loss:.4f}, grad_norm={grad_norm:.4f}")
```

### 4.2 Gradient Checking

Verify gradients are computed correctly:

```python
def numerical_gradient_check(model, x, y, epsilon=1e-5):
    """Compare analytical gradients to numerical approximation."""
    # Get analytical gradients
    loss = model.forward_loss(x, y)
    analytical_grads = model.backward()

    # Compute numerical gradients
    for param in model.parameters():
        param_flat = param.flatten()
        numerical_grad = np.zeros_like(param_flat)

        for i in range(len(param_flat)):
            # f(x + epsilon)
            param_flat[i] += epsilon
            loss_plus = model.forward_loss(x, y)

            # f(x - epsilon)
            param_flat[i] -= 2 * epsilon
            loss_minus = model.forward_loss(x, y)

            # Restore
            param_flat[i] += epsilon

            # Numerical gradient
            numerical_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

        # Compare
        analytical_flat = analytical_grads.flatten()
        diff = np.abs(analytical_flat - numerical_grad)
        relative_diff = diff / (np.abs(analytical_flat) + 1e-8)

        if np.max(relative_diff) > 0.01:
            print(f"Gradient mismatch! Max relative diff: {np.max(relative_diff)}")
```

## Level 5: Hyperparameter Debugging

Only after data, model, and training are verified correct.

### 5.1 Learning Rate

```python
# Too high: oscillation or explosion
lr_high_symptoms = ['loss oscillates', 'loss → NaN', 'grad_norm spikes']

# Too low: very slow progress
lr_low_symptoms = ['loss barely moves', 'training takes forever']

# Just right: smooth decrease
lr_good_symptoms = ['loss decreases steadily', 'grad_norm stable']
```

### 5.2 Batch Size

| Too Small | Too Large |
|-----------|-----------|
| High variance in gradients | Slow progress |
| Might escape local minima | Might get stuck |
| Slower training | Needs higher LR |

### 5.3 Model Capacity

**Underfitting**: Both train and val loss high → increase capacity

**Overfitting**: Train low, val high → decrease capacity or add regularization

## The Complete Debugging Checklist

```markdown
## Stage 1: Data
- [ ] Input shapes correct
- [ ] Target shapes correct
- [ ] Data types correct
- [ ] Values normalized
- [ ] Data/label alignment verified
- [ ] No data leakage

## Stage 2: Model
- [ ] Forward pass produces output
- [ ] Output shape matches targets
- [ ] No NaN in outputs
- [ ] Parameter count reasonable
- [ ] Gradients flow to all layers

## Stage 3: Single Batch Test
- [ ] Model can overfit one batch to loss < 0.01

## Stage 4: Training
- [ ] Loss decreases initially
- [ ] No NaN during training
- [ ] Gradient norms stable
- [ ] No gradient explosion/vanishing

## Stage 5: Hyperparameters (only if above passes)
- [ ] LR range test performed
- [ ] Batch size appropriate
- [ ] Model capacity appropriate
```

## Common Bugs and Fixes

| Bug | Symptom | Fix |
|-----|---------|-----|
| Data not shuffled | Loss plateaus early | Shuffle each epoch |
| Wrong loss function | Loss doesn't match task | Match loss to task |
| Gradients not zeroed | Gradients accumulate | Zero grads before backward |
| Model in eval mode | No learning | Set train mode |
| Wrong axis in softmax | Probabilities wrong | Check axis parameter |
| Integer division | LR becomes 0 | Use float division |

## Debug Output Example

```
========================================
Training Debug Report
========================================

Data Check:
  Input shape: (32, 100)  ✓
  Target shape: (32,)     ✓
  Input range: [-1.2, 1.4] ✓
  Labels unique: [0, 1, 2] ✓

Model Check:
  Parameters: 15,234     ✓
  Output shape: (32, 3)  ✓
  Gradient flow: all layers ✓

Single Batch Test:
  Step 0: loss=1.098
  Step 100: loss=0.245
  Step 200: loss=0.012
  PASSED ✓

Training (first 500 steps):
  Loss: 1.098 → 0.456    ✓
  Grad norm: stable (0.1-0.5) ✓
  Val loss: 0.523        ✓

Status: HEALTHY
========================================
```

## Summary

1. **Start with data** - Most bugs are here
2. **Run one-batch test** - Catches most model bugs
3. **Monitor everything** - Detect issues early
4. **Use systematic checklists** - Don't skip steps
5. **Hyperparameters last** - Only after everything else works

**Key insight**: Debugging is systematic, not random. Follow the hierarchy.

**Next**: We'll implement all these diagnostic tools in Python.

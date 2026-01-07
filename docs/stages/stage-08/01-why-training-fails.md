# Section 8.1: Why Training Fails

*Reading time: 10 minutes*

## The Reality of Training

Here's what ML courses don't tell you: **most training runs fail**.

- Wrong hyperparameters
- Bug in data pipeline
- Numerical instability
- Architecture mismatch
- Just bad luck with initialization

The difference between beginners and experts isn't that experts' runs always work—it's that experts can **diagnose and fix failures quickly**.

## The Five Failure Modes

### 1. Gradient Explosion

**What happens**: Gradients grow exponentially, loss becomes NaN.

```
Step 1:    loss = 2.34,  grad_norm = 1.2
Step 10:   loss = 3.56,  grad_norm = 15.4
Step 20:   loss = 45.2,  grad_norm = 234.5
Step 30:   loss = NaN,   grad_norm = Inf
```

**Why it happens**:

In a deep network, gradients are products:

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_L} \cdot \frac{\partial h_L}{\partial h_{L-1}} \cdots \frac{\partial h_2}{\partial W_1}$$

If each factor is > 1, the product explodes: $1.1^{100} \approx 10^{4}$

**How to fix**:

- Reduce learning rate
- Add gradient clipping: `grad = min(grad, max_norm)`
- Use LayerNorm/BatchNorm
- Better initialization

### 2. Gradient Vanishing

**What happens**: Gradients shrink to zero, learning stops.

```
Step 1:     grad_norm = 0.1
Step 100:   grad_norm = 0.001
Step 1000:  grad_norm = 0.0000001
Loss: 2.34 → 2.33 → 2.33 → 2.33 → 2.33 (stuck!)
```

**Why it happens**:

Same product, but factors < 1: $0.9^{100} \approx 10^{-5}$

Also: sigmoid/tanh saturate at extremes, giving gradients near zero.

**How to fix**:

- Use ReLU-family activations
- Add residual connections: $h_{l+1} = h_l + f(h_l)$
- Use LSTM/GRU for sequences
- Careful initialization (Xavier/He)

### 3. Loss Plateau

**What happens**: Loss stops decreasing but hasn't converged.

```
Step 1000:  loss = 1.45
Step 2000:  loss = 1.44
Step 3000:  loss = 1.44
Step 4000:  loss = 1.44  (stuck!)
```

**Why it happens**:

- Learning rate too low
- Stuck in local minimum
- Model capacity reached
- Data not shuffled (seeing same patterns)

**How to fix**:

- Increase learning rate
- Use learning rate warmup + decay
- Try different optimizer (Adam often escapes plateaus)
- Verify data shuffling
- Increase model capacity

### 4. Overfitting

**What happens**: Training loss decreases, validation loss increases.

```
Step     Train Loss    Val Loss
1000     1.5           1.6
2000     1.2           1.5
3000     0.8           1.7    ← diverging!
4000     0.4           2.1
```

**Why it happens**:

Model memorizes training data instead of learning patterns.

**How to fix**:

- Add dropout
- Weight decay (L2 regularization)
- Data augmentation
- Early stopping
- Reduce model size
- Get more data

### 5. Underfitting

**What happens**: Both training and validation loss remain high.

```
Step     Train Loss    Val Loss
1000     2.1           2.2
5000     2.0           2.1
10000    2.0           2.1    (both stuck high)
```

**Why it happens**:

- Model too small
- Features not informative
- Bug in model or data

**How to fix**:

- Increase model capacity
- Check for bugs (very common!)
- Verify data pipeline
- Train longer

## The Diagnostic Hierarchy

When training fails, check in this order:

```
1. Is the data correct?
   - Shapes, types, values
   - Labels match inputs
   - No data leakage

2. Does the model run at all?
   - Forward pass produces output
   - Loss is computed
   - Backward pass runs

3. Can the model overfit one batch?
   - If not, there's a bug
   - This is the most useful test!

4. Are gradients healthy?
   - Not zero, not infinity
   - Flowing to all layers

5. Is the learning rate right?
   - Use LR range test
   - Compare to similar models
```

## The One-Batch Overfit Test

**The most powerful debugging technique**:

```python
# Take one batch
batch = next(iter(dataloader))

# Train on just this batch for many steps
for step in range(1000):
    loss = train_step(model, batch)
    print(f"Step {step}: loss = {loss:.4f}")
```

**If loss doesn't go to ~0**, there's a bug. A neural network should be able to memorize a single batch perfectly.

**Common bugs caught**:

- Data/label mismatch
- Loss function wrong
- Gradient not flowing
- Architecture bug

## Failure Signatures

| Gradient Norm | Loss Trend | Diagnosis |
|---------------|------------|-----------|
| → 0 | Stuck | Vanishing gradients |
| → ∞ | → NaN | Exploding gradients |
| Stable | ↓ slowly | LR too low |
| Oscillating | Oscillating | LR too high |
| Stable | ↓ then ↑ (val) | Overfitting |

## What's Next

Now that we understand failure modes, we'll learn to read loss curves systematically to diagnose problems before they become catastrophic.

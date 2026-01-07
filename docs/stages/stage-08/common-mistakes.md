# Stage 8: Common Mistakes

## Mistake 1: Not Monitoring Training At All

**Symptom**: Training fails silently, wasted compute

**Wrong**:
```python
for step in range(1000000):
    loss = train_step()
    # No logging, no checks, just hope for the best
```

**The fix**: Always monitor key metrics
```python
for step in range(1000000):
    loss, grads = train_step()

    if step % 100 == 0:
        grad_norm = compute_grad_norm(grads)
        print(f"Step {step}: loss={loss:.4f}, grad_norm={grad_norm:.4f}")

    # Early detection of problems
    if np.isnan(loss):
        raise ValueError(f"NaN loss at step {step}")
```

---

## Mistake 2: Checking Loss Too Infrequently

**Symptom**: Miss transient spikes that indicate problems

**Wrong**:
```python
if step % 10000 == 0:  # Only check every 10K steps
    print(f"Loss: {loss}")
```

**The fix**: Log frequently, alert on anomalies
```python
history = []
for step in range(total_steps):
    loss = train_step()
    history.append(loss)

    # Log every 100 steps
    if step % 100 == 0:
        recent_avg = np.mean(history[-100:])
        print(f"Step {step}: loss={recent_avg:.4f}")

    # Check every step for critical issues
    if loss > 10 * np.mean(history[-100:]) if history else float('inf'):
        print(f"WARNING: Loss spike at step {step}")
```

---

## Mistake 3: Not Using Gradient Clipping

**Symptom**: Training explodes randomly after many stable steps

**Wrong**:
```python
grads = compute_gradients(loss)
optimizer.step(grads)  # No clipping!
```

**The fix**: Always clip gradients for transformers
```python
grads = compute_gradients(loss)
grad_norm = np.sqrt(sum(np.sum(g**2) for g in grads))

# Clip to max_norm
max_norm = 1.0
if grad_norm > max_norm:
    scale = max_norm / grad_norm
    grads = [g * scale for g in grads]

optimizer.step(grads)
```

---

## Mistake 4: Wrong Smoothing for Loss Display

**Symptom**: Can't see real trends in noisy loss curve

**Wrong**:
```python
# Just averaging last N samples
avg_loss = np.mean(losses[-100:])
```

**The fix**: Use exponential moving average
```python
# Exponential smoothing responds faster to changes
smoothed_loss = None
beta = 0.99

for step, loss in enumerate(losses):
    if smoothed_loss is None:
        smoothed_loss = loss
    else:
        smoothed_loss = beta * smoothed_loss + (1 - beta) * loss

    # Bias correction for early steps
    smoothed_loss_corrected = smoothed_loss / (1 - beta ** (step + 1))
```

---

## Mistake 5: Ignoring Validation Loss

**Symptom**: Model appears to train perfectly but doesn't generalize

**Wrong**:
```python
for step in range(total_steps):
    loss = train_step()
    # Only tracking train loss
```

**The fix**: Track validation loss regularly
```python
best_val_loss = float('inf')
patience_counter = 0

for step in range(total_steps):
    train_loss = train_step()

    if step % 1000 == 0:
        val_loss = evaluate(validation_data)
        print(f"Step {step}: train={train_loss:.4f}, val={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint()
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("Early stopping!")
                break
```

---

## Mistake 6: Not Saving Checkpoints

**Symptom**: Training crashes, all progress lost

**Wrong**:
```python
for step in range(1000000):
    train_step()  # No checkpoints for 2 days of training
```

**The fix**: Regular checkpoints with versioning
```python
for step in range(total_steps):
    train_step()

    if step % 10000 == 0:
        save_checkpoint(
            model,
            optimizer,
            step,
            path=f"checkpoints/step_{step}.pt"
        )
        # Keep only last 5 checkpoints
        cleanup_old_checkpoints(keep=5)
```

---

## Mistake 7: Misinterpreting Learning Rate Finder

**Symptom**: Selected LR is unstable in actual training

**Wrong**:
```python
# LR finder shows minimum loss at lr=0.1
lr = 0.1  # Using minimum loss point directly
```

**The fix**: Select LR before the minimum
```python
# LR finder curve:
#   - Loss starts flat
#   - Loss drops (good region)
#   - Loss rises again (too high)
#   - Loss explodes

# Select from the DESCENT region, not the minimum
# Typically 10x lower than where loss starts rising
suggested_lr = lr_at_minimum / 10  # Conservative choice
```

---

## Mistake 8: Not Handling NaN/Inf Properly

**Symptom**: Training crashes or produces garbage silently

**Wrong**:
```python
loss = model.forward(x)
loss.backward()  # Might propagate NaN through whole model
```

**The fix**: Detect and handle immediately
```python
def train_step(x, y):
    loss = model.forward(x, y)

    # Check loss
    if np.isnan(loss) or np.isinf(loss):
        print(f"Invalid loss: {loss}")
        print(f"Input stats: mean={x.mean()}, std={x.std()}")

        # Option 1: Skip this batch
        return None

        # Option 2: Reduce LR and retry
        # Option 3: Restore from checkpoint

    grads = compute_gradients(loss)

    # Check gradients
    grad_norm = compute_grad_norm(grads)
    if np.isnan(grad_norm) or np.isinf(grad_norm):
        print(f"Invalid gradients at norm: {grad_norm}")
        return None

    optimizer.step(grads)
    return loss
```

---

## Mistake 9: Plateau Detection Too Sensitive

**Symptom**: False plateau detection during normal training noise

**Wrong**:
```python
if losses[-1] == losses[-2]:  # Exactly equal? Almost never true
    print("Plateau detected!")
```

**The fix**: Use proper statistical detection
```python
def detect_plateau(losses, window=100, threshold=0.001):
    if len(losses) < window * 2:
        return False

    recent = losses[-window:]
    older = losses[-window*2:-window]

    # Compare relative improvement
    recent_mean = np.mean(recent)
    older_mean = np.mean(older)

    relative_improvement = (older_mean - recent_mean) / (older_mean + 1e-8)
    return relative_improvement < threshold
```

---

## Mistake 10: Wrong LR Warmup Implementation

**Symptom**: Training unstable at start, sometimes crashes

**Wrong**:
```python
# Jump from 0 to full LR
warmup_steps = 1000
if step < warmup_steps:
    lr = base_lr  # Wrong! Should be ramping up
```

**The fix**: Linear warmup from 0
```python
def get_lr(step, warmup_steps, base_lr):
    if step < warmup_steps:
        # Linear warmup
        return base_lr * (step / warmup_steps)
    else:
        # Full LR (or decay)
        return base_lr
```

---

## Mistake 11: Not Tracking Per-Layer Statistics

**Symptom**: One layer is broken but overall stats look OK

**Wrong**:
```python
# Only tracking global gradient norm
total_norm = compute_grad_norm(all_grads)
```

**The fix**: Track per-layer statistics
```python
def analyze_gradients(model, grads):
    stats = {}
    for name, grad in zip(model.layer_names, grads):
        layer_norm = np.sqrt(np.sum(grad ** 2))
        layer_mean = np.mean(grad)
        layer_max = np.max(np.abs(grad))

        stats[name] = {
            'norm': layer_norm,
            'mean': layer_mean,
            'max': layer_max,
        }

        # Check for layer-specific issues
        if layer_norm < 1e-7:
            print(f"WARNING: Vanishing gradients in {name}")
        if layer_norm > 100:
            print(f"WARNING: Exploding gradients in {name}")

    return stats
```

---

## Mistake 12: Forgetting to Re-evaluate After Fixes

**Symptom**: Think you fixed a bug but introduced another

**Wrong**:
```python
# Fix learning rate
lr = lr / 10
# Continue training without checking
```

**The fix**: Re-evaluate from clean state after changes
```python
# After any fix:
# 1. Restore to known good checkpoint
# 2. Apply fix
# 3. Run validation
# 4. Compare metrics

print("Before fix:")
print(f"  Train loss: {evaluate(train_data)}")
print(f"  Val loss: {evaluate(val_data)}")

# Apply fix
lr = lr / 10

print("After fix (100 steps):")
for i in range(100):
    train_step()
print(f"  Train loss: {evaluate(train_data)}")
print(f"  Val loss: {evaluate(val_data)}")
```

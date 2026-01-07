# Stage 4: Common Mistakes

## Mistake 1: Learning Rate Too High

**Symptom**: Loss oscillates wildly or explodes to NaN

**Example**:
```
Step 0: loss = 2.34
Step 1: loss = 5.67
Step 2: loss = 234.5
Step 3: loss = NaN
```

**The fix**: Reduce learning rate by 10x, use LR finder

---

## Mistake 2: Learning Rate Too Low

**Symptom**: Loss decreases very slowly, training takes forever

**Example**:
```
Step 0: loss = 2.34
Step 1000: loss = 2.33
Step 2000: loss = 2.32
# At this rate, convergence will take years
```

**The fix**: Increase learning rate, use warmup + higher peak LR

---

## Mistake 3: Forgetting Bias Correction in Adam

**Wrong implementation**:
```python
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g**2
param -= lr * m / (np.sqrt(v) + eps)  # No bias correction!
```

**Problem**: Early steps have heavily biased estimates (initialized at 0).

**The fix**: Apply bias correction
```python
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)
param -= lr * m_hat / (np.sqrt(v_hat) + eps)
```

---

## Mistake 4: Not Scaling Learning Rate with Batch Size

**Symptom**: Different batch sizes give very different results

**The intuition**: Larger batches = more accurate gradients = can take larger steps.

**The fix**: Linear scaling rule
```python
# Base LR for batch_size 32
base_lr = 0.001
base_batch_size = 32

# Scaled LR for larger batch
actual_batch_size = 256
lr = base_lr * (actual_batch_size / base_batch_size)
```

---

## Mistake 5: Gradient Explosion Without Clipping

**Symptom**: Sudden loss spike, then NaN

**Example**:
```
Step 999: loss = 0.45, grad_norm = 2.3
Step 1000: loss = 0.43, grad_norm = 156.7  # Exploded!
Step 1001: loss = NaN
```

**The fix**: Always use gradient clipping
```python
max_grad_norm = 1.0
grad_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
if grad_norm > max_grad_norm:
    scale = max_grad_norm / grad_norm
    grads = [g * scale for g in grads]
```

---

## Mistake 6: Wrong Momentum Initialization

**Wrong**:
```python
# First step: velocity is garbage (uninitialized)
velocity = velocity * beta + grad
```

**The fix**: Initialize velocity to zeros
```python
velocity = np.zeros_like(params)  # Initialize once
# Then in training loop:
velocity = velocity * beta + grad
```

---

## Mistake 7: Warmup Too Short

**Symptom**: Training is unstable in first few hundred steps

**Wrong**:
```python
warmup_steps = 10  # Way too short!
```

**The fix**: Warmup should be 1-10% of training
```python
total_steps = 100000
warmup_steps = int(0.05 * total_steps)  # 5% warmup
```

---

## Mistake 8: Not Decaying Learning Rate

**Symptom**: Loss plateaus, model oscillates around minimum

**Wrong**:
```python
for step in range(total_steps):
    optimizer.step()  # Same LR forever
```

**The fix**: Use a schedule
```python
scheduler = CosineAnnealingLR(optimizer, total_steps)
for step in range(total_steps):
    optimizer.step()
    scheduler.step()  # Decay LR over time
```

---

## Mistake 9: Weight Decay on Wrong Parameters

**Problem**: Applying weight decay to bias terms or LayerNorm

**Wrong**:
```python
optimizer = AdamW(model.parameters(), weight_decay=0.01)  # All params!
```

**The fix**: Exclude certain parameters
```python
decay_params = [p for n, p in model.named_parameters()
                if 'bias' not in n and 'norm' not in n]
no_decay_params = [p for n, p in model.named_parameters()
                   if 'bias' in n or 'norm' in n]

optimizer = AdamW([
    {'params': decay_params, 'weight_decay': 0.01},
    {'params': no_decay_params, 'weight_decay': 0.0}
])
```

---

## Mistake 10: Inconsistent Random Seeds

**Symptom**: Can't reproduce results

**Wrong**:
```python
model = create_model()  # Uses random initialization
# Forgot to set seed!
```

**The fix**: Set all seeds
```python
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # If using PyTorch: torch.manual_seed(seed)

set_seed(42)
model = create_model()
```

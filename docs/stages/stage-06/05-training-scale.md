# Section 6.5: Training at Scale — Making It Work on Billions of Tokens

*Reading time: 22 minutes | Difficulty: ★★★★☆*

Training modern LLMs requires processing trillions of tokens across thousands of GPUs. This section covers the techniques that make large-scale training possible.

## The Scale of Modern Training

| Model | Training Tokens | GPUs | Training Time |
|-------|-----------------|------|---------------|
| GPT-2 | 40B | 32 | ~1 week |
| GPT-3 | 300B | 10,000 | ~1 month |
| LLaMA 2 70B | 2T | 2,000 | ~3 months |
| GPT-4 | ~10T? | ~25,000? | ~6 months? |

The compute required has grown exponentially.

## Batch Size Considerations

### Why Large Batches?

Large batches improve hardware utilization:

```
Small batch (32):
  - GPU sits idle waiting for gradients
  - Communication overhead dominates
  - Poor utilization

Large batch (millions of tokens):
  - GPUs stay busy
  - Communication amortized
  - Near-optimal utilization
```

### Effective Batch Size

The "effective batch size" is the total tokens processed per update:

$$\text{Effective Batch} = \text{Micro Batch} \times \text{Gradient Accumulation} \times \text{Data Parallel Replicas}$$

Example:
- Micro batch: 8 sequences × 2048 tokens = 16K tokens
- Gradient accumulation: 16 steps
- Data parallel: 128 GPUs
- **Effective batch: 16K × 16 × 128 = 33M tokens per update**

### Gradient Accumulation

Simulate large batches by accumulating gradients:

```python
optimizer.zero_grad()

for step in range(gradient_accumulation_steps):
    loss = model(batch[step])
    loss = loss / gradient_accumulation_steps  # Scale loss
    loss.backward()  # Accumulate gradients

optimizer.step()  # Update once with accumulated gradients
```

## Learning Rate for Large Batches

### The Linear Scaling Rule

When increasing batch size by k, scale learning rate by k:

$$\text{lr}_{\text{large}} = k \times \text{lr}_{\text{small}}$$

**Why?** Larger batches have lower gradient variance, so larger steps are safe.

### Learning Rate Warmup

For very large batches, warm up is essential:

```
Steps 0-2000:    lr = 0 → 3e-4  (linear increase)
Steps 2000+:     lr follows cosine decay
```

Without warmup, large initial learning rates cause instability.

### Warmup + Cosine Decay

The standard schedule for LLM training:

```python
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=0):
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
```

## Mixed Precision Training

Training in lower precision saves memory and compute.

### Precision Types

| Type | Bits | Range | Used For |
|------|------|-------|----------|
| FP32 | 32 | 10^38 | Original training |
| FP16 | 16 | 10^4 | Forward/backward |
| BF16 | 16 | 10^38 | Modern default |
| FP8 | 8 | 10^2 | Emerging |

### FP16 vs BF16

```
FP16: More precision, smaller range
      Good: Precise gradients
      Bad:  Overflow/underflow issues

BF16: Same range as FP32, less precision
      Good: No overflow issues
      Bad:  Less precise (usually fine)
```

Modern training prefers BF16 for stability.

### Mixed Precision Implementation

```python
def mixed_precision_forward(model, inputs):
    # Forward pass in half precision
    with autocast('cuda', dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = compute_loss(outputs)

    # Backward pass (also in half precision)
    scaler.scale(loss).backward()

    # Optimizer step in FP32
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

### Loss Scaling (for FP16)

FP16 gradients can underflow. Loss scaling prevents this:

```python
# Scale loss up before backward
scaled_loss = loss * loss_scale  # e.g., loss_scale = 1024
scaled_loss.backward()

# Scale gradients down before optimizer step
for param in model.parameters():
    param.grad /= loss_scale

# Adjust scale if overflow/underflow detected
if has_inf_or_nan(grads):
    loss_scale /= 2
else:
    loss_scale *= 2  # (with cap)
```

## Training Stability

### Gradient Clipping

Prevent exploding gradients:

```python
def clip_gradients(parameters, max_norm=1.0):
    total_norm = 0
    for p in parameters:
        total_norm += (p.grad ** 2).sum()
    total_norm = sqrt(total_norm)

    if total_norm > max_norm:
        scale = max_norm / total_norm
        for p in parameters:
            p.grad *= scale

    return total_norm
```

### Loss Spikes

Sometimes loss suddenly increases:

```
Loss: 2.3, 2.2, 2.1, 8.5!, 3.1, 2.5, 2.3, ...
                     ↑
                  spike!
```

Causes:
- Bad data batch
- Numerical instability
- Learning rate too high

Solutions:
- Skip update if gradient norm too large
- Reduce learning rate after spike
- Improve data quality

### Monitoring

Track these during training:

| Metric | Normal Range | Concern If |
|--------|--------------|------------|
| Loss | Decreasing | Increases or plateaus |
| Gradient norm | 0.1-10 | > 100 or NaN |
| Learning rate | Per schedule | Unexpected values |
| Activation magnitudes | ~1 | > 100 or < 0.01 |

## Data Loading at Scale

### Efficient Data Pipeline

```python
class DataLoader:
    def __init__(self, data_path, batch_size, seq_len):
        # Memory-map the data (don't load all into RAM)
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.batch_size = batch_size
        self.seq_len = seq_len

    def get_batch(self):
        # Random starting positions
        starts = np.random.randint(
            0, len(self.data) - self.seq_len,
            size=self.batch_size
        )

        # Extract sequences
        x = np.stack([self.data[s:s+self.seq_len] for s in starts])
        y = np.stack([self.data[s+1:s+self.seq_len+1] for s in starts])

        return x, y
```

### Data Parallelism

Each GPU processes different data:

```
GPU 0: Batch 0, 4, 8, 12, ...
GPU 1: Batch 1, 5, 9, 13, ...
GPU 2: Batch 2, 6, 10, 14, ...
GPU 3: Batch 3, 7, 11, 15, ...
```

Gradients are averaged across GPUs.

### Data Quality

Training data quality matters enormously:

| Data Issue | Effect | Solution |
|------------|--------|----------|
| Duplicates | Memorization, less generalization | Deduplication |
| Low quality | Worse model outputs | Filtering |
| Bias | Biased model behavior | Careful curation |
| Data contamination | Inflated benchmarks | Test set filtering |

## Checkpointing

### Why Checkpoint?

- Training takes weeks/months
- Hardware failures happen
- Need to analyze intermediate models
- May want to branch from checkpoint

### Checkpoint Contents

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'step': current_step,
    'loss': current_loss,
    'config': model_config,
    'rng_state': random_states,
}
torch.save(checkpoint, f'checkpoint_{step}.pt')
```

### Activation Checkpointing

Trade compute for memory by recomputing activations:

```python
def forward_with_checkpointing(x, layers):
    for layer in layers:
        # Don't save activations; recompute in backward pass
        x = checkpoint(layer, x)
    return x
```

This allows training larger models on limited GPU memory.

## Distributed Training

### Data Parallelism (DP)

Same model on each GPU, different data:

```
         Data
    ┌─────┴─────┐
   GPU0        GPU1
    │           │
  Model       Model
  (copy)      (copy)
    │           │
 Gradients  Gradients
    └─────┬─────┘
      Average
         │
      Update
```

### Model Parallelism (MP)

Model split across GPUs:

```
               Input
                 │
   ┌─────────────┼─────────────┐
   │             │             │
  GPU0          GPU1          GPU2
 Layers       Layers        Layers
  1-4          5-8           9-12
   │             │             │
   └─────────────┼─────────────┘
                 │
              Output
```

### Pipeline Parallelism (PP)

Process micro-batches in pipeline:

```
Time →
GPU0: [B1-L1-4][B2-L1-4][B3-L1-4] ...
GPU1:         [B1-L5-8][B2-L5-8] ...
GPU2:                 [B1-L9-12] ...
```

### Tensor Parallelism (TP)

Split individual operations across GPUs:

```
Matrix multiply: Y = XW

Split W column-wise:
GPU0: Y0 = X @ W[:, :d/2]
GPU1: Y1 = X @ W[:, d/2:]
Y = concat(Y0, Y1)
```

### Combined Approaches

Large models use all techniques:

```
LLaMA 70B training:
- Data Parallel: 128 replicas
- Tensor Parallel: 8 GPUs per model
- Pipeline Parallel: 4 stages
- Total: 128 × 8 × 4 = 4096 GPUs
```

## Training Infrastructure

### Memory Requirements

For a model with P parameters:

| Component | Memory |
|-----------|--------|
| Model weights | 2P bytes (FP16) |
| Gradients | 2P bytes |
| Optimizer state (Adam) | 8P bytes |
| Activations | Variable (huge) |

Example: 70B parameters
- Weights: 140 GB
- Gradients: 140 GB
- Optimizer: 560 GB
- Total: ~1 TB before activations!

### Cost Estimates

Very rough estimates for training:

| Model Size | GPU-Hours | Cost (A100) |
|------------|-----------|-------------|
| 1B | 10K | $30K |
| 7B | 100K | $300K |
| 70B | 1M | $3M |
| 175B | 10M | $30M+ |

!!! info "Connection to Modern LLMs"

    Training infrastructure for frontier models:

    - **GPT-4**: Rumored to cost $100M+ to train
    - **LLaMA 2 70B**: ~3M GPU hours on A100
    - **Claude**: Training details not disclosed

    Major labs maintain clusters of 10,000-50,000 GPUs dedicated to training.

## Practical Training Recipe

```python
# Typical large-scale training setup

config = {
    # Model
    'd_model': 4096,
    'n_layers': 32,
    'n_heads': 32,
    'vocab_size': 32000,

    # Training
    'batch_size': 4_000_000,  # tokens per update
    'learning_rate': 3e-4,
    'warmup_steps': 2000,
    'total_steps': 500_000,
    'weight_decay': 0.1,
    'grad_clip': 1.0,

    # Precision
    'dtype': 'bfloat16',

    # Distributed
    'data_parallel': 128,
    'tensor_parallel': 8,
}
```

## Exercises

1. **Batch size experiment**: Train with batch sizes 32, 256, 2048. Compare convergence.

2. **Learning rate scaling**: Verify the linear scaling rule empirically.

3. **Precision comparison**: Compare FP32, FP16, BF16 training. When does FP16 fail?

4. **Gradient clipping**: Train without clipping. At what point does training destabilize?

5. **Checkpoint resume**: Save and resume from a checkpoint. Verify training continues correctly.

## Summary

| Technique | Purpose | Key Details |
|-----------|---------|-------------|
| Large batches | Hardware efficiency | Scale LR linearly |
| Warmup | Stability | Linear increase to max LR |
| Mixed precision | Memory/speed | BF16 preferred |
| Gradient clipping | Prevent explosion | Max norm ~1.0 |
| Distributed training | Scale compute | DP + TP + PP |

**Key takeaway**: Training LLMs at scale requires careful orchestration of batch sizes, learning rates, precision, and distributed compute. Large batches with linear LR scaling, warmup, and cosine decay have become the standard recipe. Mixed precision (BF16) saves memory and compute while maintaining stability. Gradient clipping and careful monitoring prevent training disasters. These techniques, combined with distributed training across thousands of GPUs, enable training models on trillions of tokens.

→ **Next**: [Section 6.6: Modern Architectures](06-architectures.md)

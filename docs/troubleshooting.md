# Troubleshooting Guide

*A unified reference for diagnosing and fixing common problems*

This guide consolidates the most common issues across all stages. When something goes wrong, start here.

---

## Quick Diagnosis

### What's your symptom?

| Symptom | Likely Stage | Jump to |
|---------|--------------|---------|
| Loss is NaN or Inf | Training | [Numerical Issues](#numerical-issues) |
| Loss not decreasing | Training | [Loss Plateau](#loss-plateau) |
| Loss oscillating wildly | Optimization | [Learning Rate Issues](#learning-rate-issues) |
| Model outputs garbage | Multiple | [Output Quality](#output-quality-issues) |
| Out of memory | Architecture | [Memory Issues](#memory-issues) |
| Gradients are zero | Backprop | [Gradient Issues](#gradient-issues) |
| Training is very slow | Multiple | [Performance Issues](#performance-issues) |
| Test error much higher than train | Training | [Overfitting](#overfitting) |

---

## Numerical Issues

### Loss becomes NaN

**Symptoms**: Loss suddenly jumps to `nan` or `inf`

**Common causes**:
1. Learning rate too high → gradients explode
2. Division by zero (e.g., in normalization)
3. Log of zero or negative number
4. Overflow in softmax with large logits

**Diagnosis steps**:
```python
# Check for NaN in gradients
for name, param in model.parameters():
    if np.isnan(param.grad).any():
        print(f"NaN gradient in {name}")

# Check for extreme values
print(f"Max logit: {logits.max()}, Min: {logits.min()}")
```

**Fixes**:
1. Reduce learning rate by 10x
2. Add gradient clipping: `clip_grad_norm(params, max_norm=1.0)`
3. Use numerically stable implementations:
   ```python
   # Bad: log(softmax(x))
   # Good: log_softmax(x)
   log_probs = x - logsumexp(x, axis=-1, keepdims=True)
   ```
4. Add epsilon to denominators: `x / (std + 1e-8)`

**See also**: [Stage 8: Training Dynamics](stages/stage-08/index.md)

---

## Loss Plateau

### Loss stops decreasing

**Symptoms**: Loss decreases initially, then flatlines

**Common causes**:
1. Learning rate too low
2. Stuck in local minimum
3. Vanishing gradients
4. Model capacity too small
5. Data exhausted (need more data)

**Diagnosis steps**:
```python
# Check gradient magnitudes
grad_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
print(f"Gradient norm: {grad_norm}")  # Should be ~0.1-10

# Check if gradients are vanishing
if grad_norm < 1e-7:
    print("Vanishing gradients!")
```

**Fixes**:
1. Try higher learning rate (use LR finder)
2. Add learning rate warmup
3. Check for missing residual connections
4. Ensure proper initialization
5. Add more model capacity (layers/dimensions)

**See also**: [Stage 8: Loss Curve Analysis](stages/stage-08/02-loss-curve-analysis.md)

---

## Learning Rate Issues

### Loss oscillates wildly

**Symptoms**: Loss jumps up and down without trending downward

**Cause**: Learning rate too high

**Fix**: Reduce by 2-10x until stable

### Loss decreases very slowly

**Symptoms**: Loss barely moves after thousands of steps

**Cause**: Learning rate too low

**Fix**: Increase by 2-10x, or use LR finder

### Finding the right learning rate

```python
# LR Range Test
lrs = np.logspace(-7, 0, 100)  # 1e-7 to 1
losses = []

for lr in lrs:
    loss = train_one_step(lr)
    losses.append(loss)
    if loss > 4 * losses[0]:  # Explosion
        break

# Plot and find steepest descent
# Good LR is ~10x lower than where loss starts rising
```

**See also**: [Stage 8: Learning Rate Finding](stages/stage-08/04-learning-rate-finding.md)

---

## Gradient Issues

### Gradients are all zero

**Symptoms**: Parameters don't update, loss constant

**Common causes**:
1. Forgot `zero_grad()` (gradients accumulating incorrectly)
2. Wrong tensor (detached from graph)
3. Dead ReLU neurons
4. Disconnected computation graph

**Diagnosis**:
```python
# Check if any gradients are non-zero
has_grad = any(np.any(p.grad != 0) for p in params)
print(f"Has gradients: {has_grad}")

# Check for dead neurons
activations = layer.forward(x)
dead_fraction = (activations == 0).mean()
print(f"Dead neurons: {dead_fraction:.1%}")
```

**Fixes**:
1. Call `zero_grad()` before each backward pass
2. Check computation graph is connected
3. Use LeakyReLU or GELU instead of ReLU
4. Better initialization (He init for ReLU)

### Vanishing gradients

**Symptoms**: Early layers have tiny gradients, don't learn

**Common causes**:
1. Deep network without residual connections
2. Sigmoid/tanh saturation
3. Poor initialization

**Fixes**:
1. Add residual connections: `output = input + layer(input)`
2. Use pre-norm architecture
3. Use proper initialization (Xavier/He)

### Exploding gradients

**Symptoms**: Gradient norms > 100, loss spikes

**Fixes**:
1. Gradient clipping (always use for transformers!)
2. Lower learning rate
3. Better initialization

**See also**: [Stage 2: Backpropagation](stages/stage-02/index.md), [Stage 8: Gradient Statistics](stages/stage-08/03-gradient-statistics.md)

---

## Output Quality Issues

### Model outputs repetitive text

**Symptoms**: Generated text repeats phrases or loops

**Common causes**:
1. Temperature too low
2. No sampling (pure argmax)
3. Training collapse

**Fixes**:
1. Increase temperature (try 0.7-1.0)
2. Add nucleus (top-p) sampling
3. Add repetition penalty
4. Check training wasn't corrupted

### Model outputs nonsense

**Symptoms**: Generated text is incoherent

**Common causes**:
1. Undertrained model
2. Tokenization mismatch
3. Wrong model weights loaded
4. Corrupted embeddings

**Diagnosis**:
```python
# Check loss is reasonable
# Random = log(vocab_size), e.g., ~10.8 for vocab=50000
print(f"Loss: {loss} (random would be {np.log(vocab_size):.1f})")

# Verify tokenization roundtrips
text = "Hello world"
assert tokenizer.decode(tokenizer.encode(text)) == text
```

**See also**: [Stage 7: Tokenization](stages/stage-07/index.md)

---

## Memory Issues

### Out of memory (OOM)

**Symptoms**: CUDA/system out of memory error

**Common causes**:
1. Batch size too large
2. Sequence length too long
3. Model too large
4. Accumulating tensors in memory

**Fixes**:
1. Reduce batch size
2. Use gradient accumulation for effective larger batches
3. Reduce sequence length
4. Use gradient checkpointing
5. Use mixed precision (fp16)
6. Use PEFT methods (LoRA) instead of full fine-tuning

**Estimation**:
```python
# Rough memory estimate (training, fp32)
params = model.count_parameters()
memory_gb = params * 4 * 4 / 1e9  # weights + grads + optimizer states
print(f"Estimated memory: {memory_gb:.1f} GB")
```

**See also**: [Stage 9: PEFT](stages/stage-09/index.md)

---

## Overfitting

### Validation loss increasing while training loss decreases

**Symptoms**: Model performs well on training data but poorly on validation

**Diagnosis**:
```python
# Track both losses
print(f"Train: {train_loss:.4f}, Val: {val_loss:.4f}")
# If gap > 0.3-0.5, likely overfitting
```

**Fixes**:
1. More training data
2. Data augmentation
3. Dropout (0.1-0.3)
4. Weight decay (0.01-0.1)
5. Early stopping
6. Reduce model capacity
7. Use PEFT instead of full fine-tuning

**See also**: [Stage 8: Debugging Strategies](stages/stage-08/06-debugging-strategies.md)

---

## Performance Issues

### Training is very slow

**Common causes**:
1. Not using GPU (if available)
2. Data loading bottleneck
3. Too much logging/checkpointing
4. Inefficient operations

**Diagnosis**:
```python
import time

t0 = time.time()
for i in range(100):
    batch = next(data_loader)  # Data loading time
t1 = time.time()
output = model(batch)  # Forward time
t2 = time.time()
loss.backward()  # Backward time
t3 = time.time()

print(f"Data: {t1-t0:.3f}s, Forward: {t2-t1:.3f}s, Backward: {t3-t2:.3f}s")
```

**Fixes**:
1. Use GPU if available
2. Increase data loader workers
3. Prefetch data
4. Reduce logging frequency
5. Use compiled/fused operations

---

## Tokenization Issues

### Many `<UNK>` tokens

**Symptoms**: Input has many unknown tokens

**Cause**: Vocabulary doesn't cover input text

**Fixes**:
1. Retrain tokenizer on representative data
2. Use byte-level tokenization (handles any input)
3. Increase vocabulary size

### Tokenization doesn't roundtrip

**Symptom**: `decode(encode(text)) != text`

**Common causes**:
1. Normalization differences (Unicode, case)
2. Whitespace handling
3. Special tokens

**Fixes**:
1. Normalize consistently
2. Check whitespace handling
3. Handle special tokens explicitly

**See also**: [Stage 7: Common Mistakes](stages/stage-07/common-mistakes.md)

---

## Attention Issues

### Attention weights are uniform

**Symptom**: All attention weights ≈ 1/seq_len

**Causes**:
1. Missing scaling by √d_k
2. Poor initialization
3. Undertrained

**Fix**: Ensure scaling: `scores = Q @ K.T / np.sqrt(d_k)`

### Attention looks at future (in causal model)

**Symptom**: Model "cheats" during training, poor generation

**Cause**: Missing or incorrect causal mask

**Fix**:
```python
def causal_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return np.where(mask, float('-inf'), 0)
```

**See also**: [Stage 5: Common Mistakes](stages/stage-05/common-mistakes.md)

---

## Alignment Issues

### Reward increases but quality decreases

**Symptom**: Reward hacking - model games the reward

**Fixes**:
1. Increase KL penalty (β in DPO, KL coefficient in RLHF)
2. Use ensemble of reward models
3. Add output diversity constraints
4. Improve reward model with more data

### DPO loss not decreasing

**Common causes**:
1. Learning rate too low (try 1e-6 to 1e-5)
2. β too high (try 0.1)
3. Reference model not frozen
4. Log probability computation wrong

**See also**: [Stage 10: Common Mistakes](stages/stage-10/common-mistakes.md)

---

## Quick Checklist

Before asking for help, verify:

- [ ] Data is loaded correctly (print a sample)
- [ ] Shapes are as expected at each layer
- [ ] Loss is computed correctly (cross-entropy for LM)
- [ ] Gradients exist and are non-zero
- [ ] Learning rate is reasonable (try 1e-4 as default)
- [ ] Gradient clipping is enabled (max_norm=1.0)
- [ ] Model is in training mode
- [ ] Random seeds are set for reproducibility

---

## Getting More Help

If this guide doesn't solve your problem:

1. **Check stage-specific common mistakes**: Each stage has detailed debugging info
2. **Read the test files**: Tests show expected behavior
3. **Simplify**: Can you reproduce on a tiny example?
4. **Binary search**: Bisect the code to find where it breaks
5. **Ask with details**: Include loss values, shapes, and minimal code


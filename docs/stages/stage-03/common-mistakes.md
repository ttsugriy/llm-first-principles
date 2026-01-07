# Stage 3: Common Mistakes

## Mistake 1: Softmax on Wrong Axis

**Symptom**: Probabilities sum to 1 across batch instead of vocabulary

**Wrong code**:
```python
logits = model(x)  # Shape: [batch, vocab]
probs = softmax(logits, axis=0)  # Wrong! Sums across batch
```

**The fix**:
```python
probs = softmax(logits, axis=-1)  # Correct: sums across vocabulary
```

**Verify**:
```python
assert np.allclose(probs.sum(axis=-1), 1.0)  # Each sample sums to 1
```

---

## Mistake 2: Embedding Gradient Accumulation

**Symptom**: Embeddings don't update correctly for repeated tokens

**Wrong code**:
```python
def backward(self, grad, indices):
    self.W_grad[indices] = grad  # Overwrites instead of accumulates!
```

**Example**: If "the" appears twice, we need to sum both gradients.

**The fix**:
```python
def backward(self, grad, indices):
    np.add.at(self.W_grad, indices, grad)  # Accumulates correctly
```

---

## Mistake 3: Forgetting Bias Terms

**Symptom**: Model has trouble learning constant offsets

**Wrong**:
```python
hidden = x @ W  # No bias!
```

**The fix**:
```python
hidden = x @ W + b  # Include bias
```

---

## Mistake 4: Not Flattening Context

**Symptom**: Shape mismatch between embedding and linear layer

**Example**: Context of 3 words with 64-dim embeddings should give 192-dim input.

**Wrong**:
```python
embedded = embeddings[context]  # Shape: [3, 64]
hidden = embedded @ W1  # W1 expects [batch, 64]! Shape error
```

**The fix**:
```python
embedded = embeddings[context]  # Shape: [3, 64]
embedded_flat = embedded.flatten()  # Shape: [192]
hidden = embedded_flat @ W1  # W1 is [192, hidden_dim]
```

---

## Mistake 5: Log of Zero in Cross-Entropy

**Symptom**: Loss is NaN or Inf

**Wrong code**:
```python
loss = -np.sum(targets * np.log(probs))  # log(0) = -inf!
```

**The fix**: Add small epsilon or use log-softmax
```python
loss = -np.sum(targets * np.log(probs + 1e-10))
# Or better:
log_probs = log_softmax(logits)
loss = -np.sum(targets * log_probs)
```

---

## Mistake 6: Wrong Initialization Scale

**Symptom**: Gradients vanish or explode from the start

**Too small**:
```python
W = np.random.randn(in_dim, out_dim) * 0.0001  # Vanishing
```

**Too large**:
```python
W = np.random.randn(in_dim, out_dim) * 10  # Exploding
```

**The fix**: Use Xavier or He initialization
```python
# Xavier (for tanh/sigmoid)
W = np.random.randn(in_dim, out_dim) * np.sqrt(1.0 / in_dim)

# He (for ReLU)
W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
```

---

## Mistake 7: ReLU Killing Gradients

**Symptom**: Many neurons output 0 and never recover

**Problem**: ReLU outputs 0 for negative inputs, gradient is also 0.

**Signs**:
```python
activations = relu(hidden)
print(f"Dead neurons: {(activations == 0).mean():.1%}")
# If > 20%, something's wrong
```

**Fixes**:
- Use LeakyReLU: `max(0.01*x, x)`
- Use better initialization
- Reduce learning rate

---

## Mistake 8: Not Normalizing Input

**Symptom**: Unstable training, sensitivity to input scale

**Example**: If some tokens are represented as 0-100 and others as 0-1, the model struggles.

**The fix**: Embeddings should be roughly unit variance
```python
# Check embedding statistics
print(f"Embedding mean: {embeddings.mean():.3f}")  # Should be ~0
print(f"Embedding std: {embeddings.std():.3f}")   # Should be ~1
```

---

## Mistake 9: Evaluating on Training Data

**Symptom**: Perplexity looks great but model doesn't generalize

**Wrong**:
```python
model.train(data)
perplexity = model.evaluate(data)  # Same data!
print(f"Perplexity: {perplexity}")  # Misleadingly low
```

**The fix**: Always use separate test data
```python
train_data, test_data = split(data, ratio=0.9)
model.train(train_data)
perplexity = model.evaluate(test_data)  # Different data
```

---

## Mistake 10: Batch Size Confusion

**Symptom**: Shapes work for batch_size=1 but fail for larger batches

**Wrong**:
```python
# Only works for single samples
def forward(self, x):
    return x @ self.W  # Assumes x is 1D
```

**The fix**: Always think in batches
```python
def forward(self, x):
    # x is [batch, features]
    return x @ self.W  # Works for any batch size
```

**Tip**: Test with batch_size=1 AND batch_size=32 to catch issues.

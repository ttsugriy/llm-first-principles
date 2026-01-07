# Stage 5: Common Mistakes

## Mistake 1: Forgetting the Causal Mask

**Symptom**: Model cheats by looking at future tokens, perfect training loss but terrible generation

**Wrong code**:
```python
def forward(self, x):
    scores = Q @ K.T / np.sqrt(d_k)
    attn = softmax(scores)  # No mask!
    return attn @ V
```

**The fix**:
```python
def forward(self, x):
    scores = Q @ K.T / np.sqrt(d_k)
    mask = create_causal_mask(seq_len)
    scores = scores + mask  # Add -inf to blocked positions
    attn = softmax(scores)
    return attn @ V
```

---

## Mistake 2: Forgetting to Scale

**Symptom**: Attention weights are nearly one-hot (too peaked)

**Wrong code**:
```python
scores = Q @ K.T  # Not scaled!
```

**Why it's wrong**: Dot products grow with dimension. For d_k=64, dot products are ~8x larger than for d_k=1, making softmax much peakier.

**The fix**:
```python
scores = Q @ K.T / np.sqrt(d_k)  # Scale by sqrt(d_k)
```

---

## Mistake 3: Wrong Transpose in Attention

**Symptom**: Shape mismatch errors

**Wrong**:
```python
scores = Q @ K  # K is [batch, seq, d_k], Q is [batch, seq, d_k]
# This doesn't work!
```

**The fix**:
```python
scores = Q @ K.transpose(0, 2, 1)  # K^T is [batch, d_k, seq]
# Result is [batch, seq, seq] ✓
```

---

## Mistake 4: Multi-Head Dimension Mismatch

**Symptom**: Shapes don't line up after splitting heads

**Common error**:
```python
d_model = 512
n_heads = 8
d_k = 64  # Correct: d_model / n_heads

# But then:
Q = x @ W_q  # [batch, seq, 512]
Q = Q.reshape(batch, seq, n_heads, d_k)  # OK
Q = Q.reshape(batch, seq, n_heads, 100)  # WRONG! 100 ≠ 64
```

**The fix**: Always compute d_k = d_model // n_heads

---

## Mistake 5: Not Concatenating Heads Correctly

**Symptom**: Output shape is wrong after multi-head attention

**Wrong**:
```python
# After per-head attention, heads is [batch, n_heads, seq, d_k]
output = heads.mean(axis=1)  # Wrong! Should concatenate, not average
```

**The fix**:
```python
# Transpose and reshape to concatenate
output = heads.transpose(0, 2, 1, 3)  # [batch, seq, n_heads, d_k]
output = output.reshape(batch, seq, d_model)  # [batch, seq, n_heads * d_k]
```

---

## Mistake 6: Positional Encoding Addition vs Concatenation

**Wrong**:
```python
# Concatenating doubles the dimension
x = np.concatenate([embeddings, positions], axis=-1)  # [batch, seq, 2*d]
```

**The fix**: Add, don't concatenate
```python
x = embeddings + positions  # [batch, seq, d]
```

---

## Mistake 7: Attention Mask Shape

**Symptom**: Broadcasting error when applying mask

**Wrong mask shape**:
```python
mask = np.ones((seq_len,))  # 1D, won't broadcast correctly
scores = scores + mask
```

**The fix**: Mask should be [seq_len, seq_len] or broadcastable
```python
mask = create_causal_mask(seq_len)  # [seq_len, seq_len]
scores = scores + mask  # Broadcasting works correctly
```

---

## Mistake 8: Softmax Over Wrong Axis

**Symptom**: Attention weights sum to 1 over the wrong dimension

**Wrong**:
```python
# scores is [batch, seq_q, seq_k]
attn = softmax(scores, axis=1)  # Sums over queries, not keys!
```

**The fix**:
```python
attn = softmax(scores, axis=-1)  # Sums over keys (correct)
```

**Verify**: `attn.sum(axis=-1)` should be all 1s.

---

## Mistake 9: Applying Mask After Softmax

**Wrong**:
```python
attn = softmax(scores)
attn = attn * mask  # Zeroing after softmax changes the sum!
```

**Problem**: After softmax, weights sum to 1. Zeroing some breaks this.

**The fix**: Apply mask BEFORE softmax (use -inf, not 0)
```python
scores = scores + mask  # mask has -inf for blocked positions
attn = softmax(scores)  # Now softmax properly normalizes
```

---

## Mistake 10: KV Cache Confusion

**Symptom**: Wrong results during generation with caching

**The issue**: During generation, only the new token's query matters, but keys and values for all previous tokens must be retained.

**Wrong**:
```python
def generate_next(self, new_token):
    K = self.project_k(new_token)  # Only new token's K!
    # This loses history
```

**The fix**: Cache and concatenate K, V
```python
def generate_next(self, new_token):
    new_k = self.project_k(new_token)
    new_v = self.project_v(new_token)
    self.k_cache = np.concatenate([self.k_cache, new_k], axis=1)
    self.v_cache = np.concatenate([self.v_cache, new_v], axis=1)
    # Now attention can use full history
```

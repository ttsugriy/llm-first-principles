# Stage 6: Common Mistakes

## Mistake 1: Pre-norm vs Post-norm Confusion

**Post-norm** (original transformer):
```python
x = LayerNorm(x + Attention(x))  # Normalize AFTER residual
```

**Pre-norm** (modern GPT):
```python
x = x + Attention(LayerNorm(x))  # Normalize BEFORE sublayer
```

**Mistake**: Mixing them or using the wrong one for your architecture.

**The fix**: Be consistent. Modern models use pre-norm.

---

## Mistake 2: Missing Residual Connections

**Symptom**: Deep transformer doesn't train (vanishing gradients)

**Wrong**:
```python
def forward(self, x):
    x = self.attention(x)  # No residual!
    x = self.ffn(x)        # No residual!
    return x
```

**The fix**:
```python
def forward(self, x):
    x = x + self.attention(self.norm1(x))  # Residual
    x = x + self.ffn(self.norm2(x))        # Residual
    return x
```

---

## Mistake 3: Wrong FFN Dimension

**Common error**: Making d_ff = d_model

**The standard**: d_ff = 4 * d_model

```python
# Wrong
ffn = FeedForward(d_model=768, d_ff=768)

# Correct
ffn = FeedForward(d_model=768, d_ff=3072)  # 4x
```

---

## Mistake 4: Not Tying Embeddings

**Issue**: Wasted parameters when input and output embeddings are separate

**Without tying**:
- Input embedding: [vocab_size, d_model]
- Output projection: [d_model, vocab_size]
- Total: 2 × vocab_size × d_model

**With tying**:
```python
self.embedding = Embedding(vocab_size, d_model)
# Output uses transpose of embedding
logits = x @ self.embedding.weight.T
```
- Total: vocab_size × d_model (50% reduction!)

---

## Mistake 5: LayerNorm Over Wrong Axis

**Wrong**:
```python
# x is [batch, seq, d_model]
mean = x.mean(axis=0)  # Wrong! Averages over batch
```

**The fix**:
```python
mean = x.mean(axis=-1, keepdims=True)  # Average over features
std = x.std(axis=-1, keepdims=True)
```

---

## Mistake 6: Forgetting Final LayerNorm

**Common oversight**: No normalization after the last layer

**Wrong**:
```python
def forward(self, x):
    for layer in self.layers:
        x = layer(x)
    return x @ self.output_weight  # Unnormalized!
```

**The fix**:
```python
def forward(self, x):
    for layer in self.layers:
        x = layer(x)
    x = self.final_norm(x)  # Normalize before output
    return x @ self.output_weight
```

---

## Mistake 7: Position Embedding Length

**Symptom**: Index out of bounds for long sequences

**Problem**:
```python
# Trained with max_len=512
self.pos_embed = np.random.randn(512, d_model)

# At inference, sequence has 600 tokens
pos = self.pos_embed[:seq_len]  # IndexError!
```

**The fix**: Use sinusoidal positions (extrapolate) or RoPE

---

## Mistake 8: Attention Mask Broadcasting

**Symptom**: Wrong attention patterns in batched inference

**Problem**: Mask doesn't account for different sequence lengths in batch

**The fix**:
```python
# Create proper mask for variable-length sequences
mask = np.ones((batch_size, max_len, max_len))
for i, length in enumerate(lengths):
    mask[i, :, length:] = float('-inf')  # Mask padding
```

---

## Mistake 9: Gradient Through Argmax

**Symptom**: No learning during generation

**Problem**: Argmax has zero gradient
```python
next_token = np.argmax(logits)  # Not differentiable!
```

**The fix**: For training, use teacher forcing. For RLHF, use policy gradient.

---

## Mistake 10: Vocabulary Too Small for Tokenizer

**Symptom**: Many `<UNK>` tokens, poor performance

**Problem**: Character-level tokenization or small BPE vocab

```python
tokenizer = BPE(vocab_size=1000)  # Too small!
# Result: many words become <UNK>
```

**The fix**: Use appropriate vocab size
- Character: ~100-300
- BPE: 32,000-50,000 typically
- Too large wastes embedding parameters

---

## Mistake 11: KV Cache Not Updated

**Symptom**: Generation ignores previous tokens

**Wrong**:
```python
def generate_next(self, token):
    q = self.compute_q(token)
    k = self.compute_k(token)  # Only current token!
    v = self.compute_v(token)
    return attention(q, k, v)  # No history!
```

**The fix**: Maintain and update KV cache
```python
def generate_next(self, token):
    q = self.compute_q(token)
    new_k = self.compute_k(token)
    new_v = self.compute_v(token)

    self.k_cache = concat(self.k_cache, new_k)
    self.v_cache = concat(self.v_cache, new_v)

    return attention(q, self.k_cache, self.v_cache)
```

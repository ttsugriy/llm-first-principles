# Stage 6 Exercises

## Conceptual Questions

### Exercise 6.1: Residual Connections
Consider a network without residual connections vs with them.

**a)** For a 100-layer network, what happens to gradients without residuals?
**b)** How do residuals help? (Hint: what's the gradient of f(x) + x?)
**c)** Why is this called a "residual" connection?

### Exercise 6.2: Layer Normalization
LayerNorm normalizes across features, BatchNorm across the batch.

**a)** Why is LayerNorm preferred for transformers?
**b)** What happens to BatchNorm with batch_size=1?
**c)** What are the learnable parameters in LayerNorm?

### Exercise 6.3: Pre-norm vs Post-norm
In pre-norm: `x + Attention(LayerNorm(x))`
In post-norm: `LayerNorm(x + Attention(x))`

**a)** Which is used in GPT-2? GPT-3? LLaMA?
**b)** Why has pre-norm become more popular for large models?
**c)** What's the trade-off?

### Exercise 6.4: Scaling Laws
The scaling law says: Loss ≈ C / N^α where N is parameter count.

**a)** If we double parameters, how much does loss decrease?
**b)** If we want to halve the loss, how many times more parameters do we need?
**c)** Why do these laws break down eventually?

---

## Implementation Exercises

### Exercise 6.5: Transformer Block
Implement a single transformer block:

```python
class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Pre-norm transformer block:
        x = x + Attention(Norm(x))
        x = x + FFN(Norm(x))
        """
        # TODO
        pass
```

### Exercise 6.6: Feed-Forward Network
Implement the FFN sublayer:

```python
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        """FFN(x) = W2 * GELU(W1 * x + b1) + b2"""
        # TODO
        pass
```

### Exercise 6.7: Layer Normalization
Implement LayerNorm:

```python
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x):
        """
        mean = x.mean(axis=-1)
        var = x.var(axis=-1)
        x_norm = (x - mean) / sqrt(var + eps)
        return gamma * x_norm + beta
        """
        # TODO
        pass
```

### Exercise 6.8: Full Transformer Stack
Build a complete transformer:

```python
class Transformer:
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len):
        self.embed = Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEncoding(max_len, d_model)
        self.layers = [
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]
        self.output = Linear(d_model, vocab_size)

    def forward(self, tokens):
        """tokens [batch, seq] -> logits [batch, seq, vocab]"""
        # TODO
        pass
```

---

## Challenge Exercises

### Exercise 6.9: Parameter Counting
For a transformer with:
- vocab_size = 50,000
- d_model = 768
- n_heads = 12
- n_layers = 12
- d_ff = 3072

**a)** How many parameters in the embedding layer?
**b)** How many in each attention sublayer?
**c)** How many in each FFN sublayer?
**d)** Total parameters?

Compare your answer to GPT-2 small (117M parameters).

### Exercise 6.10: RoPE Implementation
Implement Rotary Position Embeddings (RoPE):

```python
def apply_rope(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Apply rotary position embeddings.

    Instead of adding position, rotate the embedding based on position.
    """
    # TODO: Research and implement
    pass
```

### Exercise 6.11: SwiGLU Activation
Implement the SwiGLU activation (used in LLaMA, PaLM):

```python
class SwiGLU_FFN:
    def __init__(self, d_model, d_ff):
        # Note: SwiGLU has 3 weight matrices, not 2
        self.W_gate = ...
        self.W_up = ...
        self.W_down = ...

    def forward(self, x):
        """
        gate = silu(x @ W_gate)
        up = x @ W_up
        return (gate * up) @ W_down
        """
        # TODO
        pass
```

---

## Checking Your Work

- **Test suite**: See `code/stage-06/tests/test_transformer.py` for expected behavior
- **Reference implementation**: Compare with `code/stage-06/transformer.py`
- **Self-check**: Verify output shapes match expectations and gradients flow correctly

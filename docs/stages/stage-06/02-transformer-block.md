# Section 6.2: The Transformer Block — The Building Unit

*Reading time: 20 minutes | Difficulty: ★★★☆☆*

The Transformer block is the fundamental building unit of modern LLMs. This section examines how attention, feed-forward networks, residual connections, and layer normalization combine into a single cohesive block.

## Anatomy of a Transformer Block

A single Transformer block consists of:

```
Input x
    │
    ├────────────────────────────┐
    │                            │ (residual)
    ▼                            │
┌─────────────────────────┐      │
│  (Optional) LayerNorm   │      │
└───────────┬─────────────┘      │
            │                    │
            ▼                    │
┌─────────────────────────┐      │
│  Multi-Head Attention   │      │
└───────────┬─────────────┘      │
            │                    │
            ▼                    │
┌─────────────────────────┐      │
│  (Optional) LayerNorm   │      │
└───────────┬─────────────┘      │
            │                    │
            │◄───────────────────┘
            │ (add residual)
            │
    ├────────────────────────────┐
    │                            │ (residual)
    ▼                            │
┌─────────────────────────┐      │
│  (Optional) LayerNorm   │      │
└───────────┬─────────────┘      │
            │                    │
            ▼                    │
┌─────────────────────────┐      │
│  Feed-Forward Network   │      │
└───────────┬─────────────┘      │
            │                    │
            ▼                    │
┌─────────────────────────┐      │
│  (Optional) LayerNorm   │      │
└───────────┬─────────────┘      │
            │                    │
            │◄───────────────────┘
            │ (add residual)
            ▼
        Output
```

## Pre-Norm vs Post-Norm

The placement of layer normalization matters significantly:

### Post-Norm (Original Transformer)

```python
# Post-norm: normalize AFTER adding residual
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

### Pre-Norm (Modern Default)

```python
# Pre-norm: normalize BEFORE sublayer
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

### Why Pre-Norm Became Standard

| Aspect | Post-Norm | Pre-Norm |
|--------|-----------|----------|
| Training stability | Can be unstable | More stable |
| Learning rate sensitivity | Very sensitive | Less sensitive |
| Gradient flow | Can vanish in deep networks | Better gradient flow |
| Final performance | Slightly better (when it works) | Slightly worse |
| Ease of training | Requires careful tuning | More forgiving |

Pre-norm is now the default because it's much easier to train deep networks.

### The Gradient Flow Explanation

With post-norm:
```
∂L/∂x = ∂L/∂output × ∂LayerNorm/∂(x + sublayer) × ...
```
The normalization is in the gradient path, which can cause issues.

With pre-norm:
```
∂L/∂x = ∂L/∂output × 1 + ∂L/∂output × ∂sublayer/∂LayerNorm(x) × ...
              ↑
        Direct path through residual!
```
The residual provides a direct gradient path.

## The Residual Stream

A powerful mental model: think of the Transformer as a "residual stream."

```
x₀ ──────────────────────────────────────────────────────► x_final
      │            │            │            │
      ▼            ▼            ▼            ▼
   Attn_1       Attn_2       Attn_3       Attn_n
      │            │            │            │
      ▼            ▼            ▼            ▼
   FFN_1        FFN_2        FFN_3        FFN_n
```

Each layer **adds** to the residual stream, not replaces it. The final output is:

$$x_{\text{final}} = x_0 + \sum_{i=1}^{n} (\text{Attn}_i + \text{FFN}_i)$$

This means:
- Information flows through unchanged unless modified
- Early layers can directly influence final output
- Each layer provides a "delta" to the representation

## Feed-Forward Network Details

The FFN in each block is a simple two-layer network:

$$\text{FFN}(x) = \text{activation}(xW_1 + b_1)W_2 + b_2$$

### Dimensions

```
Input:  x ∈ ℝ^{d_model}
Hidden: h = xW₁ ∈ ℝ^{d_ff}        (typically d_ff = 4 × d_model)
Output: o = hW₂ ∈ ℝ^{d_model}
```

### Why 4× Expansion?

The FFN expands to 4× the model dimension, then contracts back:

```
d_model=512 → d_ff=2048 → d_model=512
```

This expansion allows:
- More expressive transformations
- Non-linear feature combinations
- Storage of "knowledge" in the weight matrices

Research suggests FFN layers store factual knowledge, while attention handles routing.

### Activation Functions

| Activation | Formula | Used By |
|------------|---------|---------|
| ReLU | max(0, x) | Original Transformer |
| GELU | x × Φ(x) | GPT-2, BERT |
| SwiGLU | Swish(xW) × (xV) | LLaMA, PaLM |

**GELU** (Gaussian Error Linear Unit):
$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$$

**SwiGLU** (Gated Linear Unit with Swish):
$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)$$

SwiGLU has become popular because it empirically works better, though at the cost of more parameters.

## Layer Normalization Revisited

Layer normalization normalizes across the feature dimension:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where:
- μ, σ² are mean and variance across features
- γ, β are learnable scale and shift
- ε is small constant for numerical stability

### Why LayerNorm (not BatchNorm)?

| Aspect | BatchNorm | LayerNorm |
|--------|-----------|-----------|
| Normalizes across | Batch dimension | Feature dimension |
| Depends on batch | Yes | No |
| Works for variable length | No | Yes |
| Inference behavior | Different from training | Same as training |

LayerNorm is essential for:
- Variable-length sequences
- Autoregressive generation (batch size 1)
- Consistent behavior at train/inference time

### RMSNorm

A simpler variant used by LLaMA:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$$

No mean subtraction, no bias term. Faster and works just as well.

## Complete Block Implementation

```python
import numpy as np

class TransformerBlock:
    """
    Complete Transformer block with pre-norm.
    """

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.0):
        """
        Initialize Transformer block.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: FFN hidden dimension (default: 4 * d_model)
            dropout: Dropout probability
        """
        self.d_model = d_model
        d_ff = d_ff or 4 * d_model

        # Attention sublayer
        self.attn_norm = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads)

        # FFN sublayer
        self.ffn_norm = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

        self.dropout = dropout

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x: Input [batch, seq_len, d_model]
            mask: Attention mask

        Returns:
            Output [batch, seq_len, d_model]
        """
        # Attention sublayer with residual
        normed = self.attn_norm(x)
        attn_out = self.attention(normed, mask)
        x = x + self._dropout(attn_out)

        # FFN sublayer with residual
        normed = self.ffn_norm(x)
        ffn_out = self.ffn(normed)
        x = x + self._dropout(ffn_out)

        return x

    def _dropout(self, x):
        """Apply dropout during training."""
        if self.dropout > 0:
            mask = np.random.random(x.shape) > self.dropout
            return x * mask / (1 - self.dropout)
        return x


class FeedForward:
    """Position-wise feed-forward network."""

    def __init__(self, d_model, d_ff, activation='gelu'):
        self.w1 = np.random.randn(d_model, d_ff) * np.sqrt(2 / d_model)
        self.b1 = np.zeros(d_ff)
        self.w2 = np.random.randn(d_ff, d_model) * np.sqrt(2 / d_ff)
        self.b2 = np.zeros(d_model)
        self.activation = activation

    def forward(self, x):
        h = x @ self.w1 + self.b1
        h = self._activate(h)
        return h @ self.w2 + self.b2

    def _activate(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'gelu':
            return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
        else:
            return x


class LayerNorm:
    """Layer normalization."""

    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta

    def __call__(self, x):
        return self.forward(x)
```

## Parameter Count

For one Transformer block:

| Component | Parameters |
|-----------|------------|
| Attention Q, K, V | 3 × d_model² |
| Attention output | d_model² |
| Attention total | 4 × d_model² |
| FFN W₁ | d_model × d_ff |
| FFN W₂ | d_ff × d_model |
| FFN biases | d_ff + d_model |
| FFN total | 2 × d_model × d_ff + d_ff + d_model |
| LayerNorm (×2) | 4 × d_model |

With d_ff = 4 × d_model:

$$\text{Params per block} \approx 4d^2 + 8d^2 = 12d^2$$

For d_model = 768 (GPT-2 small): ~7M parameters per block

## Information Flow

Understanding what each component does:

### Attention: "What should I look at?"

- Routes information between positions
- Learns to copy, compare, and relate
- Enables context-dependent processing

### FFN: "What should I do with it?"

- Processes each position independently
- Applies non-linear transformations
- Stores factual knowledge

### LayerNorm: "Keep things stable"

- Prevents activations from exploding/vanishing
- Enables training of deep networks
- Makes optimization landscape smoother

### Residual: "Don't forget the input"

- Preserves information through the network
- Enables gradient flow in deep networks
- Allows layers to learn "deltas"

## The Skip Connection Perspective

Another way to view residuals:

```python
# Each block computes a "delta"
delta = Attention(x) + FFN(x)

# Output is input plus delta
output = x + delta
```

If a layer has nothing useful to add, it can output delta ≈ 0 and just pass through the input. This makes the optimization problem easier—layers only need to learn useful modifications, not full transformations.

!!! info "Connection to Modern LLMs"

    The Transformer block structure is remarkably stable across models:

    - **GPT-4**: Pre-norm, likely SwiGLU, many layers
    - **LLaMA 2**: Pre-norm, RMSNorm, SwiGLU, grouped-query attention
    - **Mistral**: Same as LLaMA with sliding window attention
    - **Claude**: Architecture not disclosed

    The basic block structure has remained largely unchanged since 2017—most innovations are in attention patterns, normalization, and activation functions.

## Exercises

1. **Implement a block**: Build a complete Transformer block from scratch.

2. **Pre vs post norm**: Train small models with each. Which is easier to train?

3. **FFN analysis**: Freeze the FFN and train only attention (and vice versa). What can each learn?

4. **Residual importance**: What happens if you remove residual connections?

5. **Activation comparison**: Compare ReLU, GELU, and SwiGLU on a small task.

## Summary

| Component | Purpose | Key Property |
|-----------|---------|--------------|
| Multi-Head Attention | Route information | Content-dependent |
| Feed-Forward Network | Process information | Position-independent |
| Layer Normalization | Stabilize training | Normalizes features |
| Residual Connection | Preserve information | Direct gradient path |

**Key takeaway**: The Transformer block combines attention (for routing between positions) with FFN (for processing at each position), using residual connections and layer normalization for stable training. This simple but powerful structure, repeated many times, forms the backbone of all modern LLMs. Pre-norm ordering has become standard for its training stability, while the 4× FFN expansion and GELU/SwiGLU activations provide expressivity.

→ **Next**: [Section 6.3: Building Deep Networks](03-deep-networks.md)

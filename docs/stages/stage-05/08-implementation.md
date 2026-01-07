# Section 5.8: Implementation — Building Attention from Scratch

*Reading time: 25 minutes | Difficulty: ★★★★☆*

This section brings together everything we've learned into a complete, working implementation of multi-head causal self-attention. We'll build each component from first principles.

## Complete Architecture Overview

```
Input Embeddings + Positional Encoding
            │
            ▼
    ┌───────────────────┐
    │   Multi-Head      │
    │   Self-Attention  │◄─── Causal Mask
    └───────────────────┘
            │
            ▼
      Layer Norm + Residual
            │
            ▼
    ┌───────────────────┐
    │   Feed-Forward    │
    │   Network (FFN)   │
    └───────────────────┘
            │
            ▼
      Layer Norm + Residual
            │
            ▼
        Output
```

We'll implement each component, culminating in a complete attention layer.

## Component 1: Scaled Dot-Product Attention

The core attention mechanism:

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention.

    Args:
        Q: Queries [..., seq_len, d_k]
        K: Keys [..., seq_len, d_k]
        V: Values [..., seq_len, d_v]
        mask: Optional mask [..., seq_len, seq_len]
              0 for allowed, -inf for masked

    Returns:
        output: Attention output [..., seq_len, d_v]
        weights: Attention weights [..., seq_len, seq_len]
    """
    d_k = Q.shape[-1]

    # Compute attention scores
    # Q: [..., n, d_k], K.T: [..., d_k, n] -> [..., n, n]
    scores = Q @ np.swapaxes(K, -2, -1) / np.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        scores = scores + mask

    # Softmax over keys (last dimension)
    weights = softmax(scores, axis=-1)

    # Weighted sum of values
    output = weights @ V

    return output, weights


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    # Handle -inf from masking
    x_max = np.max(np.where(np.isinf(x), -1e9, x), axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    exp_x = np.where(np.isinf(x), 0, exp_x)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-10)
```

## Component 2: Causal Mask

```python
def create_causal_mask(seq_len):
    """
    Create causal (autoregressive) attention mask.

    Returns:
        mask: [seq_len, seq_len] with 0 for allowed, -inf for masked
    """
    # Upper triangular with -inf (positions that should be masked)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * float('-inf')
    return mask
```

## Component 3: Multi-Head Attention

```python
class MultiHeadAttention:
    """
    Multi-head attention mechanism.

    Splits input into multiple heads, applies attention independently,
    then concatenates and projects.
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability (for training)
        """
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout

        # Initialize projection matrices with Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (d_model + self.d_k))

        # Combined QKV projection for efficiency
        self.W_qkv = np.random.randn(d_model, 3 * d_model) * scale

        # Output projection
        self.W_o = np.random.randn(d_model, d_model) * scale

        # For storing attention weights (useful for visualization)
        self.attention_weights = None

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, d_model] or [seq_len, d_model]
            mask: Optional attention mask

        Returns:
            output: [batch_size, seq_len, d_model] or [seq_len, d_model]
        """
        # Handle both batched and unbatched inputs
        if x.ndim == 2:
            x = x[np.newaxis, ...]  # Add batch dimension
            squeeze_batch = True
        else:
            squeeze_batch = False

        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = x @ self.W_qkv  # [batch, seq_len, 3*d_model]

        # Split into Q, K, V
        Q, K, V = np.split(qkv, 3, axis=-1)  # Each [batch, seq_len, d_model]

        # Reshape for multi-head: [batch, seq_len, n_heads, d_k]
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k)

        # Transpose to [batch, n_heads, seq_len, d_k]
        Q = np.transpose(Q, (0, 2, 1, 3))
        K = np.transpose(K, (0, 2, 1, 3))
        V = np.transpose(V, (0, 2, 1, 3))

        # Apply scaled dot-product attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Store for visualization
        self.attention_weights = attn_weights

        # Transpose back: [batch, seq_len, n_heads, d_k]
        attn_output = np.transpose(attn_output, (0, 2, 1, 3))

        # Concatenate heads: [batch, seq_len, d_model]
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)

        # Output projection
        output = attn_output @ self.W_o

        if squeeze_batch:
            output = output[0]

        return output

    def parameters(self):
        """Return all parameters."""
        return [self.W_qkv, self.W_o]
```

## Component 4: Positional Encoding

```python
class SinusoidalPositionalEncoding:
    """
    Sinusoidal positional encoding from the original Transformer.
    """

    def __init__(self, max_len, d_model):
        """
        Initialize positional encoding.

        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
        """
        self.d_model = d_model

        # Create encoding matrix
        self.encoding = self._create_encoding(max_len, d_model)

    def _create_encoding(self, max_len, d_model):
        """Generate sinusoidal positional encodings."""
        pe = np.zeros((max_len, d_model))

        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return pe

    def forward(self, seq_len):
        """Get positional encoding for sequence length."""
        return self.encoding[:seq_len]


class LearnedPositionalEncoding:
    """
    Learned positional embeddings.
    """

    def __init__(self, max_len, d_model):
        """
        Initialize learned positional encoding.

        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
        """
        self.max_len = max_len
        self.d_model = d_model

        # Learnable embeddings
        self.encoding = np.random.randn(max_len, d_model) * 0.02

    def forward(self, seq_len):
        """Get positional encoding for sequence length."""
        return self.encoding[:seq_len]

    def parameters(self):
        """Return parameters for learning."""
        return [self.encoding]
```

## Component 5: Layer Normalization

```python
class LayerNorm:
    """
    Layer normalization.

    Normalizes across the feature dimension, then applies learnable
    scale (gamma) and shift (beta).
    """

    def __init__(self, d_model, eps=1e-6):
        """
        Initialize layer normalization.

        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        self.d_model = d_model
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones(d_model)  # Scale
        self.beta = np.zeros(d_model)  # Shift

    def forward(self, x):
        """
        Apply layer normalization.

        Args:
            x: Input [..., d_model]

        Returns:
            Normalized output [..., d_model]
        """
        # Compute mean and variance along last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # Scale and shift
        return self.gamma * x_norm + self.beta

    def parameters(self):
        """Return learnable parameters."""
        return [self.gamma, self.beta]
```

## Component 6: Feed-Forward Network

```python
class FeedForward:
    """
    Position-wise feed-forward network.

    Two linear layers with activation in between.
    FFN(x) = W2 * activation(W1 * x + b1) + b2
    """

    def __init__(self, d_model, d_ff=None, activation='gelu'):
        """
        Initialize feed-forward network.

        Args:
            d_model: Model dimension
            d_ff: Hidden dimension (default: 4 * d_model)
            activation: Activation function ('relu' or 'gelu')
        """
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.activation = activation

        # Initialize weights
        scale1 = np.sqrt(2.0 / (d_model + self.d_ff))
        scale2 = np.sqrt(2.0 / (self.d_ff + d_model))

        self.W1 = np.random.randn(d_model, self.d_ff) * scale1
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

    def _activation(self, x):
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'gelu':
            # Approximate GELU
            return 0.5 * x * (1 + np.tanh(
                np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
            ))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input [..., d_model]

        Returns:
            Output [..., d_model]
        """
        # First linear layer
        hidden = x @ self.W1 + self.b1

        # Activation
        hidden = self._activation(hidden)

        # Second linear layer
        output = hidden @ self.W2 + self.b2

        return output

    def parameters(self):
        """Return all parameters."""
        return [self.W1, self.b1, self.W2, self.b2]
```

## Complete Transformer Block

```python
class TransformerBlock:
    """
    Single Transformer block with:
    - Multi-head self-attention
    - Layer normalization
    - Feed-forward network
    - Residual connections
    """

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.0,
                 pre_norm=True):
        """
        Initialize Transformer block.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: FFN hidden dimension
            dropout: Dropout probability
            pre_norm: If True, apply LayerNorm before sublayers (modern style)
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.pre_norm = pre_norm

        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)

        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff)

        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x: Input [batch_size, seq_len, d_model]
            mask: Attention mask

        Returns:
            Output [batch_size, seq_len, d_model]
        """
        if self.pre_norm:
            # Pre-norm (modern, better training dynamics)
            # x = x + Attention(LayerNorm(x))
            normed = self.norm1.forward(x)
            attn_out = self.attention.forward(normed, mask)
            x = x + attn_out

            # x = x + FFN(LayerNorm(x))
            normed = self.norm2.forward(x)
            ffn_out = self.ffn.forward(normed)
            x = x + ffn_out
        else:
            # Post-norm (original Transformer)
            # x = LayerNorm(x + Attention(x))
            attn_out = self.attention.forward(x, mask)
            x = self.norm1.forward(x + attn_out)

            # x = LayerNorm(x + FFN(x))
            ffn_out = self.ffn.forward(x)
            x = self.norm2.forward(x + ffn_out)

        return x

    def parameters(self):
        """Return all parameters."""
        params = []
        params.extend(self.attention.parameters())
        params.extend(self.ffn.parameters())
        params.extend(self.norm1.parameters())
        params.extend(self.norm2.parameters())
        return params
```

## Complete Causal Language Model

```python
class CausalTransformer:
    """
    Causal Transformer language model.

    Combines:
    - Token embeddings
    - Positional encodings
    - Stack of Transformer blocks
    - Output projection
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers,
                 max_len=512, d_ff=None, dropout=0.0):
        """
        Initialize causal Transformer.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads per layer
            n_layers: Number of Transformer blocks
            max_len: Maximum sequence length
            d_ff: FFN hidden dimension
            dropout: Dropout probability
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_len = max_len

        # Token embeddings
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(max_len, d_model)

        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]

        # Final layer norm
        self.final_norm = LayerNorm(d_model)

        # Output projection (often tied with token embeddings)
        self.output_projection = self.token_embedding.T  # [d_model, vocab_size]

    def forward(self, token_ids):
        """
        Forward pass.

        Args:
            token_ids: Input token IDs [batch_size, seq_len] or [seq_len]

        Returns:
            logits: [batch_size, seq_len, vocab_size] or [seq_len, vocab_size]
        """
        # Handle unbatched input
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]
            squeeze_batch = True
        else:
            squeeze_batch = False

        batch_size, seq_len = token_ids.shape

        # Token embeddings
        x = self.token_embedding[token_ids]  # [batch, seq_len, d_model]

        # Add positional encoding
        pos_enc = self.pos_encoding.forward(seq_len)
        x = x + pos_enc

        # Create causal mask
        mask = create_causal_mask(seq_len)

        # Apply Transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)

        # Final layer norm
        x = self.final_norm.forward(x)

        # Project to vocabulary
        logits = x @ self.output_projection  # [batch, seq_len, vocab_size]

        if squeeze_batch:
            logits = logits[0]

        return logits

    def generate(self, prompt_ids, max_new_tokens, temperature=1.0):
        """
        Generate tokens autoregressively.

        Args:
            prompt_ids: Initial token IDs [seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated token IDs
        """
        tokens = list(prompt_ids)

        for _ in range(max_new_tokens):
            # Get current context (up to max_len)
            context = np.array(tokens[-self.max_len:])

            # Forward pass
            logits = self.forward(context)

            # Get logits for last position
            next_logits = logits[-1] / temperature

            # Sample from distribution
            probs = softmax(next_logits)
            next_token = np.random.choice(len(probs), p=probs)

            tokens.append(next_token)

        return tokens

    def parameters(self):
        """Return all parameters."""
        params = [self.token_embedding]
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_norm.parameters())
        return params
```

## Usage Example

```python
# Create a small model
model = CausalTransformer(
    vocab_size=1000,
    d_model=64,
    n_heads=4,
    n_layers=2,
    max_len=128
)

# Example input
token_ids = np.array([1, 5, 23, 7, 42])

# Forward pass
logits = model.forward(token_ids)
print(f"Input shape: {token_ids.shape}")
print(f"Output shape: {logits.shape}")  # [5, 1000]

# Generation
generated = model.generate(token_ids, max_new_tokens=10, temperature=0.8)
print(f"Generated {len(generated)} tokens")

# Visualize attention weights
block = model.blocks[0]
attention_weights = block.attention.attention_weights
print(f"Attention weights shape: {attention_weights.shape}")
# [batch=1, n_heads=4, seq_len=5, seq_len=5]
```

## Visualizing Attention

```python
import matplotlib.pyplot as plt

def visualize_attention(model, token_ids, tokens_str, layer=0, head=0):
    """
    Visualize attention patterns.

    Args:
        model: CausalTransformer
        token_ids: Input token IDs
        tokens_str: String representation of tokens
        layer: Which layer to visualize
        head: Which head to visualize
    """
    # Forward pass to populate attention weights
    _ = model.forward(token_ids)

    # Get attention weights from specified layer
    weights = model.blocks[layer].attention.attention_weights

    # Select specific head
    if weights.ndim == 4:  # [batch, heads, seq, seq]
        weights = weights[0, head]  # [seq, seq]
    elif weights.ndim == 3:  # [heads, seq, seq]
        weights = weights[head]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(weights, cmap='Blues')

    ax.set_xticks(range(len(tokens_str)))
    ax.set_yticks(range(len(tokens_str)))
    ax.set_xticklabels(tokens_str, rotation=45, ha='right')
    ax.set_yticklabels(tokens_str)

    ax.set_xlabel('Key positions')
    ax.set_ylabel('Query positions')
    ax.set_title(f'Attention weights (Layer {layer}, Head {head})')

    plt.colorbar(im)
    plt.tight_layout()
    plt.show()
```

## Performance Considerations

### Memory Usage

```
Model parameters:

- Token embeddings: vocab_size × d_model
- Per block:
  - QKV projection: d_model × 3d_model = 3d_model²
  - Output projection: d_model²
  - FFN: d_model × 4d_model × 2 = 8d_model²
  - Layer norms: 4d_model
  - Total per block: ~12d_model²

Activation memory (during forward):

- Attention scores: batch × n_heads × seq_len²
- Attention weights: batch × n_heads × seq_len²
- This is the O(n²) that limits context length!
```

### Computational Cost

```
Per attention layer:

- QKV projection: O(seq_len × d_model²)
- Attention scores: O(seq_len² × d_model)
- Softmax: O(seq_len²)
- Attention @ V: O(seq_len² × d_model)
- Output projection: O(seq_len × d_model²)

Per FFN:

- First linear: O(seq_len × d_model × d_ff)
- Second linear: O(seq_len × d_ff × d_model)

Total per layer: O(seq_len × d_model² + seq_len² × d_model)
```

## Summary

We've built a complete attention implementation with:

| Component | Purpose | Key Details |
|-----------|---------|-------------|
| Scaled dot-product attention | Core attention computation | $QK^T$/√d_k, softmax, × V |
| Multi-head attention | Multiple attention patterns | Split into h heads, concatenate |
| Positional encoding | Position information | Sinusoidal or learned |
| Layer normalization | Training stability | Normalize, scale, shift |
| Feed-forward network | Feature processing | Two layers with activation |
| Transformer block | Combine components | Attention + FFN + residuals |
| Causal Transformer | Complete model | Embeddings + blocks + output |

**Key takeaway**: Building attention from scratch reveals how each component contributes to the whole. The core idea—computing relevance via dot products, normalizing with softmax, and aggregating values—is elegant and powerful. Understanding this implementation provides the foundation for working with any modern language model.

→ **Continue to**: [Stage 6: The Complete Transformer](../stage-06/index.md) (coming soon)

# Section 5.6: Positional Encoding — Injecting Position Information

*Reading time: 20 minutes | Difficulty: ★★★★☆*

Attention is permutation-equivariant: it treats positions as a set, not a sequence. This section explains why position information is essential and how we inject it into the model.

## The Problem: Attention Has No Notion of Order

Consider self-attention on "dog bites man" vs "man bites dog":

```
Without positional info:

"dog bites man"  → Same attention scores!
"man bites dog"  → (just rows/cols permuted)
```

The attention mechanism computes:

- score(dog, bites) based on their embeddings
- score(man, bites) based on their embeddings

The **position** of dog/man doesn't affect these scores. But meaning completely changes!

### Mathematical Proof

For any permutation π of positions:

$$\text{Attention}(\pi X) = \pi \text{Attention}(X)$$

If we permute the input, the output is permuted the same way. The relative relationships are unchanged—attention doesn't know position 1 from position 5.

## Why Position Matters

| Linguistic Phenomenon | Position-Dependent? |
|----------------------|---------------------|
| Subject-verb agreement | Yes (subject comes before verb) |
| Adjective-noun order | Yes (varies by language) |
| Pronoun reference | Yes (usually refers backward) |
| Negation scope | Yes ("not" affects what follows) |

Almost everything in language depends on position!

## Solution: Add Position Information to Embeddings

The key insight: add position-specific vectors to token embeddings before attention.

$$\text{input}_i = \text{token\_embedding}_i + \text{positional\_encoding}_i$$

Now each position has unique information that attention can use.

## Sinusoidal Positional Encoding

The original Transformer uses fixed sinusoidal functions:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Where:

- pos: position in sequence (0, 1, 2, ...)
- i: dimension index (0, 1, 2, ..., d/2-1)
- d: model dimension

### Why Sinusoids?

Each dimension has a different frequency:

```
Dimension 0-1:   High frequency (changes rapidly with position)
Dimension d-2:   Low frequency (changes slowly)

Position 0: [sin(0), cos(0), sin(0), cos(0), ...]
Position 1: [sin(1/1), cos(1/1), sin(1/10000^(2/d)), cos(1/10000^(2/d)), ...]
Position 2: [sin(2/1), cos(2/1), sin(2/10000^(2/d)), cos(2/10000^(2/d)), ...]
```

### Key Properties

**1. Unique encoding per position**:
Each position gets a unique vector.

**2. Bounded values**:
All values in [-1, 1], matching embedding scale.

**3. Relative position as linear function**:
$$PE_{pos+k}$$ can be expressed as a linear function of $$PE_{pos}$$.

This means the model can learn to compute relative positions!

**Proof sketch**:
```
sin(pos + k) = sin(pos)cos(k) + cos(pos)sin(k)
cos(pos + k) = cos(pos)cos(k) - sin(pos)sin(k)
```

This is a linear transformation of [sin(pos), cos(pos)] with matrix:
```
[cos(k)   sin(k)]
[-sin(k)  cos(k)]
```

### Visualization

```
Position:  0    1    2    3    4    5    6    7    8
Dim 0:    [████ ░░░░ ████ ░░░░ ████ ░░░░ ████ ░░░░ ████]  High freq
Dim 2:    [████████ ░░░░░░░░ ████████ ░░░░░░░░ ████████]  Medium freq
Dim d-2:  [██████████████████████░░░░░░░░░░░░░░░░░░░░░░]  Low freq

(█ = positive, ░ = negative)
```

Different dimensions encode position at different scales.

### Implementation

```python
import numpy as np

def sinusoidal_positional_encoding(max_len, d_model):
    """
    Generate sinusoidal positional encodings.

    Args:
        max_len: Maximum sequence length
        d_model: Model dimension

    Returns:
        Positional encodings [max_len, d_model]
    """
    PE = np.zeros((max_len, d_model))

    position = np.arange(max_len)[:, np.newaxis]  # [max_len, 1]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    PE[:, 0::2] = np.sin(position * div_term)  # Even dimensions
    PE[:, 1::2] = np.cos(position * div_term)  # Odd dimensions

    return PE


# Example usage
max_len = 100
d_model = 64
PE = sinusoidal_positional_encoding(max_len, d_model)

print(f"PE shape: {PE.shape}")  # [100, 64]
print(f"PE[0]: {PE[0][:4]}")    # [0, 1, 0, 1]  (sin(0), cos(0), ...)
print(f"PE[1]: {PE[1][:4]}")    # [0.84, 0.54, ...]  (sin(1), cos(1), ...)
```

## Learned Positional Embeddings

An alternative: learn position embeddings just like token embeddings.

```python
class LearnedPositionalEncoding:
    """Learned positional embeddings."""

    def __init__(self, max_len, d_model):
        # Each position gets its own learnable vector
        self.PE = np.random.randn(max_len, d_model) * 0.02

    def forward(self, seq_len):
        return self.PE[:seq_len]
```

### Comparison

| Aspect | Sinusoidal | Learned |
|--------|------------|---------|
| Parameters | 0 | max_len × d |
| Extrapolation | Can extend to longer sequences | Fixed to training length |
| Expressivity | Fixed patterns | Can learn any pattern |
| Modern preference | GPT-1, T5, some variants | GPT-2, BERT, most LLMs |

Modern large models mostly use learned embeddings, but with tricks for length generalization.

## Relative Positional Encoding

Instead of encoding absolute positions, encode relative distances.

### The Intuition

"The cat sat" should have similar relationships whether it appears at positions [0,1,2] or [100,101,102].

Relative encoding: position i attending to position j uses encoding for (i-j).

### Transformer-XL Style

Modify attention scores to include relative position:

$$\text{score}_{ij} = q_i^T k_j + q_i^T r_{i-j} + u^T k_j + v^T r_{i-j}$$

Where:

- $r_{i-j}$: relative position embedding
- u, v: learnable global biases

### T5 Style (Simplified)

Add learned bias based on relative position:

$$\text{score}_{ij} = q_i^T k_j + b_{i-j}$$

Where b is a learned bias table indexed by relative position.

```python
class T5RelativePositionBias:
    """T5-style relative position bias."""

    def __init__(self, max_distance=128, n_heads=8):
        # Learn bias for each relative distance and head
        self.bias_table = np.random.randn(2 * max_distance + 1, n_heads) * 0.02
        self.max_distance = max_distance

    def forward(self, seq_len):
        """Compute relative position bias matrix."""
        # Create relative position matrix
        positions = np.arange(seq_len)
        relative_pos = positions[None, :] - positions[:, None]  # [n, n]

        # Clip to max distance
        relative_pos = np.clip(relative_pos,
                              -self.max_distance,
                               self.max_distance)

        # Shift to positive indices
        indices = relative_pos + self.max_distance

        # Look up biases
        return self.bias_table[indices]  # [n, n, n_heads]
```

## Rotary Positional Embedding (RoPE)

The modern standard for many LLMs (LLaMA, Mistral, etc.).

### Key Idea

Rotate query and key vectors based on position. Relative positions emerge from the dot product of rotated vectors.

$$q_m' = R_m q_m, \quad k_n' = R_n k_n$$

$$q_m' \cdot k_n' = q_m^T R_m^T R_n k_n = q_m^T R_{n-m} k_n$$

The rotation difference $R_{n-m}$ depends only on relative position!

### How Rotation Works

For 2D vectors:

$$R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

For higher dimensions, apply 2D rotations to pairs of dimensions.

### Implementation

```python
def rotary_embedding(x, position, base=10000):
    """
    Apply rotary positional embedding.

    Args:
        x: Input tensor [seq_len, d]
        position: Position indices [seq_len]
        base: Base for frequency computation

    Returns:
        Rotated tensor [seq_len, d]
    """
    d = x.shape[-1]
    assert d % 2 == 0, "Dimension must be even"

    # Compute frequencies
    freqs = 1.0 / (base ** (np.arange(0, d, 2) / d))  # [d/2]

    # Compute angles
    angles = position[:, None] * freqs[None, :]  # [seq_len, d/2]

    cos_angles = np.cos(angles)  # [seq_len, d/2]
    sin_angles = np.sin(angles)  # [seq_len, d/2]

    # Split x into pairs
    x1 = x[:, 0::2]  # Even dimensions
    x2 = x[:, 1::2]  # Odd dimensions

    # Apply rotation
    x_rotated = np.empty_like(x)
    x_rotated[:, 0::2] = x1 * cos_angles - x2 * sin_angles
    x_rotated[:, 1::2] = x1 * sin_angles + x2 * cos_angles

    return x_rotated
```

### Why RoPE is Popular

1. **Relative positions naturally**: No need for explicit relative position computation
2. **Efficient**: Simple element-wise operations
3. **Length extrapolation**: With NTK-aware scaling, can extend to longer sequences
4. **Linear attention compatible**: Works with efficient attention variants

!!! info "Connection to Modern LLMs"

    Most recent LLMs use RoPE or variants:

    - **LLaMA, LLaMA 2**: Standard RoPE
    - **Mistral, Mixtral**: RoPE with sliding window attention
    - **GPT-4, Claude**: Details not public, but likely relative position methods
    - **Gemini**: Uses relative position encoding

    The field has converged on relative methods for their better length generalization.

## ALiBi: Attention with Linear Biases

Another simple and effective method.

### The Idea

Add a linear penalty to attention scores based on distance:

$$\text{score}_{ij} = q_i^T k_j - m \cdot |i - j|$$

Where m is a head-specific slope.

### Implementation

```python
def alibi_attention(Q, K, V, slopes):
    """
    Attention with ALiBi positional encoding.

    Args:
        Q, K, V: Query, Key, Value matrices
        slopes: Per-head slopes for distance penalty
    """
    n = Q.shape[0]
    d_k = Q.shape[-1]

    # Standard attention scores
    scores = Q @ K.T / np.sqrt(d_k)  # [n, n]

    # Create distance matrix
    positions = np.arange(n)
    distances = positions[None, :] - positions[:, None]  # [n, n]

    # Apply linear bias (for causal, only negative distances matter)
    bias = slopes * np.abs(distances)
    scores = scores - bias

    # Rest is standard attention
    weights = softmax(scores, axis=-1)
    return weights @ V
```

### Advantages

- **Zero extra parameters**: Just a simple bias
- **Excellent length generalization**: Works on sequences 10x training length
- **Efficient**: Simple addition operation

## Combining Positional Encodings

Some architectures combine methods:

```python
class HybridPositionalEncoding:
    """Combine absolute and relative encodings."""

    def __init__(self, max_len, d_model, n_heads):
        # Absolute (added to embeddings)
        self.absolute = sinusoidal_positional_encoding(max_len, d_model)

        # Relative (added to attention scores)
        self.relative = T5RelativePositionBias(max_distance=128, n_heads=n_heads)

    def encode_input(self, X, positions):
        """Add absolute encoding to input."""
        return X + self.absolute[positions]

    def attention_bias(self, seq_len):
        """Get relative position bias for attention."""
        return self.relative.forward(seq_len)
```

## Position Encoding Summary

| Method | Parameters | Length Generalization | Modern Usage |
|--------|------------|----------------------|--------------|
| Sinusoidal | 0 | Good | Limited |
| Learned | max_len × d | Poor | Common |
| Relative (T5) | distance × heads | Good | Common |
| RoPE | 0 | Good with scaling | Very common |
| ALiBi | 0 | Excellent | Common |

## Exercises

1. **Implement sinusoidal**: Write the encoding and visualize as a heatmap.

2. **Dot product distance**: For sinusoidal PE, compute dot product between positions. What pattern emerges?

3. **Extrapolation test**: Train with max length 100, test at 200. Compare methods.

4. **RoPE derivation**: Prove that q_m' · k_n' depends only on (m-n).

5. **Design your own**: Create a positional encoding and analyze its properties.

## Summary

| Concept | Definition | Purpose |
|---------|------------|---------|
| Position encoding | Vector added/applied per position | Give attention position awareness |
| Sinusoidal | Fixed sine/cosine patterns | Unique, bounded, allows relative |
| Learned | Trainable per-position vectors | More flexible, limited length |
| Relative | Encode (i-j) not absolute i, j | Better generalization |
| RoPE | Rotate Q, K by position | Efficient relative encoding |

**Key takeaway**: Attention mechanisms are permutation-equivariant by design, treating input as a set rather than a sequence. Positional encodings inject position information, enabling the model to understand word order. Modern methods favor relative positions (RoPE, ALiBi) for better length generalization, while the original Transformer used fixed sinusoidal patterns.

→ **Next**: [Section 5.7: Causal Masking](07-causal-masking.md)

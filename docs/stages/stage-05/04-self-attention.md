# Section 5.4: Self-Attention — A Sequence Attending to Itself

*Reading time: 15 minutes | Difficulty: ★★★☆☆*

Self-attention is the key insight that powers transformers: a sequence can attend to itself, learning relationships between all positions simultaneously.

## From Cross-Attention to Self-Attention

### Cross-Attention (Original Attention)

In machine translation, attention was used between two sequences:

```
Encoder output: [h₁, h₂, h₃]  (source sentence)
Decoder state:   s            (current translation state)

Query: from decoder (what translation needs)
Keys:  from encoder (what source contains)
Values: from encoder (source information)
```

The decoder attends to the encoder—two separate sequences.

### Self-Attention

What if we use attention within a single sequence?

```
Input:  [x₁, x₂, x₃, x₄]

Query: from input (what each position needs)
Keys:  from input (what each position contains)
Values: from input (what each position should contribute)
```

Every position can attend to every other position (including itself)!

## The Self-Attention Mechanism

### The Complete Setup

Given input sequence X ∈ ℝ^{n×d}:

1. **Project to Q, K, V**:
   - Q = XW^Q (queries)
   - K = XW^K (keys)
   - V = XW^V (values)

2. **Compute attention**:
$$\text{SelfAttention}(X) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

3. **Output**: Each position gets a weighted combination of all values.

### Matrix Dimensions

```
X:     [n × d]      (input sequence)

W^Q:   [d × d_k]    (query projection)
W^K:   [d × d_k]    (key projection)
W^V:   [d × d_v]    (value projection)

Q:     [n × d_k]    (queries)
K:     [n × d_k]    (keys)
V:     [n × d_v]    (values)

QK^T:  [n × n]      (attention scores)
Output:[n × d_v]    (contextualized representations)
```

## Why Self-Attention Works

### Every Position Gets Context

After self-attention, position i contains information from all positions:

```
Before: x_i contains only information about position i

After:  output_i = Σⱼ α_ij · v_j
        Contains information from all positions,
        weighted by relevance to position i
```

### Learned Relevance

The model learns what's relevant through W^Q, W^K, W^V:

```
Position i asks: "What's relevant to me?"  → query q_i
Position j says: "Here's what I have"      → key k_j
Match score:     q_i · k_j                 → attention weight

High match → j is relevant to i → high weight
Low match  → j isn't relevant   → low weight
```

## Worked Example: "The cat sat"

Let's trace through self-attention step by step.

### Input

```python
# 3 positions, 4-dimensional embeddings
X = [
    [1, 0, 1, 0],  # "The"
    [0, 1, 0, 1],  # "cat"
    [1, 1, 0, 0],  # "sat"
]
```

### Projections (simplified 2D output)

Using learned weight matrices:

```python
Q = X @ W_Q = [
    [0.5, 0.5],   # "The" query
    [0.8, 0.2],   # "cat" query: looking for context
    [0.3, 0.9],   # "sat" query: looking for subject
]

K = X @ W_K = [
    [0.2, 0.8],   # "The" key
    [0.9, 0.3],   # "cat" key: noun, subject
    [0.1, 0.7],   # "sat" key: verb
]

V = X @ W_V = [
    [0.1, 0.9],   # "The" value
    [0.8, 0.5],   # "cat" value
    [0.4, 0.6],   # "sat" value
]
```

### Attention Scores

Compute QK^T (before scaling):

```
           The   cat   sat
    The [ 0.50  0.60  0.40 ]
    cat [ 0.32  0.78  0.22 ]
    sat [ 0.78  0.54  0.66 ]
```

After scaling by 1/√d_k = 1/√2 ≈ 0.707:

```
           The   cat   sat
    The [ 0.35  0.42  0.28 ]
    cat [ 0.23  0.55  0.16 ]
    sat [ 0.55  0.38  0.47 ]
```

### Softmax (row-wise)

```
           The   cat   sat
    The [ 0.31  0.34  0.28 ]  → "The" attends roughly evenly
    cat [ 0.27  0.42  0.24 ]  → "cat" attends most to itself
    sat [ 0.36  0.30  0.33 ]  → "sat" attends most to "The"
```

### Output

Weighted sum of values:

```
output_The = 0.31·v_The + 0.34·v_cat + 0.28·v_sat
           = [0.43, 0.66]

output_cat = 0.27·v_The + 0.42·v_cat + 0.24·v_sat
           = [0.46, 0.59]

output_sat = 0.36·v_The + 0.30·v_cat + 0.33·v_sat
           = [0.41, 0.68]
```

Each position now contains a blend of information from all positions!

## What Self-Attention Learns

Different attention heads learn different patterns:

### Syntactic Patterns

```
"The cat that I saw sat on the mat"

When processing "sat":
- High attention to "cat" (subject)
- Lower attention to "I" (not the subject of "sat")
```

### Positional Patterns

```
Some heads learn to attend to nearby tokens:

Position i attends heavily to:
- Position i-1 (previous token)
- Position i+1 (next token)
```

### Semantic Patterns

```
"The bank was closed. I couldn't deposit money at the bank."

When processing the second "bank":
- Attends to "deposit" and "money"
- Learns this "bank" means financial institution
```

!!! info "Connection to Modern LLMs"

    Researchers have found interpretable attention patterns in GPT and similar models:

    - **Induction heads**: Copy patterns from earlier in the context
    - **Previous token heads**: Always attend to the previous position
    - **Duplicate token heads**: Find repeated tokens
    - **Coreference heads**: Link pronouns to their referents

    These emerge automatically from training on language!

## The [n × n] Attention Matrix

The attention matrix is the heart of self-attention:

```
            Position 1   Position 2   ...   Position n
Position 1 [   α₁₁         α₁₂       ...      α₁ₙ    ]
Position 2 [   α₂₁         α₂₂       ...      α₂ₙ    ]
...        [   ...         ...       ...      ...    ]
Position n [   αₙ₁         αₙ₂       ...      αₙₙ    ]
```

Properties:
- Each row sums to 1 (probability distribution)
- Entry α_ij = "how much position i attends to position j"
- Diagonal elements = self-attention (attending to yourself)
- Off-diagonal = cross-position attention

## Parameter Count

For self-attention with:
- Input dimension: d
- Key/Query dimension: d_k
- Value dimension: d_v

Parameters:
- W^Q: d × d_k
- W^K: d × d_k
- W^V: d × d_v
- **Total**: d(2d_k + d_v)

Typical setting (d = d_k = d_v = 512):
- **Parameters**: 512 × 3 × 512 = 786,432 ≈ 0.8M per attention layer

## Computational Cost

| Operation | FLOPs | Memory |
|-----------|-------|--------|
| Q, K, V projection | O(nd²) | O(nd) |
| QK^T | O(n²d) | O(n²) |
| Softmax | O(n²) | O(n²) |
| Attention × V | O(n²d) | O(nd) |
| **Total** | **O(n²d)** | **O(n² + nd)** |

The quadratic O(n²) in sequence length is attention's main limitation.

## Implementation

```python
import numpy as np

def self_attention(X, W_Q, W_K, W_V):
    """
    Self-attention mechanism.

    Args:
        X: Input sequence [n, d]
        W_Q: Query projection [d, d_k]
        W_K: Key projection [d, d_k]
        W_V: Value projection [d, d_v]

    Returns:
        Output [n, d_v], attention weights [n, n]
    """
    # Project to Q, K, V
    Q = X @ W_Q  # [n, d_k]
    K = X @ W_K  # [n, d_k]
    V = X @ W_V  # [n, d_v]

    d_k = Q.shape[-1]

    # Compute scaled attention scores
    scores = Q @ K.T / np.sqrt(d_k)  # [n, n]

    # Softmax to get attention weights
    def softmax(x, axis=-1):
        x_max = x.max(axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / exp_x.sum(axis=axis, keepdims=True)

    attention_weights = softmax(scores)  # [n, n]

    # Weighted sum of values
    output = attention_weights @ V  # [n, d_v]

    return output, attention_weights


class SelfAttentionLayer:
    """Learnable self-attention layer."""

    def __init__(self, d_model, d_k=None, d_v=None):
        """
        Initialize self-attention layer.

        Args:
            d_model: Input/output dimension
            d_k: Key/query dimension (default: d_model)
            d_v: Value dimension (default: d_model)
        """
        d_k = d_k or d_model
        d_v = d_v or d_model

        # Xavier initialization
        scale_qk = np.sqrt(2.0 / (d_model + d_k))
        scale_v = np.sqrt(2.0 / (d_model + d_v))

        self.W_Q = np.random.randn(d_model, d_k) * scale_qk
        self.W_K = np.random.randn(d_model, d_k) * scale_qk
        self.W_V = np.random.randn(d_model, d_v) * scale_v

        # Output projection (to match input dimension)
        self.W_O = np.random.randn(d_v, d_model) * scale_v

    def forward(self, X):
        """Forward pass."""
        output, weights = self_attention(X, self.W_Q, self.W_K, self.W_V)

        # Project back to d_model
        output = output @ self.W_O

        return output, weights
```

## Self-Attention vs. Other Architectures

| Architecture | Context | Parallelizable | Path Length |
|--------------|---------|----------------|-------------|
| RNN | Sequential state | No | O(n) |
| CNN | Local window | Yes | O(n/k) |
| Self-Attention | All positions | Yes | O(1) |

Self-attention wins on all fronts for modeling:
- **Global context**: Every position sees every other
- **Training speed**: Fully parallelizable
- **Gradient flow**: Direct paths between any positions

## Common Variations

### Pre-Norm vs Post-Norm

```python
# Post-norm (original Transformer)
output = LayerNorm(x + SelfAttention(x))

# Pre-norm (better training dynamics)
output = x + SelfAttention(LayerNorm(x))
```

### With Residual Connection

In practice, self-attention is always used with residual:

```python
def attention_block(x, layer):
    attention_output, _ = layer.forward(x)
    return x + attention_output  # Residual connection
```

This allows gradients to flow directly and helps with training deep networks.

## Exercises

1. **Implement self-attention**: Write the forward pass from scratch.

2. **Visualize attention**: For the sentence "The cat sat on the mat", plot the attention matrix.

3. **Symmetry breaking**: If W^Q = W^K, what happens? When might this be useful?

4. **Complexity analysis**: For n=1000, d=512, compute the number of FLOPs.

5. **No values**: What if we set V = K? What does this represent?

## Summary

| Concept | Definition | Purpose |
|---------|------------|---------|
| Self-attention | Sequence attends to itself | Capture relationships within sequence |
| Q, K, V projections | XW^Q, XW^K, XW^V | Different roles for matching and retrieval |
| Attention matrix | [n × n] weights | Shows what each position attends to |
| Output | Weighted sum of values | Contextualized representations |

**Key takeaway**: Self-attention allows every position in a sequence to directly access every other position through learned query-key matching. This creates contextualized representations where each position contains relevant information from the entire sequence, enabling the model to capture long-range dependencies in a single operation.

→ **Next**: [Section 5.5: Multi-Head Attention](05-multi-head.md)

# Section 5.5: Multi-Head Attention — Multiple Perspectives

*Reading time: 18 minutes | Difficulty: ★★★☆☆*

A single attention mechanism can only focus on one type of relationship at a time. Multi-head attention runs multiple attention operations in parallel, allowing the model to jointly attend to information from different representation subspaces.

## The Limitation of Single-Head Attention

Consider processing "The cat sat on the mat because it was tired."

Different types of information are relevant:

- **Syntactic**: "sat" should attend to "cat" (subject-verb)
- **Positional**: "tired" should attend to nearby words
- **Coreference**: "it" should attend to "cat" (reference)

A single attention head must compress all these relationships into one set of weights. It can't simultaneously:

- Pay maximum attention to the subject AND
- Pay attention to nearby words AND
- Resolve references

## The Multi-Head Solution

Instead of one attention mechanism with d-dimensional keys/queries, use h parallel attention heads, each with d/h dimensions:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

where:
$$\text{head}_i = \text{Attention}($QW_i^Q$, $KW_i^K$, $VW_i^V$)$$

Each head learns to focus on different aspects of the input.

## How It Works

### Step 1: Project to Multiple Heads

For each head i:

- Q_i = $QW_i^Q$ ∈ $ℝ^{n × d_k}$
- K_i = $KW_i^K$ ∈ $ℝ^{n × d_k}$
- V_i = $VW_i^V$ ∈ $ℝ^{n × d_v}$

Where d_k = d_v = d/h typically.

### Step 2: Parallel Attention

Each head computes attention independently:

$$\text{head}_i = \text{softmax}\left(\frac{Q_i $K_i^T$}{\sqrt{d_k}}\right) V_i$$

### Step 3: Concatenate and Project

Combine all heads and project back:

$$\text{output} = [head_1; head_2; ...; head_h] W^O$$

Where $W^O$ ∈ $ℝ^{hd_v × d}$ projects back to model dimension.

## Visual Representation

```
Input X [n × d]
    │
    ├──────────────────────────────────────────────┐
    │                                              │
    ▼                                              ▼
   Head 1                    ...                  Head h
    │                                              │
    ├── Q₁ = XW¹_Q                    Q_h = XW^h_Q ─┤
    ├── K₁ = XW¹_K                    K_h = XW^h_K ─┤
    ├── V₁ = XW¹_V                    V_h = XW^h_V ─┤
    │                                              │
    ▼                                              ▼
Attention(Q₁,K₁,V₁)          ...        Attention(Q_h,K_h,V_h)
    │                                              │
    ▼                                              ▼
  [n × d_v]                  ...                [n × d_v]
    │                                              │
    └──────────────────┬───────────────────────────┘
                       │ Concat
                       ▼
                   [n × hd_v]
                       │
                       ▼ W^O
                   [n × d]
                       │
                       ▼
                   Output
```

## Worked Example

Let's trace through with h=2 heads, d=4, d_k=d_v=2.

### Input

```python
X = [
    [1, 0, 1, 0],  # Position 1
    [0, 1, 0, 1],  # Position 2
    [1, 1, 0, 0],  # Position 3
]  # [3 × 4]
```

### Head 1 Projections

```python
W1_Q = [[1, 0], [0, 1], [0, 0], [0, 0]]  # [4 × 2]
W1_K = [[0, 1], [1, 0], [0, 0], [0, 0]]
W1_V = [[1, 0], [0, 0], [1, 0], [0, 0]]

Q1 = X @ W1_Q = [[1, 0], [0, 1], [1, 1]]  # [3 × 2]
K1 = X @ W1_K = [[0, 1], [1, 0], [1, 1]]
V1 = X @ W1_V = [[2, 0], [0, 0], [1, 0]]
```

### Head 2 Projections

```python
W2_Q = [[0, 0], [0, 0], [1, 0], [0, 1]]  # Different subspace!
W2_K = [[0, 0], [0, 0], [0, 1], [1, 0]]
W2_V = [[0, 1], [0, 1], [0, 0], [0, 0]]

Q2 = X @ W2_Q = [[1, 0], [0, 1], [0, 0]]  # [3 × 2]
K2 = X @ W2_K = [[0, 1], [1, 0], [0, 0]]
V2 = X @ W2_V = [[0, 1], [0, 1], [0, 0]]
```

### Compute Attention for Each Head

**Head 1**: Focuses on first two dimensions of input
```
Attention scores (Q1 @ K1.T / sqrt(2)):
[[0.71, 0.00, 1.41],
 [0.00, 0.71, 0.71],
 [0.71, 0.71, 1.41]]

After softmax:
[[0.30, 0.15, 0.55],
 [0.21, 0.37, 0.42],
 [0.21, 0.21, 0.58]]

Output1 = attention @ V1
```

**Head 2**: Focuses on last two dimensions
```
Different attention pattern!
```

### Concatenate and Project

```python
concat = [head1; head2]  # [3 × 4]

W_O = ...  # [4 × 4]

output = concat @ W_O  # [3 × 4]
```

## What Different Heads Learn

Research on trained transformers reveals specialized heads:

### Syntactic Heads

```
"The cat that I saw yesterday sat"

Head focusing on subject-verb:
sat → cat: 0.65  (main subject)
sat → I:   0.10  (not the subject of "sat")
sat → saw: 0.05
```

### Positional Heads

```
Head focusing on adjacent tokens:
Each position strongly attends to position-1 and position+1
```

### Copy/Induction Heads

```
"[A][B] ... [A]"

When seeing second [A], head attends to [B]
(predicts what came after first [A])
```

### Rare Word Heads

```
Some heads specialize in attending to rare/important tokens
like proper nouns, numbers, or technical terms.
```

!!! info "Connection to Modern LLMs"

    In GPT-2 and similar models, researchers found:

    - **Induction heads** (copy patterns): Emerge in layer 2+ and are crucial for in-context learning
    - **Previous token heads**: Simple but important for local coherence
    - **Backup heads**: Redundant heads that provide robustness

    The model learns to allocate heads to different linguistic tasks automatically!

## Parameter Analysis

For multi-head attention with:

- Model dimension: d
- Number of heads: h
- Head dimension: d_k = d_v = d/h

**Per-head parameters**:

- $W_i^Q$: d × d_k = d × d/h = d²/h
- $W_i^K$: d × d_k = d²/h
- $W_i^V$: d × d_v = d²/h

**Total for all heads**: h × 3 × d²/h = 3d²

**Output projection**: $W^O$: hd_v × d = d × d = d²

**Grand total**: 3d² + d² = **4d²**

This is the same as having separate Q, K, V, O projections in single-head attention with dimension d. Multi-head is essentially a particular factorization.

## Why Not Just Use More Parameters?

Question: Why h heads of dimension d/h instead of 1 head of dimension d?

### Computational Efficiency

Same parameter count, but:

- Each head operates in lower dimension
- All heads compute in parallel
- Similar total compute

### Representational Power

Different heads can learn:

- Orthogonal attention patterns
- Specialized roles
- Complementary information

A single head would have to encode everything in one pattern.

### Empirical Evidence

```
Ablation on WMT translation:

Heads | BLEU Score
------|-----------
  1   |   25.8
  2   |   27.1
  4   |   27.5
  8   |   28.0 (original Transformer)
 16   |   28.0 (no improvement)
```

More heads help up to a point, then saturate.

## Implementation

```python
import numpy as np

def multi_head_attention(X, W_Qs, W_Ks, W_Vs, W_O):
    """
    Multi-head attention mechanism.

    Args:
        X: Input [n, d]
        W_Qs: List of query projections [h × (d, d_k)]
        W_Ks: List of key projections [h × (d, d_k)]
        W_Vs: List of value projections [h × (d, d_v)]
        W_O: Output projection [h*d_v, d]

    Returns:
        Output [n, d], list of attention weights [h × (n, n)]
    """
    h = len(W_Qs)
    heads = []
    attention_weights = []

    for i in range(h):
        # Project to this head's subspace
        Q_i = X @ W_Qs[i]
        K_i = X @ W_Ks[i]
        V_i = X @ W_Vs[i]

        d_k = Q_i.shape[-1]

        # Scaled dot-product attention
        scores = Q_i @ K_i.T / np.sqrt(d_k)

        # Softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        attention_weights.append(weights)

        # Weighted sum of values
        head_output = weights @ V_i
        heads.append(head_output)

    # Concatenate all heads
    concat = np.concatenate(heads, axis=-1)  # [n, h*d_v]

    # Final projection
    output = concat @ W_O  # [n, d]

    return output, attention_weights


class MultiHeadAttention:
    """Multi-head attention layer with learned parameters."""

    def __init__(self, d_model, n_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
        """
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        # Initialize projection matrices
        scale = np.sqrt(2.0 / (d_model + self.d_k))

        self.W_Qs = [np.random.randn(d_model, self.d_k) * scale
                     for _ in range(n_heads)]
        self.W_Ks = [np.random.randn(d_model, self.d_k) * scale
                     for _ in range(n_heads)]
        self.W_Vs = [np.random.randn(d_model, self.d_v) * scale
                     for _ in range(n_heads)]

        # Output projection
        self.W_O = np.random.randn(n_heads * self.d_v, d_model) * scale

    def forward(self, X, mask=None):
        """
        Forward pass.

        Args:
            X: Input [n, d_model]
            mask: Optional attention mask [n, n]

        Returns:
            Output [n, d_model], attention weights [n_heads, n, n]
        """
        return multi_head_attention(X, self.W_Qs, self.W_Ks, self.W_Vs, self.W_O)

    def parameters(self):
        """Return all parameters as a list."""
        params = self.W_Qs + self.W_Ks + self.W_Vs + [self.W_O]
        return params
```

## Efficient Implementation: Batched Projections

In practice, we batch all head projections together:

```python
class EfficientMultiHeadAttention:
    """Efficient multi-head attention using batched operations."""

    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Single large projections instead of per-head
        scale = np.sqrt(2.0 / (2 * d_model))
        self.W_QKV = np.random.randn(d_model, 3 * d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

    def forward(self, X):
        """Forward pass with efficient batched computation."""
        n = X.shape[0]

        # Single projection for all Q, K, V
        QKV = X @ self.W_QKV  # [n, 3*d_model]

        # Split into Q, K, V
        Q, K, V = np.split(QKV, 3, axis=-1)  # Each [n, d_model]

        # Reshape into heads: [n, d_model] -> [n, h, d_k] -> [h, n, d_k]
        Q = Q.reshape(n, self.n_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(n, self.n_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(n, self.n_heads, self.d_k).transpose(1, 0, 2)

        # Batched attention: [h, n, d_k] @ [h, d_k, n] -> [h, n, n]
        scores = np.einsum('hnd,hmd->hnm', Q, K) / np.sqrt(self.d_k)

        # Softmax per head
        scores_max = scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores - scores_max)
        weights = weights / weights.sum(axis=-1, keepdims=True)

        # Apply attention: [h, n, n] @ [h, n, d_k] -> [h, n, d_k]
        heads = np.einsum('hnm,hmd->hnd', weights, V)

        # Reshape back: [h, n, d_k] -> [n, h, d_k] -> [n, d_model]
        concat = heads.transpose(1, 0, 2).reshape(n, self.d_model)

        # Output projection
        output = concat @ self.W_O

        return output, weights
```

## Visualizing Multi-Head Attention

```
Sentence: "The cat sat on the mat"

Head 1 (syntactic):        Head 2 (positional):
    T c s o t m                T c s o t m
T [ ░ █ ░ ░ ░ ░ ]          T [ █ █ ░ ░ ░ ░ ]
c [ █ ░ ░ ░ ░ ░ ]          c [ █ █ █ ░ ░ ░ ]
s [ ░ █ ░ ░ ░ ░ ]          s [ ░ █ █ █ ░ ░ ]
o [ ░ ░ █ ░ ░ ░ ]          o [ ░ ░ █ █ █ ░ ]
t [ ░ ░ ░ █ ░ ░ ]          t [ ░ ░ ░ █ █ █ ]
m [ ░ ░ ░ ░ █ ░ ]          m [ ░ ░ ░ ░ █ █ ]

Head 1 learns subject-verb    Head 2 learns local context
```

## When to Use How Many Heads

| Model Size | Typical Heads | Head Dimension |
|------------|--------------|----------------|
| Small (d=256) | 4 | 64 |
| Medium (d=512) | 8 | 64 |
| Large (d=1024) | 16 | 64 |
| XL (d=2048) | 32 | 64 |

The head dimension is often kept constant (64) while scaling the number of heads with model size.

## Exercises

1. **Implement multi-head**: Write both naive and efficient versions.

2. **Visualize heads**: Train a small model and plot attention patterns for different heads.

3. **Head ablation**: What happens if you zero out different heads? Which are important?

4. **Head pruning**: After training, can you remove heads with minimal performance loss?

5. **Specialized heads**: Can you design heads that must attend to specific patterns?

## Summary

| Concept | Definition | Purpose |
|---------|------------|---------|
| Multi-head attention | h parallel attention mechanisms | Multiple representation subspaces |
| Head dimension | d_k = d/h | Reduced per-head computation |
| Concatenation | [head_1; ...; head_h] | Combine all perspectives |
| Output projection | $W^O$ | Mix head outputs |

**Key takeaway**: Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. Each head can specialize in different types of relationships (syntactic, positional, semantic), enabling richer modeling of language structure than a single attention mechanism could achieve.

→ **Next**: [Section 5.6: Positional Encoding](06-positional-encoding.md)

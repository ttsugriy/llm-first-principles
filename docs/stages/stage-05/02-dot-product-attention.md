# Section 5.2: Dot-Product Attention — The Core Mechanism

*Reading time: 20 minutes | Difficulty: ★★★☆☆*

This section derives the dot-product attention mechanism from first principles. We'll understand exactly how queries, keys, and values work together to enable dynamic context selection.

## The Setup

We have a sequence of n positions, each represented by a d-dimensional vector:

$$X = [x_1, x_2, ..., x_n] \in \mathbb{R}^{n \times d}$$

We want each position to gather information from other positions based on relevance.

## Query, Key, Value: The Three Roles

Each position plays three roles simultaneously:

### Query: "What am I looking for?"

When position i wants to gather information, it generates a **query** vector:
$$q_i = x_i W^Q$$

The query represents what kind of information position i is seeking.

### Key: "What do I contain?"

Each position j provides a **key** vector:
$$k_j = x_j W^K$$

The key advertises what information position j has.

### Value: "What should I return?"

Each position j also provides a **value** vector:
$$v_j = x_j W^V$$

If position i attends to position j, it receives j's value.

**Key insight**: Queries and keys determine *how much* to attend. Values determine *what* to retrieve.

## Computing Attention

### Step 1: Measure Similarity

How relevant is position j to position i? Use the dot product:

$$\text{score}_{ij} = q_i \cdot k_j = q_i^T k_j$$

Why dot product?
- Positive when vectors point in similar directions
- Large when both vectors have large magnitudes
- Efficient to compute (matrix multiplication)

```
Query q_i:  [1, 0, 1]     (what I'm looking for)
Key k_j:    [1, 0, 1]     (what j contains)
Dot product: 1+0+1 = 2    (high similarity!)

Key k_m:    [0, 1, 0]     (what m contains)
Dot product: 0+0+0 = 0    (low similarity)
```

### Step 2: Convert to Probabilities

The scores can be any real number. Convert to probabilities using softmax:

$$\alpha_{ij} = \frac{\exp(\text{score}_{ij})}{\sum_{k=1}^{n} \exp(\text{score}_{ik})}$$

Properties of attention weights α:
- Non-negative: α_{ij} ≥ 0
- Normalized: Σ_j α_{ij} = 1
- Differentiable: We can backpropagate through them

### Step 3: Weighted Sum of Values

The output for position i is:

$$\text{output}_i = \sum_{j=1}^{n} \alpha_{ij} v_j$$

This is a weighted average of all values, where the weights are the attention probabilities.

## The Complete Formula

Putting it together for all positions simultaneously:

$$\text{Attention}(Q, K, V) = \text{softmax}(QK^T) V$$

Where:
- Q = XW^Q ∈ ℝ^{n×d_k} (all queries)
- K = XW^K ∈ ℝ^{n×d_k} (all keys)
- V = XW^V ∈ ℝ^{n×d_v} (all values)

### Matrix Dimensions

```
Q:     [n × d_k]    (n queries, each d_k-dimensional)
K:     [n × d_k]    (n keys, each d_k-dimensional)
V:     [n × d_v]    (n values, each d_v-dimensional)

QK^T:  [n × n]      (attention scores for all pairs)
softmax(QK^T): [n × n]  (attention weights)
softmax(QK^T) × V: [n × d_v]  (output)
```

The [n × n] attention matrix shows how much each position attends to every other position.

## Worked Example

Let's compute attention for a 3-word sequence: "cat sat mat"

**Input embeddings** (simplified 2D):
```
x_cat = [1, 0]
x_sat = [0, 1]
x_mat = [1, 1]
```

**Learned projections** (identity for simplicity):
```
W^Q = W^K = W^V = I
```

So Q = K = V = X.

**Step 1: Compute scores** (QK^T):
```
           cat  sat  mat
    cat [  1    0    1  ]   (cat·cat, cat·sat, cat·mat)
    sat [  0    1    1  ]
    mat [  1    1    2  ]
```

**Step 2: Apply softmax** (row-wise):
```
           cat   sat   mat
    cat [ 0.42  0.16  0.42 ]
    sat [ 0.18  0.33  0.49 ]
    mat [ 0.18  0.18  0.64 ]
```

**Step 3: Weighted sum of values**:
```
output_cat = 0.42×[1,0] + 0.16×[0,1] + 0.42×[1,1] = [0.84, 0.58]
output_sat = 0.18×[1,0] + 0.33×[0,1] + 0.49×[1,1] = [0.67, 0.82]
output_mat = 0.18×[1,0] + 0.18×[0,1] + 0.64×[1,1] = [0.82, 0.82]
```

Each output is a blend of all values, weighted by attention!

## Why Separate Q, K, V?

Why not use the same projection for all three?

**Different roles require different representations**:

| Role | What it captures |
|------|------------------|
| Query | "What am I looking for?" — information needs |
| Key | "What do I contain?" — content summary |
| Value | "What should I return?" — actual content |

Example: The word "it" might have:
- Query: "looking for a noun, preferably animate"
- Key: "I am a pronoun"
- Value: semantic representation of "it"

The key and query don't need to match—the key says what "it" *is*, but the query asks what "it" *needs*.

## Attention as Soft Dictionary Lookup

Think of attention as a differentiable dictionary:

```python
# Hard lookup (regular dictionary)
def hard_lookup(query, dictionary):
    return dictionary[query]  # Exact match or error

# Soft lookup (attention)
def soft_lookup(query, keys, values):
    scores = [dot(query, k) for k in keys]
    weights = softmax(scores)
    return sum(w * v for w, v in zip(weights, values))
```

The soft version:
- Never fails (always returns something)
- Can combine information from multiple sources
- Is differentiable (we can learn query/key representations)

## Attention Patterns

The attention weights form interpretable patterns:

```
"The cat sat on the mat"

When processing "sat", attention might look like:
The:  ░░░      (0.05) - low relevance
cat:  ████████ (0.70) - who sat?
sat:  ░        (0.02) - self
on:   ░        (0.03) - irrelevant
the:  ░░       (0.05) - irrelevant
mat:  ░░░░     (0.15) - where?
```

Different output positions attend to different input positions based on what they need.

!!! info "Connection to Modern LLMs"

    Modern LLMs like GPT-4 learn rich attention patterns:

    - **Syntactic heads**: Track grammatical relationships
    - **Coreference heads**: Connect pronouns to referents
    - **Positional heads**: Attend to nearby tokens
    - **Induction heads**: Copy patterns from context

    These emerge automatically from training on language.

## Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| QK^T | O(n²d_k) | Matrix multiplication |
| Softmax | O(n²) | Per row |
| Attention × V | O(n²d_v) | Matrix multiplication |
| **Total** | **O(n²d)** | Quadratic in sequence length! |

The O(n²) complexity is attention's main limitation. For n=100,000 tokens, we'd need 10 billion operations just for attention. This motivates efficient attention variants (sparse attention, linear attention, etc.).

## Implementation

```python
def dot_product_attention(Q, K, V):
    """
    Basic dot-product attention (without scaling).

    Args:
        Q: Query matrix [n, d_k]
        K: Key matrix [n, d_k]
        V: Value matrix [n, d_v]

    Returns:
        Output matrix [n, d_v]
    """
    # Compute attention scores
    scores = Q @ K.T  # [n, n]

    # Convert to probabilities
    attention_weights = softmax(scores, axis=-1)  # [n, n]

    # Weighted sum of values
    output = attention_weights @ V  # [n, d_v]

    return output, attention_weights
```

## Why This Works

Attention learns to:

1. **Encode queries**: What information does each position need?
2. **Encode keys**: What information does each position provide?
3. **Match them**: Which keys satisfy which queries?
4. **Aggregate**: Combine relevant information via values

All of this is **learned end-to-end** through backpropagation. The model discovers useful attention patterns automatically.

## The Missing Piece: Scaling

There's one problem we haven't addressed. The next section shows why we need to divide by √d_k:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

## Exercises

1. **Hand compute**: For Q = [[1,0], [0,1]], K = V = [[1,1], [1,0]], compute attention output.

2. **Visualize**: Plot attention weights for a sentence. What patterns do you see?

3. **Complexity**: For n=1000, d=512, how many floating-point operations is attention?

4. **Separate projections**: What happens if W^Q = W^K? When might this be desirable?

5. **Attention entropy**: Compute the entropy of attention weights. What does high/low entropy mean?

## Summary

| Concept | Definition | Role |
|---------|------------|------|
| Query (Q) | q = xW^Q | What am I looking for? |
| Key (K) | k = xW^K | What do I contain? |
| Value (V) | v = xW^V | What should I return? |
| Score | q·k | Similarity between query and key |
| Weight | softmax(scores) | How much to attend |
| Output | Σ α_j v_j | Weighted combination of values |

**Key takeaway**: Dot-product attention computes relevance between all position pairs using learned query and key representations, then retrieves a weighted combination of value representations. This enables dynamic, content-based context selection that can be learned end-to-end.

→ **Next**: [Section 5.3: Scaled Attention](03-scaled-attention.md)

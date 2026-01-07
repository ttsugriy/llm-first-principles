# Stage 5: Attention — The Key to Modern LLMs

*Estimated reading time: 3-4 hours | Prerequisites: Stages 1-4*

## Overview

Attention is the single most important innovation in modern language models. This mechanism allows models to dynamically focus on relevant parts of the input, overcoming the fundamental limitations of fixed-context approaches.

**The central question**: How can a model learn which parts of the input are relevant for each output?

## What You'll Learn

By the end of this stage, you'll understand:

1. **Why attention is necessary** — The limitations of fixed-context models
2. **Dot-product attention** — The mathematical foundation
3. **Scaled attention** — Why we divide by √d
4. **Self-attention** — A sequence attending to itself
5. **Multi-head attention** — Multiple perspectives simultaneously
6. **Positional encoding** — Injecting position information
7. **Causal masking** — Preventing future information leakage
8. **The Transformer block** — Putting it all together

## Sections

| Section | Topic | Key Concepts |
|---------|-------|--------------|
| 5.1 | [The Attention Problem](01-attention-problem.md) | Fixed context limits, information bottleneck, alignment |
| 5.2 | [Dot-Product Attention](02-dot-product-attention.md) | Query, Key, Value, attention weights, softmax |
| 5.3 | [Scaled Attention](03-scaled-attention.md) | Variance analysis, √d scaling, numerical stability |
| 5.4 | [Self-Attention](04-self-attention.md) | Sequence-to-sequence, learned projections |
| 5.5 | [Multi-Head Attention](05-multi-head.md) | Multiple heads, concatenation, different subspaces |
| 5.6 | [Positional Encoding](06-positional-encoding.md) | Sinusoidal, learned, relative positions, RoPE |
| 5.7 | [Causal Masking](07-causal-masking.md) | Autoregressive models, masked attention |
| 5.8 | [Implementation](08-implementation.md) | Building attention from scratch |

## Key Mathematical Results

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:
- Q (Query): What am I looking for? [n × d_k]
- K (Key): What do I contain? [m × d_k]
- V (Value): What should I return? [m × d_v]
- d_k: Key/query dimension

### Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

We'll derive these formulas step by step and explain why each component is necessary.

## Connection to Modern LLMs

This stage covers the exact attention mechanism used in:

- **GPT-4, Claude, LLaMA**: Decoder-only transformers with causal self-attention
- **BERT**: Encoder with bidirectional self-attention
- **T5**: Encoder-decoder with cross-attention

Understanding attention is essential for understanding how these models work.

## Code Preview

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention from scratch.

    Args:
        Q: Query matrix [batch, seq_len, d_k]
        K: Key matrix [batch, seq_len, d_k]
        V: Value matrix [batch, seq_len, d_v]
        mask: Optional attention mask

    Returns:
        Attention output [batch, seq_len, d_v]
    """
    d_k = Q.shape[-1]

    # Compute attention scores
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    # Apply mask (for causal attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax to get attention weights
    attention_weights = softmax(scores, dim=-1)

    # Weighted sum of values
    return attention_weights @ V
```

## Prerequisites

Before starting this stage, ensure you understand:

- [ ] Neural network forward/backward passes (Stage 2-3)
- [ ] Matrix multiplication and linear algebra
- [ ] Softmax function and its properties
- [ ] Cross-entropy loss (Stage 3)
- [ ] Optimization basics (Stage 4)

## The Big Picture

```
Stage 1: Markov         → Fixed context, counting
Stage 2: Autograd       → Learning via gradients
Stage 3: Neural LM      → Continuous representations
Stage 4: Optimization   → Making learning work
Stage 5: Attention      → Dynamic context ← YOU ARE HERE
Stage 6: Transformers   → Full architecture
```

Attention solves the fundamental limitation of all previous approaches: fixed context windows. With attention, a model can dynamically decide which parts of a potentially unlimited context are relevant.

## Historical Context

- **2014**: Bahdanau et al. introduce attention for machine translation
- **2015**: Luong et al. simplify attention mechanisms
- **2017**: Vaswani et al. publish "Attention Is All You Need" — the Transformer
- **2018-present**: GPT, BERT, and the age of large language models

## Exercises Preview

1. **Implement attention**: Build scaled dot-product attention from scratch
2. **Visualize patterns**: Plot attention weights on sample text
3. **Multi-head ablation**: What happens with 1 head vs 8 vs 64?
4. **Positional encoding**: Compare sinusoidal vs learned positions
5. **Causal masking**: Implement and verify no future information leakage

## Begin

→ [Start with Section 5.1: The Attention Problem](01-attention-problem.md)

# Section 5.1: The Attention Problem — Why We Need a New Approach

*Reading time: 15 minutes | Difficulty: ★★☆☆☆*

Before diving into attention mechanisms, we need to understand why they were invented. This section examines the fundamental limitations of fixed-context models and motivates the need for dynamic, content-based context selection.

## The Fixed Context Problem

Recall our neural language model from Stage 3:

```
Input: last k characters → Embedding → Hidden layers → Next character
```

This works, but has a fatal flaw: **k is fixed**.

### Why Fixed Context Fails

Consider translating: "The cat sat on the mat because **it** was tired."

What does "it" refer to? The cat. But:
- "it" appears at position 10
- "cat" appears at position 2
- With context k=4, we only see "because it was tired"

We've lost the referent! The model has no way to connect "it" back to "cat".

```
Position:  1    2    3   4   5   6   7    8       9   10  11   12
Words:    The  cat  sat on the mat because it    was tired  .
                ↑                           ↑
                └──────── Reference ────────┘
                     (8 positions apart!)
```

### The Information Bottleneck

Even if we increase k, we hit another problem: the **hidden state bottleneck**.

In recurrent models (RNNs, LSTMs):
```
h_t = f(h_{t-1}, x_t)
```

All information from the past must squeeze through a fixed-size vector h. As the sequence grows, earlier information gets compressed and lost.

```
Input:  x₁ → x₂ → x₃ → ... → x₁₀₀₀ → x₁₀₀₁

Hidden: h₁ → h₂ → h₃ → ... → h₁₀₀₀ → h₁₀₀₁
         ↑
    Information about x₁ is
    almost entirely lost by h₁₀₀₁
```

!!! info "Connection to Modern LLMs"

    Modern LLMs like GPT-4 and Claude can handle contexts of 100,000+ tokens. This is only possible because attention allows direct connections between any two positions, bypassing the bottleneck problem entirely.

## The Alignment Problem

Machine translation highlighted another issue: **alignment**.

Consider English → French:
```
English: The    black  cat   sat
French:  Le     chat   noir  s'est assis
              ↑      ↑
           These correspond!
```

The word order differs between languages. A fixed left-to-right model struggles because:
- "black" (position 2) maps to "noir" (position 3)
- "cat" (position 3) maps to "chat" (position 2)

We need a way for the model to **look back** at relevant source words when generating each target word.

## What We Want

An ideal mechanism would:

1. **Look at all positions**: Not just the last k
2. **Select dynamically**: Different outputs need different inputs
3. **Be differentiable**: So we can learn it with gradient descent
4. **Scale efficiently**: Handle long sequences

This is exactly what attention provides.

## The Attention Intuition

Think of attention as a **soft database lookup**:

| Component | Database Analogy | Attention |
|-----------|------------------|-----------|
| Query | What am I looking for? | Current position's question |
| Key | What does each entry contain? | Each position's identifier |
| Value | What should I return? | Each position's content |
| Lookup | Find matching entries | Compute similarity scores |
| Result | Return matched values | Weighted sum of values |

The key insight: instead of a hard lookup (return one result), we do a **soft lookup** (return a weighted combination of all results, with higher weights for better matches).

## A Simple Example

Suppose we're processing "The cat sat on the mat" and we're at "sat":

**Query**: "What did the action?" (looking for the subject)

**Keys**: Each word provides a key describing itself:
- "The" → determiner
- "cat" → noun, animate
- "sat" → verb
- ...

**Attention**: The query "looking for subject" matches best with "cat" (noun, animate), so "cat" gets high attention weight.

**Value**: We retrieve information from "cat" to help understand "sat".

```
Query (sat):  "Who did this action?"
                    ↓ match
Keys:    The   cat   sat   on   the   mat
         0.05  0.80  0.05  0.03 0.04  0.03  ← attention weights
                ↑
         Best match!
```

## Historical Development

### Before Attention: Encoder-Decoder

Early sequence-to-sequence models (Sutskever et al., 2014):

```
Encoder: x₁ → x₂ → x₃ → [context vector c]
                              ↓
Decoder:                     c → y₁ → y₂ → y₃
```

**Problem**: Everything must squeeze through c, a single fixed vector.

### The Attention Solution (Bahdanau et al., 2014)

Instead of a single context vector, let the decoder look at all encoder states:

```
Encoder: x₁ → x₂ → x₃   (keep all hidden states)
          ↑    ↑    ↑
          └────┼────┼──── attention weights
               ↓
Decoder:      y₁ → y₂ → y₃
              ↑
         weighted sum of encoder states
```

At each decoder step:
1. Compute attention weights over all encoder states
2. Take weighted sum to get context
3. Use context to generate output

This was the breakthrough that enabled modern neural machine translation.

## Complexity Comparison

| Approach | Context Access | Memory | Path Length |
|----------|---------------|--------|-------------|
| Markov (k-gram) | Last k tokens | O(k) | 1 |
| RNN | All (in theory) | O(1) per step | O(n) |
| Attention | All (directly) | O(n) | O(1) |

**Path length** is crucial: how many steps must information travel?
- In RNNs, info from position 1 takes n-1 steps to reach position n
- With attention, any position can directly access any other position

This is why attention enables learning long-range dependencies.

## The Attention Equation Preview

We'll derive this fully in the next section, but here's the core formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Each part serves a purpose:
- **QK^T**: Compute similarity between queries and keys
- **softmax**: Convert similarities to probabilities (sum to 1)
- **√d_k**: Scaling factor (we'll explain why)
- **× V**: Weighted sum of values

## Why This Matters

Attention is not just an improvement—it's a paradigm shift:

| Fixed Context | Attention |
|--------------|-----------|
| Predetermined connections | Learned connections |
| Same context for all outputs | Different context per output |
| Information bottleneck | Direct access |
| Hard to parallelize (RNNs) | Fully parallelizable |

The Transformer architecture (next stage) builds entirely on attention, removing recurrence completely. This enabled:
- Massive parallelization during training
- Scaling to billions of parameters
- The modern LLM revolution

## Exercises

1. **Context limitation**: Take a paragraph and predict each word using only the previous 4 words. Note where you'd need more context.

2. **Alignment analysis**: For a sentence pair in two languages, manually mark which source words each target word depends on.

3. **Bottleneck experiment**: If you could only pass 5 numbers to summarize a paragraph, what would you choose? Feel the compression.

4. **Reference resolution**: Find 5 examples where pronouns refer to words more than 10 positions away.

## Summary

| Concept | Definition | Why It Matters |
|---------|------------|----------------|
| Fixed context | Only see last k tokens | Limits long-range understanding |
| Information bottleneck | Fixed-size state | Compresses and loses information |
| Alignment | Correspondence between positions | Word order differs across tasks |
| Attention | Dynamic context selection | Solves all the above problems |

**Key takeaway**: Fixed-context models fundamentally cannot handle long-range dependencies, variable alignment, or preserve information across long sequences. Attention provides a learnable mechanism for dynamic, content-based context selection that directly connects any two positions.

→ **Next**: [Section 5.2: Dot-Product Attention](02-dot-product-attention.md)

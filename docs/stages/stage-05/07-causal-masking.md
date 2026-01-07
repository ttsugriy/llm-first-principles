# Section 5.7: Causal Masking — Preventing Future Information Leakage

*Reading time: 15 minutes | Difficulty: ★★★☆☆*

Language models predict the next token given previous tokens. During training, we must prevent the model from "cheating" by looking at future tokens. Causal masking enforces this constraint in attention.

## The Autoregressive Constraint

### The Problem

During training, we have the complete sequence:

```
"The cat sat on the mat"

When predicting "sat":
✓ Can see: "The cat"
✗ Cannot see: "sat on the mat" (the future!)
```

But in standard self-attention, every position attends to every other position—including future ones.

### Why This Matters

If the model could see future tokens during training:

```
Training: "The cat sat" → predicting "sat"
          Model peeks at position 3, sees "sat" → trivially outputs "sat"

Inference: "The cat ___" → predicting next word
           Position 3 doesn't exist yet → model has never learned to predict!
```

Training wouldn't teach the model anything useful.

## The Solution: Causal Mask

Mask out attention to future positions:

$$\text{mask}_{ij} = \begin{cases} 0 & \text{if } j \leq i \text{ (past or present)} \\ -\infty & \text{if } j > i \text{ (future)} \end{cases}$$

Apply this mask to attention scores before softmax:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{$QK^T$}{\sqrt{d_k}} + \text{mask}\right) V$$

### How Masking Works

```
Scores before masking (position attends to all):
           pos1  pos2  pos3  pos4
    pos1 [  1.2   0.8   0.5   0.3 ]
    pos2 [  0.9   1.5   0.7   0.4 ]
    pos3 [  0.6   0.8   1.3   0.6 ]
    pos4 [  0.4   0.5   0.7   1.1 ]

Causal mask:
           pos1  pos2  pos3  pos4
    pos1 [  0    -∞    -∞    -∞  ]
    pos2 [  0     0    -∞    -∞  ]
    pos3 [  0     0     0    -∞  ]
    pos4 [  0     0     0     0  ]

Scores after masking:
           pos1  pos2  pos3  pos4
    pos1 [  1.2   -∞    -∞    -∞  ]
    pos2 [  0.9   1.5   -∞    -∞  ]
    pos3 [  0.6   0.8   1.3   -∞  ]
    pos4 [  0.4   0.5   0.7   1.1 ]

After softmax:
           pos1  pos2  pos3  pos4
    pos1 [ 1.00  0.00  0.00  0.00 ]  ← pos1 only sees itself
    pos2 [ 0.35  0.65  0.00  0.00 ]  ← pos2 sees pos1, pos2
    pos3 [ 0.21  0.26  0.53  0.00 ]  ← pos3 sees pos1, pos2, pos3
    pos4 [ 0.16  0.18  0.22  0.44 ]  ← pos4 sees all (no future)
```

The -∞ values become 0 after softmax ($e^{-∞}$ = 0).

## Causal vs. Bidirectional Attention

### Causal (Decoder-only)

```
Used by: GPT, LLaMA, Claude

Pattern:
    ████░░░░░░
    ████████░░░
    █████████░░
    ██████████
    ███████████

Each position sees only past positions.
Good for: generation, autoregressive modeling
```

### Bidirectional (Encoder-only)

```
Used by: BERT, RoBERTa

Pattern:
    ████████████
    ████████████
    ████████████
    ████████████

Each position sees all positions.
Good for: understanding, classification, NLU tasks
```

### Encoder-Decoder

```
Used by: T5, BART, original Transformer

Encoder: Bidirectional (see all)
Decoder: Causal (see past only) + cross-attention to encoder

Pattern in decoder:
    ████░░░░ + encoder
    ████████ + encoder
```

## Implementation

### Basic Causal Mask

```python
import numpy as np

def create_causal_mask(seq_len):
    """
    Create causal attention mask.

    Args:
        seq_len: Sequence length

    Returns:
        Mask of shape [seq_len, seq_len]
        0 for allowed positions, -inf for masked positions
    """
    # Create upper triangular matrix of ones (future positions)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)

    # Convert to -inf
    mask = mask * float('-inf')

    return mask


def causal_attention(Q, K, V):
    """
    Scaled dot-product attention with causal masking.

    Args:
        Q: Queries [n, d_k]
        K: Keys [n, d_k]
        V: Values [n, d_v]

    Returns:
        Output [n, d_v], attention weights [n, n]
    """
    n = Q.shape[0]
    d_k = Q.shape[-1]

    # Compute scaled scores
    scores = Q @ K.T / np.sqrt(d_k)  # [n, n]

    # Apply causal mask
    mask = create_causal_mask(n)
    scores = scores + mask

    # Softmax (handles -inf correctly)
    def stable_softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        # Replace -inf with large negative for numerical stability
        x_safe = np.where(np.isinf(x), -1e9, x)
        x_max = np.max(x_safe, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        exp_x = np.where(np.isinf(x), 0, exp_x)  # -inf becomes 0
        return exp_x / (exp_x.sum(axis=axis, keepdims=True) + 1e-10)

    attention_weights = stable_softmax(scores, axis=-1)

    # Weighted sum of values
    output = attention_weights @ V

    return output, attention_weights
```

### Visualization

```python
def visualize_causal_mask(seq_len):
    """Print causal attention pattern."""
    for i in range(seq_len):
        row = ['█' if j <= i else '░' for j in range(seq_len)]
        print(' '.join(row))

# Example
print("Causal mask for sequence length 5:")
visualize_causal_mask(5)
# Output:
# █ ░ ░ ░ ░
# █ █ ░ ░ ░
# █ █ █ ░ ░
# █ █ █ █ ░
# █ █ █ █ █
```

## Training with Causal Masking

### Parallel Training

Even with causal masking, we train on all positions in parallel:

```python
def causal_lm_loss(model, tokens):
    """
    Compute language modeling loss with causal attention.

    Args:
        model: Language model with causal attention
        tokens: Input sequence [batch, seq_len]

    Returns:
        Loss scalar
    """
    # Forward pass computes all positions in parallel
    # Causal masking prevents cheating
    logits = model(tokens[:, :-1])  # [batch, seq_len-1, vocab]

    # Each position predicts the next token
    targets = tokens[:, 1:]  # [batch, seq_len-1]

    # Cross-entropy loss
    loss = cross_entropy(logits, targets)

    return loss
```

### Why Parallel Training Works

```
Input:  "The cat sat on"
Target: "cat sat on the"

Position 0: sees "The"        → predicts "cat"   ✓
Position 1: sees "The cat"    → predicts "sat"   ✓
Position 2: sees "The cat sat" → predicts "on"   ✓
Position 3: sees "The cat sat on" → predicts "the" ✓

All predictions happen simultaneously!
Causal mask ensures each sees only valid context.
```

This is why Transformers train much faster than RNNs—no sequential dependency during training.

## Inference with Causal Masking

### The KV-Cache Optimization

During generation, we only need to compute the new token's attention:

```python
class CausalAttentionWithCache:
    """Causal attention with KV caching for efficient inference."""

    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Initialize projections
        self.W_Q = np.random.randn(d_model, d_model) * 0.02
        self.W_K = np.random.randn(d_model, d_model) * 0.02
        self.W_V = np.random.randn(d_model, d_model) * 0.02
        self.W_O = np.random.randn(d_model, d_model) * 0.02

        # Cache for K, V
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, use_cache=False):
        """
        Forward pass with optional caching.

        Args:
            x: Input [seq_len, d_model] or [1, d_model] if using cache
            use_cache: Whether to use/update cache
        """
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        if use_cache and self.k_cache is not None:
            # Append new K, V to cache
            K = np.concatenate([self.k_cache, K], axis=0)
            V = np.concatenate([self.v_cache, V], axis=0)

        if use_cache:
            self.k_cache = K
            self.v_cache = V

        # Compute attention (new position attends to all cached positions)
        d_k = Q.shape[-1]
        scores = Q @ K.T / np.sqrt(d_k)

        # Apply causal mask
        if not use_cache:  # Full sequence mode
            mask = create_causal_mask(Q.shape[0])
            scores = scores + mask

        # Softmax and output
        weights = stable_softmax(scores)
        output = weights @ V
        output = output @ self.W_O

        return output

    def reset_cache(self):
        """Clear the KV cache."""
        self.k_cache = None
        self.v_cache = None
```

### Generation Loop

```python
def generate(model, prompt_tokens, max_new_tokens):
    """
    Autoregressive generation with causal model.

    Args:
        model: Language model with causal attention
        prompt_tokens: Initial tokens [seq_len]
        max_new_tokens: Number of tokens to generate

    Returns:
        Generated sequence
    """
    tokens = list(prompt_tokens)

    # Reset cache
    model.reset_cache()

    # Process prompt
    prompt_tensor = np.array(prompt_tokens)
    _ = model(prompt_tensor, use_cache=True)

    # Generate new tokens
    for _ in range(max_new_tokens):
        # Only process last token (cache has the rest)
        last_token = np.array([tokens[-1]])
        logits = model(last_token, use_cache=True)

        # Sample next token
        next_token = sample_from_logits(logits[-1])
        tokens.append(next_token)

        if next_token == EOS_TOKEN:
            break

    return tokens
```

## Variants and Extensions

### Sliding Window Attention

For very long sequences, limit attention to a window:

```python
def sliding_window_causal_mask(seq_len, window_size):
    """
    Causal mask with limited window.

    Position i attends to positions max(0, i-window_size+1) to i.
    """
    mask = np.full((seq_len, seq_len), float('-inf'))

    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start:i+1] = 0

    return mask
```

Used by: Mistral, some efficient Transformer variants.

### Block-Sparse Attention

Combine local windows with sparse global attention:

```
Pattern:
    ████░░░░████  (local + first block)
    ████████░░░░  (local window)
    ░░░░████████  (local window)
    ████░░░░████  (local + last block)
```

Used by: Longformer, BigBird.

### Prefix-LM

Allow bidirectional attention on a prefix, causal on the rest:

```
"[Context] The answer is"
 ↑ bidirectional ↑  ↑ causal ↑
```

Used by: T5 for some tasks, instruction-following models.

!!! info "Connection to Modern LLMs"

    Modern LLMs use various attention patterns:

    - **GPT-4, Claude**: Standard causal (assumed)
    - **LLaMA 2**: Standard causal
    - **Mistral**: Sliding window causal
    - **GPT-4 Turbo**: Likely hierarchical patterns for long context

    The basic causal constraint is universal for autoregressive generation.

## Numerical Stability

### The -inf Problem

When implementing masking:

```python
# Problem: softmax of row with all -inf
scores = np.array([float('-inf'), float('-inf'), float('-inf')])
exp_scores = np.exp(scores)  # [0, 0, 0]
softmax = exp_scores / exp_scores.sum()  # 0/0 = nan!
```

### Solution: Add Small Epsilon

```python
def stable_masked_softmax(scores, mask):
    """Numerically stable softmax with mask."""
    # Apply mask
    masked_scores = scores + mask

    # Subtract max for stability (ignoring -inf)
    finite_scores = np.where(np.isinf(masked_scores), -1e9, masked_scores)
    max_score = finite_scores.max(axis=-1, keepdims=True)

    exp_scores = np.exp(masked_scores - max_score)
    exp_scores = np.where(np.isinf(masked_scores), 0, exp_scores)

    sum_exp = exp_scores.sum(axis=-1, keepdims=True) + 1e-10
    return exp_scores / sum_exp
```

## Exercises

1. **Implement causal mask**: Write a function that creates the mask matrix.

2. **Verify no leakage**: After applying causal masking, verify that gradients don't flow from future to past.

3. **Sliding window**: Implement sliding window causal attention with window size k.

4. **KV-cache speedup**: Compare generation time with and without KV caching.

5. **Prefix-LM**: Implement a mask that's bidirectional for first k tokens, causal for rest.

## Summary

| Concept | Definition | Purpose |
|---------|------------|---------|
| Causal mask | -∞ for future positions | Prevent seeing future |
| Autoregressive | Predict next given past | Language modeling |
| KV-cache | Store computed K, V | Efficient inference |
| Sliding window | Limit attention span | Long sequence efficiency |

**Key takeaway**: Causal masking prevents positions from attending to future positions, enforcing the autoregressive constraint essential for language modeling. By adding -∞ to future positions before softmax, we ensure zero attention weight on future tokens. This allows parallel training on all positions while ensuring the model learns to predict without access to future information.

→ **Next**: [Section 5.8: Implementation](08-implementation.md)

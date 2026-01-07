# Stage 5 Exercises

## Conceptual Questions

### Exercise 5.1: Attention Intuition
Consider the sentence: "The cat sat on the mat because it was tired."

**a)** When predicting "tired", which words should receive high attention?
**b)** What does "it" most likely refer to? How would attention help resolve this?
**c)** Why is this hard for a fixed-window model?

### Exercise 5.2: Scaling Factor
In scaled dot-product attention, we divide by √d_k.

**a)** If d_k = 64, what is √d_k?
**b)** Without scaling, what happens to dot products as d_k increases?
**c)** How does this affect the softmax distribution?

### Exercise 5.3: Multi-Head Purpose
Why use 8 heads of dimension 64 instead of 1 head of dimension 512?

**a)** What can multiple heads learn that a single head cannot?
**b)** Is the parameter count the same?
**c)** Give an example of different "types" of attention different heads might learn.

### Exercise 5.4: Causal Masking
For sequence ["The", "cat", "sat"]:

**a)** Draw the attention matrix (which positions can attend to which?)
**b)** What value goes in the masked positions before softmax?
**c)** Why is causal masking necessary for language modeling?

---

## Implementation Exercises

### Exercise 5.5: Dot-Product Attention
Implement basic attention:

```python
def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Args:
        Q: queries [batch, seq_q, d_k]
        K: keys [batch, seq_k, d_k]
        V: values [batch, seq_k, d_v]

    Returns:
        output [batch, seq_q, d_v]
    """
    # scores = Q @ K^T / sqrt(d_k)
    # weights = softmax(scores)
    # output = weights @ V
    # TODO
    pass
```

### Exercise 5.6: Causal Mask
Implement causal masking:

```python
def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create a mask where position i can only attend to positions <= i.

    Returns:
        mask [seq_len, seq_len] with 0 for allowed, -inf for blocked
    """
    # TODO
    pass

def masked_attention(Q, K, V, mask):
    """Apply attention with mask"""
    # TODO: add mask to scores before softmax
    pass
```

### Exercise 5.7: Multi-Head Attention
Implement full multi-head attention:

```python
class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int):
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02

    def forward(self, x):
        """
        1. Project to Q, K, V
        2. Split into heads
        3. Apply attention per head
        4. Concatenate and project
        """
        # TODO
        pass
```

### Exercise 5.8: Positional Encoding
Implement sinusoidal positional encoding:

```python
def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Returns:
        encoding [seq_len, d_model]
    """
    # TODO
    pass
```

---

## Challenge Exercises

### Exercise 5.9: Attention Visualization
Train a small attention model and visualize the attention patterns:

**a)** Create a heatmap of attention weights
**b)** Do different heads focus on different positions?
**c)** Does one head learn to attend to the previous word?

### Exercise 5.10: Attention Backward Pass
Implement the backward pass through attention:

```python
def attention_backward(Q, K, V, grad_output):
    """
    Given gradient of loss w.r.t. output, compute gradients for Q, K, V.

    Returns:
        (grad_Q, grad_K, grad_V)
    """
    # Forward values (from cache)
    scores = Q @ K.T / np.sqrt(d_k)
    attn = softmax(scores)
    # output = attn @ V

    # Backward:
    # TODO: compute gradients
    pass
```

### Exercise 5.11: Relative Positional Encoding
Research and implement relative positional encoding (as used in Transformer-XL):

```python
def relative_attention(Q, K, V, pos_embed):
    """
    Attention with relative position representations.

    Instead of absolute position, encode relative distance between tokens.
    """
    # TODO: Advanced exercise
    pass
```

---

## Solutions

Solutions are available in `code/stage-05/solutions/`.

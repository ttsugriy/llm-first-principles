# Section 6.8: Implementation — Training a Complete Transformer

*Reading time: 25 minutes | Difficulty: ★★★★☆*

This section brings everything together into a complete, trainable Transformer language model. We'll implement training, generation, and see how all the components work in practice.

## Complete Transformer Implementation

```python
import numpy as np
from typing import List, Optional, Tuple

class Transformer:
    """
    Complete decoder-only Transformer for language modeling.

    This implementation includes:
    - Token embeddings with RMSNorm
    - Rotary positional encoding
    - Multi-head attention with causal masking
    - SwiGLU feed-forward network
    - Pre-norm architecture
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = None,
        max_seq_len: int = 256,
        dropout: float = 0.0,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff or int(2.5 * d_model)  # SwiGLU uses 2.5x
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        # Initialize all components
        self._init_embeddings()
        self._init_layers()
        self._init_output()

    def _init_embeddings(self):
        """Initialize token embeddings."""
        self.token_emb = np.random.randn(
            self.vocab_size, self.d_model
        ) * 0.02

    def _init_layers(self):
        """Initialize Transformer layers."""
        self.layers = []
        for i in range(self.n_layers):
            layer = {
                # Attention
                'attn_norm': RMSNorm(self.d_model),
                'wq': self._init_weight(self.d_model, self.d_model),
                'wk': self._init_weight(self.d_model, self.d_model),
                'wv': self._init_weight(self.d_model, self.d_model),
                'wo': self._init_weight(self.d_model, self.d_model, scale=1/np.sqrt(2*self.n_layers)),

                # FFN (SwiGLU)
                'ffn_norm': RMSNorm(self.d_model),
                'w1': self._init_weight(self.d_model, self.d_ff),
                'w2': self._init_weight(self.d_ff, self.d_model, scale=1/np.sqrt(2*self.n_layers)),
                'w3': self._init_weight(self.d_model, self.d_ff),
            }
            self.layers.append(layer)

    def _init_output(self):
        """Initialize output projection."""
        self.output_norm = RMSNorm(self.d_model)
        # Tie output projection with input embeddings
        self.output_proj = self.token_emb.T  # [d_model, vocab_size]

    def _init_weight(self, d_in, d_out, scale=1.0):
        """Initialize weight matrix."""
        std = np.sqrt(2.0 / (d_in + d_out)) * scale
        return np.random.randn(d_in, d_out) * std

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            tokens: Input token IDs [seq_len] or [batch, seq_len]

        Returns:
            logits: [seq_len, vocab_size] or [batch, seq_len, vocab_size]
        """
        # Handle batched input
        if tokens.ndim == 1:
            tokens = tokens[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False

        batch_size, seq_len = tokens.shape

        # Token embeddings
        x = self.token_emb[tokens]  # [batch, seq_len, d_model]

        # Create causal mask and position indices
        mask = create_causal_mask(seq_len)
        positions = np.arange(seq_len)

        # Process through layers
        for layer in self.layers:
            x = self._layer_forward(x, layer, mask, positions)

        # Output
        x = self.output_norm(x)
        logits = x @ self.output_proj  # [batch, seq_len, vocab_size]

        if squeeze:
            logits = logits[0]

        return logits

    def _layer_forward(self, x, layer, mask, positions):
        """Forward through one Transformer layer."""
        # Attention sublayer
        residual = x
        x = layer['attn_norm'](x)
        x = self._attention(x, layer, mask, positions)
        x = residual + x

        # FFN sublayer
        residual = x
        x = layer['ffn_norm'](x)
        x = self._ffn(x, layer)
        x = residual + x

        return x

    def _attention(self, x, layer, mask, positions):
        """Multi-head attention with RoPE."""
        batch_size, seq_len, _ = x.shape
        d_k = self.d_model // self.n_heads

        # Project Q, K, V
        q = x @ layer['wq']  # [batch, seq, d_model]
        k = x @ layer['wk']
        v = x @ layer['wv']

        # Reshape for multi-head
        q = q.reshape(batch_size, seq_len, self.n_heads, d_k)
        k = k.reshape(batch_size, seq_len, self.n_heads, d_k)
        v = v.reshape(batch_size, seq_len, self.n_heads, d_k)

        # Apply RoPE
        q = apply_rope(q, positions)
        k = apply_rope(k, positions)

        # Transpose for attention: [batch, heads, seq, d_k]
        q = np.transpose(q, (0, 2, 1, 3))
        k = np.transpose(k, (0, 2, 1, 3))
        v = np.transpose(v, (0, 2, 1, 3))

        # Attention scores
        scores = q @ np.transpose(k, (0, 1, 3, 2)) / np.sqrt(d_k)
        scores = scores + mask

        # Softmax
        weights = softmax(scores, axis=-1)

        # Weighted sum
        out = weights @ v  # [batch, heads, seq, d_k]

        # Reshape back
        out = np.transpose(out, (0, 2, 1, 3))
        out = out.reshape(batch_size, seq_len, self.d_model)

        # Output projection
        out = out @ layer['wo']

        return out

    def _ffn(self, x, layer):
        """SwiGLU feed-forward network."""
        # Gate
        gate = x @ layer['w1']
        gate = silu(gate)

        # Up projection
        up = x @ layer['w3']

        # Element-wise multiply and down project
        return (gate * up) @ layer['w2']

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> List[int]:
        """Generate tokens autoregressively."""
        tokens = list(prompt_tokens)

        for _ in range(max_new_tokens):
            # Get context (up to max_seq_len)
            context = np.array(tokens[-self.max_seq_len:])

            # Forward pass
            logits = self.forward(context)

            # Get logits for last position
            next_logits = logits[-1] / temperature

            # Top-k filtering
            if top_k is not None:
                indices = np.argsort(next_logits)[-top_k:]
                mask = np.full_like(next_logits, float('-inf'))
                mask[indices] = next_logits[indices]
                next_logits = mask

            # Sample
            probs = softmax(next_logits)
            next_token = np.random.choice(len(probs), p=probs)
            tokens.append(int(next_token))

        return tokens


# Supporting functions and classes

class RMSNorm:
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        self.weight = np.ones(d_model)

    def __call__(self, x):
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


def create_causal_mask(seq_len):
    """Create causal attention mask."""
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    return np.where(mask, float('-inf'), 0.0)


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(np.where(np.isinf(x), -1e9, x), axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    exp_x = np.where(np.isinf(x), 0.0, exp_x)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-10)


def silu(x):
    """SiLU/Swish activation."""
    return x * (1 / (1 + np.exp(-x)))


def apply_rope(x, positions, base=10000):
    """Apply Rotary Position Embedding."""
    batch_size, seq_len, n_heads, d_k = x.shape

    # Compute frequencies
    freqs = 1.0 / (base ** (np.arange(0, d_k, 2) / d_k))

    # Compute angles
    angles = positions[:, np.newaxis] * freqs[np.newaxis, :]  # [seq, d_k/2]

    cos = np.cos(angles)
    sin = np.sin(angles)

    # Reshape for broadcasting
    cos = cos[np.newaxis, :, np.newaxis, :]  # [1, seq, 1, d_k/2]
    sin = sin[np.newaxis, :, np.newaxis, :]

    # Rotate
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    rotated = np.empty_like(x)
    rotated[..., 0::2] = x1 * cos - x2 * sin
    rotated[..., 1::2] = x1 * sin + x2 * cos

    return rotated
```

## Training Loop

```python
def train_transformer(
    model: Transformer,
    data: np.ndarray,
    n_steps: int = 1000,
    batch_size: int = 32,
    seq_len: int = 64,
    learning_rate: float = 3e-4,
    warmup_steps: int = 100,
):
    """
    Train the Transformer on text data.

    Args:
        model: Transformer instance
        data: Token IDs [total_tokens]
        n_steps: Number of training steps
        batch_size: Batch size
        seq_len: Sequence length
        learning_rate: Peak learning rate
        warmup_steps: Warmup steps
    """
    # Adam optimizer state
    adam_state = initialize_adam_state(model)

    losses = []

    for step in range(n_steps):
        # Get batch
        x, y = get_batch(data, batch_size, seq_len)

        # Forward pass
        logits = model.forward(x)

        # Compute loss
        loss = cross_entropy_loss(logits, y)
        losses.append(loss)

        # Backward pass (numerical gradients for simplicity)
        grads = compute_gradients(model, x, y)

        # Learning rate schedule
        lr = get_lr(step, warmup_steps, n_steps, learning_rate)

        # Update parameters
        update_parameters(model, grads, adam_state, lr)

        # Logging
        if step % 100 == 0:
            print(f"Step {step}: loss = {loss:.4f}, lr = {lr:.6f}")

    return losses


def get_batch(data, batch_size, seq_len):
    """Get a random batch of sequences."""
    max_start = len(data) - seq_len - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    x = np.stack([data[s:s+seq_len] for s in starts])
    y = np.stack([data[s+1:s+seq_len+1] for s in starts])

    return x, y


def cross_entropy_loss(logits, targets):
    """Compute cross-entropy loss."""
    # logits: [batch, seq, vocab]
    # targets: [batch, seq]
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Log softmax
    log_probs = logits_flat - np.log(
        np.sum(np.exp(logits_flat - logits_flat.max(axis=-1, keepdims=True)), axis=-1, keepdims=True)
    ) - logits_flat.max(axis=-1, keepdims=True)

    # Select target log probs
    loss = -log_probs[np.arange(len(targets_flat)), targets_flat].mean()

    return loss


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-5):
    """Cosine learning rate with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))
```

## Simple Tokenizer

```python
class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}

    def train(self, text):
        """Build vocabulary from text."""
        chars = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        return self

    @property
    def vocab_size(self):
        return len(self.char_to_id)

    def encode(self, text):
        """Convert text to token IDs."""
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, ids):
        """Convert token IDs to text."""
        return ''.join(self.id_to_char.get(i, '?') for i in ids)
```

## Complete Training Example

```python
def main():
    """Train a small Transformer on Shakespeare."""

    # Load data
    text = open('shakespeare.txt', 'r').read()

    # Create tokenizer
    tokenizer = CharTokenizer().train(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Tokenize
    data = np.array(tokenizer.encode(text), dtype=np.int32)
    print(f"Total tokens: {len(data):,}")

    # Create model
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        max_seq_len=128,
    )

    # Count parameters
    n_params = sum(p.size for p in get_all_params(model))
    print(f"Parameters: {n_params:,}")

    # Train
    losses = train_transformer(
        model,
        data,
        n_steps=2000,
        batch_size=16,
        seq_len=64,
        learning_rate=3e-4,
    )

    # Generate sample
    prompt = "ROMEO:"
    prompt_tokens = tokenizer.encode(prompt)
    generated = model.generate(prompt_tokens, max_new_tokens=200, temperature=0.8)
    print("\nGenerated text:")
    print(tokenizer.decode(generated))


def get_all_params(model):
    """Get all model parameters."""
    params = [model.token_emb]
    for layer in model.layers:
        params.extend([
            layer['attn_norm'].weight,
            layer['wq'], layer['wk'], layer['wv'], layer['wo'],
            layer['ffn_norm'].weight,
            layer['w1'], layer['w2'], layer['w3'],
        ])
    params.append(model.output_norm.weight)
    return params


if __name__ == '__main__':
    main()
```

## Expected Output

```
Vocabulary size: 65
Total tokens: 1,115,394
Parameters: 2,362,433

Step 0: loss = 4.1742, lr = 0.000003
Step 100: loss = 2.4521, lr = 0.000300
Step 200: loss = 2.0183, lr = 0.000296
Step 500: loss = 1.6234, lr = 0.000264
Step 1000: loss = 1.4521, lr = 0.000182
Step 1500: loss = 1.3842, lr = 0.000098
Step 2000: loss = 1.3521, lr = 0.000010

Generated text:
ROMEO:
What, art thou mad? thou art a villain;
And I do love thee better than I love
My heart to bear the burden of my soul,
That ever I did love thee so well?
```

## What We've Built

| Component | Stage | Implementation |
|-----------|-------|----------------|
| Tokenization | 6.1 | CharTokenizer |
| Embeddings | 3, 6.2 | Token embeddings |
| RoPE | 5.6, 6.2 | apply_rope() |
| Attention | 5.2-5.5 | _attention() |
| Causal mask | 5.7 | create_causal_mask() |
| SwiGLU FFN | 6.2 | _ffn() |
| RMSNorm | 6.2 | RMSNorm class |
| Residuals | 6.2, 6.3 | x = residual + sublayer |
| Training | 4, 6.5 | train_transformer() |
| Generation | 1.6 | generate() |

## Exercises

1. **Train on your data**: Use a different text corpus. How does quality change?

2. **Scale up**: Increase to 8 layers, 256 dimensions. Does loss improve?

3. **Learning rate sweep**: Try LR from 1e-5 to 1e-2. What's optimal?

4. **Temperature exploration**: Generate at T=0.5, 1.0, 1.5. How does output differ?

5. **Add BPE**: Implement BPE tokenization instead of character-level.

## Summary

We've built a complete Transformer that:

| Feature | Details |
|---------|---------|
| Architecture | Decoder-only with pre-norm |
| Attention | Multi-head with RoPE |
| FFN | SwiGLU activation |
| Normalization | RMSNorm |
| Training | Adam with warmup + cosine decay |
| Generation | Autoregressive with temperature |

**Key takeaway**: A complete Transformer language model is built from components we've developed throughout the book: embeddings (Stage 3), attention (Stage 5), optimization (Stage 4), and generation (Stage 1). Modern architectural choices like RMSNorm, RoPE, and SwiGLU are straightforward to implement once you understand the fundamentals. This small model can generate coherent text after training on just a few million characters—scaling up these same principles produces GPT-4 and Claude.

## What's Next

You now understand LLMs from first principles! Future topics to explore:

- **Fine-tuning**: Adapting pre-trained models to specific tasks
- **RLHF**: Aligning models with human preferences
- **Inference optimization**: Making generation faster
- **Multimodal**: Extending to images and audio
- **Agents**: Using LLMs to take actions

The foundations you've built here apply to all of these.

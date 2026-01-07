"""
Stage 6: Complete Transformer Implementation

This module provides a complete, trainable Transformer language model
implementing modern architectural choices:
- RMSNorm (instead of LayerNorm)
- Rotary Positional Encoding (RoPE)
- SwiGLU activation in FFN
- Pre-norm architecture
- Grouped-Query Attention (optional)

All components are implemented from scratch using only NumPy.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
import sys
import os

# Add Stage 5 for reuse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage-05'))

# =============================================================================
# Utilities
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax that handles -inf from masking."""
    x_safe = np.where(np.isinf(x), -1e9, x)
    x_max = np.max(x_safe, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    exp_x = np.where(np.isinf(x), 0.0, exp_x)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-10)


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU/Swish activation: x * sigmoid(x)."""
    return x * (1 / (1 + np.exp(-np.clip(x, -500, 500))))


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def create_causal_mask(seq_len: int) -> np.ndarray:
    """Create causal attention mask."""
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    return np.where(mask, float('-inf'), 0.0)


# =============================================================================
# Normalization
# =============================================================================

class RMSNorm:
    """
    Root Mean Square Layer Normalization.

    Simpler and faster than LayerNorm, used in LLaMA.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = np.ones(dim)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight

    def parameters(self) -> List[np.ndarray]:
        return [self.weight]


class LayerNorm:
    """Standard Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = np.ones(dim)
        self.bias = np.zeros(dim)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.weight * (x - mean) / np.sqrt(var + self.eps) + self.bias

    def parameters(self) -> List[np.ndarray]:
        return [self.weight, self.bias]


# =============================================================================
# Positional Encoding
# =============================================================================

def apply_rope(
    x: np.ndarray,
    positions: np.ndarray,
    base: float = 10000.0
) -> np.ndarray:
    """
    Apply Rotary Position Embedding (RoPE).

    Args:
        x: Input tensor [batch, seq_len, n_heads, d_k]
        positions: Position indices [seq_len]
        base: Base for frequency computation

    Returns:
        Rotated tensor of same shape
    """
    d_k = x.shape[-1]

    # Compute frequencies
    freqs = 1.0 / (base ** (np.arange(0, d_k, 2) / d_k))

    # Compute angles: [seq_len, d_k/2]
    angles = positions[:, np.newaxis] * freqs[np.newaxis, :]

    cos = np.cos(angles)
    sin = np.sin(angles)

    # Reshape for broadcasting: [1, seq_len, 1, d_k/2]
    cos = cos[np.newaxis, :, np.newaxis, :]
    sin = sin[np.newaxis, :, np.newaxis, :]

    # Rotate
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    rotated = np.empty_like(x)
    rotated[..., 0::2] = x1 * cos - x2 * sin
    rotated[..., 1::2] = x1 * sin + x2 * cos

    return rotated


class SinusoidalPositionalEncoding:
    """Fixed sinusoidal positional encoding."""

    def __init__(self, max_len: int, d_model: int):
        self.encoding = self._create_encoding(max_len, d_model)

    def _create_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        pe = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def __call__(self, seq_len: int) -> np.ndarray:
        return self.encoding[:seq_len]


# =============================================================================
# Attention
# =============================================================================

class MultiHeadAttention:
    """
    Multi-head attention with optional Grouped-Query Attention (GQA).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of query heads
            n_kv_heads: Number of key/value heads (for GQA). Default: same as n_heads
            dropout: Dropout probability
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout

        assert d_model % n_heads == 0
        assert n_heads % self.n_kv_heads == 0

        # Projections
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.wq = np.random.randn(d_model, n_heads * self.d_k) * scale
        self.wk = np.random.randn(d_model, self.n_kv_heads * self.d_k) * scale
        self.wv = np.random.randn(d_model, self.n_kv_heads * self.d_k) * scale
        self.wo = np.random.randn(n_heads * self.d_k, d_model) * scale

        self.attention_weights = None

    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input [batch, seq_len, d_model]
            mask: Attention mask
            positions: Position indices for RoPE

        Returns:
            Output [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = x @ self.wq  # [batch, seq, n_heads * d_k]
        k = x @ self.wk  # [batch, seq, n_kv_heads * d_k]
        v = x @ self.wv

        # Reshape
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        k = k.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = v.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)

        # Apply RoPE if positions provided
        if positions is not None:
            q = apply_rope(q, positions)
            k = apply_rope(k, positions)

        # Expand KV heads for GQA
        if self.n_kv_heads < self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = np.repeat(k, n_rep, axis=2)
            v = np.repeat(v, n_rep, axis=2)

        # Transpose: [batch, n_heads, seq, d_k]
        q = np.transpose(q, (0, 2, 1, 3))
        k = np.transpose(k, (0, 2, 1, 3))
        v = np.transpose(v, (0, 2, 1, 3))

        # Attention scores
        scores = q @ np.transpose(k, (0, 1, 3, 2)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask

        weights = softmax(scores, axis=-1)
        self.attention_weights = weights

        # Apply attention
        out = weights @ v  # [batch, n_heads, seq, d_k]

        # Reshape and project
        out = np.transpose(out, (0, 2, 1, 3))
        out = out.reshape(batch_size, seq_len, self.n_heads * self.d_k)
        out = out @ self.wo

        return out

    def parameters(self) -> List[np.ndarray]:
        return [self.wq, self.wk, self.wv, self.wo]


# =============================================================================
# Feed-Forward Network
# =============================================================================

class SwiGLUFFN:
    """
    SwiGLU Feed-Forward Network used in LLaMA.

    FFN(x) = (SiLU(x @ W1) * (x @ W3)) @ W2
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        # LLaMA uses 2.67x multiplier (after accounting for extra W3)
        self.d_ff = d_ff or int(2.67 * d_model)

        scale1 = np.sqrt(2.0 / (d_model + self.d_ff))
        scale2 = np.sqrt(2.0 / (self.d_ff + d_model))

        self.w1 = np.random.randn(d_model, self.d_ff) * scale1  # Gate
        self.w2 = np.random.randn(self.d_ff, d_model) * scale2  # Down
        self.w3 = np.random.randn(d_model, self.d_ff) * scale1  # Up

    def __call__(self, x: np.ndarray) -> np.ndarray:
        gate = silu(x @ self.w1)
        up = x @ self.w3
        return (gate * up) @ self.w2

    def parameters(self) -> List[np.ndarray]:
        return [self.w1, self.w2, self.w3]


class GELUMLPFFN:
    """Standard GELU Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        self.d_ff = d_ff or 4 * d_model

        scale1 = np.sqrt(2.0 / (d_model + self.d_ff))
        scale2 = np.sqrt(2.0 / (self.d_ff + d_model))

        self.w1 = np.random.randn(d_model, self.d_ff) * scale1
        self.b1 = np.zeros(self.d_ff)
        self.w2 = np.random.randn(self.d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return gelu(x @ self.w1 + self.b1) @ self.w2 + self.b2

    def parameters(self) -> List[np.ndarray]:
        return [self.w1, self.b1, self.w2, self.b2]


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock:
    """
    Single Transformer block with pre-norm architecture.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        layer_idx: int = 0,
        n_layers: int = 1
    ):
        self.d_model = d_model

        # Normalization
        norm_cls = RMSNorm if use_rmsnorm else LayerNorm
        self.attn_norm = norm_cls(d_model)
        self.ffn_norm = norm_cls(d_model)

        # Attention
        self.attention = MultiHeadAttention(d_model, n_heads, n_kv_heads, dropout)

        # FFN
        if use_swiglu:
            self.ffn = SwiGLUFFN(d_model, d_ff)
        else:
            self.ffn = GELUMLPFFN(d_model, d_ff)

        # Scale residual projections
        scale = 1.0 / np.sqrt(2 * n_layers)
        self.attention.wo *= scale
        if hasattr(self.ffn, 'w2'):
            self.ffn.w2 *= scale

    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # Attention sublayer
        residual = x
        x = self.attn_norm(x)
        x = self.attention(x, mask, positions)
        x = residual + x

        # FFN sublayer
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x

    def parameters(self) -> List[np.ndarray]:
        params = []
        params.extend(self.attn_norm.parameters())
        params.extend(self.attention.parameters())
        params.extend(self.ffn_norm.parameters())
        params.extend(self.ffn.parameters())
        return params


# =============================================================================
# Complete Transformer
# =============================================================================

class Transformer:
    """
    Complete decoder-only Transformer for language modeling.

    Implements modern architectural choices from LLaMA:
    - RMSNorm
    - Rotary Positional Encoding (RoPE)
    - SwiGLU activation
    - Pre-norm architecture
    - Optional Grouped-Query Attention
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_kv_heads: Optional[int] = None,
        n_layers: int = 4,
        d_ff: Optional[int] = None,
        max_seq_len: int = 256,
        dropout: float = 0.0,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        tie_embeddings: bool = True
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.tie_embeddings = tie_embeddings

        # Token embeddings
        self.token_emb = np.random.randn(vocab_size, d_model) * 0.02

        # Transformer blocks
        self.layers = [
            TransformerBlock(
                d_model, n_heads, n_kv_heads, d_ff, dropout,
                use_swiglu, use_rmsnorm, i, n_layers
            )
            for i in range(n_layers)
        ]

        # Output
        norm_cls = RMSNorm if use_rmsnorm else LayerNorm
        self.output_norm = norm_cls(d_model)

        if not tie_embeddings:
            self.output_proj = np.random.randn(d_model, vocab_size) * 0.02
        else:
            self.output_proj = None  # Will use token_emb.T

    def __call__(self, tokens: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            tokens: Input token IDs [batch, seq_len] or [seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size] or [seq_len, vocab_size]
        """
        # Handle unbatched
        squeeze = tokens.ndim == 1
        if squeeze:
            tokens = tokens[np.newaxis, :]

        batch_size, seq_len = tokens.shape

        # Embeddings
        x = self.token_emb[tokens]

        # Create mask and positions
        mask = create_causal_mask(seq_len)
        positions = np.arange(seq_len)

        # Process through layers
        for layer in self.layers:
            x = layer(x, mask, positions)

        # Output
        x = self.output_norm(x)

        if self.tie_embeddings:
            logits = x @ self.token_emb.T
        else:
            logits = x @ self.output_proj

        if squeeze:
            logits = logits[0]

        return logits

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> List[int]:
        """
        Generate tokens autoregressively.

        Args:
            prompt_tokens: Starting token IDs
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Sample from top k tokens
            top_p: Nucleus sampling threshold

        Returns:
            Complete sequence including prompt
        """
        tokens = list(prompt_tokens)

        for _ in range(max_new_tokens):
            context = np.array(tokens[-self.max_seq_len:])
            logits = self(context)

            # Last position
            next_logits = logits[-1] / temperature

            # Top-k filtering
            if top_k is not None:
                indices = np.argsort(next_logits)[-top_k:]
                mask = np.full_like(next_logits, float('-inf'))
                mask[indices] = next_logits[indices]
                next_logits = mask

            # Top-p (nucleus) sampling
            if top_p is not None:
                sorted_indices = np.argsort(next_logits)[::-1]
                sorted_logits = next_logits[sorted_indices]
                sorted_probs = softmax(sorted_logits)
                cumsum = np.cumsum(sorted_probs)

                # Find cutoff
                cutoff_idx = np.searchsorted(cumsum, top_p)
                mask = np.full_like(next_logits, float('-inf'))
                mask[sorted_indices[:cutoff_idx + 1]] = next_logits[sorted_indices[:cutoff_idx + 1]]
                next_logits = mask

            # Sample
            probs = softmax(next_logits)
            next_token = np.random.choice(len(probs), p=probs)
            tokens.append(int(next_token))

        return tokens

    def parameters(self) -> List[np.ndarray]:
        """Return all model parameters."""
        params = [self.token_emb]
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend(self.output_norm.parameters())
        if not self.tie_embeddings:
            params.append(self.output_proj)
        return params

    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.size for p in self.parameters())


# =============================================================================
# Tokenization
# =============================================================================

class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

    def train(self, text: str) -> 'CharTokenizer':
        """Build vocabulary from text."""
        chars = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.id_to_char.get(i, '?') for i in ids)


class BPETokenizer:
    """Simple Byte-Pair Encoding tokenizer."""

    def __init__(self, vocab_size: int = 1000):
        self.target_vocab_size = vocab_size
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[bytes, int] = {}
        self.inverse_vocab: Dict[int, bytes] = {}

    def train(self, texts: List[str]) -> 'BPETokenizer':
        """Learn BPE merges from texts."""
        # Start with byte vocabulary
        for i in range(256):
            self.vocab[bytes([i])] = i
            self.inverse_vocab[i] = bytes([i])
        next_id = 256

        # Encode texts as bytes
        corpus = [list(text.encode('utf-8')) for text in texts]

        while len(self.vocab) < self.target_vocab_size:
            # Count pairs
            pair_counts: Dict[Tuple[int, int], int] = {}
            for tokens in corpus:
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            # Find most frequent
            best_pair = max(pair_counts, key=pair_counts.get)

            # Create merged token
            t1 = self.inverse_vocab[best_pair[0]]
            t2 = self.inverse_vocab[best_pair[1]]
            new_token = t1 + t2

            self.merges[best_pair] = next_id
            self.vocab[new_token] = next_id
            self.inverse_vocab[next_id] = new_token
            next_id += 1

            # Apply merge
            new_corpus = []
            for tokens in corpus:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                        new_tokens.append(self.merges[best_pair])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                new_corpus.append(new_tokens)
            corpus = new_corpus

        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        tokens = list(text.encode('utf-8'))

        for pair, merged_id in self.merges.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(merged_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def decode(self, ids: List[int]) -> str:
        result = b''
        for i in ids:
            result += self.inverse_vocab.get(i, b'?')
        return result.decode('utf-8', errors='replace')


# =============================================================================
# Training
# =============================================================================

def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Compute cross-entropy loss."""
    if logits.ndim == 3:
        batch, seq_len, vocab = logits.shape
        logits = logits.reshape(-1, vocab)
        targets = targets.reshape(-1)

    # Log softmax
    logits_max = logits.max(axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(logits - logits_max), axis=-1, keepdims=True))
    log_probs = logits - logits_max - log_sum_exp

    # Select targets
    loss = -log_probs[np.arange(len(targets)), targets].mean()
    return float(loss)


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float = 1e-5
) -> float:
    """Cosine learning rate with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss."""
    return np.exp(loss)


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate the Transformer with a simple example."""
    np.random.seed(42)

    # Create a small model
    model = Transformer(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=32
    )

    print(f"Model parameters: {model.count_parameters():,}")

    # Test forward pass
    tokens = np.array([1, 5, 10, 15, 20])
    logits = model(tokens)
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {logits.shape}")

    # Test generation
    generated = model.generate([1, 2, 3], max_new_tokens=10, temperature=1.0)
    print(f"Generated: {generated}")

    # Test loss
    targets = np.array([5, 10, 15, 20, 25])
    loss = cross_entropy_loss(logits, targets)
    print(f"Loss: {loss:.4f}")
    print(f"Perplexity: {compute_perplexity(loss):.2f}")


if __name__ == '__main__':
    demo()

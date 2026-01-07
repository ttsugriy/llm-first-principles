"""
Capstone: Trainable Transformer Language Model

This module implements a complete, trainable transformer language model
with manual backpropagation. It ties together all concepts from Stages 1-6:

- Stage 1: Language modeling as next-token prediction, perplexity
- Stage 2: Backpropagation and gradient computation
- Stage 3: Neural networks learn continuous representations
- Stage 4: Optimization with Adam
- Stage 5: Attention mechanisms
- Stage 6: Modern transformer architecture

The key insight: we implement backward() methods for each layer,
computing gradients manually (what autodiff does automatically).
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


# =============================================================================
# Utilities
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU/Swish activation: x * sigmoid(x)."""
    sig = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    return x * sig


def silu_backward(x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    """Gradient of SiLU activation."""
    sig = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    return grad_output * (sig + x * sig * (1 - sig))


def create_causal_mask(seq_len: int) -> np.ndarray:
    """Create causal attention mask."""
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    return np.where(mask, float('-inf'), 0.0)


# =============================================================================
# Parameter Container
# =============================================================================

@dataclass
class Parameter:
    """A learnable parameter with gradient storage."""
    data: np.ndarray
    grad: Optional[np.ndarray] = None

    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill(0)
        else:
            self.grad = np.zeros_like(self.data)


# =============================================================================
# RMSNorm with Backprop
# =============================================================================

class RMSNorm:
    """Root Mean Square Layer Normalization with backprop."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = Parameter(np.ones(dim))

        # Cache for backward
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: x_norm = x / rms(x) * weight"""
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        x_norm = x / rms
        out = x_norm * self.weight.data

        # Cache for backward
        self.cache = {'x': x, 'rms': rms, 'x_norm': x_norm}
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        x = self.cache['x']
        rms = self.cache['rms']
        x_norm = self.cache['x_norm']
        d = x.shape[-1]

        # Gradient w.r.t. weight
        self.weight.grad = np.sum(grad_output * x_norm, axis=tuple(range(grad_output.ndim - 1)))

        # Gradient w.r.t. x
        # d/dx (x / rms * w) where rms = sqrt(mean(x^2) + eps)
        grad_x_norm = grad_output * self.weight.data

        # d/dx (x / rms)
        grad_x = grad_x_norm / rms
        grad_rms = -np.sum(grad_x_norm * x / (rms ** 2), axis=-1, keepdims=True)

        # d/drms (sqrt(mean(x^2)))
        grad_mean_sq = grad_rms / (2 * rms)
        grad_x += grad_mean_sq * 2 * x / d

        return grad_x

    def parameters(self) -> List[Parameter]:
        return [self.weight]


# =============================================================================
# Embedding with Backprop
# =============================================================================

class Embedding:
    """Token embedding layer with backprop."""

    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = Parameter(np.random.randn(vocab_size, d_model) * 0.02)
        self.cache = {}

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """Forward: lookup token embeddings."""
        self.cache['tokens'] = tokens
        return self.weight.data[tokens]

    def backward(self, grad_output: np.ndarray) -> None:
        """Backward: accumulate gradients into embedding matrix."""
        tokens = self.cache['tokens']
        if self.weight.grad is None:
            self.weight.grad = np.zeros_like(self.weight.data)

        # Accumulate gradients for each token
        np.add.at(self.weight.grad, tokens, grad_output)

    def parameters(self) -> List[Parameter]:
        return [self.weight]


# =============================================================================
# Multi-Head Attention with Backprop
# =============================================================================

class MultiHeadAttention:
    """Multi-head self-attention with manual backprop."""

    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.wq = Parameter(np.random.randn(d_model, d_model) * scale)
        self.wk = Parameter(np.random.randn(d_model, d_model) * scale)
        self.wv = Parameter(np.random.randn(d_model, d_model) * scale)
        self.wo = Parameter(np.random.randn(d_model, d_model) * scale)

        self.cache = {}

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass."""
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = x @ self.wq.data  # [B, S, D]
        k = x @ self.wk.data
        v = x @ self.wv.data

        # Reshape to [B, H, S, D_k]
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Attention scores
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)  # [B, H, S, S]

        if mask is not None:
            scores = scores + mask

        attn = softmax(scores, axis=-1)  # [B, H, S, S]

        # Apply attention
        context = attn @ v  # [B, H, S, D_k]

        # Reshape back
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Output projection
        out = context @ self.wo.data

        # Cache for backward
        self.cache = {
            'x': x, 'q': q, 'k': k, 'v': v,
            'scores': scores, 'attn': attn, 'context': context, 'mask': mask
        }

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through attention."""
        x = self.cache['x']
        q, k, v = self.cache['q'], self.cache['k'], self.cache['v']
        attn = self.cache['attn']
        context = self.cache['context']

        batch_size, seq_len, _ = x.shape

        # Gradient w.r.t. output projection
        self.wo.grad = context.reshape(-1, self.d_model).T @ grad_output.reshape(-1, self.d_model)
        grad_context = grad_output @ self.wo.data.T

        # Reshape gradient
        grad_context = grad_context.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        grad_context = grad_context.transpose(0, 2, 1, 3)  # [B, H, S, D_k]

        # Gradient through attention @ v
        grad_attn = grad_context @ v.transpose(0, 1, 3, 2)  # [B, H, S, S]
        grad_v = attn.transpose(0, 1, 3, 2) @ grad_context  # [B, H, S, D_k]

        # Gradient through softmax
        # d_softmax(x)_i = softmax_i * (d_i - sum_j(softmax_j * d_j))
        grad_scores = attn * (grad_attn - np.sum(grad_attn * attn, axis=-1, keepdims=True))
        grad_scores = grad_scores / np.sqrt(self.d_k)

        # Gradient through Q @ K^T
        grad_q = grad_scores @ k  # [B, H, S, D_k]
        grad_k = grad_scores.transpose(0, 1, 3, 2) @ q  # [B, H, S, D_k]

        # Reshape back to [B, S, D]
        grad_q = grad_q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        grad_k = grad_k.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        grad_v = grad_v.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Gradient w.r.t. projections
        x_flat = x.reshape(-1, self.d_model)
        self.wq.grad = x_flat.T @ grad_q.reshape(-1, self.d_model)
        self.wk.grad = x_flat.T @ grad_k.reshape(-1, self.d_model)
        self.wv.grad = x_flat.T @ grad_v.reshape(-1, self.d_model)

        # Gradient w.r.t. input
        grad_x = (grad_q @ self.wq.data.T +
                  grad_k @ self.wk.data.T +
                  grad_v @ self.wv.data.T)

        return grad_x

    def parameters(self) -> List[Parameter]:
        return [self.wq, self.wk, self.wv, self.wo]


# =============================================================================
# Feed-Forward Network with Backprop
# =============================================================================

class FeedForward:
    """SwiGLU Feed-Forward Network with backprop."""

    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        self.d_ff = d_ff or int(2.67 * d_model)

        scale1 = np.sqrt(2.0 / (d_model + self.d_ff))
        scale2 = np.sqrt(2.0 / (self.d_ff + d_model))

        self.w1 = Parameter(np.random.randn(d_model, self.d_ff) * scale1)  # Gate
        self.w2 = Parameter(np.random.randn(self.d_ff, d_model) * scale2)  # Down
        self.w3 = Parameter(np.random.randn(d_model, self.d_ff) * scale1)  # Up

        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward: SwiGLU(x) = (SiLU(x @ W1) * (x @ W3)) @ W2"""
        gate_pre = x @ self.w1.data
        gate = silu(gate_pre)
        up = x @ self.w3.data
        hidden = gate * up
        out = hidden @ self.w2.data

        self.cache = {'x': x, 'gate_pre': gate_pre, 'gate': gate, 'up': up, 'hidden': hidden}
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward through SwiGLU."""
        x = self.cache['x']
        gate_pre = self.cache['gate_pre']
        gate = self.cache['gate']
        up = self.cache['up']
        hidden = self.cache['hidden']

        # Gradient w.r.t. W2
        hidden_flat = hidden.reshape(-1, self.d_ff)
        self.w2.grad = hidden_flat.T @ grad_output.reshape(-1, self.w2.data.shape[1])

        # Gradient w.r.t. hidden
        grad_hidden = grad_output @ self.w2.data.T

        # Gradient through gate * up
        grad_gate = grad_hidden * up
        grad_up = grad_hidden * gate

        # Gradient through SiLU
        grad_gate_pre = silu_backward(gate_pre, grad_gate)

        # Gradient w.r.t. W1 and W3
        x_flat = x.reshape(-1, x.shape[-1])
        self.w1.grad = x_flat.T @ grad_gate_pre.reshape(-1, self.d_ff)
        self.w3.grad = x_flat.T @ grad_up.reshape(-1, self.d_ff)

        # Gradient w.r.t. x
        grad_x = (grad_gate_pre @ self.w1.data.T + grad_up @ self.w3.data.T)

        return grad_x

    def parameters(self) -> List[Parameter]:
        return [self.w1, self.w2, self.w3]


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock:
    """Single transformer block with backprop."""

    def __init__(self, d_model: int, n_heads: int, d_ff: Optional[int] = None):
        self.attn_norm = RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.cache = {}

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward: pre-norm transformer block."""
        # Attention sublayer
        x_norm = self.attn_norm.forward(x)
        attn_out = self.attention.forward(x_norm, mask)
        x = x + attn_out

        # FFN sublayer
        x_norm = self.ffn_norm.forward(x)
        ffn_out = self.ffn.forward(x_norm)
        x = x + ffn_out

        self.cache = {'residual1': x - ffn_out, 'residual0': x - ffn_out - attn_out}
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward through transformer block."""
        # FFN sublayer backward
        grad_ffn_out = grad_output
        grad_x_norm = self.ffn.backward(grad_ffn_out)
        grad_x = grad_output + self.ffn_norm.backward(grad_x_norm)

        # Attention sublayer backward
        grad_attn_out = grad_x
        grad_x_norm = self.attention.backward(grad_attn_out)
        grad_x = grad_x + self.attn_norm.backward(grad_x_norm)

        return grad_x

    def parameters(self) -> List[Parameter]:
        params = []
        params.extend(self.attn_norm.parameters())
        params.extend(self.attention.parameters())
        params.extend(self.ffn_norm.parameters())
        params.extend(self.ffn.parameters())
        return params


# =============================================================================
# Complete Trainable Transformer
# =============================================================================

class TrainableTransformer:
    """
    Complete trainable transformer language model.

    This is the capstone: a full transformer with forward and backward
    passes, ready for training with any optimizer from Stage 4.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: Optional[int] = None,
        max_seq_len: int = 256,
        tie_embeddings: bool = True
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.tie_embeddings = tie_embeddings

        # Token embedding
        self.embedding = Embedding(vocab_size, d_model)

        # Transformer blocks
        self.layers = [
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]

        # Output norm
        self.output_norm = RMSNorm(d_model)

        # Output projection (or tied with embeddings)
        if not tie_embeddings:
            self.output_proj = Parameter(np.random.randn(d_model, vocab_size) * 0.02)
        else:
            self.output_proj = None

        self.cache = {}

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """
        Forward pass: tokens -> logits.

        Args:
            tokens: [batch_size, seq_len] token indices

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Handle 1D input
        if tokens.ndim == 1:
            tokens = tokens[np.newaxis, :]

        batch_size, seq_len = tokens.shape

        # Embedding
        x = self.embedding.forward(tokens)  # [B, S, D]

        # Create causal mask
        mask = create_causal_mask(seq_len)

        # Process through layers
        for layer in self.layers:
            x = layer.forward(x, mask)

        # Output norm
        x = self.output_norm.forward(x)

        # Output projection
        if self.tie_embeddings:
            logits = x @ self.embedding.weight.data.T
        else:
            logits = x @ self.output_proj.data

        self.cache = {'x_final': x}
        return logits

    def backward(self, grad_logits: np.ndarray) -> None:
        """
        Backward pass: compute gradients for all parameters.

        Args:
            grad_logits: gradient of loss w.r.t. logits [B, S, V]
        """
        x_final = self.cache['x_final']

        # Gradient through output projection
        if self.tie_embeddings:
            # grad_x = grad_logits @ W_emb
            grad_x = grad_logits @ self.embedding.weight.data
            # grad_W_emb += x_final.T @ grad_logits (accumulated in embedding.backward)
            self.cache['output_grad'] = x_final.reshape(-1, self.d_model).T @ grad_logits.reshape(-1, self.vocab_size)
        else:
            grad_x = grad_logits @ self.output_proj.data.T
            self.output_proj.grad = x_final.reshape(-1, self.d_model).T @ grad_logits.reshape(-1, self.vocab_size)

        # Gradient through output norm
        grad_x = self.output_norm.backward(grad_x)

        # Gradient through layers (reverse order)
        for layer in reversed(self.layers):
            grad_x = layer.backward(grad_x)

        # Gradient through embedding
        self.embedding.backward(grad_x)

        # Add output gradient to embedding if tied
        if self.tie_embeddings:
            self.embedding.weight.grad += self.cache['output_grad'].T

    def parameters(self) -> List[Parameter]:
        """Return all learnable parameters."""
        params = self.embedding.parameters()
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend(self.output_norm.parameters())
        if not self.tie_embeddings:
            params.append(self.output_proj)
        return params

    def zero_grad(self) -> None:
        """Zero all gradients."""
        for p in self.parameters():
            p.zero_grad()

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.data.size for p in self.parameters())

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> List[int]:
        """Generate tokens autoregressively."""
        tokens = list(prompt_tokens)

        for _ in range(max_new_tokens):
            # Get context (truncate if needed)
            context = np.array(tokens[-self.max_seq_len:])

            # Forward pass
            logits = self.forward(context)

            # Get last position logits
            next_logits = logits[0, -1] / temperature

            # Sample
            probs = softmax(next_logits)
            next_token = np.random.choice(len(probs), p=probs)
            tokens.append(int(next_token))

        return tokens


# =============================================================================
# Loss Function
# =============================================================================

def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute cross-entropy loss and gradient.

    Args:
        logits: [batch, seq_len, vocab_size] or [seq_len, vocab_size]
        targets: [batch, seq_len] or [seq_len] target token indices

    Returns:
        (loss, grad_logits): scalar loss and gradient w.r.t. logits
    """
    original_shape = logits.shape

    # Flatten
    if logits.ndim == 3:
        batch, seq_len, vocab = logits.shape
        logits = logits.reshape(-1, vocab)
        targets = targets.reshape(-1)
    else:
        seq_len, vocab = logits.shape
        batch = 1

    n = logits.shape[0]

    # Log softmax (numerically stable)
    logits_max = logits.max(axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(logits - logits_max), axis=-1, keepdims=True))
    log_probs = logits - logits_max - log_sum_exp

    # Cross-entropy loss
    loss = -np.mean(log_probs[np.arange(n), targets])

    # Gradient: d/d_logits = softmax(logits) - one_hot(targets)
    probs = np.exp(log_probs)
    grad = probs.copy()
    grad[np.arange(n), targets] -= 1
    grad = grad / n  # Average gradient

    # Reshape gradient
    grad = grad.reshape(original_shape)

    return float(loss), grad


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    return np.exp(loss)


# =============================================================================
# Tokenizer
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


# =============================================================================
# Demo
# =============================================================================

if __name__ == '__main__':
    np.random.seed(42)

    # Create a small model
    model = TrainableTransformer(
        vocab_size=50,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=32
    )

    print(f"Model parameters: {model.count_parameters():,}")

    # Test forward pass
    tokens = np.array([[1, 5, 10, 15, 20]])
    logits = model.forward(tokens)
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {logits.shape}")

    # Test loss and backward
    targets = np.array([[5, 10, 15, 20, 25]])
    loss, grad = cross_entropy_loss(logits, targets)
    print(f"Loss: {loss:.4f}")
    print(f"Perplexity: {compute_perplexity(loss):.2f}")

    # Test backward
    model.zero_grad()
    model.backward(grad)

    # Check gradients exist
    total_grad_norm = sum(np.sum(p.grad ** 2) for p in model.parameters())
    print(f"Total gradient norm: {np.sqrt(total_grad_norm):.4f}")

    print("\nAll forward/backward passes working!")

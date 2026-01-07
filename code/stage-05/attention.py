"""
Stage 5: Attention Mechanisms - Complete Implementation

This module implements all attention components from first principles:
- Scaled dot-product attention
- Multi-head attention
- Positional encoding (sinusoidal and learned)
- Causal masking
- Layer normalization
- Feed-forward networks
- Complete Transformer block
- Causal language model
"""

import numpy as np
from typing import Optional, List, Tuple


# =============================================================================
# Core Utilities
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax that handles -inf from masking.

    Args:
        x: Input array
        axis: Axis along which to compute softmax

    Returns:
        Softmax probabilities
    """
    # Handle -inf values by replacing with large negative for max computation
    x_safe = np.where(np.isinf(x), -1e9, x)
    x_max = np.max(x_safe, axis=axis, keepdims=True)

    exp_x = np.exp(x - x_max)
    # Set -inf positions back to 0
    exp_x = np.where(np.isinf(x), 0.0, exp_x)

    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-10)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit activation.

    Approximate version: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation."""
    return np.maximum(0, x)


# =============================================================================
# Attention Core
# =============================================================================

def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal (autoregressive) attention mask.

    Prevents positions from attending to future positions.

    Args:
        seq_len: Sequence length

    Returns:
        Mask of shape [seq_len, seq_len] with 0 for allowed, -inf for masked
    """
    # Create upper triangular matrix with True for positions to mask
    mask_bool = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    # Use np.where to avoid NaN from 0 * -inf
    mask = np.where(mask_bool, float('-inf'), 0.0)
    return mask


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention mechanism.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q: Queries [..., seq_len, d_k]
        K: Keys [..., seq_len, d_k]
        V: Values [..., seq_len, d_v]
        mask: Optional mask [..., seq_len, seq_len]
              0 for allowed positions, -inf for masked

    Returns:
        output: Attention output [..., seq_len, d_v]
        weights: Attention weights [..., seq_len, seq_len]
    """
    d_k = Q.shape[-1]

    # Compute attention scores: Q @ K^T
    scores = Q @ np.swapaxes(K, -2, -1) / np.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        scores = scores + mask

    # Softmax over keys (last dimension)
    weights = softmax(scores, axis=-1)

    # Weighted sum of values
    output = weights @ V

    return output, weights


# =============================================================================
# Positional Encoding
# =============================================================================

class SinusoidalPositionalEncoding:
    """
    Fixed sinusoidal positional encoding from "Attention Is All You Need".

    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """

    def __init__(self, max_len: int, d_model: int):
        """
        Initialize positional encoding.

        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
        """
        self.max_len = max_len
        self.d_model = d_model
        self.encoding = self._create_encoding(max_len, d_model)

    def _create_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """Generate sinusoidal positional encodings."""
        pe = np.zeros((max_len, d_model))

        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return pe

    def __call__(self, seq_len: int) -> np.ndarray:
        """Get positional encoding for sequence length."""
        return self.encoding[:seq_len]


class LearnedPositionalEncoding:
    """
    Learned positional embeddings.

    Each position has a learnable embedding vector.
    """

    def __init__(self, max_len: int, d_model: int):
        """
        Initialize learned positional encoding.

        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
        """
        self.max_len = max_len
        self.d_model = d_model
        self.encoding = np.random.randn(max_len, d_model) * 0.02

    def __call__(self, seq_len: int) -> np.ndarray:
        """Get positional encoding for sequence length."""
        return self.encoding[:seq_len]

    def parameters(self) -> List[np.ndarray]:
        """Return learnable parameters."""
        return [self.encoding]


class RotaryPositionalEncoding:
    """
    Rotary Position Embedding (RoPE).

    Applies rotation to query and key vectors based on position.
    Relative positions emerge from dot product of rotated vectors.
    """

    def __init__(self, d_model: int, base: float = 10000.0):
        """
        Initialize RoPE.

        Args:
            d_model: Model dimension (must be even)
            base: Base for frequency computation
        """
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        self.d_model = d_model
        self.base = base

        # Precompute frequency bands
        self.freqs = 1.0 / (base ** (np.arange(0, d_model, 2) / d_model))

    def __call__(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Apply rotary embedding to input.

        Args:
            x: Input tensor [..., seq_len, d_model]
            positions: Position indices [seq_len]

        Returns:
            Rotated tensor [..., seq_len, d_model]
        """
        seq_len = positions.shape[0]

        # Compute angles: [seq_len, d_model/2]
        angles = positions[:, np.newaxis] * self.freqs[np.newaxis, :]

        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        # Split into pairs
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        # Apply rotation
        x_rotated = np.empty_like(x)
        x_rotated[..., 0::2] = x1 * cos_angles - x2 * sin_angles
        x_rotated[..., 1::2] = x1 * sin_angles + x2 * cos_angles

        return x_rotated


# =============================================================================
# Layer Normalization
# =============================================================================

class LayerNorm:
    """
    Layer normalization.

    Normalizes across the feature dimension, then applies learnable
    scale (gamma) and shift (beta).
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initialize layer normalization.

        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        self.d_model = d_model
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization.

        Args:
            x: Input [..., d_model]

        Returns:
            Normalized output [..., d_model]
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        x_norm = (x - mean) / np.sqrt(var + self.eps)

        return self.gamma * x_norm + self.beta

    def parameters(self) -> List[np.ndarray]:
        """Return learnable parameters."""
        return [self.gamma, self.beta]


# =============================================================================
# Multi-Head Attention
# =============================================================================

class MultiHeadAttention:
    """
    Multi-head attention mechanism.

    Splits input into multiple heads, applies attention independently,
    then concatenates and projects back to model dimension.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability (not used in forward, for compatibility)
        """
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout

        # Xavier initialization
        scale = np.sqrt(2.0 / (d_model + self.d_k))

        # Combined QKV projection for efficiency
        self.W_qkv = np.random.randn(d_model, 3 * d_model) * scale

        # Output projection
        self.W_o = np.random.randn(d_model, d_model) * scale

        # Store attention weights for visualization
        self.attention_weights = None

    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input [batch_size, seq_len, d_model] or [seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Output [batch_size, seq_len, d_model] or [seq_len, d_model]
        """
        # Handle unbatched input
        squeeze_batch = False
        if x.ndim == 2:
            x = x[np.newaxis, ...]
            squeeze_batch = True

        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = x @ self.W_qkv  # [batch, seq_len, 3*d_model]
        Q, K, V = np.split(qkv, 3, axis=-1)

        # Reshape for multi-head: [batch, seq_len, n_heads, d_k]
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k)

        # Transpose to [batch, n_heads, seq_len, d_k]
        Q = np.transpose(Q, (0, 2, 1, 3))
        K = np.transpose(K, (0, 2, 1, 3))
        V = np.transpose(V, (0, 2, 1, 3))

        # Apply scaled dot-product attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Store weights for visualization
        self.attention_weights = attn_weights

        # Transpose back: [batch, seq_len, n_heads, d_k]
        attn_output = np.transpose(attn_output, (0, 2, 1, 3))

        # Concatenate heads: [batch, seq_len, d_model]
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)

        # Output projection
        output = attn_output @ self.W_o

        if squeeze_batch:
            output = output[0]

        return output

    def parameters(self) -> List[np.ndarray]:
        """Return all parameters."""
        return [self.W_qkv, self.W_o]


# =============================================================================
# Feed-Forward Network
# =============================================================================

class FeedForward:
    """
    Position-wise feed-forward network.

    FFN(x) = activation(x @ W1 + b1) @ W2 + b2
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        activation: str = 'gelu'
    ):
        """
        Initialize feed-forward network.

        Args:
            d_model: Model dimension
            d_ff: Hidden dimension (default: 4 * d_model)
            activation: Activation function ('relu' or 'gelu')
        """
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.activation_name = activation

        if activation == 'gelu':
            self.activation = gelu
        elif activation == 'relu':
            self.activation = relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Initialize weights
        scale1 = np.sqrt(2.0 / (d_model + self.d_ff))
        scale2 = np.sqrt(2.0 / (self.d_ff + d_model))

        self.W1 = np.random.randn(d_model, self.d_ff) * scale1
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input [..., d_model]

        Returns:
            Output [..., d_model]
        """
        hidden = x @ self.W1 + self.b1
        hidden = self.activation(hidden)
        output = hidden @ self.W2 + self.b2
        return output

    def parameters(self) -> List[np.ndarray]:
        """Return all parameters."""
        return [self.W1, self.b1, self.W2, self.b2]


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock:
    """
    Single Transformer block.

    Consists of:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization
    - Residual connections
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        pre_norm: bool = True
    ):
        """
        Initialize Transformer block.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: FFN hidden dimension
            dropout: Dropout probability
            pre_norm: If True, apply LayerNorm before sublayers (modern style)
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.pre_norm = pre_norm

        # Components
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input [batch_size, seq_len, d_model]
            mask: Attention mask

        Returns:
            Output [batch_size, seq_len, d_model]
        """
        if self.pre_norm:
            # Pre-norm style (modern, better training dynamics)
            x = x + self.attention(self.norm1(x), mask)
            x = x + self.ffn(self.norm2(x))
        else:
            # Post-norm style (original Transformer)
            x = self.norm1(x + self.attention(x, mask))
            x = self.norm2(x + self.ffn(x))

        return x

    def parameters(self) -> List[np.ndarray]:
        """Return all parameters."""
        params = []
        params.extend(self.attention.parameters())
        params.extend(self.ffn.parameters())
        params.extend(self.norm1.parameters())
        params.extend(self.norm2.parameters())
        return params


# =============================================================================
# Complete Causal Transformer
# =============================================================================

class CausalTransformer:
    """
    Causal Transformer language model.

    Combines:
    - Token embeddings
    - Positional encodings
    - Stack of Transformer blocks
    - Output projection to vocabulary
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_len: int = 512,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        pos_encoding: str = 'sinusoidal'
    ):
        """
        Initialize causal Transformer.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads per layer
            n_layers: Number of Transformer blocks
            max_len: Maximum sequence length
            d_ff: FFN hidden dimension
            dropout: Dropout probability
            pos_encoding: Type of positional encoding ('sinusoidal' or 'learned')
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_len = max_len

        # Token embeddings
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02

        # Positional encoding
        if pos_encoding == 'sinusoidal':
            self.pos_encoding = SinusoidalPositionalEncoding(max_len, d_model)
        elif pos_encoding == 'learned':
            self.pos_encoding = LearnedPositionalEncoding(max_len, d_model)
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")

        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]

        # Final layer norm
        self.final_norm = LayerNorm(d_model)

        # Output projection (weight-tied with token embeddings)
        # self.output_projection shares weights with token_embedding.T

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            token_ids: Input token IDs [batch_size, seq_len] or [seq_len]

        Returns:
            logits: [batch_size, seq_len, vocab_size] or [seq_len, vocab_size]
        """
        # Handle unbatched input
        squeeze_batch = False
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]
            squeeze_batch = True

        batch_size, seq_len = token_ids.shape

        # Token embeddings
        x = self.token_embedding[token_ids]  # [batch, seq_len, d_model]

        # Add positional encoding
        pos_enc = self.pos_encoding(seq_len)
        x = x + pos_enc

        # Create causal mask
        mask = create_causal_mask(seq_len)

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final layer norm
        x = self.final_norm(x)

        # Project to vocabulary (using tied weights)
        logits = x @ self.token_embedding.T  # [batch, seq_len, vocab_size]

        if squeeze_batch:
            logits = logits[0]

        return logits

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> List[int]:
        """
        Generate tokens autoregressively.

        Args:
            prompt_ids: Initial token IDs [seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, sample only from top-k tokens

        Returns:
            Generated token IDs as list
        """
        tokens = list(prompt_ids)

        for _ in range(max_new_tokens):
            # Get context (up to max_len)
            context = np.array(tokens[-self.max_len:])

            # Forward pass
            logits = self(context)

            # Get logits for last position
            next_logits = logits[-1] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                top_k_indices = np.argsort(next_logits)[-top_k:]
                mask = np.full_like(next_logits, float('-inf'))
                mask[top_k_indices] = next_logits[top_k_indices]
                next_logits = mask

            # Sample from distribution
            probs = softmax(next_logits)
            next_token = np.random.choice(len(probs), p=probs)
            tokens.append(int(next_token))

        return tokens

    def parameters(self) -> List[np.ndarray]:
        """Return all parameters."""
        params = [self.token_embedding]

        if isinstance(self.pos_encoding, LearnedPositionalEncoding):
            params.extend(self.pos_encoding.parameters())

        for block in self.blocks:
            params.extend(block.parameters())

        params.extend(self.final_norm.parameters())

        return params

    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.size for p in self.parameters())


# =============================================================================
# Training Utilities
# =============================================================================

def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute cross-entropy loss for language modeling.

    Args:
        logits: Predicted logits [seq_len, vocab_size]
        targets: Target token IDs [seq_len]

    Returns:
        Average cross-entropy loss
    """
    # Numerically stable log-softmax
    logits_max = logits.max(axis=-1, keepdims=True)
    log_probs = logits - logits_max - np.log(
        np.sum(np.exp(logits - logits_max), axis=-1, keepdims=True)
    )

    # Select log probabilities of target tokens
    seq_len = targets.shape[0]
    target_log_probs = log_probs[np.arange(seq_len), targets]

    # Return negative log likelihood
    return -np.mean(target_log_probs)


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    return np.exp(loss)


# =============================================================================
# Visualization Utilities
# =============================================================================

def visualize_attention_weights(
    weights: np.ndarray,
    tokens: Optional[List[str]] = None,
    head: int = 0
) -> str:
    """
    Create ASCII visualization of attention weights.

    Args:
        weights: Attention weights [n_heads, seq_len, seq_len] or [seq_len, seq_len]
        tokens: Optional token strings for labels
        head: Which head to visualize (if multi-head)

    Returns:
        ASCII string representation
    """
    # Select head if multi-head
    if weights.ndim == 4:  # [batch, heads, seq, seq]
        weights = weights[0, head]
    elif weights.ndim == 3:  # [heads, seq, seq]
        weights = weights[head]

    seq_len = weights.shape[0]

    # Create visualization
    lines = []

    # Header row
    if tokens:
        header = "     " + " ".join(f"{t[:4]:>4}" for t in tokens)
    else:
        header = "     " + " ".join(f"{i:>4}" for i in range(seq_len))
    lines.append(header)

    # Weight rows
    for i in range(seq_len):
        if tokens:
            row = f"{tokens[i][:4]:>4} "
        else:
            row = f"{i:>4} "

        for j in range(seq_len):
            w = weights[i, j]
            if w > 0.5:
                row += "████ "
            elif w > 0.25:
                row += "▓▓▓ "
            elif w > 0.1:
                row += "▒▒ "
            elif w > 0.05:
                row += "░ "
            else:
                row += "   "

        lines.append(row)

    return "\n".join(lines)


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Core functions
    'softmax',
    'gelu',
    'relu',
    'create_causal_mask',
    'scaled_dot_product_attention',

    # Positional encoding
    'SinusoidalPositionalEncoding',
    'LearnedPositionalEncoding',
    'RotaryPositionalEncoding',

    # Layers
    'LayerNorm',
    'MultiHeadAttention',
    'FeedForward',
    'TransformerBlock',

    # Model
    'CausalTransformer',

    # Utilities
    'cross_entropy_loss',
    'compute_perplexity',
    'visualize_attention_weights',
]

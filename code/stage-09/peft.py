"""
Stage 9: Fine-tuning & Parameter-Efficient Methods

This module implements parameter-efficient fine-tuning (PEFT) methods:
- LoRA (Low-Rank Adaptation) - The most popular PEFT method
- Adapters - Bottleneck layers inserted into transformers
- Prefix Tuning - Learned prefix embeddings
- Prompt Tuning - Learned soft prompts

Key insight: Modern LLMs have billions of parameters, but fine-tuning
often only needs to update a tiny fraction. PEFT methods exploit this
by training only a small number of new parameters.

Why this matters:
- Full fine-tuning of a 7B model requires ~28GB of GPU memory
- LoRA can fine-tune the same model with ~8GB
- Storage: save 1% of parameters instead of 100%
- Can merge multiple LoRAs for different tasks

"LoRA is not an approximation - it's often as good or better than full fine-tuning."
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


# =============================================================================
# Low-Rank Adaptation (LoRA)
# =============================================================================

class LoRALayer:
    """
    Low-Rank Adaptation layer.

    The key insight from Hu et al. (2021):
    - Weight updates during fine-tuning have low "intrinsic rank"
    - Instead of updating W directly, learn W + BA where B, A are low-rank
    - B: (d, r), A: (r, k) where r << min(d, k)

    For a pretrained weight W ∈ R^(d×k):
        h = Wx                  # Original forward
        h = Wx + BAx            # With LoRA
        h = (W + BA)x           # Can merge after training!

    During training:
    - W is frozen (no gradients)
    - Only A and B are trained
    - Parameter count: r(d+k) instead of dk

    Example:
        Original: 4096 × 4096 = 16.7M parameters
        LoRA r=8: 8 × (4096 + 4096) = 65K parameters (0.4%!)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        """
        Initialize LoRA layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: LoRA rank (r). Higher = more capacity, more params
            alpha: Scaling factor. The update is scaled by alpha/rank
            dropout: Dropout probability on LoRA path
        """
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout_prob = dropout

        # LoRA matrices
        # A: (rank, in_features) - initialized with Gaussian
        # B: (out_features, rank) - initialized with zeros
        # This means LoRA starts as identity (no change to pretrained weights)
        self.A = np.random.randn(rank, in_features) * 0.01
        self.B = np.zeros((out_features, rank))

        # Gradients
        self.A_grad: Optional[np.ndarray] = None
        self.B_grad: Optional[np.ndarray] = None

        # Cache for backward
        self.cache: Dict[str, Any] = {}

    def forward(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Forward pass: compute Wx + scaling * (B @ A @ x)

        Args:
            x: Input [batch, seq, in_features] or [seq, in_features]
            W: Frozen pretrained weights [out_features, in_features]

        Returns:
            Output with LoRA adaptation
        """
        # Original path (frozen)
        base_output = x @ W.T

        # LoRA path
        # x @ A.T gives [batch, seq, rank]
        # then @ B.T gives [batch, seq, out_features]
        lora_output = (x @ self.A.T) @ self.B.T

        # Cache for backward
        self.cache = {'x': x, 'W': W, 'Ax': x @ self.A.T}

        return base_output + self.scaling * lora_output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: compute gradients for A and B.

        Note: We don't compute gradients for W (it's frozen).

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradient w.r.t. input (for chain rule)
        """
        x = self.cache['x']
        W = self.cache['W']
        Ax = self.cache['Ax']

        # Scale gradient for LoRA path
        scaled_grad = grad_output * self.scaling

        # Reshape for batch matrix multiply
        original_shape = x.shape
        if x.ndim == 3:
            batch, seq, _ = x.shape
            x_flat = x.reshape(-1, self.in_features)
            Ax_flat = Ax.reshape(-1, self.rank)
            grad_flat = scaled_grad.reshape(-1, self.out_features)
        else:
            x_flat = x
            Ax_flat = Ax
            grad_flat = scaled_grad

        # Gradient w.r.t. B: grad @ Ax
        # B is (out_features, rank), grad is (N, out_features), Ax is (N, rank)
        self.B_grad = grad_flat.T @ Ax_flat  # (out_features, rank)

        # Gradient w.r.t. A: B.T @ grad @ x
        # A is (rank, in_features)
        grad_Ax = grad_flat @ self.B  # (N, rank)
        self.A_grad = grad_Ax.T @ x_flat  # (rank, in_features)

        # Gradient w.r.t. input (for frozen W and LoRA)
        # d/dx (Wx + s*BAx) = W + s*BA
        grad_x = grad_output @ W  # From frozen path
        grad_x = grad_x + scaled_grad @ self.B @ self.A  # From LoRA path

        return grad_x.reshape(original_shape) if x.ndim == 3 else grad_x

    def merge_weights(self, W: np.ndarray) -> np.ndarray:
        """
        Merge LoRA weights into base weights for inference.

        After training, you can merge W + scaling * B @ A
        to get a single weight matrix with no inference overhead.

        Args:
            W: Original pretrained weights

        Returns:
            Merged weights
        """
        return W + self.scaling * (self.B @ self.A)

    def parameters(self) -> List[np.ndarray]:
        """Return trainable parameters."""
        return [self.A, self.B]

    def gradients(self) -> List[Optional[np.ndarray]]:
        """Return gradients."""
        return [self.A_grad, self.B_grad]

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.A_grad = None
        self.B_grad = None

    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return self.A.size + self.B.size

    def compression_ratio(self) -> float:
        """Ratio of LoRA params to full fine-tuning params."""
        full_params = self.in_features * self.out_features
        lora_params = self.num_parameters()
        return lora_params / full_params


# =============================================================================
# LoRA-Enhanced Linear Layer
# =============================================================================

class LoRALinear:
    """
    Linear layer with optional LoRA adaptation.

    This wraps a pretrained linear layer and adds LoRA on top.
    The pretrained weights are frozen; only LoRA weights are trained.
    """

    def __init__(
        self,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ):
        """
        Initialize LoRA-enhanced linear layer.

        Args:
            weight: Pretrained weights [out_features, in_features]
            bias: Optional pretrained bias [out_features]
            lora_rank: LoRA rank
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout on LoRA path
        """
        self.weight = weight  # Frozen
        self.bias = bias  # Frozen
        self.out_features, self.in_features = weight.shape

        self.lora = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with LoRA."""
        output = self.lora.forward(x, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass (only through LoRA)."""
        return self.lora.backward(grad_output)

    def merge(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Merge LoRA into base weights."""
        merged_weight = self.lora.merge_weights(self.weight)
        return merged_weight, self.bias

    def parameters(self) -> List[np.ndarray]:
        return self.lora.parameters()

    def gradients(self) -> List[Optional[np.ndarray]]:
        return self.lora.gradients()

    def zero_grad(self) -> None:
        self.lora.zero_grad()


# =============================================================================
# Adapter Layers
# =============================================================================

class Adapter:
    """
    Adapter layer (Houlsby et al., 2019).

    Adapters are small bottleneck layers inserted into transformers:
        h = h + f(h @ W_down) @ W_up

    where:
    - W_down: (d_model, bottleneck_dim) - projects down
    - W_up: (bottleneck_dim, d_model) - projects back up
    - f: nonlinearity (usually ReLU or GELU)

    Typically inserted after attention and FFN sublayers.
    """

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int = 64,
        activation: str = 'relu',
    ):
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.activation = activation

        # Initialize with small weights (adapter starts near identity)
        scale = 0.01
        self.W_down = np.random.randn(d_model, bottleneck_dim) * scale
        self.W_up = np.random.randn(bottleneck_dim, d_model) * scale

        self.cache: Dict[str, Any] = {}

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'gelu':
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        else:
            return x

    def _activate_backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Backward through activation."""
        if self.activation == 'relu':
            return grad * (x > 0).astype(float)
        elif self.activation == 'gelu':
            # Approximate GELU derivative
            cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
            pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
            return grad * (cdf + x * pdf)
        else:
            return grad

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: x + adapter(x)

        Args:
            x: Input [batch, seq, d_model]

        Returns:
            Output with adapter residual
        """
        # Down projection
        down = x @ self.W_down  # [batch, seq, bottleneck]

        # Activation
        activated = self._activate(down)

        # Up projection
        up = activated @ self.W_up  # [batch, seq, d_model]

        # Residual connection
        self.cache = {'x': x, 'down': down, 'activated': activated}

        return x + up

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through adapter."""
        x = self.cache['x']
        down = self.cache['down']
        activated = self.cache['activated']

        # Gradient flows through residual and adapter path
        grad_up = grad_output  # Gradient to adapter output

        # Gradient through W_up
        original_shape = activated.shape
        activated_flat = activated.reshape(-1, self.bottleneck_dim)
        grad_up_flat = grad_up.reshape(-1, self.d_model)

        self.W_up_grad = activated_flat.T @ grad_up_flat

        # Gradient through activation
        grad_activated = grad_up @ self.W_up.T
        grad_down = self._activate_backward(down, grad_activated)

        # Gradient through W_down
        x_flat = x.reshape(-1, self.d_model)
        grad_down_flat = grad_down.reshape(-1, self.bottleneck_dim)

        self.W_down_grad = x_flat.T @ grad_down_flat

        # Gradient to input (residual + adapter)
        grad_x = grad_output + grad_down @ self.W_down.T

        return grad_x

    def parameters(self) -> List[np.ndarray]:
        return [self.W_down, self.W_up]

    def gradients(self) -> List[np.ndarray]:
        return [self.W_down_grad, self.W_up_grad]

    def num_parameters(self) -> int:
        return self.W_down.size + self.W_up.size


# =============================================================================
# Prefix Tuning
# =============================================================================

class PrefixTuning:
    """
    Prefix Tuning (Li & Liang, 2021).

    Instead of modifying weights, prepend learned "prefix" vectors
    to the key and value sequences in attention.

    For each attention layer:
        K = [P_k; K]  # Prepend prefix keys
        V = [P_v; V]  # Prepend prefix values

    The model attends to learned prefix tokens, which steer behavior.

    Advantages:
    - No weight modification
    - Very few parameters
    - Can interpolate between tasks

    Typically uses an MLP to generate prefixes from a smaller embedding,
    which we simplify here to direct prefix parameters.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_head: int,
        prefix_length: int = 10,
    ):
        """
        Initialize prefix tuning.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_head: Dimension per head
            prefix_length: Number of prefix tokens
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_head = d_head
        self.prefix_length = prefix_length

        # Prefix parameters for each layer
        # Shape: [num_layers, 2, prefix_length, num_heads, d_head]
        # The 2 is for key and value prefixes
        self.prefix = np.random.randn(
            num_layers, 2, prefix_length, num_heads, d_head
        ) * 0.01

        self.prefix_grad: Optional[np.ndarray] = None

    def get_prefix(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get key and value prefixes for a layer.

        Args:
            layer_idx: Which transformer layer

        Returns:
            (prefix_keys, prefix_values) each of shape [prefix_length, num_heads, d_head]
        """
        prefix_k = self.prefix[layer_idx, 0]
        prefix_v = self.prefix[layer_idx, 1]
        return prefix_k, prefix_v

    def parameters(self) -> List[np.ndarray]:
        return [self.prefix]

    def num_parameters(self) -> int:
        return self.prefix.size


# =============================================================================
# Prompt Tuning
# =============================================================================

class PromptTuning:
    """
    Prompt Tuning (Lester et al., 2021).

    Learn "soft prompts" - continuous embeddings prepended to input.
    Unlike discrete prompts ("Classify this text:"), soft prompts are
    learned vectors that can represent concepts not expressible in words.

    Input sequence becomes: [soft_prompt; input_tokens]

    Even simpler than prefix tuning - just learned embeddings at input.
    """

    def __init__(
        self,
        d_model: int,
        prompt_length: int = 20,
        init_from_vocab: Optional[np.ndarray] = None,
    ):
        """
        Initialize prompt tuning.

        Args:
            d_model: Model embedding dimension
            prompt_length: Number of soft prompt tokens
            init_from_vocab: Optional vocab embeddings for initialization
        """
        self.d_model = d_model
        self.prompt_length = prompt_length

        if init_from_vocab is not None:
            # Initialize from random vocab embeddings
            indices = np.random.choice(len(init_from_vocab), prompt_length)
            self.prompt = init_from_vocab[indices].copy()
        else:
            # Random initialization
            self.prompt = np.random.randn(prompt_length, d_model) * 0.01

        self.prompt_grad: Optional[np.ndarray] = None

    def forward(self, input_embeds: np.ndarray) -> np.ndarray:
        """
        Prepend soft prompt to input embeddings.

        Args:
            input_embeds: [batch, seq_len, d_model]

        Returns:
            [batch, prompt_length + seq_len, d_model]
        """
        batch_size = input_embeds.shape[0]

        # Expand prompt for batch
        prompt_expanded = np.broadcast_to(
            self.prompt[np.newaxis, :, :],
            (batch_size, self.prompt_length, self.d_model)
        ).copy()

        # Concatenate
        return np.concatenate([prompt_expanded, input_embeds], axis=1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            grad_output: Gradient of full sequence [batch, prompt_len + seq_len, d_model]

        Returns:
            Gradient for original input [batch, seq_len, d_model]
        """
        # Gradient for prompt
        prompt_grad = grad_output[:, :self.prompt_length, :]
        self.prompt_grad = prompt_grad.sum(axis=0)  # Sum over batch

        # Gradient for input (pass through)
        return grad_output[:, self.prompt_length:, :]

    def parameters(self) -> List[np.ndarray]:
        return [self.prompt]

    def num_parameters(self) -> int:
        return self.prompt.size


# =============================================================================
# Fine-tuning Utilities
# =============================================================================

def count_trainable_parameters(model_params: List[np.ndarray]) -> int:
    """Count total trainable parameters."""
    return sum(p.size for p in model_params)


def count_frozen_parameters(frozen_params: List[np.ndarray]) -> int:
    """Count frozen parameters."""
    return sum(p.size for p in frozen_params)


def compute_parameter_efficiency(
    trainable: int,
    total: int,
) -> Dict[str, float]:
    """
    Compute parameter efficiency metrics.

    Returns:
        Dictionary with efficiency metrics
    """
    return {
        'trainable_params': trainable,
        'total_params': total,
        'trainable_ratio': trainable / total if total > 0 else 0,
        'efficiency_gain': total / trainable if trainable > 0 else float('inf'),
    }


def apply_lora_to_attention(
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    wo: np.ndarray,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
) -> Dict[str, LoRALinear]:
    """
    Apply LoRA to attention weight matrices.

    Args:
        wq, wk, wv, wo: Attention weight matrices
        rank: LoRA rank
        alpha: LoRA scaling factor
        target_modules: Which matrices to apply LoRA to (default: ['q', 'v'])

    Returns:
        Dictionary of LoRA-wrapped layers
    """
    if target_modules is None:
        target_modules = ['q', 'v']  # Default: query and value (most common)

    layers = {}
    weights = {'q': wq, 'k': wk, 'v': wv, 'o': wo}

    for name, weight in weights.items():
        if name in target_modules:
            layers[name] = LoRALinear(weight, lora_rank=rank, lora_alpha=alpha)
        else:
            # Keep as frozen linear (no LoRA)
            layers[name] = weight

    return layers


# =============================================================================
# Comparison and Analysis
# =============================================================================

def compare_peft_methods(
    d_model: int = 4096,
    num_layers: int = 32,
    num_heads: int = 32,
    lora_rank: int = 8,
    adapter_bottleneck: int = 64,
    prefix_length: int = 10,
    prompt_length: int = 20,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare parameter counts across PEFT methods.

    This shows why PEFT is so powerful for large models.
    """
    d_head = d_model // num_heads

    # Full fine-tuning: all attention weights
    # 4 matrices per layer: Q, K, V, O
    full_params = num_layers * 4 * d_model * d_model

    # LoRA on Q and V
    lora_params = num_layers * 2 * lora_rank * (d_model + d_model)

    # Adapters (one after attention, one after FFN)
    adapter_params = num_layers * 2 * 2 * d_model * adapter_bottleneck

    # Prefix tuning
    prefix_params = num_layers * 2 * prefix_length * num_heads * d_head

    # Prompt tuning
    prompt_params = prompt_length * d_model

    results = {
        'full_fine_tuning': {
            'params': full_params,
            'ratio': 1.0,
            'description': 'Update all attention weights',
        },
        'lora': {
            'params': lora_params,
            'ratio': lora_params / full_params,
            'description': f'LoRA rank={lora_rank} on Q,V',
        },
        'adapters': {
            'params': adapter_params,
            'ratio': adapter_params / full_params,
            'description': f'Bottleneck={adapter_bottleneck}',
        },
        'prefix_tuning': {
            'params': prefix_params,
            'ratio': prefix_params / full_params,
            'description': f'{prefix_length} prefix tokens',
        },
        'prompt_tuning': {
            'params': prompt_params,
            'ratio': prompt_params / full_params,
            'description': f'{prompt_length} soft prompts',
        },
    }

    return results


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate PEFT methods."""
    print("=" * 60)
    print("Stage 9: Fine-tuning & Parameter-Efficient Methods")
    print("=" * 60)

    # 1. LoRA Demo
    print("\n1. LoRA (Low-Rank Adaptation)")
    print("-" * 40)

    # Simulate a pretrained weight matrix
    np.random.seed(42)
    d_model = 256
    pretrained_W = np.random.randn(d_model, d_model) * 0.02

    # Create LoRA layer
    lora = LoRALayer(d_model, d_model, rank=8, alpha=16)

    print(f"Original weight shape: {pretrained_W.shape}")
    print(f"LoRA A shape: {lora.A.shape}")
    print(f"LoRA B shape: {lora.B.shape}")
    print(f"LoRA parameters: {lora.num_parameters():,}")
    print(f"Full parameters: {d_model * d_model:,}")
    print(f"Compression: {lora.compression_ratio():.2%}")

    # Forward pass
    x = np.random.randn(2, 16, d_model)
    output = lora.forward(x, pretrained_W)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Backward pass
    grad = np.random.randn(*output.shape)
    grad_x = lora.backward(grad)
    print(f"Gradients computed: A={lora.A_grad.shape}, B={lora.B_grad.shape}")

    # Merge weights
    merged = lora.merge_weights(pretrained_W)
    print(f"Merged weight shape: {merged.shape}")

    # 2. Adapter Demo
    print("\n2. Adapter Layers")
    print("-" * 40)

    adapter = Adapter(d_model, bottleneck_dim=64)
    output = adapter.forward(x)
    print(f"Adapter parameters: {adapter.num_parameters():,}")
    print(f"Bottleneck dimension: 64")

    # 3. Prompt Tuning Demo
    print("\n3. Prompt Tuning")
    print("-" * 40)

    prompt_tuning = PromptTuning(d_model, prompt_length=20)
    prompted = prompt_tuning.forward(x)
    print(f"Original sequence length: {x.shape[1]}")
    print(f"With soft prompt: {prompted.shape[1]}")
    print(f"Prompt parameters: {prompt_tuning.num_parameters():,}")

    # 4. Method Comparison
    print("\n4. PEFT Method Comparison (7B model scale)")
    print("-" * 40)

    comparison = compare_peft_methods(
        d_model=4096,
        num_layers=32,
        num_heads=32,
        lora_rank=8,
    )

    print(f"{'Method':<20} {'Parameters':<15} {'Ratio':<10} Description")
    print("-" * 70)
    for method, info in comparison.items():
        print(f"{method:<20} {info['params']:>12,} {info['ratio']:>8.4%}  {info['description']}")


if __name__ == '__main__':
    demo()

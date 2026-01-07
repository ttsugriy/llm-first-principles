# Section 9.6: Implementation

*Reading time: 15 minutes*

## Overview

In this section, we implement four PEFT methods from scratch:

1. **LoRA**: Low-rank adaptation
2. **Adapters**: Bottleneck layers
3. **Prefix Tuning**: Learned key/value prefixes
4. **Prompt Tuning**: Soft input prompts

All code is available in `code/stage-09/peft.py`.

## LoRA Implementation

The core LoRA layer with full forward and backward passes:

```python
class LoRALayer:
    """Low-Rank Adaptation layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # LoRA matrices
        # A: random init, B: zero init (start as identity)
        self.A = np.random.randn(rank, in_features) * 0.01
        self.B = np.zeros((out_features, rank))

        self.cache = {}

    def forward(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """h = Wx + scaling * BAx"""
        # Original path (frozen)
        base_output = x @ W.T

        # LoRA path
        lora_output = (x @ self.A.T) @ self.B.T

        # Cache for backward
        self.cache = {'x': x, 'W': W, 'Ax': x @ self.A.T}

        return base_output + self.scaling * lora_output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients for A and B."""
        x = self.cache['x']
        W = self.cache['W']
        Ax = self.cache['Ax']

        scaled_grad = grad_output * self.scaling

        # Handle batched inputs
        if x.ndim == 3:
            x_flat = x.reshape(-1, self.in_features)
            Ax_flat = Ax.reshape(-1, self.rank)
            grad_flat = scaled_grad.reshape(-1, self.out_features)
        else:
            x_flat, Ax_flat, grad_flat = x, Ax, scaled_grad

        # Gradient w.r.t. B
        self.B_grad = grad_flat.T @ Ax_flat

        # Gradient w.r.t. A
        grad_Ax = grad_flat @ self.B
        self.A_grad = grad_Ax.T @ x_flat

        # Gradient w.r.t. input
        grad_x = grad_output @ W + scaled_grad @ self.B @ self.A

        return grad_x

    def merge_weights(self, W: np.ndarray) -> np.ndarray:
        """Merge LoRA into base weights for inference."""
        return W + self.scaling * (self.B @ self.A)

    def num_parameters(self) -> int:
        return self.A.size + self.B.size
```

### LoRA-Enhanced Linear Layer

Wrapping a pretrained layer with LoRA:

```python
class LoRALinear:
    """Linear layer with LoRA adaptation."""

    def __init__(
        self,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
    ):
        self.weight = weight  # Frozen
        self.bias = bias      # Frozen
        self.out_features, self.in_features = weight.shape

        self.lora = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=lora_rank,
            alpha=lora_alpha,
        )

    def forward(self, x):
        output = self.lora.forward(x, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output

    def merge(self):
        """Return merged weight and bias."""
        return self.lora.merge_weights(self.weight), self.bias
```

## Adapter Implementation

```python
class Adapter:
    """Bottleneck adapter layer."""

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int = 64,
    ):
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim

        # Small initialization (near-identity at start)
        scale = 0.01
        self.W_down = np.random.randn(d_model, bottleneck_dim) * scale
        self.W_up = np.random.randn(bottleneck_dim, d_model) * scale

        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x + adapter(x)"""
        # Down-project
        down = x @ self.W_down

        # ReLU activation
        activated = np.maximum(0, down)

        # Up-project
        up = activated @ self.W_up

        # Cache for backward
        self.cache = {'x': x, 'down': down, 'activated': activated}

        # Residual connection
        return x + up

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients."""
        x = self.cache['x']
        down = self.cache['down']
        activated = self.cache['activated']

        # Gradient through up-projection
        activated_flat = activated.reshape(-1, self.bottleneck_dim)
        grad_flat = grad_output.reshape(-1, self.d_model)
        self.W_up_grad = activated_flat.T @ grad_flat

        # Gradient through ReLU
        grad_activated = grad_output @ self.W_up.T
        grad_down = grad_activated * (down > 0)

        # Gradient through down-projection
        x_flat = x.reshape(-1, self.d_model)
        grad_down_flat = grad_down.reshape(-1, self.bottleneck_dim)
        self.W_down_grad = x_flat.T @ grad_down_flat

        # Gradient to input (residual + adapter path)
        return grad_output + grad_down @ self.W_down.T

    def num_parameters(self) -> int:
        return self.W_down.size + self.W_up.size
```

## Prefix Tuning Implementation

```python
class PrefixTuning:
    """Learned key/value prefixes for attention."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_head: int,
        prefix_length: int = 10,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_head = d_head
        self.prefix_length = prefix_length

        # Shape: [num_layers, 2 (K,V), prefix_length, num_heads, d_head]
        self.prefix = np.random.randn(
            num_layers, 2, prefix_length, num_heads, d_head
        ) * 0.01

    def get_prefix(self, layer_idx: int):
        """Get K and V prefixes for a layer."""
        prefix_k = self.prefix[layer_idx, 0]
        prefix_v = self.prefix[layer_idx, 1]
        return prefix_k, prefix_v

    def num_parameters(self) -> int:
        return self.prefix.size
```

## Prompt Tuning Implementation

```python
class PromptTuning:
    """Learned soft prompts prepended to input."""

    def __init__(
        self,
        d_model: int,
        prompt_length: int = 20,
        init_from_vocab: Optional[np.ndarray] = None,
    ):
        self.d_model = d_model
        self.prompt_length = prompt_length

        if init_from_vocab is not None:
            # Initialize from random vocabulary embeddings
            indices = np.random.choice(len(init_from_vocab), prompt_length)
            self.prompt = init_from_vocab[indices].copy()
        else:
            self.prompt = np.random.randn(prompt_length, d_model) * 0.01

    def forward(self, input_embeds: np.ndarray) -> np.ndarray:
        """Prepend soft prompts to input."""
        batch_size = input_embeds.shape[0]

        # Expand prompt for batch
        prompt_expanded = np.broadcast_to(
            self.prompt[np.newaxis, :, :],
            (batch_size, self.prompt_length, self.d_model)
        ).copy()

        return np.concatenate([prompt_expanded, input_embeds], axis=1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradient for prompts."""
        # Gradient for prompt (sum over batch)
        self.prompt_grad = grad_output[:, :self.prompt_length].sum(axis=0)

        # Pass through gradient for actual input
        return grad_output[:, self.prompt_length:]

    def num_parameters(self) -> int:
        return self.prompt.size
```

## Utility Functions

### Apply LoRA to Attention

```python
def apply_lora_to_attention(
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    wo: np.ndarray,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: List[str] = ['q', 'v'],
) -> Dict[str, Any]:
    """Apply LoRA to attention weight matrices."""
    layers = {}
    weights = {'q': wq, 'k': wk, 'v': wv, 'o': wo}

    for name, weight in weights.items():
        if name in target_modules:
            layers[name] = LoRALinear(weight, lora_rank=rank, lora_alpha=alpha)
        else:
            layers[name] = weight  # Keep frozen

    return layers
```

### Compare PEFT Methods

```python
def compare_peft_methods(
    d_model: int = 4096,
    num_layers: int = 32,
    num_heads: int = 32,
    lora_rank: int = 8,
    adapter_bottleneck: int = 64,
    prefix_length: int = 10,
    prompt_length: int = 20,
) -> Dict[str, Dict]:
    """Compare parameter counts across PEFT methods."""

    d_head = d_model // num_heads

    # Full fine-tuning: all attention weights (Q, K, V, O per layer)
    full_params = num_layers * 4 * d_model * d_model

    # LoRA on Q and V
    lora_params = num_layers * 2 * lora_rank * (d_model + d_model)

    # Adapters (one after attention, one after FFN)
    adapter_params = num_layers * 2 * 2 * d_model * adapter_bottleneck

    # Prefix tuning
    prefix_params = num_layers * 2 * prefix_length * num_heads * d_head

    # Prompt tuning
    prompt_params = prompt_length * d_model

    return {
        'full': {'params': full_params, 'ratio': 1.0},
        'lora': {'params': lora_params, 'ratio': lora_params / full_params},
        'adapters': {'params': adapter_params, 'ratio': adapter_params / full_params},
        'prefix': {'params': prefix_params, 'ratio': prefix_params / full_params},
        'prompt': {'params': prompt_params, 'ratio': prompt_params / full_params},
    }
```

## Running the Demo

```bash
cd code/stage-09
python peft.py
```

Output:

```
============================================================
Stage 9: Fine-tuning & Parameter-Efficient Methods
============================================================

1. LoRA (Low-Rank Adaptation)
----------------------------------------
Original weight shape: (256, 256)
LoRA A shape: (8, 256)
LoRA B shape: (256, 8)
LoRA parameters: 4,096
Full parameters: 65,536
Compression: 6.25%
Input shape: (2, 16, 256)
Output shape: (2, 16, 256)
Gradients computed: A=(8, 256), B=(256, 8)

2. Adapter Layers
----------------------------------------
Adapter parameters: 32,768
Bottleneck dimension: 64

3. Prompt Tuning
----------------------------------------
Original sequence length: 16
With soft prompt: 36
Prompt parameters: 5,120

4. PEFT Method Comparison (7B model scale)
----------------------------------------
Method               Parameters      Ratio     Description
----------------------------------------------------------------------
full_fine_tuning     2,147,483,648  100.0000%  Update all attention weights
lora                     4,194,304    0.1953%  LoRA rank=8 on Q,V
adapters                33,554,432    1.5625%  Bottleneck=64
prefix_tuning            2,621,440    0.1221%  10 prefix tokens
prompt_tuning               81,920    0.0038%  20 soft prompts
```

## Summary

| Method | Key Classes | Parameters |
|--------|-------------|------------|
| LoRA | `LoRALayer`, `LoRALinear` | A, B matrices |
| Adapters | `Adapter` | W_down, W_up |
| Prefix Tuning | `PrefixTuning` | prefix tensor |
| Prompt Tuning | `PromptTuning` | prompt embeddings |

All implementations follow the same pattern:

1. Initialize trainable parameters
2. Implement forward pass
3. Implement backward pass
4. Provide parameter count

## Exercises

1. **Extend LoRA**: Apply to feed-forward layers
2. **LoRA dropout**: Add dropout to the LoRA path
3. **Adapter variants**: Implement parallel adapters
4. **Prefix MLP**: Generate prefixes from a learned embedding via MLP
5. **Combination**: Implement LoRA + Prompt Tuning together
6. **Quantization**: Add 4-bit LoRA (QLoRA)

## What's Next

With PEFT methods mastered, we're ready for Stage 10: Alignment—teaching models to be helpful, harmless, and honest through RLHF and related techniques.

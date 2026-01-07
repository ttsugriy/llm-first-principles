# Stage 9 Exercises

## Conceptual Questions

### Exercise 9.1: The Fine-tuning Problem
Consider a 7B parameter model that you want to fine-tune for a specific task.

**a)** How much GPU memory is needed for full fine-tuning in FP16?
**b)** Why does fine-tuning all parameters often lead to worse generalization?
**c)** What does "catastrophic forgetting" mean in this context?

### Exercise 9.2: LoRA Mathematics
For LoRA with rank r=8 applied to a 4096x4096 weight matrix:

**a)** How many trainable parameters does LoRA add?
**b)** What percentage is this of the original matrix?
**c)** After training, can we recover a single weight matrix? How?

### Exercise 9.3: Method Comparison
Compare LoRA, Adapters, and Prompt Tuning:

**a)** Which method adds parameters during inference?
**b)** Which method is easiest to combine for multi-task models?
**c)** Which method requires changing the model architecture?

### Exercise 9.4: Rank Selection
Why is rank important in LoRA?

**a)** What happens if rank is too low?
**b)** What happens if rank is too high?
**c)** How would you choose rank for a new task?

---

## Implementation Exercises

### Exercise 9.5: LoRA Layer
Implement the core LoRA layer:

```python
class LoRALayer:
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16):
        """
        Initialize LoRA layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: LoRA rank (r)
            alpha: Scaling factor
        """
        self.scaling = alpha / rank

        # TODO: Initialize A and B matrices
        # A: (rank, in_features) - Gaussian init
        # B: (out_features, rank) - Zero init
        pass

    def forward(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Forward pass: Wx + scaling * (B @ A @ x)

        Args:
            x: Input [batch, seq, in_features]
            W: Frozen pretrained weights [out_features, in_features]
        """
        # TODO
        pass

    def merge(self, W: np.ndarray) -> np.ndarray:
        """Merge LoRA into base weights for inference."""
        # TODO
        pass
```

### Exercise 9.6: LoRA Backward Pass
Implement gradients for LoRA:

```python
def lora_backward(
    grad_output: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    scaling: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward pass through LoRA.

    Args:
        grad_output: Gradient of loss w.r.t. output
        x: Input (cached from forward)
        A, B: LoRA matrices
        scaling: LoRA scaling factor

    Returns:
        (grad_A, grad_B, grad_x)
    """
    # TODO: Compute gradients
    # Note: W is frozen, no gradient needed
    pass
```

### Exercise 9.7: Adapter Layer
Implement an adapter bottleneck layer:

```python
class Adapter:
    def __init__(self, d_model: int, bottleneck: int = 64):
        """
        Adapter layer: down -> activation -> up -> residual

        Args:
            d_model: Model dimension
            bottleneck: Bottleneck dimension
        """
        # TODO: Initialize W_down, W_up
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward: x + up(relu(down(x)))
        """
        # TODO
        pass
```

### Exercise 9.8: Prompt Tuning
Implement soft prompt tuning:

```python
class PromptTuning:
    def __init__(self, d_model: int, prompt_length: int = 20):
        """
        Initialize soft prompts.

        Args:
            d_model: Embedding dimension
            prompt_length: Number of soft prompt tokens
        """
        # TODO: Initialize learnable prompt embeddings
        pass

    def forward(self, input_embeddings: np.ndarray) -> np.ndarray:
        """
        Prepend soft prompts to input.

        Args:
            input_embeddings: [batch, seq, d_model]

        Returns:
            [batch, prompt_length + seq, d_model]
        """
        # TODO
        pass
```

---

## Challenge Exercises

### Exercise 9.9: LoRA for Multi-Head Attention
Apply LoRA to a full multi-head attention layer:

```python
class LoRAMultiHeadAttention:
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rank: int = 8,
        target_modules: List[str] = ['q', 'v'],
    ):
        """
        Multi-head attention with LoRA on selected projections.

        Args:
            d_model: Model dimension
            n_heads: Number of heads
            rank: LoRA rank
            target_modules: Which projections to apply LoRA to
                           ('q', 'k', 'v', 'o')
        """
        # TODO: Initialize attention with LoRA on target modules
        pass

    def forward(self, x: np.ndarray, mask=None) -> np.ndarray:
        """Forward pass with LoRA-enhanced projections."""
        # TODO
        pass

    def merge_lora(self):
        """Merge all LoRA weights into base weights."""
        # TODO
        pass

    def count_trainable_params(self) -> int:
        """Count only LoRA parameters."""
        # TODO
        pass
```

### Exercise 9.10: Parameter Efficiency Analysis
Build a tool to analyze PEFT efficiency:

```python
def analyze_peft_efficiency(
    model_config: Dict[str, int],  # d_model, n_layers, n_heads
    method: str,  # 'lora', 'adapter', 'prefix', 'prompt'
    method_config: Dict[str, Any],  # rank, bottleneck, length, etc.
) -> Dict[str, Any]:
    """
    Analyze parameter efficiency of a PEFT method.

    Returns:
        - trainable_params: Number of trainable parameters
        - total_params: Total model parameters
        - efficiency_ratio: trainable / total
        - memory_estimate: Approximate memory savings
    """
    # TODO
    pass
```

### Exercise 9.11: LoRA Merging and Switching
Implement LoRA weight management for multi-task:

```python
class LoRAManager:
    def __init__(self, base_model):
        self.base_model = base_model
        self.adapters = {}  # task_name -> LoRA weights

    def add_adapter(self, name: str, lora_weights: Dict):
        """Register a new LoRA adapter."""
        # TODO
        pass

    def switch_adapter(self, name: str):
        """Switch to a different adapter."""
        # TODO
        pass

    def merge_adapter(self, name: str) -> None:
        """Permanently merge adapter into base weights."""
        # TODO
        pass

    def combine_adapters(self, names: List[str], weights: List[float]) -> None:
        """Linearly combine multiple adapters."""
        # TODO: Implement weighted average of adapters
        pass
```

### Exercise 9.12: QLoRA Basics
Explore quantized LoRA concepts:

```python
def quantize_to_4bit(weight: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Quantize weight to 4-bit representation.

    Returns:
        (quantized_weight, scale, zero_point)
    """
    # TODO: Implement simple 4-bit quantization
    pass

def dequantize_4bit(
    quantized: np.ndarray,
    scale: float,
    zero_point: float
) -> np.ndarray:
    """Dequantize 4-bit weight back to float."""
    # TODO
    pass

class QLoRALayer:
    """LoRA with 4-bit quantized base weights."""
    def __init__(self, weight_4bit, scale, zero_point, rank: int = 8):
        # TODO: Store quantized weights and LoRA in full precision
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: Dequantize on-the-fly and apply LoRA
        pass
```

---

## Solutions

Solutions are available in `code/stage-09/solutions/`.

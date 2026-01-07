# Section 9.2: LoRA (Low-Rank Adaptation)

*Reading time: 15 minutes*

## The Big Idea

LoRA (Hu et al., 2021) is elegantly simple:

Instead of updating $W$ directly, learn the *update* as a low-rank matrix:

$$W' = W + \Delta W = W + BA$$

Where:

- $W \in \mathbb{R}^{d \times k}$: Frozen pretrained weights
- $B \in \mathbb{R}^{d \times r}$: Trainable down-projection
- $A \in \mathbb{R}^{r \times k}$: Trainable up-projection
- $r \ll \min(d, k)$: The rank (typically 4-64)

## Why It Works

### The Low-Rank Hypothesis

The change in weights during fine-tuning has low intrinsic rank.

Think of it this way: fine-tuning adjusts the model slightly to perform a new task. These adjustments form patterns—they're not random noise.

Patterns can be captured with low-rank matrices.

### Visual Intuition

```
Original W (4096 × 4096 = 16.7M params)
┌────────────────────────────────────────┐
│                                        │
│     [Full matrix - frozen]             │
│                                        │
└────────────────────────────────────────┘
                    +
LoRA: B × A (r=8: 65K params)
┌────┐   ┌────────────────────────────────┐
│    │   │                                │
│ B  │ × │              A                 │
│    │   │                                │
└────┘   └────────────────────────────────┘
(4096×8)        (8×4096)
```

## The Math

### Forward Pass

Standard linear layer:

$$h = Wx$$

With LoRA:

$$h = Wx + \frac{\alpha}{r}BAx$$

Where $\frac{\alpha}{r}$ is a scaling factor.

### Why Scaling?

When you change rank $r$, you want behavior to stay consistent.

- Higher $r$: More capacity, but $BA$ product changes magnitude
- $\alpha$ controls the "strength" of adaptation
- Dividing by $r$ normalizes across different ranks

**Rule of thumb**: Set $\alpha = 2r$ (so scaling = 2)

### Initialization

Critical for stable training:

- $A$: Random Gaussian with small variance
- $B$: **Zeros**

Why zeros for $B$? So that $BA = 0$ at initialization. The model starts exactly as the pretrained model.

## Implementation

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
        self.rank = rank
        self.scaling = alpha / rank

        # LoRA matrices
        self.A = np.random.randn(rank, in_features) * 0.01
        self.B = np.zeros((out_features, rank))  # Start at zero!

    def forward(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """h = Wx + scaling * BAx"""
        base = x @ W.T                    # Original path (frozen)
        lora = (x @ self.A.T) @ self.B.T  # LoRA path
        return base + self.scaling * lora

    def backward(self, grad_output, x):
        """Compute gradients for A and B only."""
        Ax = x @ self.A.T

        # Gradient w.r.t. B: grad @ Ax
        self.B_grad = grad_output.T @ Ax * self.scaling

        # Gradient w.r.t. A: B.T @ grad @ x
        grad_Ax = grad_output @ self.B
        self.A_grad = grad_Ax.T @ x * self.scaling

        return grad_output @ W  # Pass gradient to input

    def merge_weights(self, W):
        """Merge LoRA into base weights for inference."""
        return W + self.scaling * (self.B @ self.A)
```

## Where to Apply LoRA

LoRA can be applied to any linear layer. Common choices:

### Attention Layers

| Target | Effect |
|--------|--------|
| Query (Q) | How tokens attend |
| Key (K) | What tokens are attended to |
| Value (V) | What information is retrieved |
| Output (O) | How attention output is projected |

**Most common**: Apply to Q and V only.

Why Q and V?

- Q determines *what* the model looks for
- V determines *what* information is extracted
- K and O are often less important for task adaptation

### Feed-Forward Layers

Less common, but can help for knowledge-intensive tasks.

## Hyperparameters

### Rank (r)

| Rank | Params (4096 dim) | Use Case |
|------|-------------------|----------|
| 4 | 32K | Simple tasks, small data |
| 8 | 65K | General purpose |
| 16 | 131K | Complex tasks |
| 64 | 524K | Very complex tasks |

**Start with r=8**. Increase if underfitting.

### Alpha (α)

Controls adaptation strength:

- α = r: Scaling = 1 (conservative)
- α = 2r: Scaling = 2 (common default)

**Rule**: If using learning rate that worked for full fine-tuning, start with α = 2r.

### Target Modules

Which layers to apply LoRA:

```python
# Conservative (fewer params, less risk)
target_modules = ['q_proj', 'v_proj']

# Aggressive (more params, more capacity)
target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                  'up_proj', 'down_proj']
```

## The Magic: Weight Merging

After training, you can **merge** LoRA weights into the base model:

```python
W_merged = W + scaling * (B @ A)
```

Benefits:

- **Zero inference overhead**: Same speed as original model
- **Deployable anywhere**: Just swap the weights
- **Stackable**: Train multiple LoRAs, merge selectively

## Training Procedure

```python
# 1. Load pretrained model
model = load_pretrained("llama-7b")

# 2. Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# 3. Add LoRA layers
for layer in model.attention_layers:
    layer.q_proj = LoRALinear(layer.q_proj, rank=8)
    layer.v_proj = LoRALinear(layer.v_proj, rank=8)

# 4. Train only LoRA parameters
optimizer = Adam([p for p in model.lora_parameters()])

for batch in dataset:
    loss = model(batch)
    loss.backward()
    optimizer.step()

# 5. Optionally merge for deployment
for layer in model.attention_layers:
    layer.q_proj = layer.q_proj.merge()
    layer.v_proj = layer.v_proj.merge()
```

## Comparison to Full Fine-Tuning

| Aspect | Full Fine-Tuning | LoRA |
|--------|------------------|------|
| Trainable params | 100% | ~0.1% |
| GPU memory | Very high | Low |
| Training speed | Slower | Faster |
| Storage per task | Full model | ~10MB |
| Inference speed | Same | Same (merged) |
| Quality | Baseline | 95-100% |

## When LoRA Works Best

**Good for**:

- Instruction following
- Style adaptation
- Domain-specific tasks
- Limited compute/memory

**Might need more**:

- Learning entirely new knowledge
- Very different output formats
- Extreme domain shift

## Common Mistakes

1. **Forgetting to freeze base weights**: Training everything defeats the purpose
2. **Rank too low**: Underfitting. Try increasing rank.
3. **Rank too high**: Overfitting. Also slower and more memory.
4. **Wrong learning rate**: LoRA often needs higher LR than full fine-tuning (try 1e-4 to 1e-3)
5. **Not merging for production**: Keep LoRA separate during experimentation, merge for deployment

## Summary

| Component | Purpose |
|-----------|---------|
| A matrix | Projects input to low-rank space |
| B matrix | Projects from low-rank space to output |
| Rank r | Controls capacity (trade-off) |
| Alpha α | Controls adaptation strength |
| Scaling | Normalizes across ranks |
| Merging | Eliminates inference overhead |

**Key insight**: LoRA exploits the fact that fine-tuning updates are inherently low-rank. By parameterizing them as such, we get massive efficiency gains with minimal quality loss.

**Next**: We'll explore adapters, another approach to parameter-efficient fine-tuning.

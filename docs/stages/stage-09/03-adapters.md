# Section 9.3: Adapter Layers

*Reading time: 10 minutes*

## The Adapter Idea

Adapters (Houlsby et al., 2019) take a different approach from LoRA:

Instead of modifying existing weights, **insert new layers** into the model.

```
Original Transformer Layer:
    Input → Attention → Add & Norm → FFN → Add & Norm → Output

With Adapters:
    Input → Attention → Add & Norm → [Adapter] → FFN → Add & Norm → [Adapter] → Output
                                      ↑                                ↑
                                 (inserted)                       (inserted)
```

## Adapter Architecture

An adapter is a small bottleneck network:

```
Input (d_model)
    │
    ▼
Down-project (d_model → bottleneck)
    │
    ▼
Nonlinearity (ReLU/GELU)
    │
    ▼
Up-project (bottleneck → d_model)
    │
    ▼
+ Residual connection
    │
    ▼
Output (d_model)
```

The bottleneck creates an information squeeze—the adapter must learn a compressed representation.

## The Math

$$h' = h + f(h W_{down}) W_{up}$$

Where:

- $h \in \mathbb{R}^{d}$: Input (d_model dimensional)
- $W_{down} \in \mathbb{R}^{d \times b}$: Down-projection (bottleneck dimension b)
- $W_{up} \in \mathbb{R}^{b \times d}$: Up-projection back to d_model
- $f$: Nonlinearity (usually ReLU or GELU)
- Residual connection ensures the adapter can "pass through" if needed

## Implementation

```python
class Adapter:
    """Bottleneck adapter layer."""

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int = 64,
        activation: str = 'relu',
    ):
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim

        # Initialize small (near-identity at start)
        scale = 0.01
        self.W_down = np.random.randn(d_model, bottleneck_dim) * scale
        self.W_up = np.random.randn(bottleneck_dim, d_model) * scale

    def forward(self, x):
        """x + adapter(x)"""
        down = x @ self.W_down          # Project down
        activated = np.maximum(0, down)  # ReLU
        up = activated @ self.W_up       # Project up
        return x + up                    # Residual

    def backward(self, grad_output):
        # Gradient flows through both residual and adapter path
        # ... (compute gradients for W_down and W_up)
        pass
```

## Bottleneck Size Trade-offs

| Bottleneck | Params per Adapter | Capacity | Speed |
|------------|-------------------|----------|-------|
| 16 | 2 × 4096 × 16 = 131K | Low | Fast |
| 64 | 2 × 4096 × 64 = 524K | Medium | Medium |
| 256 | 2 × 4096 × 256 = 2.1M | High | Slower |

**Default recommendation**: Start with bottleneck = 64.

## Where to Insert Adapters

### Original Placement (Houlsby)

Two adapters per transformer layer:

1. After multi-head attention (before layer norm)
2. After feed-forward network (before layer norm)

### Efficient Variant (Pfeiffer)

One adapter per layer:

- After FFN only
- Nearly as effective, half the parameters

```python
class TransformerLayerWithAdapter:
    def forward(self, x):
        # Attention
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)

        # Adapter (Pfeiffer placement)
        adapted = self.adapter(ffn_out)

        x = self.norm2(x + adapted)
        return x
```

## Adapters vs LoRA

| Aspect | Adapters | LoRA |
|--------|----------|------|
| Where | New layers inserted | Modifies existing weights |
| Architecture | Changes model structure | Same structure |
| Inference | Slight overhead | Zero overhead (merged) |
| Params | ~1% | ~0.1% |
| Flexibility | More architectural freedom | Simpler |

### When to Choose Adapters

- When you want to swap adapters at inference time
- When you need more capacity than LoRA provides
- When you're comfortable with slight inference overhead

### When to Choose LoRA

- When you need minimal overhead
- When you want to merge weights for deployment
- When parameter count is critical

## Adapter Fusion

Multiple adapters can be combined:

```python
def fused_adapter_forward(x, adapters, weights):
    """Combine multiple adapters with learned weights."""
    outputs = [adapter.forward(x) for adapter in adapters]
    weighted = sum(w * out for w, out in zip(weights, outputs))
    return x + weighted
```

This allows:

- Training task-specific adapters
- Combining them for multi-task inference
- Dynamic weighting based on input

## Training Adapters

```python
# Freeze all original parameters
for param in model.parameters():
    param.requires_grad = False

# Add adapters
for layer in model.layers:
    layer.adapter = Adapter(d_model=4096, bottleneck_dim=64)

# Train only adapter parameters
optimizer = Adam(
    [p for layer in model.layers for p in layer.adapter.parameters()]
)

for batch in dataset:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

## Adapter Initialization

Good initialization is crucial:

**Small weights**: So adapter starts near identity (minimal impact)

```python
# Common initialization
W_down = np.random.randn(d_model, bottleneck) * 0.01
W_up = np.random.randn(bottleneck, d_model) * 0.01
```

**Zero up-projection**: Forces adapter to learn from scratch

```python
W_down = np.random.randn(d_model, bottleneck) * 0.01
W_up = np.zeros((bottleneck, d_model))  # Zero!
```

## Inference Considerations

Unlike LoRA, adapters cannot be "merged" into base weights. Options:

**1. Keep adapters separate**

- Switch adapters for different tasks
- Slight inference overhead (extra matrix multiplies)

**2. Distillation**

- Train a student model without adapters
- Student mimics adapted model behavior
- No inference overhead, but requires extra training

## Common Mistakes

1. **Bottleneck too small**: Underfitting, adapter can't capture task
2. **Bottleneck too large**: Overfitting, also slower
3. **Missing residual**: Without residual, adapter can't "pass through"
4. **Wrong placement**: Pfeiffer placement is more efficient
5. **Initialization too large**: Destroys pretrained behavior

## Summary

| Component | Purpose |
|-----------|---------|
| Down-projection | Compress to bottleneck |
| Nonlinearity | Add capacity |
| Up-projection | Expand back to d_model |
| Residual | Allow pass-through |

**Key insight**: Adapters insert trainable modules that can learn task-specific transformations while the original model remains frozen.

**Next**: We'll explore prefix tuning and prompt tuning—methods that don't modify weights at all.

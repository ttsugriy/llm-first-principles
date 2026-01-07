# Section 6.3: Building Deep Networks — Stacking Layers

*Reading time: 18 minutes | Difficulty: ★★★☆☆*

Modern LLMs stack dozens or even hundreds of Transformer blocks. This section examines how to build deep networks that train stably and what depth provides.

## Why Depth?

### Depth vs Width

Given a fixed parameter budget, should we go deep (more layers) or wide (larger dimensions)?

```
Option A: 6 layers, d_model=2048   (~150M params)
Option B: 24 layers, d_model=1024  (~150M params)
```

Empirically, **depth wins** for most tasks:

| Property | Wide & Shallow | Narrow & Deep |
|----------|----------------|---------------|
| Representational power | Similar | Similar |
| Sample efficiency | Worse | Better |
| Compositional reasoning | Harder | Easier |
| Training stability | Easier | Harder |

### What Depth Provides

Each layer can perform a different type of computation:

```
Layer 1-4:   Low-level patterns (syntax, local context)
Layer 5-12:  Mid-level features (phrases, entities)
Layer 13-24: High-level reasoning (relationships, inference)
```

Deep networks can compose these computations hierarchically.

## Modern Model Depths

| Model | Layers | d_model | Heads | Parameters |
|-------|--------|---------|-------|------------|
| GPT-2 Small | 12 | 768 | 12 | 124M |
| GPT-2 Medium | 24 | 1024 | 16 | 355M |
| GPT-2 Large | 36 | 1280 | 20 | 774M |
| GPT-2 XL | 48 | 1600 | 25 | 1.5B |
| LLaMA 7B | 32 | 4096 | 32 | 7B |
| LLaMA 70B | 80 | 8192 | 64 | 70B |
| GPT-4 | ~120? | ~12K? | ~96? | ~1.8T? |

The trend: more layers, more parameters, more capability.

## Initialization: Starting Right

Proper initialization is crucial for training deep networks.

### The Problem

With random initialization:

```
Layer 1 output: variance = 1
Layer 2 output: variance = 2     (grows!)
Layer 10 output: variance = 1024  (explodes!)

OR

Layer 1 output: variance = 1
Layer 2 output: variance = 0.5   (shrinks!)
Layer 10 output: variance = 0.001 (vanishes!)
```

### Xavier/Glorot Initialization

For linear layers, initialize weights to maintain variance:

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

Or uniformly:

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

### Kaiming/He Initialization

For ReLU networks (accounts for the fact that ReLU zeros half the values):

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$$

### Transformer-Specific Initialization

Modern Transformers often use:

1. **Standard initialization** for most weights
2. **Scaled initialization** for residual projections

```python
def init_weights(layer, n_layers):
    """
    Initialize weights for Transformer.

    Residual projections are scaled by 1/√(2n_layers)
    to prevent output variance from growing with depth.
    """
    std = 0.02  # Base standard deviation

    # Standard layers
    layer.weight = np.random.randn(*shape) * std

    # Residual projections (attention out, FFN out)
    if is_residual_projection:
        layer.weight = np.random.randn(*shape) * std / np.sqrt(2 * n_layers)
```

The 1/√(2n_layers) factor ensures that even with many layers, the output variance doesn't explode.

## Gradient Flow in Deep Networks

### The Vanishing Gradient Problem

Without residuals, gradients must flow through many layers:

$$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_n} \cdot \prod_{i=1}^{n-1} \frac{\partial x_{i+1}}{\partial x_i}$$

If each ∂x_{i+1}/∂x_i < 1, the product vanishes exponentially.

### Residual Connections Save the Day

With residuals:

$$x_{i+1} = x_i + f_i(x_i)$$

$$\frac{\partial x_{i+1}}{\partial x_i} = 1 + \frac{\partial f_i}{\partial x_i}$$

Even if ∂f_i/∂x_i is small, the gradient is at least 1!

```
Without residuals:
  gradient = 0.9^100 = 2.6 × 10^-5  (vanishes!)

With residuals:
  gradient ≥ 1  (preserved!)
```

### Gradient Visualization

```
Deep network WITHOUT residuals:
Layer 1  ████████████████  (large gradient)
Layer 2  ███████████       (smaller)
Layer 5  ███               (small)
Layer 10 ░                 (vanishing!)

Deep network WITH residuals:
Layer 1  ████████████████
Layer 2  ████████████████
Layer 5  ████████████████
Layer 10 ████████████████  (all healthy!)
```

## Layer Normalization Placement (Revisited)

For very deep networks, pre-norm is essential:

```python
# Pre-norm: gradients flow through residual path
def prenorm_block(x):
    x = x + Attention(LayerNorm(x))  # Gradient = 1 + ...
    x = x + FFN(LayerNorm(x))        # Gradient = 1 + ...
    return x

# Post-norm: gradient must flow through LayerNorm
def postnorm_block(x):
    x = LayerNorm(x + Attention(x))  # Gradient through LN!
    x = LayerNorm(x + FFN(x))        # Again through LN!
    return x
```

### Why Pre-Norm Scales Better

| Depth | Post-Norm | Pre-Norm |
|-------|-----------|----------|
| 6 layers | Works fine | Works fine |
| 24 layers | Needs careful tuning | Easy to train |
| 96 layers | Very difficult | Still works |
| 200+ layers | Basically impossible | Possible |

## Depth-Specific Techniques

### μP (Maximal Update Parameterization)

A systematic way to set hyperparameters that transfer across model sizes:

- Learning rates scale with width
- Initialization scales with depth
- Enables training very large models without extensive tuning

### Depth-wise Learning Rates

Some research suggests different learning rates per layer:

```python
def get_layer_lr(layer_idx, base_lr, n_layers):
    # Later layers might need smaller LR
    return base_lr * (0.9 ** (n_layers - layer_idx))
```

### Stochastic Depth

Randomly drop entire layers during training:

```python
def forward_with_stochastic_depth(x, layers, drop_prob=0.1):
    for layer in layers:
        if training and random() < drop_prob:
            continue  # Skip this layer
        x = layer(x)
    return x
```

This acts as regularization and can speed up training.

## What Each Layer Learns

Research on probing Transformers reveals layer specialization:

### Early Layers (1-4)

- Part-of-speech tagging
- Named entity recognition
- Local syntactic patterns
- Character/subword patterns

### Middle Layers (5-16)

- Dependency parsing
- Coreference resolution
- Semantic roles
- Entity relationships

### Later Layers (17+)

- Task-specific representations
- Complex reasoning
- Abstract concepts
- Output formatting

### Visualization

```
Task: Question Answering

Layer 1:  [tokens are processed individually]
Layer 4:  [local patterns emerge: "What is", "?"]
Layer 8:  [entities linked: "Einstein" → "physicist"]
Layer 12: [question understood: asking about birthdate]
Layer 16: [answer located in context]
Layer 20: [answer formatted for output]
```

## Deep Network Implementation

```python
class DeepTransformer:
    """
    Deep Transformer with proper initialization and stability.
    """

    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=12,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1
    ):
        self.n_layers = n_layers
        self.d_model = d_model

        # Embeddings
        self.token_emb = self._init_embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(max_seq_len, d_model)

        # Transformer blocks
        self.layers = []
        for i in range(n_layers):
            block = TransformerBlock(d_model, n_heads, d_ff, dropout)
            self._init_block(block, n_layers)
            self.layers.append(block)

        # Final norm and output
        self.final_norm = LayerNorm(d_model)
        self.output_proj = self._init_linear(d_model, vocab_size)

    def _init_embedding(self, vocab_size, d_model):
        """Initialize embedding with scaled weights."""
        emb = np.random.randn(vocab_size, d_model) * 0.02
        return emb

    def _init_linear(self, d_in, d_out):
        """Initialize linear layer."""
        std = np.sqrt(2.0 / (d_in + d_out))
        return np.random.randn(d_in, d_out) * std

    def _init_block(self, block, n_layers):
        """
        Initialize block with scaled residual projections.
        """
        # Scale residual projections by 1/sqrt(2*n_layers)
        scale = 1.0 / np.sqrt(2 * n_layers)

        # Attention output projection
        block.attention.W_o *= scale

        # FFN output projection
        block.ffn.w2 *= scale

    def forward(self, tokens):
        """Forward pass through deep network."""
        # Embed
        x = self.token_emb[tokens]
        x = x + self.pos_enc(len(tokens))

        # Create causal mask
        mask = create_causal_mask(len(tokens))

        # Process through all layers
        for layer in self.layers:
            x = layer.forward(x, mask)

        # Final norm and project
        x = self.final_norm(x)
        logits = x @ self.output_proj

        return logits
```

## Scaling Considerations

### Memory

Deep networks require more memory:

```
Memory ≈ batch_size × seq_len × d_model × n_layers × 2
                                                    ↑
                                            (activations + gradients)
```

Techniques to manage memory:
- Gradient checkpointing
- Mixed precision training
- Activation recomputation

### Compute

Each layer adds compute:

```
FLOPs per layer ≈ 12 × d_model² × seq_len  (approximate)
Total FLOPs ≈ n_layers × 12 × d_model² × seq_len
```

### Training Time

Deeper networks take longer per step but may need fewer steps:

```
Steps to convergence × Time per step
        ↓                     ↑
  (may decrease)        (increases)
```

!!! info "Connection to Modern LLMs"

    The deepest production models:

    - **GPT-4**: Rumored to have 120+ layers (unconfirmed)
    - **LLaMA 70B**: 80 layers
    - **Claude**: Unknown, likely 80+ layers

    Training such deep models requires:
    - Careful initialization (μP or similar)
    - Mixed precision (FP16/BF16 with FP32 accumulation)
    - Gradient checkpointing
    - Distributed training across many GPUs

## Exercises

1. **Depth ablation**: Train models with 2, 4, 8, 16 layers. Plot loss vs depth.

2. **Initialization experiment**: Compare random init vs scaled init on a 24-layer model.

3. **Gradient flow**: Measure gradient norms at each layer. Are they stable?

4. **Layer probing**: Freeze all but one layer. Which layer is most important for your task?

5. **Remove a layer**: What happens if you remove layer 6 from a trained 12-layer model?

## Summary

| Concept | Definition | Why It Matters |
|---------|------------|----------------|
| Depth | Number of layers | More compositional computation |
| Initialization | Weight starting values | Prevents explosion/vanishing |
| Residual scaling | 1/√(2n) for residual projections | Stable deep networks |
| Pre-norm | Normalize before sublayers | Better gradient flow |
| Layer specialization | Different layers learn different things | Hierarchical processing |

**Key takeaway**: Building deep Transformer networks requires careful attention to initialization, normalization placement, and gradient flow. Residual connections provide a direct gradient path that enables training networks with 100+ layers. Proper scaling of residual projections (1/√(2n_layers)) prevents activation variance from growing with depth. These techniques enable the very deep networks that power modern LLMs.

→ **Next**: [Section 6.4: Pre-training Objectives](04-pretraining.md)

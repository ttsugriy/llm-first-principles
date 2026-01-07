# Capstone: End-to-End Transformer Training

*Putting it all together: A complete, trainable transformer from first principles*

## Overview

This capstone project brings together everything from Stages 1-10 into a single, complete transformer language model that you can train from scratch. Unlike using a framework like PyTorch with autodiff, we implement every component by hand—including the backward pass.

This is the final test of your understanding: if you can follow this code, you truly understand how transformers work at a fundamental level.

## What This Demonstrates

| Stage | Concept | Where It Appears |
|-------|---------|------------------|
| 1 | Language modeling, perplexity | Loss function, evaluation |
| 2 | Backpropagation | Manual `backward()` methods |
| 3 | Neural networks | Embeddings, linear layers |
| 4 | Adam optimizer | Training loop |
| 5 | Attention | Multi-head self-attention |
| 6 | Transformers | Full architecture |
| 7 | Tokenization | Character tokenizer (extendable to BPE) |
| 8 | Training dynamics | Learning rate schedules, gradient clipping |

## The Core Insight

The key insight of this capstone is that **autodiff is not magic**. Every `backward()` method we implement is exactly what PyTorch does automatically. By writing it ourselves, we understand:

1. What gets cached during forward pass
2. How gradients flow backward through each operation
3. Why certain architectural choices matter for gradient flow
4. The computational cost of training vs. inference

## Architecture Choices

The model uses modern architectural patterns:

| Choice | What It Is | Why It Matters |
|--------|------------|----------------|
| **RMSNorm** | Root mean square normalization | Simpler than LayerNorm, faster |
| **SwiGLU** | Gated linear unit with SiLU | Better performance than GELU |
| **Pre-norm** | Normalize before sublayer | More stable training for deep networks |
| **Tied embeddings** | Input and output share weights | 50% parameter reduction |
| **Causal masking** | Can only attend to past tokens | Enables autoregressive generation |

## File Structure

```
code/capstone/
├── model.py          # Complete trainable transformer
├── train.py          # Training script with logging
└── tests/
    └── test_capstone.py  # 23 comprehensive tests
```

## Key Components

### Parameter Container

Every learnable parameter is wrapped in a `Parameter` class that stores both data and gradients:

```python
@dataclass
class Parameter:
    data: np.ndarray
    grad: Optional[np.ndarray] = None

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
```

### Manual Backward Pass

Each layer implements its own backward pass using the chain rule:

```python
class FeedForward:
    def forward(self, x):
        # Cache values needed for backward
        self.cache = {'x': x, 'hidden': hidden, ...}
        return output

    def backward(self, grad_output):
        # Retrieve cached values
        x = self.cache['x']

        # Compute parameter gradients
        self.w1.grad = ...
        self.w2.grad = ...

        # Compute input gradient for chain rule
        return grad_input
```

### Training Loop

The complete training loop follows the pattern you learned in Stage 4:

```python
for epoch in range(epochs):
    for inputs, targets in batches:
        # Forward pass
        logits = model.forward(inputs)
        loss, grad = cross_entropy_loss(logits, targets)

        # Backward pass
        model.zero_grad()
        model.backward(grad)

        # Gradient clipping (Stage 8)
        clip_grad_norm(params, max_norm=1.0)

        # Optimizer step (Stage 4)
        optimizer.step()
        scheduler.step()
```

## Model Sizes

| Configuration | Parameters | Use Case |
|---------------|------------|----------|
| Tiny (default) | ~800K | Learning, debugging |
| Small | ~3M | Character-level text |
| Medium | ~12M | Actual generation |

For comparison:
- GPT-2 Small: 117M parameters
- GPT-2 XL: 1.5B parameters
- LLaMA 7B: 7B parameters

Our capstone is a *toy model* for learning, not production.

## Quick Start

```bash
# Navigate to capstone directory
cd code/capstone

# Run with default settings (trains on Shakespeare)
python train.py

# Custom training
python train.py --epochs 50 --lr 3e-4 --d-model 256 --n-layers 6

# Train on your own text
python train.py --text-file path/to/your/text.txt --epochs 100

# Run tests
python tests/test_capstone.py
```

## Understanding Through Tests

The test suite (`test_capstone.py`) covers:

1. **Utility functions**: softmax, silu, causal mask
2. **Component tests**: RMSNorm, Attention, FFN
3. **Full model tests**: Forward shape, parameter count
4. **Gradient tests**: Numerical gradient checking
5. **Training tests**: Convergence on simple data

Running the tests is a great way to verify your understanding.

## Extending the Capstone

After mastering the basics, try these extensions:

1. **Replace CharTokenizer with BPE** (Stage 7)
2. **Add training diagnostics** (Stage 8)
3. **Implement LoRA fine-tuning** (Stage 9)
4. **Add KV-cache for faster generation**
5. **Implement Flash Attention for memory efficiency**

## Connection to Production Systems

Everything in this capstone maps directly to production LLMs:

| This Capstone | Production (PyTorch/JAX) |
|---------------|--------------------------|
| `model.forward()` | Same, but compiled |
| `model.backward()` | Automatic differentiation |
| `Adam` optimizer | Same algorithm |
| `WarmupCosine` schedule | Standard practice |
| `CharTokenizer` | BPE/SentencePiece |
| 4 layers, 128 dim | 80+ layers, 4096+ dim |
| NumPy on CPU | CUDA/TPU tensors |

The only differences are:
1. **Scale**: More layers, larger dimensions
2. **Hardware**: GPUs/TPUs instead of CPU
3. **Engineering**: Compiled kernels, distributed training
4. **Tokenization**: Subword instead of character

## Key Takeaways

1. **Autodiff is just chain rule automation** - We can implement it manually
2. **Caching is essential** - Forward pass saves values for backward
3. **Gradient flow matters** - Architecture choices affect trainability
4. **Scale is the difference** - Same algorithms, more compute
5. **Everything connects** - Each stage builds on previous ones

## Next Steps

Congratulations on completing the capstone! You now have a deep understanding of how language models work. Consider:

1. **Read the GPT-2 paper** with fresh eyes
2. **Explore the Hugging Face Transformers library** source code
3. **Try training larger models** on cloud GPUs
4. **Implement modern improvements** like RoPE, GQA, or MoE
5. **Contribute to open-source LLM projects**

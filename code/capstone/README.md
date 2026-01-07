# Capstone: End-to-End Transformer Training

This capstone brings together all concepts from Stages 1-6 into a complete,
working transformer language model that you can train from scratch.

## What This Demonstrates

| Stage | Concept | Where It Appears |
|-------|---------|------------------|
| 1 | Language modeling, perplexity | Loss function, evaluation |
| 2 | Backpropagation | Manual `backward()` methods |
| 3 | Neural networks | Embeddings, linear layers |
| 4 | Adam optimizer | Training loop |
| 5 | Attention | Multi-head self-attention |
| 6 | Transformers | Full architecture |

## Key Files

- **`model.py`** - Trainable transformer with manual backpropagation
- **`train.py`** - End-to-end training script with logging
- **`tests/test_capstone.py`** - Comprehensive test suite (23 tests)

## Quick Start

```bash
# Run with default settings (Shakespeare excerpt)
python train.py

# Custom training
python train.py --epochs 50 --lr 3e-4 --d-model 256 --n-layers 6

# Train on your own text
python train.py --text-file path/to/your/text.txt --epochs 100
```

## Architecture

The model uses modern architectural choices:

- **RMSNorm** (not LayerNorm) - simpler, faster
- **SwiGLU activation** - better than GELU for FFN
- **Pre-norm** - residual stream before normalization
- **Tied embeddings** - output weights share with input embeddings

## Understanding the Code

### Manual Backpropagation

Unlike PyTorch which uses autodiff, we implement `backward()` manually for each layer.
This is exactly what autodiff does under the hood:

```python
class FeedForward:
    def forward(self, x):
        self.cache = {'x': x, ...}  # Save for backward
        return output

    def backward(self, grad_output):
        x = self.cache['x']
        # Compute gradients using chain rule
        self.w1.grad = ...
        return grad_input
```

### Training Loop

The training loop follows the standard pattern:

```python
for epoch in range(epochs):
    for inputs, targets in batches:
        # Forward
        logits = model.forward(inputs)
        loss, grad = cross_entropy_loss(logits, targets)

        # Backward
        model.zero_grad()
        model.backward(grad)

        # Update
        clip_grad_norm(params, max_norm)
        optimizer.step()
        scheduler.step()
```

## Running Tests

```bash
python tests/test_capstone.py
```

This runs 23 tests covering:
- Utility functions (softmax, silu)
- Individual components (RMSNorm, Attention, FFN)
- Full model forward/backward
- Numerical gradient checking
- Training convergence

## Model Size

With default settings (`d_model=128, n_heads=4, n_layers=4`):

- **Parameters**: ~800K
- **Training speed**: ~45 steps/second (CPU)

For comparison, GPT-2 small has 117M parameters. This is a toy model for
learning, not for practical use.

## Next Steps

After understanding this capstone:

1. **Scale up**: Try larger models on more data
2. **Add features**: Implement KV-cache, Flash Attention
3. **Try BPE**: Replace character tokenizer with BPE
4. **Fine-tune**: Implement LoRA or other PEFT methods

## Connection to Modern LLMs

Everything here maps directly to production systems:

| This Capstone | GPT/LLaMA |
|---------------|-----------|
| `model.forward()` | Forward pass |
| `model.backward()` | Autodiff (PyTorch) |
| `Adam` | Same algorithm |
| `WarmupCosineScheduler` | Standard schedule |
| `CharTokenizer` | BPE/SentencePiece |
| 4 layers, 128 dim | 80+ layers, 4096+ dim |

The only difference is scale and engineering optimization.

# Section 9.1: The Fine-Tuning Problem

*Reading time: 10 minutes*

## Why Fine-Tune?

Pretrained LLMs know a lot about language but nothing about *your* specific task:

- They don't know your company's terminology
- They don't follow your preferred output format
- They haven't seen your private data

**Fine-tuning** adapts a pretrained model to your specific needs.

## The Naive Approach: Full Fine-Tuning

Just train all parameters on your data:

```python
for batch in task_data:
    loss = model.forward(batch)
    gradients = model.backward(loss)
    for param, grad in zip(model.parameters, gradients):
        param -= learning_rate * grad  # Update ALL parameters
```

### The Problems

**1. Memory**

A 7B parameter model in fp32:

- Weights: 7B × 4 bytes = 28GB
- Gradients: 7B × 4 bytes = 28GB
- Optimizer states (Adam): 7B × 8 bytes = 56GB
- **Total: ~112GB** just for training!

Even with fp16/bf16, you need expensive hardware.

**2. Storage**

Each fine-tuned version is a complete copy:

- Customer support bot: 28GB
- Code assistant: 28GB
- Medical Q&A: 28GB
- **10 tasks = 280GB** of model storage

**3. Catastrophic Forgetting**

Fine-tuning too aggressively destroys general capabilities:

```
Before: "What is the capital of France?" → "Paris"
After:  "What is the capital of France?" → "Our return policy..."
```

The model forgets what it knew.

**4. Overfitting**

With billions of parameters and small task datasets:

```
Dataset size: 1,000 examples
Parameters: 7,000,000,000

Ratio: 7,000,000 parameters per example!
```

Massive overfitting is almost guaranteed.

## The Core Insight

Research by Aghajanyan et al. (2021) showed:

> The intrinsic dimensionality of fine-tuning is remarkably low.

What does this mean?

The difference between pretrained weights and fine-tuned weights lives in a **low-dimensional subspace**.

$$W_{fine-tuned} - W_{pretrained} \approx \Delta W$$

Where $\Delta W$ can be represented with far fewer than $n \times m$ parameters.

## Visualizing the Insight

Imagine weight space as a high-dimensional landscape:

```
Full parameter space (7B dimensions)
    │
    ▼
╔══════════════════════════════════════════╗
║                                          ║
║     ●  pretrained                        ║
║      \                                   ║
║       \──────▶ fine-tuned               ║
║        (small update)                    ║
║                                          ║
╚══════════════════════════════════════════╝

The actual update path is LOW RANK:
It doesn't explore most of the 7B dimensions.
```

## The Solution: PEFT

Instead of updating all parameters, update only:

1. **A small number of new parameters** (LoRA, adapters)
2. **Input representations** (prompt tuning)
3. **Attention patterns** (prefix tuning)

These methods exploit the low intrinsic dimensionality of fine-tuning.

## Comparison

| Approach | Trainable Params | Memory | Storage per Task | Risk |
|----------|-----------------|--------|------------------|------|
| Full fine-tuning | 100% | Very high | Full model | High |
| LoRA | ~0.1% | Low | ~10MB | Low |
| Adapters | ~1% | Medium | ~100MB | Low |
| Prompt tuning | ~0.001% | Very low | ~100KB | Very low |

## The PEFT Philosophy

**Don't change what works—add what's needed.**

The pretrained model already knows:

- Grammar and syntax
- World knowledge
- Reasoning patterns

Fine-tuning only needs to teach:

- Task-specific behavior
- Domain terminology
- Output format

This can be done with a tiny fraction of the parameters.

## Mathematical Foundation

For a weight matrix $W \in \mathbb{R}^{m \times n}$:

**Full fine-tuning**: Learn $W' = W + \Delta W$ where $\Delta W$ has $mn$ parameters

**LoRA**: Learn $W' = W + BA$ where:

- $B \in \mathbb{R}^{m \times r}$
- $A \in \mathbb{R}^{r \times n}$
- $r \ll \min(m, n)$

Parameters: $r(m + n)$ instead of $mn$

Example:

- $m = n = 4096$ (typical LLM dimension)
- $r = 8$ (common LoRA rank)
- Full: $4096 \times 4096 = 16.7M$ parameters
- LoRA: $8 \times (4096 + 4096) = 65K$ parameters
- **256× reduction!**

## Summary

| Problem | PEFT Solution |
|---------|---------------|
| Memory | Freeze most params, train few |
| Storage | Save only new params |
| Forgetting | Pretrained weights untouched |
| Overfitting | Far fewer params than data |

**Key insight**: Fine-tuning doesn't require exploring the full parameter space—just the right low-dimensional subspace.

**Next**: We'll dive into LoRA, the most popular PEFT method.

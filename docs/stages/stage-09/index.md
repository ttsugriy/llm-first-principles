# Stage 9: Parameter-Efficient Fine-Tuning (PEFT)

*Adapting large models without breaking the bank*

## Overview

Modern LLMs have billions of parameters. Fine-tuning them all is:

- **Expensive**: A 7B model needs ~28GB just for weights in fp32
- **Slow**: Updating billions of parameters takes time
- **Wasteful**: Most parameters don't need to change much

PEFT methods solve this by training only a tiny fraction of parameters while keeping most of the model frozen.

> "Fine-tuning 1% of parameters can achieve 99% of full fine-tuning performance."

## The Key Insight

Research shows that weight updates during fine-tuning have **low intrinsic rank**. This means:

- The change from pretrained weights to fine-tuned weights can be approximated with far fewer parameters
- We don't need to update 7 billion parameters—a few million carefully placed parameters suffice

## Methods We'll Cover

| Method | Key Idea | Parameters |
|--------|----------|------------|
| LoRA | Low-rank weight updates | ~0.1-1% |
| Adapters | Bottleneck layers | ~1-5% |
| Prefix Tuning | Learned key/value prefixes | ~0.01% |
| Prompt Tuning | Soft input prompts | ~0.001% |

## Why This Matters

For a 7B parameter model:

| Method | Trainable Params | GPU Memory |
|--------|------------------|------------|
| Full fine-tuning | 7B | ~28GB |
| LoRA (r=8) | ~4M | ~8GB |
| Prompt tuning | ~80K | ~2GB |

That's the difference between needing a $10,000 GPU and a $500 one.

## Learning Objectives

By the end of this stage, you will:

1. Understand why PEFT works (the low-rank hypothesis)
2. Implement LoRA from scratch
3. Implement adapters with bottleneck architecture
4. Understand prefix and prompt tuning
5. Know when to use each method

## Sections

1. [The Fine-Tuning Problem](01-fine-tuning-problem.md) - Why full fine-tuning is hard
2. [LoRA: Low-Rank Adaptation](02-lora.md) - The most popular PEFT method
3. [Adapter Layers](03-adapters.md) - Bottleneck modules
4. [Prefix and Prompt Tuning](04-prefix-and-prompt.md) - Learning soft prompts
5. [Choosing a Method](05-choosing-a-method.md) - Trade-offs and recommendations
6. [Implementation](06-implementation.md) - Building PEFT from scratch

## Prerequisites

- Understanding of transformer architecture (Stage 6)
- Familiarity with backpropagation (Stage 2)
- Experience with optimization (Stage 4)

## Key Insight

> PEFT isn't about approximating full fine-tuning—it's about finding the right subspace for adaptation. Often, this subspace is tiny compared to the full parameter space.

## Code & Resources

| Resource | Description |
|----------|-------------|
| [`code/stage-09/peft.py`](https://github.com/ttsugriy/llm-first-principles/blob/main/code/stage-09/peft.py) | LoRA, Adapters, and Prompt Tuning |
| [`code/stage-09/tests/`](https://github.com/ttsugriy/llm-first-principles/tree/main/code/stage-09/tests) | Test suite |
| [Exercises](exercises.md) | Practice problems |
| [Common Mistakes](common-mistakes.md) | Debugging guide |

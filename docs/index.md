# Building LLMs from First Principles

**A rigorous, bottom-up approach to understanding language models.**

This book derives every concept from first principles. No hand-waving. No "it's well known that..." Every formula is explained, every claim is proven.

## What Makes This Different

Most LLM tutorials tell you *what* to do. This book shows you *why* it works:

- **Full mathematical derivations** - Chain rule proved by induction, MLE derived with Lagrange multipliers, smoothing derived from Bayesian priors
- **Code from scratch** - Every algorithm implemented in pure NumPy with comprehensive test suites
- **First principles pedagogy** - Each concept builds only on what's already been covered
- **Exercises & common mistakes** - Practice problems and debugging guides for every stage
- **Modern connections** - See how Markov chains connect directly to GPT-4 and Claude
- **Interactive tools** - Explore concepts with live visualizations

## The Complete Journey

| Stage | Topic | Key Concepts | Exercises |
|-------|-------|--------------|-----------|
| 1 | [Markov Chains](stages/stage-01/index.md) | Probability, MLE, perplexity, temperature | 11 |
| 2 | [Automatic Differentiation](stages/stage-02/index.md) | Derivatives, chain rule, autograd | 11 |
| 3 | [Neural Language Models](stages/stage-03/index.md) | Embeddings, softmax, cross-entropy | 11 |
| 4 | [Optimization](stages/stage-04/index.md) | SGD, momentum, Adam, learning rate schedules | 11 |
| 5 | [Attention](stages/stage-05/index.md) | Dot-product attention, multi-head, causal masking | 11 |
| 6 | [The Complete Transformer](stages/stage-06/index.md) | Transformer blocks, LayerNorm, scaling laws | 11 |
| 7 | [Tokenization](stages/stage-07/index.md) | BPE, WordPiece, Unigram, vocabulary size | 12 |
| 8 | [Training Dynamics](stages/stage-08/index.md) | Loss curves, gradient statistics, debugging | 12 |
| 9 | [Parameter-Efficient Fine-Tuning](stages/stage-09/index.md) | LoRA, adapters, prefix tuning | 12 |
| 10 | [Alignment](stages/stage-10/index.md) | Reward modeling, RLHF, DPO | 12 |
| **Capstone** | [End-to-End Transformer](capstone/index.md) | Complete trainable model from scratch | - |

## Learning Paths

Choose a path based on your goals:

### The Fundamentals Path (Stages 1-6)
*Best for: Understanding how transformers work*

Progress through the core stages in order. By the end, you'll understand:
- How language models predict next tokens
- Why attention is the key innovation
- What makes transformers trainable and scalable

**Time estimate**: 20-30 hours of focused study

### The Practitioner Path (Stages 7-10)
*Best for: People who want to fine-tune and deploy models*

After completing fundamentals, focus on:
- Stage 7: How tokenization affects model performance
- Stage 8: Debugging training issues
- Stage 9: Fine-tuning without training all parameters
- Stage 10: Aligning models with human preferences

**Prerequisites**: Stages 1-6 or equivalent experience

### The Deep Dive Path (All Stages + Capstone)
*Best for: Researchers and those building from scratch*

Complete all stages and the capstone project, which involves:
- Implementing every backward pass manually
- Training a complete transformer on real text
- Understanding exactly what autodiff does under the hood

**Time estimate**: 40-60 hours

## Quick Links

- [Glossary](glossary.md) - Key terms and notation reference
- [Interactive Tools](interactive/index.md) - Autograd visualizer, temperature explorer
- [Capstone Project](capstone/index.md) - Put it all together

## Prerequisites

- Basic Python programming
- High school algebra
- Curiosity about how things work

No deep learning experience required. We build everything from the ground up.

## How to Use This Book

Each stage is self-contained but builds on previous stages:

1. **Read the theory** - Understand the mathematical foundations
2. **Study the code** - See how theory translates to implementation
3. **Do the exercises** - Solidify understanding through practice
4. **Review common mistakes** - Learn from typical errors
5. **Reflect** - Connect new concepts to the bigger picture

The book follows Polya's problem-solving method:

- **Understand** the problem
- **Devise** a plan
- **Execute** the plan
- **Reflect** on the solution

## What You'll Build

By the end of this book, you will have implemented:

- A Markov chain text generator
- An automatic differentiation engine
- A neural language model with embeddings
- Optimizers (SGD, Adam) and learning rate schedulers
- Multi-head self-attention from scratch
- A complete transformer architecture
- BPE tokenization
- Training diagnostics and debugging tools
- LoRA fine-tuning
- DPO alignment training
- A **complete trainable transformer** with manual backpropagation

## Get Started

<div class="grid cards" markdown>

-   **New to ML?**

    ---

    Start from the beginning with probability and Markov chains.

    [:octicons-arrow-right-24: Stage 1: Markov Chains](stages/stage-01/index.md)

-   **Know the basics?**

    ---

    Jump to attention and transformers.

    [:octicons-arrow-right-24: Stage 5: Attention](stages/stage-05/index.md)

-   **Want to fine-tune?**

    ---

    Learn modern fine-tuning techniques.

    [:octicons-arrow-right-24: Stage 9: PEFT](stages/stage-09/index.md)

-   **Ready to build?**

    ---

    Dive into the capstone project.

    [:octicons-arrow-right-24: Capstone](capstone/index.md)

</div>

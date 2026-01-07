# Stage 1: The Simplest Language Model

**Markov Chains — Where It All Begins**

This stage builds a language model from absolute first principles. Every concept is derived, not stated. By the end, you'll understand not just *what* language models do, but *why* they work mathematically.

!!! tip "Reading Time"
    60-90 minutes | Prerequisites: Basic Python, high school algebra

## Overview

We begin with the simplest possible language model: a Markov chain. Despite its simplicity, this model introduces concepts that underpin *all* modern LLMs:

- **Autoregressive factorization** — the same decomposition used by GPT, LLaMA, and Claude
- **Maximum likelihood estimation** — the same training objective
- **Cross-entropy loss** — the same loss function
- **Temperature sampling** — the same generation technique
- **Perplexity** — the same evaluation metric

The difference between a Markov chain and GPT-4 isn't in *what* they compute, but *how* they compute it.

## What You'll Learn

1. **Probability Foundations** — Kolmogorov axioms, conditional probability, chain rule (proved by induction)

2. **The Language Modeling Problem** — Why exponential space makes direct modeling impossible

3. **Maximum Likelihood Estimation** — Full Lagrangian derivation proving counting = optimal

4. **Information Theory** — Entropy, cross-entropy, KL divergence derived from axioms

5. **Perplexity** — The standard evaluation metric, with the "effective vocabulary" interpretation

6. **Temperature Sampling** — From softmax to Boltzmann distribution

7. **Implementation** — Complete ~150-line implementation with every design decision explained

8. **The Fundamental Trade-offs** — Why we need neural networks

## Key Formulas

| Concept | Formula | Intuition |
|---------|---------|-----------|
| Chain Rule | $P(x_{1:n}) = \prod_i P(x_i \mid x_{<i})$ | Factor into conditionals |
| MLE | $P(b\mid a) = \frac{\text{count}(a,b)}{\text{count}(a,\cdot)}$ | Counting is optimal |
| Cross-entropy | $H(P,Q) = -\mathbb{E}_P[\log Q]$ | Average surprise |
| Perplexity | $\exp(H)$ | Effective vocabulary |
| Temperature | $P'(x) \propto P(x)^{1/T}$ | Control randomness |

## The Central Insight

!!! important "The Fundamental Trade-off"
    **More context → better predictions**

    **More context → sparser observations**

    We prove this quantitatively and show experimental data. This limitation is why we need neural networks: they can *generalize* from similar patterns rather than requiring exact matches.

## Begin Reading

Start with the foundations:

→ [Section 1.1: Probability Foundations](01-probability-foundations.md)

## Code & Resources

| Resource | Description |
|----------|-------------|
| [`code/stage-01/markov.py`](https://github.com/ttsugriy/llm-first-principles/blob/main/code/stage-01/markov.py) | Reference implementation |
| [`code/stage-01/tests/`](https://github.com/ttsugriy/llm-first-principles/tree/main/code/stage-01/tests) | Test suite |
| [Exercises](exercises.md) | Practice problems |
| [Common Mistakes](common-mistakes.md) | Debugging guide |

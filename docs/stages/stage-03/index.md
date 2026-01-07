# Stage 3: Neural Language Models

## From Counting to Learning: Building Your First Neural Language Model

In Stage 1, we built language models by counting n-grams. In Stage 2, we built the automatic differentiation system that enables neural networks to learn. Now we bring them together.

**Stage 3 builds a complete neural language model from scratch**—using only the autograd system we developed ourselves. No PyTorch, no TensorFlow. Just first principles.

## What We'll Build

A character-level neural language model that:

- Learns continuous representations (embeddings) of characters
- Uses feed-forward neural networks to predict the next character
- Trains via gradient descent using our own autograd
- Outperforms our Stage 1 Markov baselines

## Sections

### [3.1: Why Neural? The Limits of Counting](01-why-neural.md)
The curse of dimensionality makes n-grams fundamentally limited. We need a new approach.

### [3.2: Embeddings — From Discrete to Continuous](02-embeddings.md)
How to represent characters as vectors. The key insight that enables neural language modeling.

### [3.3: Feed-Forward Neural Networks](03-feed-forward.md)
Building blocks of deep learning: linear layers, activations, and the universal approximation theorem.

### [3.4: Cross-Entropy Loss and Maximum Likelihood](04-cross-entropy.md)
Deriving the loss function from first principles. Proving it's equivalent to MLE from Stage 1.

### [3.5: Building a Character-Level Neural LM](05-implementation.md)
Complete implementation using our Stage 2 autograd. ~300 lines of code for a working language model.

### [3.6: Training Dynamics](06-training-dynamics.md)
Learning rates, initialization, batching, regularization. The art and science of making networks learn.

### [3.7: Evaluation and Comparison](07-evaluation.md)
Rigorous comparison with Stage 1 Markov models. Proving the neural advantage with numbers.

## Prerequisites

- **Stage 1**: Probability, MLE, perplexity
- **Stage 2**: Derivatives, chain rule, autograd

## Key Takeaways

By the end of this stage, you will understand:

1. **Why neural beats n-gram**: Continuous representations enable generalization
2. **Embeddings deeply**: How similar tokens get similar vectors automatically
3. **Network architecture**: How layers combine to form universal function approximators
4. **Training from scratch**: Gradient descent, learning rates, and regularization
5. **Empirical validation**: How to properly compare models

## The Journey So Far

| Stage | Topic | Key Insight |
|-------|-------|-------------|
| 1 | Markov Chains | Language modeling = probability over sequences |
| 2 | Automatic Differentiation | Gradients enable iterative learning |
| **3** | **Neural Language Models** | **Continuous representations beat discrete counting** |
| 4 | (Coming) | Recurrent networks for unbounded context |

## Let's Begin

We start by understanding exactly why counting-based models hit a wall, and how continuous representations offer a way forward.

[→ Start with Section 3.1: Why Neural?](01-why-neural.md)

## Code & Resources

| Resource | Description |
|----------|-------------|
| [`code/stage-03/neural_lm.py`](https://github.com/ttsugriy/llm-first-principles/blob/main/code/stage-03/neural_lm.py) | Reference implementation |
| [`code/stage-03/tests/`](https://github.com/ttsugriy/llm-first-principles/tree/main/code/stage-03/tests) | Test suite |
| [Exercises](exercises.md) | Practice problems |
| [Common Mistakes](common-mistakes.md) | Debugging guide |

# Stage 2: Automatic Differentiation

## From Calculus to Code: Building the Foundation of Deep Learning

In Stage 1, we built a Markov chain language model and found optimal parameters through counting—a closed-form solution. Neural networks are fundamentally different: there's no closed-form solution. We must *search* for good parameters by iteratively improving them.

This search requires knowing: **if I change a parameter slightly, how does the output change?**

This is the domain of **automatic differentiation**—the technique that makes training neural networks possible. By the end of this stage, you'll understand exactly how PyTorch and TensorFlow compute gradients, because you'll have built the same system from scratch.

## What We'll Build

A complete automatic differentiation engine that can:

- Track computations as they happen
- Build computational graphs automatically
- Compute gradients via reverse-mode differentiation
- Train neural networks using gradient descent

## Sections

### [2.1: What is a Derivative?](01-what-is-derivative.md)
The geometric and algebraic foundations. Why derivatives matter for optimization, and how they connect to machine learning.

### [2.2: Derivative Rules from First Principles](02-derivative-rules.md)
Deriving the power, product, quotient, and exponential rules from the limit definition. We don't just state rules—we prove them.

### [2.3: The Chain Rule — The Heart of Backpropagation](03-chain-rule.md)
The most important derivative rule for deep learning. How derivatives chain through compositions, and why this leads directly to backpropagation.

### [2.4: Computational Graphs](04-computational-graphs.md)
Representing computation as directed acyclic graphs. Forward passes, backward passes, and gradient accumulation.

### [2.5: Forward Mode vs Reverse Mode](05-forward-vs-reverse.md)
Two fundamentally different ways to apply the chain rule. Why reverse mode is exponentially faster for neural networks.

### [2.6: Building Autograd from Scratch](06-autograd-from-scratch.md)
~100 lines of code that implement complete automatic differentiation. Building and training neural networks with our own engine.

### [2.7: Testing and Validation](07-testing-validation.md)
How to verify your gradients are correct. Numerical checking, property-based testing, and debugging strategies.

## Prerequisites

- Basic calculus (we'll derive everything from limits)
- Python programming
- Completion of Stage 1 (for context on why we need this)

## Key Takeaways

By the end of this stage, you will understand:

1. **Derivatives from first principles**: Not just rules to memorize, but why they work
2. **The chain rule deeply**: How it enables differentiating any composition
3. **Computational graphs**: The data structure behind modern deep learning
4. **Why reverse mode wins**: The complexity analysis that explains backpropagation's efficiency
5. **How to build autograd**: The ~100 lines of core code that power gradient computation
6. **Testing gradients**: Essential techniques for verifying correctness

## The Journey So Far

| Stage | Topic | Key Insight |
|-------|-------|-------------|
| 1 | Markov Chains | Language modeling is probability estimation over sequences |
| **2** | **Automatic Differentiation** | **Gradients enable iterative optimization—no closed-form needed** |
| 3 | (Coming) | Building our first neural language model |

## Let's Begin

The derivative is where it all starts. Understanding it deeply—not just as a formula, but as a concept—unlocks everything that follows.

[→ Start with Section 2.1: What is a Derivative?](01-what-is-derivative.md)

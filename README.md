# Building LLMs from First Principles

[![Tests](https://github.com/ttsugriy/llm-first-principles/actions/workflows/tests.yml/badge.svg)](https://github.com/ttsugriy/llm-first-principles/actions/workflows/tests.yml)
[![Deploy](https://github.com/ttsugriy/llm-first-principles/actions/workflows/deploy-book.yml/badge.svg)](https://github.com/ttsugriy/llm-first-principles/actions/workflows/deploy-book.yml)

A comprehensive educational resource teaching Large Language Model development from absolute first principles, with full mathematical derivations, working implementations, and systematic analysis.

**[Read Online →](https://ttsugriy.github.io/llm-first-principles/)**

**Author:** Taras Tsugrii | [Substack](https://softwarebits.substack.com/)

---

## Philosophy

> "Performance is the product of deep understanding of foundations."

- Every formula is **derived**, not just stated
- Every algorithm is **implemented from scratch** with tests
- Every design decision includes **trade-off analysis**
- Historical context shows **who developed these ideas and why**
- Modern connections show **how basics relate to GPT-4 and Claude**

---

## Quick Start

### Read Online

Visit the [GitHub Pages site](https://ttsugriy.github.io/llm-first-principles/) for beautifully formatted content with LaTeX math rendering.

### Build Locally

```bash
# Clone the repo
git clone https://github.com/ttsugriy/llm-first-principles.git
cd llm-first-principles

# Install dependencies
pip install -r requirements.txt

# Build the book
mkdocs build

# Serve locally with hot reload
mkdocs serve
# Open http://127.0.0.1:8000
```

### Run Stage 1 Code

```bash
cd code/stage-01

# Run the demo
python3 main.py

# Run the test suite
python3 tests/test_markov.py

# See data utilities
python3 data.py
```

### Run Stage 2 Code

```bash
cd code/stage-02

# Run autograd tests (50 tests)
python3 tests/test_value.py

# Use autograd in your code
python3 -c "from value import Value; x = Value(3.0); y = x**2; y.backward(); print(f'dy/dx = {x.grad}')"
```

### Run Stage 3 Code

```bash
cd code/stage-03

# Run neural LM tests (26 tests)
python3 tests/test_neural_lm.py

# Run the demo (trains a small character-level LM)
python3 neural_lm.py
```

### Run Stage 4 Code

```bash
cd code/stage-04

# Run optimizer tests
python3 tests/test_optimizers.py

# Use optimizers in your code
python3 -c "from optimizers import AdamW; print('Optimizers ready!')"
```

### Run Stage 5 Code

```bash
cd code/stage-05

# Run attention tests (53 tests)
python3 tests/test_attention.py

# Use attention in your code
python3 -c "from attention import CausalTransformer; model = CausalTransformer(100, 32, 4, 2); print('Attention ready!')"
```

### Run Stage 6 Code

```bash
cd code/stage-06

# Run transformer tests (60+ tests)
python3 tests/test_transformer.py

# Use the transformer in your code
python3 -c "from transformer import Transformer; model = Transformer(vocab_size=100); print(f'Params: {model.count_parameters():,}')"
```

### Run Benchmarks

```bash
cd benchmarks

# Run benchmarks and see results
python3 run_benchmarks.py

# Generate visualization plots (requires matplotlib)
python3 visualize.py
```

### Interactive Notebooks

Stage 1 includes a [marimo](https://marimo.io) interactive notebook:

```bash
pip install marimo numpy matplotlib
marimo run code/stage-01/stage_01_markov_interactive.py
```

---

## Content Overview

### Stage 1: Markov Chains ✅

The complete first stage covers:

| Section | Topic | Key Concepts |
|---------|-------|--------------|
| 1.1 | Probability Foundations | Axioms, conditional probability, chain rule |
| 1.2 | Language Modeling Problem | What is a language model, exponential space problem |
| 1.3 | MLE Derivation | Likelihood, log-likelihood, Lagrange multipliers |
| 1.3b | Smoothing | Dirichlet priors, Laplace smoothing, backoff |
| 1.4 | Information Theory | Entropy, cross-entropy, KL divergence |
| 1.5 | Perplexity | Evaluation metric, effective vocabulary interpretation |
| 1.6 | Temperature Sampling | Generation, temperature scaling, greedy vs sampling |
| 1.7 | Implementation | Complete working code with explanations |
| 1.8 | Trade-offs | Context-sparsity trade-off, limitations |
| 1.9 | Why Neural Networks | Motivation for moving beyond counting |

**Code includes:**
- `markov.py` — MarkovChain and SmoothedMarkovChain classes
- `evaluate.py` — Perplexity computation
- `generate.py` — Text generation with temperature
- `data.py` — Data loading and preprocessing utilities
- `tests/test_markov.py` — Comprehensive test suite (19 tests)

### Stage 2: Automatic Differentiation ✅

Understanding how neural networks learn:
- What is a derivative?
- Derivative rules and the chain rule
- Computational graphs
- Forward vs. reverse mode autodiff
- Building autograd from scratch

**Code includes:**
- `value.py` — Value class with automatic differentiation
- `tests/test_value.py` — Comprehensive test suite (50 tests)

### Stage 3: Neural Language Models ✅

Building neural approaches:
- Why neural networks generalize
- Word embeddings
- Feed-forward language models
- Cross-entropy loss
- Training dynamics

**Code includes:**
- `neural_lm.py` — CharacterLM, Embedding, Linear layers
- `tests/test_neural_lm.py` — Comprehensive test suite (26 tests)

### Stage 4: Optimization ✅

Making learning work:

| Section | Topic | Key Concepts |
|---------|-------|--------------|
| 4.1 | The Optimization Problem | Loss landscapes, saddle points, condition number |
| 4.2 | Gradient Descent | Derivation, convergence analysis, learning rate |
| 4.3 | Stochastic Gradient Descent | Mini-batches, noise as regularization, variance |
| 4.4 | Momentum | Physics intuition, Nesterov acceleration |
| 4.5 | Adaptive Learning Rates | AdaGrad, RMSprop, Adam, AdamW |
| 4.6 | Learning Rate Schedules | Warmup, cosine decay, restarts |
| 4.7 | Implementation | Optimizers from scratch |
| 4.8 | Practical Considerations | Initialization, clipping, debugging |

**Code includes:**
- `optimizers.py` — SGD, Adam, AdamW, schedulers
- `tests/test_optimizers.py` — Comprehensive test suite

### Stage 5: Attention Mechanisms ✅

The attention mechanism that powers transformers:

| Section | Topic | Key Concepts |
|---------|-------|--------------|
| 5.1 | The Attention Problem | Fixed context limits, information bottleneck |
| 5.2 | Dot-Product Attention | Query, Key, Value, attention weights |
| 5.3 | Scaled Attention | √d scaling, variance analysis |
| 5.4 | Self-Attention | Sequence attending to itself |
| 5.5 | Multi-Head Attention | Multiple heads, different subspaces |
| 5.6 | Positional Encoding | Sinusoidal, learned, RoPE, ALiBi |
| 5.7 | Causal Masking | Autoregressive constraints, KV-cache |
| 5.8 | Implementation | Complete attention from scratch |

**Code includes:**
- `attention.py` — Full attention implementation
- `tests/test_attention.py` — Comprehensive test suite (53 tests)

### Stage 6: The Complete Transformer ✅

Bringing all components together into a complete LLM:

| Section | Topic | Key Concepts |
|---------|-------|--------------|
| 6.1 | Tokenization | BPE, WordPiece, vocabulary size trade-offs |
| 6.2 | Transformer Block | Pre-norm, residual stream, FFN expansion |
| 6.3 | Deep Networks | Initialization, gradient flow, layer specialization |
| 6.4 | Pre-training Objectives | Causal LM, masked LM, span corruption |
| 6.5 | Training at Scale | Large batches, mixed precision, distributed training |
| 6.6 | Modern Architectures | GPT, LLaMA, Mistral, Mixtral (MoE), GQA |
| 6.7 | Scaling Laws | Chinchilla, compute-optimal training, emergent abilities |
| 6.8 | Implementation | Complete Transformer with modern features |

**Code includes:**
- `transformer.py` — Complete Transformer with RMSNorm, RoPE, SwiGLU, GQA
- `tests/test_transformer.py` — Comprehensive test suite (60+ tests)

### Stages 7-18: Coming Soon

| Spiral | Theme | Stages |
|--------|-------|--------|
| 2 | Training Dynamics | Optimization, Stability |
| 3 | Transformers | Attention, Multi-head, Positional Encoding |
| 4 | Making It Fast | Memory, Flash Attention, Distributed |
| 5 | Modern Practice | Architectures, RLHF, Inference |

---

## Repository Structure

```
llm-first-principles/
├── docs/                           # MkDocs source files
│   ├── index.md                    # Home page
│   ├── glossary.md                 # Key terms reference
│   ├── interactive/                # Interactive visualizations
│   │   ├── autograd.html           # Autograd visualizer
│   │   ├── temperature.html        # Temperature explorer
│   │   ├── ngram.html              # N-gram state machine
│   │   ├── optimizer.html          # Gradient descent visualizer
│   │   └── attention.html          # Attention visualizer
│   └── stages/                     # Stage content
│       ├── stage-01/               # Markov chains
│       ├── stage-02/               # Autodiff
│       ├── stage-03/               # Neural LMs
│       ├── stage-04/               # Optimization
│       ├── stage-05/               # Attention
│       └── stage-06/               # Complete Transformer
├── code/
│   ├── stage-01/                   # Stage 1 implementation
│   │   ├── markov.py               # Core Markov chain class
│   │   ├── evaluate.py             # Perplexity computation
│   │   ├── generate.py             # Text generation
│   │   ├── data.py                 # Data utilities
│   │   ├── main.py                 # Demo script
│   │   └── tests/                  # Test suite
│   │       └── test_markov.py
│   ├── stage-02/                   # Stage 2 implementation
│   │   ├── value.py                # Autograd Value class
│   │   └── tests/
│   │       └── test_value.py
│   ├── stage-03/                   # Stage 3 implementation
│   │   ├── neural_lm.py            # Neural language model
│   │   └── tests/
│   │       └── test_neural_lm.py
│   ├── stage-04/                   # Stage 4 implementation
│   │   ├── optimizers.py           # All optimizers from scratch
│   │   └── tests/
│   │       └── test_optimizers.py
│   ├── stage-05/                   # Stage 5 implementation
│   │   ├── attention.py            # Complete attention from scratch
│   │   └── tests/
│   │       └── test_attention.py
│   └── stage-06/                   # Stage 6 implementation
│       ├── transformer.py          # Complete Transformer
│       └── tests/
│           └── test_transformer.py
├── benchmarks/                     # Reproducible benchmarks
│   ├── run_benchmarks.py           # Run all benchmarks
│   └── visualize.py                # Generate plots
├── solutions/                      # Exercise solutions
│   └── stage-01-solutions.md
├── .github/workflows/              # CI/CD
│   ├── deploy-book.yml             # Deploy to GitHub Pages
│   └── tests.yml                   # Run test suite
├── mkdocs.yml                      # MkDocs configuration
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Features

### Mathematical Rigor
- Chain rule proved by induction
- MLE derived with Lagrange multipliers
- Smoothing derived from Bayesian priors
- Information theory from first principles

### Working Code
- Every algorithm implemented from scratch
- Comprehensive test suite
- Reproducible benchmarks
- Well-documented with type hints

### Interactive Tools
- **Autograd Visualizer**: Watch gradients flow through computational graphs
- **Temperature Explorer**: See how temperature affects sampling
- **N-gram State Machine**: Visualize Markov chains as state machines
- **Optimizer Visualizer**: Compare SGD, momentum, Adam on different loss surfaces
- **Attention Visualizer**: See how queries match keys in self-attention

### Pedagogical Features
- Reading time estimates for each section
- Difficulty ratings (★☆☆☆☆ to ★★★★★)
- Historical notes on who developed key ideas
- "Common Mistakes" callouts
- "Connection to Modern LLMs" boxes
- Exercises with solutions

---

## Development

### Running Tests

```bash
# Run Stage 1 tests
python3 code/stage-01/tests/test_markov.py
# Expected output: "Results: 19/19 tests passed"

# Run Stage 2 tests
python3 code/stage-02/tests/test_value.py
# Expected output: "Results: 50 passed, 0 failed"

# Run Stage 3 tests
python3 code/stage-03/tests/test_neural_lm.py
# Expected output: "Results: 26 passed, 0 failed"

# Run Stage 4 tests
python3 code/stage-04/tests/test_optimizers.py
# Expected output: all tests pass

# Run Stage 5 tests
python3 code/stage-05/tests/test_attention.py
# Expected output: "Results: 53 passed, 0 failed"

# Run Stage 6 tests
python3 code/stage-06/tests/test_transformer.py
# Expected output: "Results: 60+ passed, 0 failed"
```

### Building the Book

```bash
# Install dependencies
pip install mkdocs-material pymdown-extensions

# Build static site
mkdocs build

# Serve with hot reload for development
mkdocs serve
```

### Contributing

Issues and PRs welcome! Please:

1. Run tests before submitting
2. Follow existing code style (type hints, docstrings)
3. Add tests for new functionality
4. Update documentation as needed

---

## Acknowledgments

This project draws inspiration from:
- Andrej Karpathy's "Zero to Hero" series
- Sebastian Raschka's "Build a Large Language Model from Scratch"
- The Stanford CS224N course
- Goodfellow, Bengio, and Courville's "Deep Learning"

---

## License

- Content: CC BY-NC-SA 4.0
- Code: MIT License

---

*Built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)*

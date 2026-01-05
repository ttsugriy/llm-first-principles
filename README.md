# Building LLMs from First Principles

A comprehensive educational resource teaching Large Language Model development from absolute first principles, with full mathematical derivations, performance-focused implementation, and systematic trade-off analysis.

**[Read Online →](https://ttsugriy.github.io/llm-first-principles/)**

**Author:** Taras Tsugrii | [Substack](https://softwarebits.substack.com/)

---

## Philosophy

> "Performance is the product of deep understanding of foundations."

Every formula is derived. Every algorithm is implemented from scratch. Every design decision is analyzed for performance trade-offs.

---

## Quick Start

### Read Online

Visit the [GitHub Pages site](https://ttsugriy.github.io/llm-first-principles/) for formatted content with proper math rendering.

### Build Locally

```bash
# Clone the repo
git clone https://github.com/ttsugriy/llm-first-principles.git
cd llm-first-principles

# Install jupyter-book
pip install jupyter-book

# Build the book
jupyter-book build .

# Open in browser
open _build/html/index.html
```

### Run Stage 1 Code

```bash
cd code/stage-01
python3 main.py
```

### Interactive Notebooks

Stage 1 includes a [marimo](https://marimo.io) interactive notebook:

```bash
pip install marimo numpy matplotlib
marimo run code/stage-01/stage_01_markov_interactive.py
```

---

## Repository Structure

```
llm-first-principles/
├── _config.yml                     # Jupyter Book configuration
├── _toc.yml                        # Table of contents
├── intro.md                        # Book introduction
├── stages/
│   ├── stage-01/                   # Stage 1: Markov Chains
│   │   ├── index.md                # Stage overview
│   │   ├── 01-probability-foundations.md
│   │   ├── 02-language-modeling-problem.md
│   │   ├── 03-mle-derivation.md
│   │   ├── 04-information-theory.md
│   │   ├── 05-perplexity.md
│   │   ├── 06-temperature-sampling.md
│   │   ├── 07-implementation.md
│   │   └── 08-trade-offs.md
│   └── stage-02-preview.md         # Coming soon
├── code/
│   └── stage-01/                   # Stage 1 implementation
│       ├── markov.py               # MarkovChain class
│       ├── generate.py             # Text generation
│       ├── evaluate.py             # Perplexity computation
│       └── main.py                 # Demo script
├── .github/workflows/
│   └── deploy-book.yml             # Auto-deploy to GitHub Pages
└── planning/                       # Project planning docs
    ├── 00-PROJECT-OVERVIEW.md
    ├── 01-SPIRAL-STRUCTURE.md
    └── ...
```

---

## Project Summary

### What This Is

A **first-principles approach** to teaching LLM development that:

1. **Derives all mathematics** from foundations (no "it's well known that...")
2. **Implements everything from scratch** (no magic libraries)
3. **Analyzes performance throughout** (every formula gets a FLOP count)
4. **Uses spiral learning** (concepts revisited with increasing depth)
5. **Follows Pólya's problem-solving method** (understand → plan → execute → reflect)
6. **Applies Tufte's design principles** (clear, honest, integrated presentation)

### Structure: 5 Spirals, 18 Stages

| Spiral | Theme | Stages | Focus |
|--------|-------|--------|-------|
| 1 | Foundations | 1-4 | Markov → Neural LM |
| 2 | Training | 5-6 | Optimization, Stability |
| 3 | Transformer | 7-10 | Attention, Architecture |
| 4 | Making It Fast | 11-13 | Memory, Distributed |
| 5 | Modern Practice | 14-18 | Architectures, Alignment, Inference |

---

## Current Status

- ✅ Stage 1: Markov Chains (complete with 8 comprehensive sections)
- 🚧 Stage 2: Automatic Differentiation (coming soon)
- 📋 Stages 3-18: Planned

---

## Contributing

Issues and PRs welcome! See the [GitHub repository](https://github.com/ttsugriy/llm-first-principles).

---

## License

Content: [TBD]
Code: MIT License

---

*Built with [Jupyter Book](https://jupyterbook.org)*

# Building LLMs from First Principles

A comprehensive educational resource teaching Large Language Model development from absolute first principles, with full mathematical derivations, performance-focused implementation, and systematic trade-off analysis.

**[Read Online →](https://YOUR_USERNAME.github.io/llm-first-principles/)**

**Author:** Taras Tsugrii | [Substack](https://softwarebits.substack.com/)

---

## Philosophy

> "Performance is the product of deep understanding of foundations."

Every formula is derived. Every algorithm is implemented from scratch. Every design decision is analyzed for performance trade-offs.

---

## Quick Start

### Read Online

Visit the [GitHub Pages site](https://YOUR_USERNAME.github.io/llm-first-principles/) for formatted content.

### Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/llm-first-principles.git
cd llm-first-principles

# Run Stage 1 demo
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
├── docs/                           # GitHub Pages site
│   ├── index.html                  # Landing page
│   └── stage-01/                   # Stage 1 content
├── code/
│   └── stage-01/                   # Stage 1 implementation
│       ├── markov.py               # MarkovChain class
│       ├── generate.py             # Text generation
│       ├── evaluate.py             # Perplexity computation
│       ├── main.py                 # Demo script
│       └── stage_01_markov_interactive.py  # marimo notebook
├── stages/
│   └── stage-01-markov-chains.md   # Stage 1 content draft
├── 00-PROJECT-OVERVIEW.md          # Vision and goals
├── 01-SPIRAL-STRUCTURE.md          # Detailed stage breakdown (1-10)
├── 01-SPIRAL-STRUCTURE-PART2.md    # Detailed stage breakdown (11-18)
├── 02-PEDAGOGICAL-FRAMEWORK.md     # Pólya + Tufte methodology
├── 03-MATHEMATICAL-FOUNDATIONS.md  # All derivations catalog
├── 04-RESEARCH-NOTES.md            # References
└── 05-OPEN-QUESTIONS.md            # Decisions and open items
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

### The Gap We Fill

No existing resource combines:
- Complete mathematical rigor
- From-scratch implementation
- Performance/trade-off focus
- Modern architecture coverage (RoPE, GQA, Flash Attention, etc.)
- Full training pipeline through RLHF/DPO

### Structure: 5 Spirals, 18 Stages

| Spiral | Theme | Stages | Focus |
|--------|-------|--------|-------|
| 1 | Foundations | 1-4 | Markov → Neural LM |
| 2 | Training | 5-6 | Optimization, Stability |
| 3 | Transformer | 7-10 | Attention, Architecture |
| 4 | Making It Fast | 11-13 | Memory, Distributed |
| 5 | Modern Practice | 14-18 | Architectures, Alignment, Inference |

**Total estimated content:** 170,000-210,000 words (~500 pages)

---

## Key Documents

### For Understanding the Vision
→ Read `00-PROJECT-OVERVIEW.md`

### For Understanding the Content
→ Read `01-SPIRAL-STRUCTURE.md` and `01-SPIRAL-STRUCTURE-PART2.md`

### For Understanding the Methodology
→ Read `02-PEDAGOGICAL-FRAMEWORK.md`

### For Mathematical Reference
→ Read `03-MATHEMATICAL-FOUNDATIONS.md`

### For Sources and References
→ Read `04-RESEARCH-NOTES.md`

### For Open Decisions
→ Read `05-OPEN-QUESTIONS.md`

### For First Content Draft
→ Read `stages/stage-01-markov-chains.md`

---

## Core Philosophy

> "Performance is the product of deep understanding of foundations"

Every architectural decision in LLMs traces to:
- A **mathematical property** (attention enables direct position access)
- A **hardware constraint** (memory bandwidth limits inference)
- A **trade-off resolution** (vocabulary size vs. sequence length)

We teach all three perspectives together.

---

## Next Steps

### Immediate (Writing Phase)
1. [ ] Complete Stage 1 draft with code
2. [ ] Write Stage 2 (Automatic Differentiation)
3. [ ] Write Stage 3 (Backpropagation Deep Dive)
4. [ ] Write Stage 4 (Neural Language Model)
5. [ ] Gather feedback on Spiral 1

### Short-term
- [ ] Decide publication format (Substack vs dedicated site)
- [ ] Set up code repository
- [ ] Create first visualizations

### Medium-term
- [ ] Complete all 18 stages
- [ ] Full revision pass
- [ ] Compile into book format

---

## For AI Assistants (Continuity Notes)

If you're an AI continuing work on this project:

1. **Read these files first:**
   - `00-PROJECT-OVERVIEW.md` — overall vision
   - `02-PEDAGOGICAL-FRAMEWORK.md` — how to write content
   - `05-OPEN-QUESTIONS.md` — decisions still needed

2. **Current status:**
   - Planning documents complete
   - Stage 1 draft started (see `stages/stage-01-markov-chains.md`)
   - Code not yet written

3. **Key constraints:**
   - Follow Pólya's 4-step structure for each section
   - Apply Tufte's information design principles
   - Every formula needs derivation AND performance analysis
   - Spiral learning: simple first, deep later
   - Use Python/NumPy/PyTorch only

4. **Author's voice:**
   - First person plural ("we")
   - Enthusiastic but honest
   - Performance-focused perspective
   - Short stories style (see softwarebits.substack.com)

---

## Contact

**Author:** Taras Tsugrii
**Substack:** https://softwarebits.substack.com/

---

## License

Content: [TBD]
Code: MIT License

---

*Last updated: 2026-01-03*

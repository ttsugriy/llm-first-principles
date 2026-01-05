# Research: Educational Formats and Teaching Methodologies

## Overview

This document summarizes research on how technical educational materials (especially ML/programming) are taught, covering formats, tools, and pedagogical approaches that could inform "Building LLMs from First Principles."

---

## 1. Major Teaching Formats

### 1.1 Traditional Formats

| Format | Examples | Pros | Cons |
|--------|----------|------|------|
| **Book (Print/PDF)** | Goodfellow's Deep Learning, CLRS | Comprehensive, portable, prestige | Static, no interaction, outdated quickly |
| **Video Lectures** | Karpathy, 3Blue1Brown, Stanford | Engaging, shows process, personality | Linear, can't modify, hard to reference |
| **Blog Posts/Articles** | Substack, Medium, personal blogs | Accessible, quick to produce, SEO | Scattered, varying quality, no progression |
| **Academic Papers** | arXiv, conferences | Rigorous, peer-reviewed | Dense, assumes background, not pedagogical |

### 1.2 Interactive Formats

| Format | Examples | Pros | Cons |
|--------|----------|------|------|
| **Jupyter Notebooks** | d2l.ai, Fast.ai, tutorials | Code + prose, runnable, modifiable | Version control issues, state problems |
| **Executable Books** | Jupyter Book, Quarto | Professional output, web-based, live code | Setup complexity, maintenance |
| **Explorable Explanations** | Distill.pub, Nicky Case | Deep engagement, interactive viz | High production cost, web-only |
| **Interactive Playgrounds** | TensorFlow Playground | Immediate feedback, no setup | Limited scope, can't customize |

---

## 2. Detailed Format Analysis

### 2.1 Jupyter Notebooks

**What they are:** Documents combining executable code, rich text, equations, and visualizations.

**Strengths:**
- Reader can modify and run code
- See actual outputs, not just described outputs
- Supports LaTeX math, markdown, images
- Standard in ML community

**Weaknesses:**
- Hidden state issues (cells run out of order)
- Poor version control (JSON format)
- Difficult to maintain across Python versions
- Not reproducible without environment management

**Best practices:**
- Clear cell ordering with dependencies
- Include `requirements.txt` or `environment.yml`
- Restart-and-run-all before publishing
- Use Google Colab for accessibility

**Examples:**
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) — Harvard NLP's line-by-line implementation
- [d2l.ai](https://d2l.ai/) — Complete textbook in notebook format
- Fast.ai courses — All material as notebooks

---

### 2.2 Jupyter Book / MyST

**What it is:** A tool to build publication-quality books from Jupyter notebooks and Markdown files.

**Key features:**
- Converts notebooks to beautiful HTML/PDF
- Cross-references between pages
- Built-in citations and bibliography
- Supports JupyterLite for in-browser execution
- Theming and customization

**When to use:**
- Creating a cohesive multi-chapter resource
- Need professional web presence
- Want PDF export option
- Multiple authors/contributors

**Examples:**
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- Many university course websites

**Setup:** `pip install jupyter-book` then `jupyter-book create mybook/`

---

### 2.3 Quarto

**What it is:** Next-generation scientific publishing system (successor to R Markdown).

**Key features:**
- Works with Python, R, Julia, Observable
- Multiple output formats (HTML, PDF, Word, slides)
- Built-in citations, cross-references
- Excellent for academic publishing
- VS Code integration

**When to use:**
- Need multi-format output
- Academic/research context
- Mixed language projects

---

### 2.4 marimo Notebooks

**What it is:** Reactive Python notebook alternative to Jupyter.

**Key features:**
- **Reactive:** Cells automatically update when dependencies change
- **Reproducible:** Deterministic execution, no hidden state
- **Git-friendly:** Stored as pure `.py` files
- **Interactive widgets:** Built-in sliders, dropdowns, tables
- **Deployable:** Can run as standalone web apps

**Advantages over Jupyter:**
- No more "run all cells in order" problems
- Version control just works
- Can import notebooks as Python modules
- Modern, AI-native editor

**When to use:**
- Building interactive tutorials
- Need reproducibility guarantees
- Want Git-friendly notebooks
- Teaching with live parameter exploration

**Example use case for LLM book:**
```python
import marimo as mo

# Interactive temperature slider
temp = mo.ui.slider(0.1, 2.0, value=1.0, label="Temperature")

# Visualization updates automatically when temp changes
@mo.cell
def show_samples():
    samples = generate(model, temperature=temp.value, n=5)
    return mo.md(f"**Samples at T={temp.value}:**\n" + "\n".join(samples))
```

---

### 2.5 Explorable Explanations

**What they are:** Interactive documents where readers actively explore concepts through manipulation.

**Pioneered by:** Bret Victor ("Explorable Explanations" essay, 2011)

**Key practitioners:**
- **Nicky Case** — Game theory, anxiety, complex systems
- **Distill.pub** — ML research visualization
- **3Blue1Brown** — Math visualization (video format)

**Principles:**
1. **Active, not passive:** Reader manipulates parameters, not just reads
2. **Immediate feedback:** Changes show instant results
3. **Play-based:** Exploration feels like a game
4. **Progressive disclosure:** Complexity revealed gradually

**Examples relevant to ML:**
- [TensorFlow Playground](https://playground.tensorflow.org/) — Neural network visualization
- [Attention Visualization](https://distill.pub/2016/augmented-rnns/) — Interactive attention diagrams
- [t-SNE Explained](https://distill.pub/2016/misread-tsne/) — Interactive dimensionality reduction

**Production cost:** High. Requires custom JavaScript/web development for each interactive element.

---

### 2.6 Annotated Implementations

**What they are:** Code with extensive inline documentation explaining every line.

**Examples:**
- **[The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)** — 400 lines, every line explained
- **[labml.ai](https://github.com/labmlai/annotated_deep_learning_paper_implementations)** — 60+ papers with side-by-side notes
- **Karpathy's nanoGPT** — Clean code with comments

**Format characteristics:**
- Code and explanation interleaved
- Often as notebook or literate programming document
- Runnable end-to-end
- Follows paper structure but as code

**Strengths:**
- See exactly how theory becomes code
- Complete, working implementations
- Can modify and experiment

**This is likely the closest format to your vision.**

---

### 2.7 Video + Notebook Hybrid

**What it is:** Video walkthrough with accompanying notebook for hands-on work.

**Examples:**
- **Karpathy's Zero to Hero** — YouTube + GitHub repo
- **Fast.ai** — Videos + notebooks + book
- **3Blue1Brown + Manim** — Visualization code available

**Why it works:**
- Video provides narrative, personality, thinking process
- Notebook provides hands-on practice
- Multiple modalities reinforce learning

**Production cost:** High (video production + notebook maintenance)

---

## 3. Teaching Methodologies

### 3.1 Fast.ai's Top-Down Approach

**Philosophy:** "Show a working example first, then dig into details."

**Structure:**
1. Lesson 1: Train a state-of-the-art model in 5 lines
2. Subsequent lessons: Peel back layers, understand internals
3. Later lessons: Build from scratch

**Why it works:**
- Immediate success builds confidence
- Context makes details meaningful
- "Whole game" before "fundamentals"

**For your book:** Consider starting each spiral with a working example, then diving deep.

---

### 3.2 Literate Programming (Knuth)

**Philosophy:** "Programs should be written for humans to read, and only incidentally for machines to execute."

**Implementation:**
- WEB/CWEB systems
- Modern: nbdev, Quarto, Jupyter Book

**Key insight:** The explanation is primary; code is embedded in explanation (not vice versa).

**For your book:** Your mathematical derivations interleaved with code embodies this philosophy.

---

### 3.3 Mastery Learning (Khan Academy)

**Philosophy:** Don't move on until current concept is mastered.

**Implementation:**
- Exercises with immediate feedback
- Multiple attempts allowed
- Prerequisite system
- Progress tracking

**For your book:** Consider adding exercises with solutions, checkpoint questions.

---

### 3.4 Spaced Repetition

**Philosophy:** Review concepts at increasing intervals to cement long-term memory.

**Implementation:**
- Anki flashcards for key concepts
- Exercises that recall earlier material
- Spiral curriculum naturally provides this

**For your book:** Your spiral approach already incorporates this — concepts revisited at increasing depth.

---

### 3.5 Explorable Learning (Bret Victor / Nicky Case)

**Philosophy:** Understanding comes from playing with a system, not reading about it.

**Key principles:**
1. Make the invisible visible
2. Let readers manipulate parameters
3. Show immediate consequences
4. Design for exploration, not instruction

**For your book:** Could add interactive elements for key concepts (attention weights, perplexity curves, etc.)

---

## 4. Notable Examples to Study

### 4.1 Crafting Interpreters (Robert Nystrom)

**URL:** https://craftinginterpreters.com/

**Format:** Free online book with all code inline

**Structure:**
- Two complete implementations (tree-walk, bytecode)
- Design Notes sections for theory
- Challenges at end of chapters
- Beautiful typography and diagrams

**What makes it great:**
- Complete, runnable implementations
- Theory and practice balanced
- Incremental complexity
- Strong narrative voice
- Free online, print available

**Most similar to your vision.** Study its chapter structure carefully.

---

### 4.2 d2l.ai (Dive into Deep Learning)

**URL:** https://d2l.ai/

**Format:** Online interactive textbook (notebooks)

**Structure:**
- 22 chapters covering DL fundamentals to modern architectures
- Math in appendix
- Exercises per chapter
- Multi-framework (PyTorch, MXNet, JAX, TensorFlow)

**What makes it great:**
- Truly comprehensive
- Adopted by 500+ universities
- All code runnable
- Active maintenance

**Gap your book fills:** d2l math is in appendix, not integrated. You integrate math throughout.

---

### 4.3 labml.ai Annotated Implementations

**URL:** https://nn.labml.ai/

**Format:** Side-by-side code and notes

**Structure:**
- One page per paper/technique
- Code on left, notes on right
- Type hints and documentation
- Runnable via pip install

**What makes it great:**
- Covers 60+ papers
- Consistent format
- Production-quality code
- Active updates

**Your differentiation:** More narrative, derivations, performance analysis. labml is reference, not tutorial.

---

### 4.4 Distill.pub

**URL:** https://distill.pub/

**Format:** Interactive web articles

**Structure:**
- Long-form articles with rich visualizations
- Interactive diagrams
- Academic quality with web-native presentation
- Peer reviewed

**What makes it great:**
- Gold standard for interactive ML explanation
- Beautiful, thoughtful design
- Each article is a significant effort

**Unfortunately:** Publication paused. But the format is aspirational.

**Your opportunity:** Blog-post frequency with Distill-inspired quality.

---

## 5. Recommended Format for Your Project

Based on this research, here's my recommendation:

### Primary Format: Hybrid Approach

```
┌─────────────────────────────────────────────────────────────┐
│                    PRIMARY: JUPYTER BOOK                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Narrative + Math + Code + Visualizations              │  │
│  │  • Markdown chapters with embedded notebooks           │  │
│  │  • Full derivations in prose                           │  │
│  │  • Code blocks that can be copied/run                  │  │
│  │  • Generated figures from actual code runs             │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  OUTPUT FORMATS:                                              │
│  • Beautiful website (primary distribution)                   │
│  • PDF/ePub for offline reading                              │
│  • Google Colab links for each chapter                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│               SUPPLEMENTARY: MARIMO NOTEBOOKS                 │
│  • Interactive explorations                                   │
│  • Parameter sliders for concepts                            │
│  • "Play with this" sections                                 │
│  • Deploy as standalone apps                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                OPTIONAL: SUBSTACK EXCERPTS                    │
│  • Publish condensed versions as articles                    │
│  • Build audience, get feedback                              │
│  • Drive traffic to full Jupyter Book                        │
└─────────────────────────────────────────────────────────────┘
```

### Why This Combination?

1. **Jupyter Book** for main content:
   - Professional output
   - Code + prose + math integration
   - PDF export for book publication
   - Web-first but not web-only

2. **marimo** for interactive elements:
   - Reactive exploration without hidden state
   - Git-friendly
   - Can deploy as apps
   - Modern, maintained

3. **Substack** for marketing/feedback:
   - Existing audience
   - Email distribution
   - Comment/feedback mechanism
   - Condensed versions drive to full content

### Alternative: Pure marimo

If you want maximum interactivity and are okay with web-only:

- Write everything in marimo notebooks
- Deploy as web apps
- Each stage is an interactive app
- Readers manipulate parameters, see results

**Trade-off:** No PDF, requires server (or WASM), but maximum engagement.

---

## 6. Specific Recommendations

### For Stage 1 (Markov Chains)

Consider building a marimo notebook that lets readers:
- Adjust n-gram order with slider
- See perplexity change in real-time
- Toggle between train/test
- Sample text at different temperatures
- Visualize transition matrix

This would be more engaging than static code + output.

### For Mathematical Derivations

Use collapsible sections in Jupyter Book:
```markdown
```{admonition} Derivation: MLE = Counting
:class: dropdown

[Full derivation here]
```
```

Readers who want depth can expand; others can skip.

### For Code

Follow "Annotated Transformer" style:
- Every line has a purpose
- Comments explain "why" not "what"
- Shapes documented
- Can be copied and run standalone

### For Exercises

Include at end of each chapter:
- Conceptual (verify understanding)
- Implementation (extend code)
- Research (explore further)

With solutions in appendix or separate file.

---

## 7. Tools to Investigate Further

| Tool | What For | Priority |
|------|----------|----------|
| **Jupyter Book** | Main publication platform | High |
| **marimo** | Interactive explorations | High |
| **Quarto** | Alternative to Jupyter Book | Medium |
| **nbdev** | If you want library from notebooks | Low |
| **Streamlit** | For standalone demos | Medium |
| **Observable** | For JS-based interactives | Low |

---

## 8. Next Steps

1. **Prototype Stage 1** in both formats:
   - Jupyter Book version (narrative + code)
   - marimo version (interactive)
   - Compare experience

2. **Get reader feedback** on format preference

3. **Decide primary distribution** based on prototype results

4. **Build toolchain** for chosen format

---

## Sources

- [fast.ai Course](https://course.fast.ai/)
- [nbdev](https://www.fast.ai/posts/2019-11-27-nbdev.html)
- [marimo](https://marimo.io/)
- [Jupyter Book](https://jupyterbook.org/)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [labml.ai](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
- [d2l.ai](https://d2l.ai/)
- [Distill.pub](https://www.ycombinator.com/blog/distill-an-interactive-visual-journal-for-machine-learning-research)
- [Crafting Interpreters](https://craftinginterpreters.com/)
- [Nicky Case](https://ncase.me/)
- [TensorFlow Playground](https://playground.tensorflow.org/)
- [Explorable Explanations](https://en.wikipedia.org/wiki/Explorable_explanation)

---

*Last updated: 2026-01-04*

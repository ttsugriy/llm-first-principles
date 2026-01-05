# Building LLMs from First Principles
## A Performance-Focused, Mathematically Rigorous Approach

**Author:** Taras Tsugrii
**Project Started:** January 2026
**Status:** Planning Phase

---

## Vision Statement

This project creates a comprehensive educational resource that teaches building Large Language Models from absolute first principles. Unlike existing resources that focus primarily on code or assume mathematical background, this work:

1. **Derives every mathematical concept from foundations** — no formula appears without proof or derivation
2. **Implements everything from scratch** — connecting code directly to mathematics
3. **Analyzes performance and trade-offs throughout** — hardware constraints shape design decisions
4. **Uses spiral pedagogy** — concepts introduced simply, then revisited with increasing depth
5. **Follows Pólya's problem-solving methodology** — understand → plan → execute → reflect
6. **Applies Tufte's information design principles** — high data-ink ratio, integrated visualizations

---

## Core Philosophy

> "Performance is the product of deep understanding of foundations"

This philosophy drives the entire structure. We don't optimize after the fact — we understand constraints from the beginning and let them inform design.

**The fundamental insight:** Every architectural decision in modern LLMs can be traced to either:
- A mathematical property (attention enables direct position access)
- A hardware constraint (memory bandwidth limits inference speed)
- A trade-off resolution (vocabulary size vs. sequence length)

By teaching all three perspectives simultaneously, we create practitioners who can reason about novel situations rather than just apply known patterns.

---

## Target Audience

### Primary Audiences

1. **ML Engineers in Production**
   - Currently learning through trial and error (OOMs, slow training)
   - Need principled understanding to debug and optimize
   - Prerequisites: Basic programming, some calculus exposure

2. **Systems Engineers Moving to ML**
   - Understand hardware but not ML abstractions
   - Need the bridge from "how GPUs work" to "how transformers use GPUs"
   - Prerequisites: Strong systems background

3. **Researchers Seeking Intuition**
   - Can derive math but lack performance intuition
   - Need mental models for feasibility
   - Prerequisites: Graduate-level ML background

4. **Ambitious Students**
   - Want deep understanding, not just API usage
   - Have time to work through foundations
   - Prerequisites: Calculus, linear algebra, basic probability

### What We Assume
- Comfort with Python programming
- Willingness to engage with mathematics
- Access to a computer (GPU helpful but not required for early stages)

### What We Don't Assume
- Prior deep learning experience
- Advanced mathematics beyond basic calculus
- Understanding of hardware architecture

---

## Unique Differentiators

### vs. Sebastian Raschka's "Build LLM from Scratch"
- **Raschka:** Excellent practical implementation, limited mathematical depth
- **This work:** Full mathematical derivations, performance analysis throughout

### vs. Andrej Karpathy's "Zero to Hero"
- **Karpathy:** Brilliant intuition-building, assumes calculus, code-focused
- **This work:** Derives the calculus, adds hardware perspective, trade-off analysis

### vs. Goodfellow's "Deep Learning"
- **Goodfellow:** Strong theory, pre-transformer, no implementation
- **This work:** Modern architectures, complete implementations, performance focus

### vs. d2l.ai "Dive into Deep Learning"
- **d2l:** Good balance but math in appendix, not integrated
- **This work:** Math derived just-in-time, always motivated

### The Gap We Fill
No single resource exists that:
- Starts from pure mathematics and builds to modern LLMs
- Implements from scratch while explaining the math
- Discusses trade-offs systematically with quantification
- Covers the full modern stack (tokenization → RLHF → inference)
- Unifies theoretical and systems perspectives

---

## Pedagogical Framework

### Spiral Learning Model

Rather than linear progression, we return to concepts with increasing depth:

```
SPIRAL 1: "It works!"
├── Build character-level Markov chain
├── Build neural language model
├── See generation quality improve
└── Question: "Why does this work?"

SPIRAL 2: "Now I understand why"
├── Derive backpropagation fully
├── Understand optimization landscape
├── Master training stability
└── Question: "Can we do better?"

SPIRAL 3: "The attention revolution"
├── Why RNNs struggle
├── Attention from first principles
├── Full transformer architecture
└── Question: "This is slow and memory-hungry"

SPIRAL 4: "Making it fast"
├── Memory analysis
├── Flash Attention derivation
├── Distributed training
└── Question: "What does scale give us?"

SPIRAL 5: "Scale and alignment"
├── Scaling laws
├── Modern architectures
├── RLHF and DPO
├── Inference optimization
└── Complete understanding
```

### Pólya's Four Steps (Applied to Each Section)

1. **Understand the Problem**
   - What are we trying to achieve?
   - What are the constraints?
   - What would success look like?

2. **Devise a Plan**
   - Have we seen a related problem?
   - Can we solve something simpler first?
   - What approaches might work?

3. **Carry Out the Plan**
   - Implement carefully
   - Verify each step
   - Check against expectations

4. **Look Back**
   - Can we derive it differently?
   - What are the limitations?
   - How does this connect to other ideas?

### Tufte's Information Design Principles

1. **High Data-Ink Ratio:** Every diagram earns its place
2. **Minimize Chartjunk:** No decorative elements
3. **Graphical Integrity:** Accurate representations only
4. **Small Multiples:** Show variations systematically
5. **Integrate Text and Graphics:** Equations in prose, not separate
6. **Layered Depth:** Main text + margin notes + appendices

---

## Content Structure Overview

See `01-SPIRAL-STRUCTURE.md` for detailed breakdown.

### Spiral 1: Foundations & First Models (Stages 1-4)
- Markov chains and probability
- Automatic differentiation
- Backpropagation deep dive
- Neural language models

### Spiral 2: Training Dynamics (Stages 5-6)
- Optimization theory
- Training stability

### Spiral 3: The Transformer (Stages 7-10)
- Attention from first principles
- Multi-head attention and architecture
- Positional encodings
- Tokenization

### Spiral 4: Making It Fast (Stages 11-13)
- Memory analysis
- Memory optimization
- Distributed training

### Spiral 5: Modern Practice (Stages 14-18)
- Architecture evolution
- Fine-tuning
- Alignment (RLHF/DPO)
- Inference optimization
- Quantization

---

## Production Format Options

### Option A: Article Series (Recommended Start)
- Publish on Substack (softwarebits.substack.com)
- ~5,000-15,000 words per stage
- Release cadence: 1-2 per month
- Allows iteration based on feedback
- Builds audience incrementally

### Option B: Book
- Traditional publisher or self-published
- ~100,000-150,000 words total
- Requires completing significant portion before release
- Higher barrier but more comprehensive

### Option C: Hybrid
- Start with article series
- Compile and expand into book
- Best of both worlds

**Recommendation:** Start with Option A, compile into Option C.

---

## Technical Requirements

### For Writing
- Markdown with LaTeX math support
- Code syntax highlighting
- Diagram tools (TikZ, Mermaid, or custom)

### For Code Examples
- Python 3.10+
- PyTorch (primary framework)
- NumPy for low-level examples
- No high-level abstractions (no HuggingFace transformers library)

### For Readers (Minimum)
- Python environment
- Ability to run Jupyter notebooks
- CPU sufficient for early stages

### For Readers (Full Experience)
- GPU access (Colab, cloud, or local)
- For distributed training sections: multi-GPU or theoretical treatment

---

## Success Metrics

### Quality Metrics
- Every mathematical claim is proved or derived
- Every code example runs and produces expected output
- Every trade-off is quantified where possible
- Reader can predict behavior in novel situations

### Engagement Metrics (if publishing as articles)
- Completion rate per article
- Reader feedback quality
- Questions asked (indicate unclear areas)
- Code contributions / corrections

---

## Timeline (Tentative)

### Phase 1: Planning & Spiral 1 (Months 1-3)
- Complete planning documents ← Current
- Write Stages 1-4
- Gather initial feedback

### Phase 2: Spirals 2-3 (Months 4-6)
- Write Stages 5-10
- Refine based on feedback

### Phase 3: Spirals 4-5 (Months 7-10)
- Write Stages 11-18
- Complete first draft

### Phase 4: Revision & Publication (Months 11-12)
- Comprehensive revision
- Code verification
- Publication

---

## File Structure

```
llm-first-principles-book/
├── 00-PROJECT-OVERVIEW.md          # This file
├── 01-SPIRAL-STRUCTURE.md          # Detailed spiral/stage breakdown
├── 02-PEDAGOGICAL-FRAMEWORK.md     # Pólya + Tufte methodology
├── 03-MATHEMATICAL-FOUNDATIONS.md  # Math prerequisites and derivations
├── 04-RESEARCH-NOTES.md            # References and source material
├── 05-OPEN-QUESTIONS.md            # Unresolved issues and decisions
├── stages/
│   ├── stage-01-markov-chains.md
│   ├── stage-02-autodiff.md
│   ├── ... (one per stage)
│   └── stage-18-quantization.md
├── code/
│   ├── stage-01/
│   ├── stage-02/
│   └── ...
└── assets/
    ├── diagrams/
    └── figures/
```

---

## Next Steps

1. ☐ Complete `01-SPIRAL-STRUCTURE.md` with all stages detailed
2. ☐ Complete `02-PEDAGOGICAL-FRAMEWORK.md` with methodology guide
3. ☐ Create `stages/stage-01-markov-chains.md` as first draft
4. ☐ Write code for Stage 1
5. ☐ Gather feedback on Stage 1 before proceeding

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-03 | 0.1 | Initial planning document created |


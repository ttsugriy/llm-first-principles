# Pedagogical Framework

## Overview

This document details the pedagogical principles underlying "Building LLMs from First Principles." Our approach synthesizes three complementary frameworks:

1. **Spiral Learning** — Return to concepts with increasing depth
2. **Pólya's Problem-Solving** — Structured approach to understanding
3. **Tufte's Information Design** — Clear, honest, integrated presentation

---

## 1. Spiral Learning Model

### The Core Idea

Traditional linear curricula present topics once, in depth, then move on. This creates two problems:
- **Cognitive overload:** Too much depth too early
- **Fragile knowledge:** Concepts not reinforced, easily forgotten

Spiral learning (Bruner, 1960) solves both by revisiting concepts at increasing levels of sophistication.

### Our Spiral Structure

```
SPIRAL 1: First Encounter
├── Concept introduced simply
├── Working implementation
├── "It works!" moment
└── Seeds of deeper questions

SPIRAL 2: Mechanism
├── Return with "how does it work?"
├── Mathematical foundations
├── Deeper implementation
└── Understanding of trade-offs

SPIRAL 3: Mastery
├── Return with "can we do better?"
├── Optimization and edge cases
├── Production considerations
└── Connection to broader theory
```

### Example: The Softmax Function

**Spiral 1 (Stage 4):**
> "To convert scores to probabilities, we use softmax: exp(zᵢ)/Σexp(zⱼ). This gives us numbers that sum to 1 and are positive."

**Spiral 2 (Stage 5):**
> "Why softmax and not just normalization? Softmax has a beautiful connection to maximum entropy distributions. Also, there's a numerical stability issue — the log-sum-exp trick."

**Spiral 3 (Stage 7):**
> "In attention, softmax creates the weighted average. The √d scaling prevents softmax from becoming too peaked. The temperature parameter controls entropy."

**Spiral 4 (Stage 12):**
> "Flash Attention computes softmax incrementally — the 'online softmax' algorithm. This enables memory efficiency without approximation."

### Implementation Guidelines

1. **First encounter should work**
   - Provide complete, runnable code
   - Reader achieves success before understanding fully
   - Builds confidence and motivation

2. **Questions seed next encounter**
   - End sections with "but why?" or "what if?"
   - Create anticipation for deeper treatment
   - Make returns feel motivated, not repetitive

3. **Later spirals assume earlier ones**
   - Reference previous treatment explicitly
   - Build on established vocabulary
   - Don't re-derive everything

4. **Cross-references are explicit**
   - "As we saw in Stage 3..."
   - "We'll return to this in Stage 12 when we optimize memory"
   - Helps readers navigate non-linearly

---

## 2. Pólya's Problem-Solving Framework

### The Four Steps

George Pólya's "How to Solve It" (1945) remains the gold standard for teaching mathematical problem-solving. We adapt it for technical learning:

#### Step 1: Understand the Problem

**Questions to ask:**
- What are we trying to achieve?
- What are the inputs and desired outputs?
- What constraints exist?
- What would success look like?
- Have we seen a similar problem?

**In our context:**
- "We want to model language. What does that mean mathematically?"
- "We want to train faster. What's currently limiting us?"
- "We want better generation. How do we measure 'better'?"

**Writing pattern:**
```markdown
## The Problem

[Clear statement of what we're trying to achieve]

**Given:** [What we have to work with]

**Find:** [What we need to produce]

**Constraints:** [Limitations we must respect]

**Success looks like:** [How we'll know we've succeeded]
```

#### Step 2: Devise a Plan

**Questions to ask:**
- Have we seen this problem before?
- Can we solve a simpler version first?
- Can we work backwards from the solution?
- What tools/techniques might apply?
- Can we decompose into subproblems?

**In our context:**
- "Modeling full language is hard. What if we only looked at the last token?"
- "Training is unstable. What if we normalized activations?"
- "O(n²) attention is slow. Can we tile the computation?"

**Writing pattern:**
```markdown
## The Approach

**Simplification:** [Easier version of the problem]

**Key insight:** [The crucial observation that unlocks the solution]

**Plan:**
1. [First step]
2. [Second step]
3. [etc.]

**Why this might work:** [Reasoning]

**What could go wrong:** [Potential issues to watch for]
```

#### Step 3: Carry Out the Plan

**Guidance:**
- Execute step by step
- Verify each step before proceeding
- Check against expectations
- If stuck, return to Step 2

**In our context:**
- Implement the code
- Run experiments
- Verify outputs match expectations
- Debug discrepancies

**Writing pattern:**
```markdown
## Implementation

### Step 1: [Name]

[Explanation]

```python
[code]
```

**Verification:** [How to check this is correct]

**Expected output:** [What you should see]

### Step 2: [Name]
...
```

#### Step 4: Look Back

**Questions to ask:**
- Can we check the result?
- Can we derive the result differently?
- What did we learn?
- Can we use this method for other problems?
- What are the limitations?

**In our context:**
- "We showed counting = MLE. Can we also show gradient descent converges to the same answer?"
- "Attention gives O(n²). Are there O(n) alternatives? What do we lose?"
- "This worked for small models. What breaks at scale?"

**Writing pattern:**
```markdown
## Reflection

### Alternative Derivation
[Different way to reach the same result]

### Verification
[Independent check of correctness]

### Connections
- This relates to [X] because...
- In Stage [N], we'll see how this extends to...

### Limitations
- This approach assumes...
- It breaks when...
- We'll address this in Stage [N]

### Key Takeaways
1. [Main lesson]
2. [Secondary lesson]
3. [Surprising insight]
```

### Pólya's Heuristics We'll Use

| Heuristic | Application |
|-----------|-------------|
| **Solve a simpler problem** | Markov → Neural LM → Transformer |
| **Draw a picture** | Computational graphs, attention patterns |
| **Work backwards** | DPO derived from RLHF objective |
| **Use a variable** | Parameterize with learnable weights |
| **Look for patterns** | Scaling laws, architectural patterns |
| **Decompose** | Transformer = Attention + FFN + Norm + Residual |
| **Check special cases** | Single token, single head, batch size 1 |

---

## 3. Tufte's Information Design Principles

### Core Principles

Edward Tufte's work on data visualization and information design provides guidance for presenting complex technical material.

#### Principle 1: High Data-Ink Ratio

**Definition:** Maximize the proportion of ink (or pixels) that represents actual information.

**For us:**
- No decorative diagrams
- Every figure conveys information
- Remove gridlines, borders, unnecessary elements
- Code examples should be minimal but complete

**Example:**

❌ Bad:
```
╔═══════════════════════════════════════════╗
║                                           ║
║   ┌─────────┐         ┌─────────┐        ║
║   │  Input  │  ───▶   │ Output  │        ║
║   └─────────┘         └─────────┘        ║
║                                           ║
╚═══════════════════════════════════════════╝
```

✅ Good:
```
Input ──▶ [Transform] ──▶ Output
```

#### Principle 2: Graphical Integrity

**Definition:** Visual representations must accurately reflect the underlying data.

**For us:**
- Axes must start at zero when showing comparisons
- Log scales should be labeled clearly
- Don't cherry-pick favorable results
- Show variance/error bars when relevant

**Example:**

❌ Bad: "Our method is 10× faster!" [Y-axis starts at 90%]

✅ Good: "Our method reduces latency from 100ms to 60ms (40% reduction)" [Y-axis 0-100]

#### Principle 3: Minimize Non-Data-Ink

**Definition:** Remove elements that don't contribute to understanding.

**For us:**
- No "Figure 1 shows the architecture" (just show it)
- Remove unnecessary code comments
- Eliminate redundant text + figure combinations
- Trust the reader

#### Principle 4: Small Multiples

**Definition:** Show variation by repeating the same visual structure with different data.

**For us:**
- Compare optimizers with same graph structure, different curves
- Show attention patterns across heads as grid
- Display scaling with consistent axes

**Example:**

```
Perplexity vs Training Steps

Adam          AdamW         SGD
100│╲         100│╲         100│╲
   │ ╲           │ ╲           │ ╲
50 │  ╲       50 │  ╲       50 │   ╲
   │   ╲___      │   ╲___      │    ╲____
 0 └────────   0 └────────   0 └────────
   0    100K     0    100K     0    100K
```

#### Principle 5: Integrate Text and Graphics

**Definition:** Don't separate prose from visual elements.

**For us:**
- Equations in text, not numbered and referenced
- Code inline with explanation
- Diagrams next to relevant prose
- Avoid "see Appendix B for proof"

**Example:**

❌ Bad:
> The attention weight αᵢⱼ is computed according to Equation 3.2 (see Figure 7).

✅ Good:
> The attention weight αᵢⱼ = softmax(qᵢ·kⱼ/√d) tells us how much position i should attend to position j:
> ```
> α[i,j] = softmax(Q[i] @ K.T / sqrt(d))  # shape: [seq, seq]
> ```

#### Principle 6: Layered Depth

**Definition:** Provide multiple levels of detail for different reading depths.

**For us:**
- Main text: readable flow, key insights
- Margin notes: additional details, caveats
- Code blocks: full implementations
- Appendices: complete proofs, extended derivations

**Structure:**
```markdown
# Main Concept

[Core explanation that all readers need]

> 💡 **Key insight:** [The essential takeaway]

<details>
<summary>📐 Mathematical derivation</summary>

[Full derivation for those who want it]

</details>

<details>
<summary>🔧 Implementation details</summary>

[Detailed code for those implementing]

</details>

---
**Notes:**
- [Additional context]
- [Historical aside]
- [Connection to research]
```

---

## 4. Writing Patterns

### Pattern: Introducing a New Concept

```markdown
## [Concept Name]

### The Problem

[What problem does this concept solve?]

### The Intuition

[Simple, possibly imprecise explanation]

[Analogy if helpful]

### The Mathematics

[Precise definition]

[Key equations with explanation]

### The Implementation

```python
[code]
```

### The Trade-offs

[When to use, when not to use, what you're trading off]

### Looking Ahead

[How this connects to later topics]
```

### Pattern: Deriving a Result

```markdown
## Deriving [Result]

### Starting Point

We know: [established facts]

We want to show: [goal]

### Step 1: [Name]

[reasoning]

[equation]

### Step 2: [Name]

[reasoning]

[equation]

...

### Conclusion

Therefore: [final result]

### Verification

We can check this by: [independent check]

### Interpretation

This tells us: [what it means]
```

### Pattern: Analyzing Trade-offs

```markdown
## Trade-off: [X] vs [Y]

### The Tension

[Why we can't have both]

### Option A: Favor [X]

**How:** [what we do]
**Consequence:** [what happens to Y]
**When appropriate:** [use cases]

### Option B: Favor [Y]

**How:** [what we do]
**Consequence:** [what happens to X]
**When appropriate:** [use cases]

### Finding the Balance

[Practical guidance]

| Setting | [X] | [Y] | Recommendation |
|---------|-----|-----|----------------|
| ...     | ... | ... | ...            |
```

### Pattern: Comparing Approaches

```markdown
## Comparison: [A] vs [B] vs [C]

### Overview

| Aspect     | A   | B   | C   |
|------------|-----|-----|-----|
| Property 1 | ... | ... | ... |
| Property 2 | ... | ... | ... |
| Best for   | ... | ... | ... |

### [Approach A]

**Idea:** [core concept]
**Strengths:** [what it does well]
**Weaknesses:** [limitations]

### [Approach B]
...

### When to Use What

[Decision flowchart or heuristics]
```

---

## 5. Code Standards

### Philosophy

- **Complete but minimal:** Every code block should run, but include only what's necessary
- **Self-explanatory:** Prefer clear names over comments
- **PyTorch-first:** Use PyTorch for all neural network code
- **NumPy for fundamentals:** Use NumPy when teaching concepts before NN
- **No magic:** Avoid high-level libraries that hide mechanics

### Style Guide

```python
# Imports at top, grouped
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple

# Constants in CAPS
VOCAB_SIZE = 10000
EMBED_DIM = 256

# Type hints for clarity
def attention(
    query: torch.Tensor,    # [batch, seq_q, dim]
    key: torch.Tensor,      # [batch, seq_k, dim]
    value: torch.Tensor,    # [batch, seq_k, dim]
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:          # [batch, seq_q, dim]
    """
    Scaled dot-product attention.

    Computes: softmax(QK^T / sqrt(d)) V
    """
    d = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value)
```

### What to Include

- Shape comments for tensors
- Brief docstring for non-obvious functions
- Assertion checks for shapes in educational code
- Simple test at end of file

### What to Exclude

- Extensive error handling (distracts)
- Logging (not essential)
- Configuration management (add complexity)
- Multi-file organization (within a section)

---

## 6. Visual Standards

### Diagrams

- Use simple ASCII or Mermaid for inline diagrams
- Complex diagrams as separate SVG files
- Consistent color scheme:
  - Blue: inputs
  - Green: learnable parameters
  - Orange: activations
  - Red: gradients/errors

### Plots

- Matplotlib with clean style
- No gridlines unless essential
- Large, readable axis labels
- Legend only if multiple series
- Save as SVG for quality

### Equations

- LaTeX for complex equations
- Inline for simple (e.g., O(n²))
- Define all notation on first use
- Number only if referenced elsewhere

---

## 7. Voice and Tone

### Voice

- **First person plural ("we"):** Creates collaboration with reader
- **Active voice:** "We compute the gradient" not "The gradient is computed"
- **Direct:** "This fails because..." not "It might be observed that..."

### Tone

- **Confident but humble:** We know this; there's more to learn
- **Curious:** "Why does this work?" is celebrated
- **Honest about limitations:** "This approach fails when..."
- **Encouraging:** "This is tricky—let's work through it carefully"

### What to Avoid

- ❌ "Obviously..." (nothing is obvious)
- ❌ "Simply..." (implies reader should find it easy)
- ❌ "It is well known that..." (either explain or cite)
- ❌ Jargon without definition
- ❌ Unnecessary hedging ("perhaps", "might", "could")

---

## 8. Quality Checklist

Before finalizing any section:

### Content
- [ ] Does every concept serve the learning objectives?
- [ ] Is every mathematical claim proved or derived?
- [ ] Are trade-offs quantified where possible?
- [ ] Are connections to other stages explicit?

### Pólya Structure
- [ ] Is the problem clearly stated?
- [ ] Is the approach motivated?
- [ ] Is the implementation verified?
- [ ] Is there reflection and connection?

### Tufte Principles
- [ ] Does every figure earn its place?
- [ ] Is text-graphic integration smooth?
- [ ] Is depth layered appropriately?
- [ ] Are representations honest?

### Code
- [ ] Does all code run?
- [ ] Are shapes documented?
- [ ] Is the code minimal but complete?
- [ ] Are there verification checks?

### Writing
- [ ] Is the voice consistent?
- [ ] Is jargon defined?
- [ ] Are forward/backward references explicit?
- [ ] Is the length appropriate?

---

## 9. Example Section Structure

Here's a complete example of how a section should be structured:

```markdown
# Stage X: [Topic Name]

## Overview

[1-2 paragraph summary of what this stage covers and why it matters]

**Learning Objectives:**
- [Objective 1]
- [Objective 2]
- [Objective 3]

**Prerequisites:**
- Stage [N]: [Concept needed]
- Stage [M]: [Concept needed]

---

## 1. The Problem

[Pólya Step 1: Understand the Problem]

### What We're Trying to Achieve

[Clear problem statement]

### Why It's Hard

[What makes this non-trivial]

### Success Criteria

[How we'll know we've succeeded]

---

## 2. The Approach

[Pólya Step 2: Devise a Plan]

### Key Insight

[The crucial observation]

### Our Plan

1. [Step 1]
2. [Step 2]
3. [Step 3]

---

## 3. The Details

[Pólya Step 3: Carry Out the Plan]

### 3.1 [First Major Topic]

[Explanation + derivation + code]

### 3.2 [Second Major Topic]

[Explanation + derivation + code]

---

## 4. Experiments

### Setup

[What we'll test]

### Results

[What we found]

### Analysis

[What it means]

---

## 5. Reflection

[Pólya Step 4: Look Back]

### What We Learned

[Key takeaways]

### Alternative Approaches

[Other ways to solve this]

### Limitations

[When this doesn't work]

### Connections

- **Backward:** [How this builds on earlier stages]
- **Forward:** [How this enables later stages]

---

## Summary

[Concise recap]

## Exercises

1. [Exercise for reinforcement]
2. [Exercise for extension]
3. [Exercise for connection]

## Further Reading

- [Resource 1]: [Why it's useful]
- [Resource 2]: [Why it's useful]
```

---

## 10. Revision Process

### First Draft
- Focus on correctness and completeness
- Don't worry about polish
- Include all derivations, even if messy
- Mark uncertain areas with [TODO]

### Second Draft
- Verify all code runs
- Verify all math is correct
- Add visualizations
- Improve flow and transitions

### Third Draft
- Apply Tufte principles (cut ruthlessly)
- Apply Pólya structure checks
- Verify spiral connections
- Polish prose

### Final Review
- Fresh-eyes read-through
- Check against quality checklist
- Verify forward/backward references
- Test with target reader if possible

---

This framework should guide all content creation for the project. When in doubt, return to the core principles:
- **Spiral:** Is the concept introduced appropriately for this level?
- **Pólya:** Does the structure support problem-solving thinking?
- **Tufte:** Is the presentation clear, honest, and integrated?

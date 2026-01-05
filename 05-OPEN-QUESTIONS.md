# Open Questions and Future Work

## Overview

This document tracks unresolved questions, decisions to be made, and areas requiring further research or discussion. Use this to ensure nothing falls through the cracks.

---

## 1. Content Decisions

### 1.1 Scope Questions

#### Q1: How much hardware detail?
**Status:** Open
**Options:**
- A) Minimal — focus on algorithms, mention hardware constraints abstractly
- B) Moderate — explain memory hierarchy, GPU architecture basics
- C) Deep — include CUDA programming, roofline models, kernel optimization

**Considerations:**
- Target audience varies (some are systems people, some are ML people)
- Deep hardware coverage could be its own book
- Performance focus suggests at least moderate coverage

**Tentative decision:** B (Moderate) with optional deep-dive sections

---

#### Q2: Include RNNs/LSTMs or skip directly to attention?
**Status:** Decided (Skip, with mention)
**Decision:** Skip detailed RNN coverage. Mention briefly in "why attention won" context.
**Rationale:**
- RNNs are historical for LLMs
- Time better spent on transformers
- Can reference other resources for RNN background

---

#### Q3: Cover vision transformers (ViT)?
**Status:** Decided (No)
**Decision:** Focus exclusively on language models.
**Rationale:**
- Keeps scope manageable
- Language has enough depth
- ViT could be a follow-up project

---

#### Q4: How much RLHF implementation detail?
**Status:** Open
**Options:**
- A) Conceptual only — derive math, explain algorithm, no full implementation
- B) Toy implementation — implement on small model with synthetic preferences
- C) Full implementation — production-quality RLHF training

**Considerations:**
- Full RLHF is complex (reward model training, PPO, etc.)
- DPO is simpler and increasingly popular
- Conceptual understanding may be sufficient for most readers

**Tentative decision:** B for DPO, A for RLHF (with pointers to implementations)

---

#### Q5: Include reasoning/chain-of-thought?
**Status:** Open
**Options:**
- A) No — out of scope
- B) Brief mention in post-training section
- C) Dedicated stage on reasoning capabilities

**Considerations:**
- Reasoning is increasingly important
- But implementation details are less clear than other topics
- Could be added as Stage 19 if space permits

**Tentative decision:** B (Brief mention, may expand later)

---

### 1.2 Pedagogical Decisions

#### Q6: Code-first or math-first for each section?
**Status:** Decided (Alternating based on topic)
**Decision:**
- Spiral 1: Code-first (build intuition)
- Spiral 2+: Math-first (understand deeply)

---

#### Q7: Use Jupyter notebooks or standalone scripts?
**Status:** Open
**Options:**
- A) Notebooks — interactive, inline outputs
- B) Scripts — cleaner, more reusable
- C) Both — notebooks for exploration, scripts for final implementations

**Considerations:**
- Notebooks good for learning, bad for version control
- Scripts good for production, less interactive
- Many readers prefer notebooks for ML

**Tentative decision:** C (Both)

---

#### Q8: What programming language for low-level examples?
**Status:** Decided (Python with NumPy)
**Decision:** Python/NumPy for clarity, PyTorch for neural networks
**Rationale:**
- Python is most accessible
- NumPy sufficient for autodiff concepts
- PyTorch for actual training

---

### 1.3 Presentation Decisions

#### Q9: Include exercises?
**Status:** Decided (Yes)
**Decision:** 2-4 exercises per stage
**Types:**
- Conceptual (verify understanding)
- Implementation (extend code)
- Analysis (explore trade-offs)

---

#### Q10: Solutions to exercises?
**Status:** Open
**Options:**
- A) No solutions (encourage struggle)
- B) Hints only
- C) Full solutions (for self-study)
- D) Solutions in separate document/appendix

**Tentative decision:** D (Separate solutions document)

---

#### Q11: Mathematical notation consistency?
**Status:** In Progress
**See:** `03-MATHEMATICAL-FOUNDATIONS.md` Notation Reference
**Outstanding issues:**
- Vectors: bold vs arrow notation
- Matrices: capital bold vs capital only
- Indexing: 0-based or 1-based?

**Decision needed by:** Stage 1 completion

---

## 2. Technical Questions

### 2.1 Implementation Questions

#### Q12: Which tokenizer to implement in detail?
**Status:** Decided (BPE)
**Decision:** Implement BPE fully, mention others
**Rationale:**
- BPE most common (GPT, Llama use variants)
- Simple enough to implement from scratch
- Unigram/WordPiece similar conceptually

---

#### Q13: What model size to target for examples?
**Status:** Open
**Options:**
- A) Tiny (< 10M params) — runs on CPU
- B) Small (10M-100M) — needs GPU but fits on consumer hardware
- C) Medium (100M-1B) — needs good GPU
- D) Vary by stage

**Considerations:**
- Larger models show more realistic behavior
- But accessibility is important
- Most concepts work at any scale

**Tentative decision:** D (Tiny for Spiral 1-2, Small for Spiral 3-4, discussion of larger for Spiral 5)

---

#### Q14: Which pre-training dataset for examples?
**Status:** Open
**Options:**
- A) Shakespeare (traditional, small)
- B) TinyStories (designed for small models)
- C) Wikipedia subset (real-world, larger)
- D) Custom curated dataset

**Considerations:**
- Shakespeare is traditional but artificial
- TinyStories designed for this use case
- Wikipedia more realistic but messier

**Tentative decision:** A for Spiral 1, B for Spiral 3+

---

### 2.2 Research Questions

#### Q15: How to present scaling laws — empirical or derived?
**Status:** Open
**Question:** Scaling laws are empirical findings. Should we:
- A) Present as empirical (here's what we observe)
- B) Attempt theoretical justification (here's why it might be)
- C) Both (observation + speculation)

**Considerations:**
- Full theoretical understanding doesn't exist
- Some theoretical frameworks (e.g., statistical mechanics analogies)
- Honesty requires acknowledging empirical nature

**Tentative decision:** C (Both, clearly marked)

---

#### Q16: How deep on Flash Attention?
**Status:** Open
**Question:** Flash Attention involves complex CUDA programming. How much to show?
- A) Algorithm only (pseudocode level)
- B) Simplified Python implementation (not performant but correct)
- C) Actual CUDA kernel (complex but real)

**Considerations:**
- Real Flash Attention requires CUDA expertise
- Python version would be slow but educational
- Algorithm understanding is the key insight

**Tentative decision:** B (Python implementation) with discussion of C

---

#### Q17: Include speculative decoding proofs?
**Status:** Open
**Question:** Speculative decoding has elegant probability proofs. Include?
- A) Yes, full derivation
- B) State result, sketch proof
- C) Just explain algorithm

**Tentative decision:** B (State and sketch)

---

## 3. Production Questions

### 3.1 Format and Distribution

#### Q18: Primary format?
**Status:** Open
**Options:**
- A) Substack articles (your existing platform)
- B) Dedicated website/blog
- C) Book (self-published)
- D) Book (traditional publisher)
- E) Hybrid (articles first, book later)

**Considerations:**
- Substack has existing audience
- Dedicated site gives more control
- Book has higher prestige but slower
- Hybrid allows iteration

**Tentative decision:** E (Hybrid — Substack first, book later)

---

#### Q19: Open source the code?
**Status:** Decided (Yes)
**Decision:** All code MIT licensed on GitHub
**Rationale:**
- Encourages learning
- Allows contributions/corrections
- Standard practice for educational content

---

#### Q20: Paywalled or free?
**Status:** Open
**Options:**
- A) All free
- B) Free with paid extras (solutions, code, etc.)
- C) Freemium (early stages free, later paid)
- D) Fully paid

**Considerations:**
- Free maximizes reach
- Paid provides sustainability
- Freemium common model

**Tentative decision:** B or C (Some free, some paid)

---

### 3.2 Collaboration

#### Q21: Accept contributions?
**Status:** Open
**Options:**
- A) Solo author
- B) Accept corrections/typos via GitHub
- C) Accept content contributions
- D) Co-authors for specific sections

**Considerations:**
- Consistent voice important
- Community can catch errors
- Some sections might benefit from expert contributions

**Tentative decision:** B (Corrections via GitHub)

---

#### Q22: Peer review process?
**Status:** Open
**Question:** How to ensure technical accuracy?
**Options:**
- A) Self-review only
- B) Informal expert review (friends/colleagues)
- C) Formal review (paid technical reviewers)
- D) Community review (draft releases)

**Tentative decision:** B + D (Expert friends + community drafts)

---

## 4. Specific Content Questions

### 4.1 Stage-Specific Questions

#### Stage 1: Markov Chains
- [ ] Include smoothing techniques (Kneser-Ney, etc.) or just Laplace?
- [ ] How much n-gram history? (probably just bigram/trigram examples)

#### Stage 2: Autodiff
- [ ] Include dual numbers for forward mode?
- [ ] How far to extend tensor operations?

#### Stage 3: Backprop
- [ ] Include all attention gradients or just the key ones?
- [ ] Batch norm backward — full derivation or summarize?

#### Stage 5: Optimization
- [ ] Include learning rate finders (like Smith's)?
- [ ] Cover Sophia, Lion, or just Adam family?

#### Stage 7: Attention
- [ ] Include cross-attention or just self-attention?
- [ ] Discuss efficient attention variants (linear, sparse)?

#### Stage 9: Position Encodings
- [ ] How deep on RoPE derivation? (complex number formulation)
- [ ] Cover NTK-aware scaling for length extrapolation?

#### Stage 12: Memory Optimization
- [ ] Include memory-efficient attention (not Flash)?
- [ ] Cover sequence parallelism?

#### Stage 13: Distributed
- [ ] Include FSDP or just conceptual ZeRO?
- [ ] Cover expert parallelism (MoE)?

#### Stage 16: Alignment
- [ ] Cover IPO, ORPO, or just RLHF + DPO?
- [ ] Include constitutional AI?

#### Stage 18: Quantization
- [ ] Cover GGML/GGUF formats?
- [ ] Include activation quantization?

---

## 5. Feedback Collection

### 5.1 Questions for Early Readers

1. Is the pacing appropriate?
2. Where do you get confused?
3. What's missing that you expected?
4. What could be cut?
5. Are the code examples runnable?
6. Is the math level appropriate?

### 5.2 Metrics to Track

- Time to complete each stage
- Points where readers stop
- Questions asked (indicate unclear areas)
- Code issues reported
- Suggestions for improvement

---

## 6. Future Extensions

### 6.1 Potential Additional Stages

- Stage 19: Multimodal (vision-language)
- Stage 20: Retrieval Augmented Generation (RAG)
- Stage 21: Agents and tool use
- Stage 22: Evaluation and benchmarking
- Stage 23: Safety and interpretability

### 6.2 Potential Companion Projects

- Video series (Karpathy-style)
- Interactive notebooks (Colab/Kaggle)
- Study group curriculum
- Certification program

---

## 7. Decision Log

Track decisions as they're made:

| ID | Question | Decision | Date | Rationale |
|----|----------|----------|------|-----------|
| Q2 | RNNs? | Skip with mention | 2026-01-03 | Focus on transformers |
| Q3 | ViT? | No | 2026-01-03 | Language focus |
| Q6 | Code vs math first | Alternating | 2026-01-03 | Spiral-dependent |
| Q8 | Language | Python/NumPy/PyTorch | 2026-01-03 | Accessibility |
| Q9 | Exercises | Yes, 2-4 per stage | 2026-01-03 | Reinforce learning |
| Q12 | Tokenizer | BPE | 2026-01-03 | Most common |
| Q19 | Open source | Yes, MIT | 2026-01-03 | Standard practice |

---

## 8. Blockers and Dependencies

### Current Blockers
1. None yet (planning phase)

### Dependencies
- Stage N depends on Stage N-1 (mostly sequential)
- Stage 11-13 can be partially parallelized
- Stage 15-16 depend on 14 but could be written in parallel

---

## 9. Review Schedule

| Item | Review Frequency | Next Review |
|------|------------------|-------------|
| Open Questions | Weekly | 2026-01-10 |
| Content Decisions | Per stage completion | After Stage 1 |
| Technical Questions | As they arise | - |
| Production Questions | Monthly | 2026-02-01 |

---

*Last updated: 2026-01-03*

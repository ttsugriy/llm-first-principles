# Section 6.7: Scaling Laws — How Performance Relates to Compute

*Reading time: 18 minutes | Difficulty: ★★★★☆*

Scaling laws describe how model performance changes with compute, parameters, and data. Understanding these laws enables efficient allocation of training resources and prediction of model capabilities.

## The Discovery of Scaling Laws

In 2020, OpenAI discovered that LLM loss follows predictable power laws:

$$L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty$$

Where:

- L = Cross-entropy loss
- N = Number of parameters
- D = Number of training tokens
- N_c, D_c = Critical scaling constants
- $α_N$, $α_D$ = Scaling exponents (~0.076, ~0.095)
- L_∞ = Irreducible loss (entropy of natural text)

## The Three Axes of Scaling

### 1. Parameters (N)

More parameters → Lower loss (diminishing returns)

```
Loss vs Parameters (log-log scale):

Loss  │╲
      │ ╲
      │  ╲
      │   ╲.......
      │
      └─────────────
        10M 100M 1B 10B 100B  Parameters
```

### 2. Data (D)

More training tokens → Lower loss (diminishing returns)

```
Loss vs Training Tokens:

Loss  │╲
      │ ╲
      │  ╲
      │   ╲.......
      │
      └─────────────
        1B  10B 100B 1T 10T   Tokens
```

### 3. Compute (C)

More FLOPs → Lower loss

$$C \approx 6ND$$

(Approximate FLOPs for training: 6 × parameters × tokens)

## The Original Scaling Laws (Kaplan et al., 2020)

### Key Findings

1. **Power law scaling**: Performance improves as a power of resources
2. **Smooth scaling**: No sudden jumps or plateaus
3. **Predictable**: Can extrapolate from small to large models
4. **Compute-optimal**: There's an optimal N/D ratio for fixed compute

### Compute-Optimal Allocation

Given compute budget C, how to split between N and D?

**Kaplan's finding**: Scale parameters faster than data

$$N_{opt} \propto C^{0.73}$$
$$D_{opt} \propto C^{0.27}$$

This suggested training very large models on relatively less data.

## Chinchilla: Revised Scaling Laws

In 2022, DeepMind's Chinchilla paper revised these findings:

### The Chinchilla Insight

Kaplan's models were **undertrained**. With optimal training:

$$N_{opt} \propto C^{0.5}$$
$$D_{opt} \propto C^{0.5}$$

**New rule**: Parameters and tokens should scale equally!

### The 20× Rule

For compute-optimal training:

$$D \approx 20 \times N$$

Train on 20 tokens per parameter.

| Model | Parameters | Optimal Tokens |
|-------|------------|----------------|
| 1B | 1B | 20B |
| 7B | 7B | 140B |
| 70B | 70B | 1.4T |
| 175B | 175B | 3.5T |

### Chinchilla vs Gopher

```
Same compute budget:

Gopher:    280B params, 300B tokens  ← Undertrained
Chinchilla: 70B params, 1.4T tokens  ← Compute-optimal

Result: Chinchilla outperformed Gopher on almost all benchmarks
        despite being 4× smaller!
```

## Implications for Training

### Cost vs Quality Trade-off

```
Fixed compute budget: 10^23 FLOPs

Option A: 100B model, 170B tokens
  - Training cost: Fixed
  - Inference cost: HIGH (100B params)
  - Quality: Good

Option B: 10B model, 1.7T tokens
  - Training cost: Fixed
  - Inference cost: LOW (10B params)
  - Quality: BETTER (compute-optimal)

Winner: Smaller, better-trained models for most use cases
```

### Inference Cost Matters

For deployed models, inference cost often dominates:

$$\text{Total Cost} = \text{Training Cost} + \text{Inference Cost} \times \text{Queries}$$

Chinchilla suggests training smaller models longer, which reduces inference cost.

### Modern Practice

LLaMA's approach reflects Chinchilla:

| Model | Parameters | Training Tokens | Tokens/Param |
|-------|------------|-----------------|--------------|
| LLaMA 7B | 7B | 1T | 143× |
| LLaMA 13B | 13B | 1T | 77× |
| LLaMA 2 70B | 70B | 2T | 29× |

These are heavily overtrained relative to compute-optimal, prioritizing inference efficiency.

## Emergent Abilities

### What Are Emergent Abilities?

Some capabilities appear suddenly at scale:

```
Performance vs Scale:

Accuracy │          ╭──
         │         ╱
         │        ╱
         │-------╯    ← Emergence!
         │
         └─────────────
           1B  10B  100B   Parameters
```

### Examples of Emergence

| Ability | Emerges Around |
|---------|---------------|
| In-context learning | 10B+ params |
| Chain-of-thought reasoning | 100B+ params |
| Arithmetic | 10B+ params |
| Code generation | 10B+ params |
| Truthful QA improvement | 100B+ params |

### Are Emergent Abilities Real?

Recent research suggests emergence may be a metric artifact:

- With different metrics, scaling can look smooth
- Log-log plots can hide gradual improvement
- But qualitative jumps in capability are real

## Beyond Simple Scaling

### Scaling with Architecture

Some architectures scale better:

```
Dense models:   L ∝ N^{-0.076}
MoE models:     L ∝ N_{active}^{-0.04}  (flatter, but more params)
```

### Scaling with Data Quality

Better data → Better scaling:

```
Random web:     L = L_0 + (D/D_0)^{-0.1}
Curated data:   L = L_0 + (D/D_0)^{-0.15}  ← Steeper improvement
```

Data quality may matter more than quantity beyond a point.

### Scaling with Modality

Multimodal models show different scaling:

```
Text-only:      ~0.076 exponent
Text + Images:  ~0.08 exponent  (may scale better)
```

## Practical Scaling Predictions

### Loss Prediction

Given a small model's performance, predict large model:

```python
def predict_loss(small_loss, small_params, large_params, exponent=0.076):
    """Predict loss for larger model."""
    ratio = small_params / large_params
    improvement = ratio ** exponent
    # This is approximate; actual formula is more complex
    return small_loss * improvement

# Example
small_loss = 3.5  # 125M model loss
large_loss = predict_loss(3.5, 125e6, 1e9)
print(f"Predicted 1B loss: {large_loss:.2f}")  # ~3.2
```

### Compute Requirements

Estimate compute for target loss:

```python
def compute_for_loss(target_loss, current_loss, current_compute):
    """Estimate compute to reach target loss."""
    # Assuming L ∝ C^{-0.05}
    ratio = (current_loss / target_loss) ** (1 / 0.05)
    return current_compute * ratio

# Example: 10× loss reduction needs ~10^20× compute
```

## Scaling Law Caveats

### What Scaling Laws Don't Capture

| Limitation | Reality |
|------------|---------|
| Task-specific performance | Scaling helps more for some tasks |
| Safety/alignment | More scale ≠ safer |
| Efficiency innovations | Architecture matters |
| Data quality | Not all tokens equal |
| Evaluation metrics | Loss ≠ usefulness |

### Breaking Scaling Laws

Innovations can break expected scaling:

- **Flash Attention**: Same compute, longer context
- **MoE**: More params for same compute
- **Better data**: More performance per token
- **Distillation**: Small model with large model quality

## The Frontier

### Current State (2024)

```
Frontier models: ~1T+ parameters, ~10T+ tokens
Compute: ~10^25 FLOPs

Still seeing:

- Continued improvement with scale
- New emergent capabilities
- No sign of ceiling yet
```

### Future Predictions

```
2025: ~10^26 FLOPs, more emergent abilities?
2026: ~10^27 FLOPs, approaching human-level on more tasks?
????: Unknown ceiling
```

!!! info "Connection to Modern LLMs"

    Scaling law implications:

    - **GPT-4**: Trained near compute-optimal (assumed)
    - **LLaMA**: Overtrained for inference efficiency
    - **Claude**: Scaling strategy not disclosed
    - **Mistral**: Very efficient via architecture innovations

    Major labs use scaling laws to plan multi-year training investments.

## Exercises

1. **Fit scaling law**: Train models of size 10M, 100M, 1B on same data. Fit power law.

2. **Chinchilla analysis**: For your GPU budget, compute optimal N and D.

3. **Emergence search**: Find a task that shows emergence in your small models.

4. **Data efficiency**: Compare training on 1B high-quality vs 10B low-quality tokens.

5. **Extrapolation**: From 100M model results, predict 1B model performance. Verify.

## Summary

| Concept | Definition | Implication |
|---------|------------|-------------|
| Scaling law | L ∝ $N^{-α}$ | Performance predictable |
| Chinchilla rule | D ≈ 20N | Train longer on less data |
| Compute-optimal | Balance N and D | Don't over-parameterize |
| Emergence | Sudden capability gain | Scale unlocks abilities |

**Key takeaway**: LLM performance follows predictable power laws in compute, parameters, and data. The Chinchilla scaling laws show that parameters and data should scale equally—the old approach of training huge models on little data was suboptimal. For practical deployment, overtrained smaller models often give better cost/performance trade-offs. Understanding scaling laws enables efficient allocation of training resources and prediction of model capabilities.

→ **Next**: [Section 6.8: Implementation](08-implementation.md)

# Section 1.6: Generating Text — Sampling and Temperature

We've learned how to train a model and evaluate it. Now: how do we use it to generate text?

## The Generation Problem

Given a trained model P(next | context), we want to produce new text that "sounds like" the training data.

**Autoregressive generation**:
1. Start with initial context (e.g., ⟨START⟩)
2. Sample next token from P(token | context)
3. Append sampled token to context
4. Repeat until ⟨END⟩ or maximum length

But step 2 hides a crucial choice: *how* do we sample from P(token | context)?

## Greedy Decoding: The Obvious Approach

**Greedy decoding**: Always pick the highest-probability token.

$$x_t = \arg\max_{x} P(x | \text{context})$$


**Problems with greedy**:

1. **Repetitive**: Once you pick a common pattern, you keep repeating it.
   "The the the the the..."

2. **No diversity**: Running generation twice gives identical output.

3. **Misses good sequences**: The most likely sequence isn't always found by greedily picking most likely tokens.

**Example**: Suppose at position t we can choose:
- "The" with P("The") = 0.3, and after "The", the best continuation has P = 0.2
  - Path probability: 0.3 × 0.2 = 0.06
- "A" with P("A") = 0.2, and after "A", the best continuation has P = 0.5
  - Path probability: 0.2 × 0.5 = 0.10

Greedy picks "The" at the first step (since 0.3 > 0.2), but the complete sequence starting with "A" has higher probability (0.10 > 0.06)! This is why greedy decoding doesn't guarantee finding the globally most likely sequence.

## Ancestral Sampling: The Theoretically Correct Approach

**Ancestral sampling**: Sample each token from the full distribution.

$$x_t \sim P(x | \text{context})$$


This produces samples from the true model distribution—exactly what the model learned.

**How to sample from a discrete distribution**:
1. List all tokens with their probabilities: P(t₁), P(t₂), ...
2. Draw a random number r uniformly from [0, 1]
3. Find the token where the cumulative probability crosses r

**Python implementation**:
```python
import random

def sample(distribution):
    """Sample from a probability distribution (dict: token -> prob)."""
    r = random.random()  # Uniform [0, 1)
    cumulative = 0.0
    for token, prob in distribution.items():
        cumulative += prob
        if r < cumulative:
            return token
    return token  # Handle floating point errors
```

Or using the standard library:
```python
import random
tokens = list(distribution.keys())
probs = list(distribution.values())
return random.choices(tokens, weights=probs, k=1)[0]
```

## The Problem with Pure Sampling

Pure ancestral sampling can produce low-quality text because it *includes* the low-probability tokens.

If P("the" | context) = 0.1 and P("xyzzy" | context) = 0.001, pure sampling will occasionally output "xyzzy"—rare but possible.

Over many tokens, unlikely events accumulate, producing incoherent text.

**We want control** over how "random" vs "deterministic" the generation is.

## Temperature: Controlling Randomness

**Temperature** is a parameter that rescales the probability distribution before sampling.

Given probabilities P(t) for each token t, the temperature-scaled distribution is:

$$P_T(t) = \frac{P(t)^{1/T}}{\sum_{t'} P(t')^{1/T}}$$


Or equivalently, working in log-space:

$$P_T(t) = \frac{\exp(\log P(t) / T)}{\sum_{t'} \exp(\log P(t') / T)}$$


**What temperature does**:

| Temperature | Effect |
|-------------|--------|
| T → 0 | Distribution becomes one-hot (greedy) |
| T = 1 | Original distribution (no change) |
| T > 1 | Distribution becomes flatter (more random) |
| T → ∞ | Distribution becomes uniform |

## Deriving the Temperature Formula

Where does this formula come from? It's inspired by statistical mechanics.

### The Softmax Function

First, let's understand softmax. Given "logits" (unnormalized log-probabilities) z₁, z₂, ..., zₙ:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$


This converts arbitrary real numbers into a probability distribution.

**Properties**:
1. All outputs positive (due to exponential)
2. Outputs sum to 1 (due to normalization)
3. Larger zᵢ → larger probability

### Adding Temperature

Temperature divides the logits before softmax:

$$\text{softmax}(z_i / T) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$


**Why this works**:
- Dividing by T > 1 makes logits smaller → differences smaller → distribution flatter
- Dividing by T < 1 makes logits larger → differences larger → distribution sharper

### Connection to Statistical Mechanics

In physics, the Boltzmann distribution gives the probability of a system being in state i with energy Eᵢ:

$$P(i) = \frac{e^{-E_i / kT}}{Z}$$


where T is temperature and k is Boltzmann's constant.

- High temperature: System explores many states (high entropy)
- Low temperature: System settles into low-energy states

Our language model temperature is exactly analogous: high T means exploring more options, low T means sticking to high-probability options.

## Visualizing Temperature Effects

Consider this distribution: P(A) = 0.5, P(B) = 0.3, P(C) = 0.15, P(D) = 0.05

After temperature scaling:

| Token | T=0.5 | T=1.0 | T=2.0 | T→∞ |
|-------|-------|-------|-------|-----|
| A | 0.69 | 0.50 | 0.35 | 0.25 |
| B | 0.24 | 0.30 | 0.29 | 0.25 |
| C | 0.06 | 0.15 | 0.22 | 0.25 |
| D | 0.01 | 0.05 | 0.14 | 0.25 |

**Observations**:
- T=0.5: "A" dominates even more (69% vs 50%)
- T=2.0: Distribution is more uniform
- T→∞: All tokens equally likely (25% each)

## The Temperature Limit: T → 0

As T → 0, the distribution becomes a one-hot vector pointing at the highest-probability token.

**Proof**: Let z₁ > z₂ > ... > zₙ (sorted logits).

$$\lim_{T \to 0} \frac{e^{z_i/T}}{\sum_j e^{z_j/T}} = \lim_{T \to 0} \frac{e^{z_i/T}}{e^{z_1/T}(1 + \sum_{j>1} e^{(z_j-z_1)/T})}$$


Since z₁ > zⱼ for j > 1, the terms e^{(zⱼ-z₁)/T} → 0 as T → 0.

For i = 1: limit = 1
For i > 1: limit = 0

So T → 0 gives greedy decoding.

## Implementation

```python
import math

def apply_temperature(distribution, temperature):
    """Apply temperature to a probability distribution.

    Args:
        distribution: dict mapping token -> probability
        temperature: float > 0

    Returns:
        New distribution with temperature applied
    """
    if temperature == 1.0:
        return distribution

    # Work in log-space for numerical stability
    log_probs = {t: math.log(p + 1e-10) / temperature
                 for t, p in distribution.items()}

    # Subtract max for numerical stability (log-sum-exp trick)
    max_log = max(log_probs.values())
    exp_probs = {t: math.exp(lp - max_log)
                 for t, lp in log_probs.items()}

    # Normalize
    total = sum(exp_probs.values())
    return {t: p / total for t, p in exp_probs.items()}
```

**The log-sum-exp trick**: We subtract max before exponentiating to prevent overflow. This doesn't change the result because:

$$\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i} e^{-c}}{\sum_j e^{z_j} e^{-c}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$


## Other Sampling Strategies

Temperature isn't the only way to control generation:

### Top-k Sampling
Only sample from the k highest-probability tokens.

```python
def top_k(distribution, k):
    sorted_tokens = sorted(distribution.items(),
                          key=lambda x: -x[1])[:k]
    total = sum(p for _, p in sorted_tokens)
    return {t: p/total for t, p in sorted_tokens}
```

### Nucleus (Top-p) Sampling
Sample from the smallest set of tokens whose cumulative probability exceeds p.

```python
def top_p(distribution, p):
    sorted_tokens = sorted(distribution.items(),
                          key=lambda x: -x[1])
    cumulative = 0.0
    result = {}
    for token, prob in sorted_tokens:
        result[token] = prob
        cumulative += prob
        if cumulative >= p:
            break
    total = sum(result.values())
    return {t: prob/total for t, prob in result.items()}
```

### Combining Strategies
Modern LLMs often use combinations: apply temperature, then top-p, then sample.

## Temperature in Practice

| Use case | Recommended T |
|----------|---------------|
| Code generation | 0.0 - 0.3 (deterministic) |
| Factual Q&A | 0.3 - 0.7 (focused) |
| Creative writing | 0.7 - 1.0 (diverse) |
| Brainstorming | 1.0 - 1.5 (exploratory) |

**ChatGPT/Claude defaults**: Usually around T=0.7 to 1.0 for general use.

## Summary

| Concept | What it does | When to use |
|---------|--------------|-------------|
| Greedy (T=0) | Always pick max | Deterministic output needed |
| Low T (0.3-0.7) | Mostly high-prob tokens | Focused, coherent text |
| T=1.0 | Original distribution | Match training distribution |
| High T (>1.0) | Flatter distribution | Creative, diverse output |
| Top-k | Only top k tokens | Prevent rare token disasters |
| Top-p | Cumulative probability threshold | Adaptive vocabulary size |

**Key takeaways**:
1. Temperature controls the exploration-exploitation tradeoff
2. T=1 samples from the learned distribution
3. Lower T = more deterministic, higher T = more random
4. The formula comes from statistical mechanics / softmax

Next: Let's implement all of this from scratch.

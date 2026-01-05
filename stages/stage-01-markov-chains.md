# Stage 1: The Simplest Language Model (Markov Chains)

## DRAFT STATUS: Outline Complete, Content In Progress

---

## Overview

This stage introduces language modeling through the simplest possible approach: counting. We build Markov chain models that capture local patterns in text, establishing core concepts that will echo through every subsequent stage: probability distributions, training as optimization, autoregressive generation, and the fundamental trade-offs of modeling sequences.

**Estimated length:** 8,000-10,000 words
**Estimated read time:** 45-60 minutes
**Code:** ~400 lines

---

## Learning Objectives

By the end of this stage, you will be able to:

1. Define language modeling as a probability distribution over sequences
2. Derive the autoregressive factorization from the chain rule of probability
3. Prove that counting-based training is maximum likelihood estimation
4. Implement a complete Markov chain language model from scratch
5. Generate text using ancestral sampling
6. Evaluate models using perplexity and interpret its meaning
7. Analyze the context-length vs. sparsity trade-off

---

## Prerequisites

- Basic Python programming
- High school probability (what "probability" means)
- Familiarity with dictionaries and basic data structures

**No calculus or linear algebra required for this stage.**

---

## Chapter Outline

### 1. The Problem (Pólya Step 1)

#### 1.1 What is Language Modeling?

*Opening hook:* Start with a simple question — given "The cat sat on the", what word comes next?

**Content:**
- Language modeling = predicting what comes next
- Formally: learning P(x₁, x₂, ..., xₙ) over all possible sequences
- This is impossibly large: |V|ⁿ possibilities

**Key insight:** We don't need to store probabilities for every sequence. We can factorize.

#### 1.2 The Chain Rule of Probability

**Content:**
- State and derive the chain rule
- P(x₁, x₂, x₃) = P(x₁) · P(x₂|x₁) · P(x₃|x₁, x₂)
- General form: P(x₁:ₙ) = ∏ᵢ P(xᵢ | x₁:ᵢ₋₁)

**Derivation:**
- Full proof from conditional probability definition
- Visual diagram of the factorization

**Implication:**
- We've converted one impossible distribution into n conditional distributions
- But each conditional still depends on all previous tokens!

#### 1.3 The Markov Assumption

**Content:**
- Simplifying assumption: the future depends only on the recent past
- Order-1 (bigram): P(xᵢ | x₁:ᵢ₋₁) ≈ P(xᵢ | xᵢ₋₁)
- Order-k: P(xᵢ | x₁:ᵢ₋₁) ≈ P(xᵢ | xᵢ₋ₖ:ᵢ₋₁)

**Trade-off introduction:**
- Higher order = more context = better predictions
- But higher order = more possible histories = sparser data

**This is wrong, but useful:**
- Language has long-range dependencies
- "The cat that sat on the mat next to the dog ... **was** sleeping"
- But wrong assumptions can still be useful (foreshadow: neural nets)

#### 1.4 Success Criteria

**What would a good language model do?**
- Assign high probability to real text
- Assign low probability to nonsense
- Generate plausible text when sampled

**How will we measure?**
- Perplexity (to be derived)
- Visual inspection of samples

---

### 2. The Approach (Pólya Step 2)

#### 2.1 Our Plan

1. Collect training text
2. Count all transitions (previous → current)
3. Normalize counts to probabilities
4. Use the probabilities to generate or evaluate

**Key insight:** Training = counting. This is maximum likelihood estimation.

#### 2.2 Data Representation

**For bigram (order-1):**
- Store as dictionary: `counts[prev][curr] = number of occurrences`
- Convert to probabilities: `prob[prev][curr] = counts[prev][curr] / sum(counts[prev])`

**Visualization:**
- Transition matrix for small vocabulary
- Show sparsity at higher orders

---

### 3. Implementation (Pólya Step 3)

#### 3.1 Loading and Preprocessing Text

```python
# code/stage-01/data.py

def load_text(filepath: str) -> str:
    """Load text file and return as string."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def preprocess(text: str, lowercase: bool = True) -> str:
    """Basic text preprocessing."""
    if lowercase:
        text = text.lower()
    return text

def tokenize_chars(text: str) -> list[str]:
    """Character-level tokenization."""
    return list(text)

def tokenize_words(text: str) -> list[str]:
    """Word-level tokenization (simple)."""
    return text.split()
```

**Discussion:**
- Character vs word tokenization trade-offs
- Vocabulary size implications

#### 3.2 Building the Markov Chain

```python
# code/stage-01/markov.py

from collections import defaultdict, Counter
from typing import Dict, List

class MarkovChain:
    """N-gram language model using Markov assumption."""

    def __init__(self, order: int = 1):
        """
        Initialize Markov chain.

        Args:
            order: Number of previous tokens to condition on (1 = bigram)
        """
        self.order = order
        self.counts: Dict[tuple, Counter] = defaultdict(Counter)
        self.totals: Dict[tuple, int] = defaultdict(int)
        self.vocab: set = set()

    def train(self, tokens: List[str]) -> None:
        """
        Train on a sequence of tokens by counting transitions.

        Args:
            tokens: List of tokens (characters or words)
        """
        # Add special start/end tokens
        padded = ['<START>'] * self.order + tokens + ['<END>']

        # Count all n-grams
        for i in range(len(padded) - self.order):
            history = tuple(padded[i:i + self.order])
            next_token = padded[i + self.order]

            self.counts[history][next_token] += 1
            self.totals[history] += 1
            self.vocab.add(next_token)

    def probability(self, history: tuple, next_token: str) -> float:
        """
        Get probability P(next_token | history).

        Args:
            history: Tuple of previous tokens
            next_token: Token to get probability for

        Returns:
            Probability (0 if never seen)
        """
        if history not in self.counts:
            return 0.0
        return self.counts[history][next_token] / self.totals[history]

    def get_distribution(self, history: tuple) -> Dict[str, float]:
        """Get full probability distribution given history."""
        if history not in self.counts:
            return {}
        total = self.totals[history]
        return {
            token: count / total
            for token, count in self.counts[history].items()
        }
```

**Verification:**
- Check that probabilities sum to 1
- Examine sample distributions

#### 3.3 Generating Text

```python
# code/stage-01/generate.py

import random
from typing import List, Optional

def sample_from_distribution(dist: Dict[str, float]) -> str:
    """Sample a token from a probability distribution."""
    tokens = list(dist.keys())
    probs = list(dist.values())
    return random.choices(tokens, weights=probs, k=1)[0]

def generate(
    model: MarkovChain,
    max_length: int = 100,
    temperature: float = 1.0,
    seed: Optional[str] = None
) -> str:
    """
    Generate text from the Markov chain.

    Args:
        model: Trained MarkovChain
        max_length: Maximum tokens to generate
        temperature: Sampling temperature (1.0 = normal, <1 = sharper, >1 = flatter)
        seed: Optional starting text

    Returns:
        Generated text as string
    """
    # Initialize history
    if seed:
        tokens = list(seed)
        history = tuple(tokens[-model.order:])
    else:
        history = tuple(['<START>'] * model.order)

    generated = list(seed) if seed else []

    for _ in range(max_length):
        dist = model.get_distribution(history)

        if not dist:
            break

        # Apply temperature
        if temperature != 1.0:
            dist = apply_temperature(dist, temperature)

        next_token = sample_from_distribution(dist)

        if next_token == '<END>':
            break

        generated.append(next_token)
        history = tuple(list(history)[1:] + [next_token])

    return ''.join(generated)

def apply_temperature(dist: Dict[str, float], temp: float) -> Dict[str, float]:
    """Apply temperature to distribution."""
    import math
    # Convert to log space, divide by temp, convert back
    log_probs = {k: math.log(v + 1e-10) / temp for k, v in dist.items()}
    max_log = max(log_probs.values())
    exp_probs = {k: math.exp(v - max_log) for k, v in log_probs.items()}
    total = sum(exp_probs.values())
    return {k: v / total for k, v in exp_probs.items()}
```

**Discussion:**
- Ancestral sampling explained
- Temperature parameter intuition
- Greedy vs stochastic decoding

#### 3.4 Evaluating with Perplexity

```python
# code/stage-01/evaluate.py

import math
from typing import List

def compute_perplexity(model: MarkovChain, tokens: List[str]) -> float:
    """
    Compute perplexity of the model on a sequence.

    Perplexity = exp(-1/N * sum(log P(x_i | history)))

    Args:
        model: Trained MarkovChain
        tokens: Sequence to evaluate

    Returns:
        Perplexity (lower is better)
    """
    padded = ['<START>'] * model.order + tokens + ['<END>']

    log_prob_sum = 0.0
    n_tokens = 0

    for i in range(len(padded) - model.order):
        history = tuple(padded[i:i + model.order])
        next_token = padded[i + model.order]

        prob = model.probability(history, next_token)

        if prob == 0:
            # Handle unseen n-grams
            return float('inf')

        log_prob_sum += math.log(prob)
        n_tokens += 1

    avg_log_prob = log_prob_sum / n_tokens
    perplexity = math.exp(-avg_log_prob)

    return perplexity
```

**Derivation of perplexity:**
- Start from cross-entropy: H(p,q) = -Σ p(x) log q(x)
- For empirical distribution: H = -1/N Σ log q(xᵢ)
- Perplexity = exp(H) = exp(-1/N Σ log q(xᵢ))

**Interpretation:**
- If perplexity = 50, model is as uncertain as choosing among 50 equally likely tokens
- Uniform random over vocabulary k gives perplexity k
- Lower is better

---

### 4. Experiments and Analysis

#### 4.1 Experimental Setup

**Dataset:** Shakespeare (or similar literary text)
**Tokenization:** Character-level
**Orders to test:** 1, 2, 3, 5, 10

#### 4.2 Results

**Table: Perplexity vs Order**
| Order | Train Perplexity | Test Perplexity | Unique States |
|-------|------------------|-----------------|---------------|
| 1     | ~15              | ~15             | 27            |
| 2     | ~8               | ~9              | ~700          |
| 3     | ~5               | ~7              | ~10,000       |
| 5     | ~2               | ~15             | ~100,000      |
| 10    | ~1               | ∞               | ~500,000      |

**Observation:** Overfitting! Training perplexity drops, test increases.

#### 4.3 Sample Quality

**Order-1 samples:**
> "the and to of a in that is was he for it with as his they be at one have this from or had by not but what all were we when your can said there use an each which she do how their if will up other about out many then them these so some her would make like has look two more write go see number no way could people my than first water been call who oil its now find long down day did get come made may part"

**Order-3 samples:**
> "the king's son, and then to be a man of the world. 'tis true, i am a man of great..."

**Order-5 samples:**
> "the duke of york, and see the coronation of king richard iii..."

**Observation:** Quality improves with order, but so does memorization.

#### 4.4 Visualizations

1. **Transition matrix heatmap** (order-1)
2. **Perplexity vs order** (train and test)
3. **State space size** (exponential growth)

---

### 5. Reflection (Pólya Step 4)

#### 5.1 What We Learned

**Key insight 1:** Training is counting
- Maximum likelihood estimation for categorical distributions
- Elegant equivalence: counting = optimization

**Key insight 2:** Perplexity as branching factor
- Intuitive interpretation of model quality
- Connection to information theory (will explore in later stages)

**Key insight 3:** The fundamental trade-off
- More context → better predictions
- More context → sparser observations
- No free lunch

#### 5.2 The MLE Equivalence (Proof)

**Claim:** Counting transitions and normalizing is maximum likelihood estimation.

**Proof:**
Given observations x₁, ..., xₙ, the likelihood of a bigram model is:

L(θ) = ∏ᵢ P(xᵢ | xᵢ₋₁; θ)

where θ contains all transition probabilities θ_{a→b} = P(b|a).

The log-likelihood is:

log L(θ) = Σᵢ log P(xᵢ | xᵢ₋₁)
         = Σₐ Σᵦ count(a,b) · log θ_{a→b}

To maximize subject to Σᵦ θ_{a→b} = 1 for each a, use Lagrange multipliers:

∂/∂θ_{a→b} [log L - λₐ(Σᵦ θ_{a→b} - 1)] = count(a,b)/θ_{a→b} - λₐ = 0

Therefore: θ_{a→b} = count(a,b) / λₐ

From constraint: Σᵦ θ_{a→b} = 1 implies λₐ = count(a,·)

Thus: θ*_{a→b} = count(a,b) / count(a,·) ∎

**Implication:** Our counting procedure is optimal under the modeling assumption.

#### 5.3 Limitations

1. **Fixed context:** Can't look back arbitrarily far
2. **No generalization:** Exact matches only, no "similar" patterns
3. **Exponential growth:** Higher orders quickly become impractical
4. **Real language:** Has long-range dependencies we can't capture

**What we need:** A way to represent patterns that generalizes, captures long-range dependencies, and scales.

*This is what neural networks provide.*

#### 5.4 Looking Ahead

- **Stage 2:** How do we learn anything more complex? (autodiff)
- **Stage 4:** Neural language models (same task, learned patterns)
- **Stage 7:** Attention (arbitrary-range dependencies)

**The concepts that carry forward:**
- Autoregressive factorization (used in all language models)
- Next-token prediction as objective (GPT's entire training objective)
- Perplexity as evaluation metric (standard for LLMs)
- Temperature sampling (used in ChatGPT, Claude, etc.)

---

## Exercises

### Exercise 1.1: Smoothing (Conceptual)
What happens when we encounter an n-gram we never saw during training? Our model assigns probability 0, leading to infinite perplexity. Implement Laplace (add-one) smoothing:

P_smooth(b|a) = (count(a,b) + 1) / (count(a,·) + |V|)

Show that this is equivalent to placing a uniform prior over transitions.

### Exercise 1.2: Order Sweep (Implementation)
Train character-level Markov chains with orders 1, 2, 3, 4, 5 on a text of your choice. Plot:
- Training perplexity vs order
- Test perplexity vs order
- Number of unique histories vs order

Explain the pattern you observe.

### Exercise 1.3: Word-Level Model (Extension)
Modify the code to work with word-level tokenization instead of character-level. Compare:
- What order is now reasonable?
- How does vocabulary size affect things?
- What happens with rare words?

### Exercise 1.4: Temperature Analysis (Analysis)
Generate 10 samples each at temperatures 0.5, 1.0, and 2.0. Characterize the difference. What is the mathematical relationship between temperature and entropy of the output distribution?

---

## Summary

| Concept | Formal Definition | Intuition |
|---------|-------------------|-----------|
| Language Model | P(x₁, ..., xₙ) | Probability of text |
| Chain Rule | ∏ᵢ P(xᵢ\|x<ᵢ) | Factor into conditionals |
| Markov Assumption | P(xᵢ\|x<ᵢ) ≈ P(xᵢ\|xᵢ₋ₖ:ᵢ₋₁) | Only recent past matters |
| Training | Count and normalize | MLE = counting |
| Perplexity | exp(-1/N Σ log P) | Effective vocabulary size |
| Temperature | Sharpen/flatten distribution | Control randomness |

---

## Code Files

```
code/stage-01/
├── data.py          # Text loading and preprocessing
├── markov.py        # MarkovChain class
├── generate.py      # Text generation
├── evaluate.py      # Perplexity computation
├── experiments.py   # Run experiments
├── visualize.py     # Create plots
└── main.py          # Complete example
```

---

## Further Reading

- **Jurafsky & Martin, Chapter 3:** N-gram Language Models (comprehensive textbook treatment)
- **Shannon (1948):** "A Mathematical Theory of Communication" (information theory foundations)
- **Chen & Goodman (1996):** "An Empirical Study of Smoothing Techniques" (practical smoothing)

---

## Next Stage Preview

We've built a working language model with pure counting. But it can't generalize — it only works with exact pattern matches. What if we could learn a function that maps patterns to probabilities, enabling generalization?

That requires optimization. And optimization requires gradients. And computing gradients efficiently requires **automatic differentiation**.

**Stage 2: Automatic Differentiation** will give us the tools to learn from data.

---

## Writing Notes (for author)

### Tone
- Enthusiastic discovery ("This is elegant!")
- Honest about limitations ("This is wrong, but usefully wrong")
- Building anticipation for what's next

### Key moments
1. The "counting = MLE" reveal (should feel surprising then obvious)
2. The overfitting pattern (should be visceral from experiments)
3. The long-range dependency example (should motivate later work)

### Visualizations to create
- [ ] Transition matrix heatmap
- [ ] Perplexity curves (train vs test)
- [ ] Sample quality comparison
- [ ] State space growth

### Code to verify
- [ ] All code runs on clean Python 3.10+
- [ ] Perplexity values are reasonable
- [ ] Generation quality matches expectations

---

*Draft version 0.1 — 2026-01-03*

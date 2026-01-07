# Stage 1 Exercises

## Conceptual Questions

### Exercise 1.1: Order Trade-offs
Consider a language model for autocomplete suggestions.

**Question**: You're building autocomplete for a search engine. Users type 2-5 characters and expect instant suggestions. Would you use a unigram, bigram, or trigram model? Justify your choice considering:

- Response time requirements
- Quality of suggestions
- Memory constraints on mobile devices

### Exercise 1.2: Probability Chains
Given this tiny corpus: "the cat sat on the mat"

**a)** Calculate P("the" | "on") for a bigram model
**b)** What is P("cat" | "the", bigram)?
**c)** Why is P("dog" | "the") = 0 a problem?

### Exercise 1.3: Information Content
A language model assigns these probabilities:

- P("Paris" | "The capital of France is") = 0.85
- P("London" | "The capital of France is") = 0.05
- P("Berlin" | "The capital of France is") = 0.03

**a)** Which word carries the most "surprise" (information)?
**b)** Calculate the surprisal (in bits) for each word
**c)** If the model assigns P("Paris") = 0.001 without context, how much does context help?

### Exercise 1.4: Temperature Intuition
Without running code, predict what happens:

**a)** T = 0.1 with uniform distribution [0.25, 0.25, 0.25, 0.25]
**b)** T = 10.0 with peaked distribution [0.9, 0.05, 0.03, 0.02]
**c)** At what temperature does any distribution become uniform?

---

## Implementation Exercises

### Exercise 1.5: Trigram Implementation
Extend the bigram model to a trigram model.

```python
class TrigramModel:
    def __init__(self):
        self.counts = {}  # (w1, w2) -> {w3: count}
        self.context_totals = {}

    def train(self, text: str) -> None:
        # TODO: Implement trigram counting
        pass

    def probability(self, word: str, context: Tuple[str, str]) -> float:
        # TODO: Return P(word | context)
        pass
```

**Test**: Train on "the cat sat on the mat the cat ran" and verify:
- P("cat" | "the", "the") should be higher than P("sat" | "the", "the")

### Exercise 1.6: Perplexity Calculator
Implement a function to compute perplexity on held-out data:

```python
def compute_perplexity(model, test_text: str) -> float:
    """
    Compute perplexity of model on test_text.

    Perplexity = exp(-1/N * sum(log P(w_i | context)))
    """
    # TODO: Implement
    pass
```

**Test**: A model that assigns uniform probability 1/V should have perplexity = V

### Exercise 1.7: Add-k Smoothing
Implement add-k smoothing with tunable k:

```python
def smoothed_probability(
    word: str,
    context: str,
    counts: dict,
    vocab_size: int,
    k: float = 1.0
) -> float:
    """
    P_smooth(word | context) = (count + k) / (total + k * V)
    """
    # TODO: Implement
    pass
```

**Experiment**: Plot perplexity vs k for k in [0.001, 0.01, 0.1, 1.0, 10.0]

### Exercise 1.8: Interpolation
Implement linear interpolation between unigram and bigram:

```python
def interpolated_probability(
    word: str,
    context: str,
    unigram_model,
    bigram_model,
    lambda_1: float = 0.5
) -> float:
    """
    P(w|c) = λ₁ * P_bi(w|c) + (1-λ₁) * P_uni(w)
    """
    # TODO: Implement
    pass
```

---

## Challenge Exercises

### Exercise 1.9: Optimal λ via Cross-Validation
Implement a function that finds the optimal interpolation weight:

```python
def find_optimal_lambda(
    train_text: str,
    val_text: str,
    lambdas: List[float]
) -> float:
    """
    Find λ that minimizes perplexity on validation set.
    """
    # TODO: Train models, evaluate each λ, return best
    pass
```

### Exercise 1.10: Corpus Analysis
Download a real corpus (e.g., first chapter of a book from Project Gutenberg).

**a)** Train unigram, bigram, and trigram models
**b)** Compare perplexity on held-out text
**c)** Generate 100 characters from each model with T=1.0
**d)** Which produces more coherent text? Why?

### Exercise 1.11: Context Length Explosion
For a vocabulary of size V=10,000:

**a)** How many possible bigram contexts are there?
**b)** How many possible 5-gram contexts?
**c)** If you have 1 million tokens of training data, what fraction of 5-gram contexts will you observe?
**d)** This is called the "curse of dimensionality." How does it motivate neural approaches?

---

## Checking Your Work

- **Test suite**: See `code/stage-01/tests/test_markov.py` for expected behavior
- **Reference implementation**: Compare with `code/stage-01/markov.py`
- **Self-check**: Verify perplexity calculation matches examples in documentation

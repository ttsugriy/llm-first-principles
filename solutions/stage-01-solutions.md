# Exercise Solutions

This directory contains solutions to exercises from each stage. Solutions are hidden by default to encourage attempting problems first.

## How to Use

1. **Try the exercise first** — Struggle is part of learning
2. **Check hints if stuck** — Each solution starts with hints
3. **Study the full solution** — Understand *why* it works
4. **Learn from common mistakes** — Each solution lists pitfalls

---

## Stage 1: Markov Chains

### Exercise 1.3.1: Implement Laplace Smoothing

**Problem**: Modify the `MarkovChain` class to support Laplace smoothing.

**Hints**:
1. The formula is: P(b|a) = (count(a,b) + α) / (count(a,·) + α|V|)
2. You need to know the vocabulary size |V|
3. α = 1 is standard Laplace smoothing

**Common Mistakes**:
- Forgetting to add α|V| to the denominator (probabilities won't sum to 1!)
- Using wrong vocabulary size (should include `<END>` token)
- Not handling the case where history was never seen

**Solution**:

```python
class SmoothedMarkovChain(MarkovChain):
    """Markov chain with Laplace (add-α) smoothing."""

    def __init__(self, order: int = 1, alpha: float = 1.0):
        super().__init__(order)
        self.alpha = alpha

    def probability(self, history: Tuple[str, ...], next_token: str) -> float:
        """
        Get smoothed probability P(next_token | history).

        Uses the Bayesian formula:
        P(b|a) = (count(a,b) + α) / (count(a,·) + α|V|)
        """
        count = self.counts[history][next_token]  # 0 if unseen
        total = self.totals[history]  # 0 if history unseen
        vocab_size = len(self.vocab)

        # Handle edge case: if vocab is empty, return uniform
        if vocab_size == 0:
            return 0.0

        return (count + self.alpha) / (total + self.alpha * vocab_size)

    def get_distribution(self, history: Tuple[str, ...]) -> Dict[str, float]:
        """Get smoothed distribution over all vocabulary."""
        result = {}
        for token in self.vocab:
            result[token] = self.probability(history, token)
        return result
```

**Verification**:
```python
model = SmoothedMarkovChain(order=1, alpha=1.0)
model.train(list("ab"))  # vocab = {a, b, <END>}, |V| = 3

# P(b|a) = (1 + 1) / (1 + 1*3) = 2/4 = 0.5
# P(<END>|a) = (0 + 1) / (1 + 1*3) = 1/4 = 0.25
# P(a|a) = (0 + 1) / (1 + 1*3) = 1/4 = 0.25
# Sum = 0.5 + 0.25 + 0.25 = 1.0 ✓

assert abs(model.probability(("a",), "b") - 0.5) < 1e-10
```

---

### Exercise 1.5.1: Derive Perplexity = Effective Vocabulary Size

**Problem**: Prove that for a uniform distribution over V items, perplexity = |V|.

**Hints**:
1. Start with the definition: PPL = exp(H(p, q))
2. For a uniform distribution, P(x) = 1/|V| for all x
3. Cross-entropy and entropy are equal when p = q

**Solution**:

For a uniform distribution p over |V| items:

$$p(x) = \frac{1}{|V|} \text{ for all } x$$

The entropy is:
$$H(p) = -\sum_{x} p(x) \log p(x) = -\sum_{x} \frac{1}{|V|} \log \frac{1}{|V|}$$

$$= -|V| \cdot \frac{1}{|V|} \cdot \log \frac{1}{|V|} = -\log \frac{1}{|V|} = \log |V|$$

Therefore, perplexity is:
$$\text{PPL} = \exp(H(p)) = \exp(\log |V|) = |V| \quad \blacksquare$$

**Intuition**: A model with perplexity K is as uncertain as if it were choosing uniformly among K equally likely options at each step.

---

### Exercise 1.6.1: Temperature Limits

**Problem**: Prove that as T → 0, temperature sampling approaches argmax, and as T → ∞, it approaches uniform.

**Hints**:
1. Temperature scaling: P_T(x) ∝ P(x)^(1/T)
2. For T → 0, consider what happens to P^(1/T) when P < 1
3. For T → ∞, consider what (1/T) approaches

**Solution**:

Let P = (p₁, p₂, ..., pₙ) be the original distribution with p₁ ≥ p₂ ≥ ... ≥ pₙ.

After temperature scaling:
$$P_T(x_i) = \frac{p_i^{1/T}}{\sum_j p_j^{1/T}}$$

**Case T → 0**:

If p₁ > p₂ (unique maximum), as T → 0:
- For the maximum: p₁^(1/T) → ∞ at the fastest rate
- For others: p_i^(1/T) → 0 (since p_i < 1 and 1/T → ∞)

Therefore:
$$P_T(x_1) = \frac{p_1^{1/T}}{p_1^{1/T} + \sum_{j>1} p_j^{1/T}} \to \frac{\infty}{\infty + 0} = 1$$

This is argmax (deterministic selection of the highest probability token).

**Case T → ∞**:

As T → ∞, 1/T → 0, so:
$$p_i^{1/T} = p_i^0 = 1 \text{ for all } i$$

Therefore:
$$P_T(x_i) = \frac{1}{\sum_j 1} = \frac{1}{n}$$

This is the uniform distribution. ∎

---

### Exercise 1.8.1: State Space Analysis

**Problem**: For a character-level bigram model with vocabulary size |V| = 27, how many possible states are there? How many parameters?

**Solution**:

**States**: Each state is a unique history (single character for bigram).
- States = |V| = 27

**Parameters**: Each state can transition to any character or `<END>`.
- Parameters = |V| × (|V| + 1) = 27 × 28 = 756

But wait—we also need to account for the `<START>` state:
- Total states = |V| + 1 = 28 (including START)
- Transitions from START: |V| = 27 (can't go to END immediately)
- Transitions from each character: |V| + 1 = 28 (including END)

**Total parameters**: 27 + 27 × 28 = 27 + 756 = 783

However, this counts many impossible transitions. In practice, only observed transitions are stored (sparse representation), so the actual number of non-zero parameters is O(number of unique bigrams in training data).

---

### Exercise 1.8.2: Optimal Order

**Problem**: Plot train and test perplexity vs. order for the Shakespeare sample. Find the optimal order.

**Solution**:

```python
import matplotlib.pyplot as plt
from markov import MarkovChain, SmoothedMarkovChain
from evaluate import compute_perplexity
from data import get_sample_data, tokenize_chars, train_test_split

# Prepare data
text = get_sample_data('shakespeare').lower()
tokens = tokenize_chars(text)
train, test = train_test_split(tokens, test_ratio=0.2)

# Sweep orders
orders = range(1, 8)
train_ppls = []
test_ppls = []

for order in orders:
    model = SmoothedMarkovChain(order=order, alpha=0.1)
    model.train(train)

    train_ppls.append(compute_perplexity(model, train))
    test_ppls.append(compute_perplexity(model, test))

# Find optimal order (minimum test perplexity)
optimal_order = orders[test_ppls.index(min(test_ppls))]
print(f"Optimal order: {optimal_order}")
print(f"Test perplexity at optimal: {min(test_ppls):.2f}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(orders, train_ppls, 'b-o', label='Train')
plt.plot(orders, test_ppls, 'r-o', label='Test')
plt.axvline(x=optimal_order, color='g', linestyle='--',
            label=f'Optimal ({optimal_order})')
plt.xlabel('Order')
plt.ylabel('Perplexity')
plt.title('Perplexity vs. Model Order')
plt.legend()
plt.grid(True)
plt.savefig('order_sweep.png')
```

**Expected results**:
- Train perplexity decreases monotonically (higher order = more memorization)
- Test perplexity has a U-shape: decreases until order ~2-3, then increases
- Optimal order depends on dataset size; larger datasets support higher orders

---

## Stage 2: Automatic Differentiation

*(Solutions for Stage 2 exercises will be added when that stage is complete)*

---

## General Tips

1. **Test incrementally**: Don't write 50 lines then test. Test each function as you write it.

2. **Use small examples**: Before testing on real data, verify correctness on tiny examples where you can compute the answer by hand.

3. **Check edge cases**: Empty input, single token, unseen contexts.

4. **Verify mathematical properties**:
   - Probabilities should sum to 1
   - Perplexity should be ≥ 1
   - Gradients should match finite differences

5. **Read the theory section if stuck**: The exercises are designed to reinforce the theory. Re-reading often provides the insight needed.

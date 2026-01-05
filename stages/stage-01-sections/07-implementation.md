# Section 1.7: Implementation — Building It From Scratch

Now we implement everything we've learned. Every line of code will be explained.

## Design Decisions

Before coding, let's make explicit choices:

**1. Vocabulary level**: Character or word?
- Character: |V| ≈ 100, can use higher orders, no unknown tokens
- Word: |V| ≈ 50,000, richer semantics, but sparse

We'll use **character-level** for this stage—it lets us explore higher-order models without running out of data.

**2. Data structure for counts**:
- Dense matrix: |V|^k × |V| entries, mostly zeros
- Sparse dictionary: Only store observed transitions

We'll use **nested dictionaries**—efficient for sparse data.

**3. Special tokens**:
- ⟨START⟩: Marks beginning of sequence (so we can predict first real token)
- ⟨END⟩: Marks end of sequence (so model learns when to stop)

## The Complete Implementation

### Part 1: Data Structures

```python
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import math
import random
```

**Why these imports**:
- `defaultdict`: Creates missing keys automatically (avoids KeyError)
- `Counter`: Efficiently counts occurrences
- `Dict, List, Tuple`: Type hints for documentation
- `math`: For log, exp
- `random`: For sampling

### Part 2: The MarkovChain Class

```python
class MarkovChain:
    """
    N-gram language model using the Markov assumption.

    This class implements training (counting), probability queries,
    text generation, and evaluation (perplexity).
    """

    def __init__(self, order: int = 1):
        """
        Initialize an empty Markov chain.

        Args:
            order: Number of previous tokens to condition on.
                   order=1 is bigram, order=2 is trigram, etc.

        Data structures:
            counts[context][token] = how many times token follows context
            totals[context] = total transitions from context
            vocab = set of all tokens seen
        """
        if order < 1:
            raise ValueError(f"Order must be ≥ 1, got {order}")

        self.order = order

        # counts[context_tuple][next_token] = count
        # Using defaultdict so we can write counts[c][t] += 1
        # without checking if c or t exist
        self.counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)

        # totals[context_tuple] = sum of all counts from this context
        # Cached for efficiency (avoid recomputing sum each time)
        self.totals: Dict[Tuple[str, ...], int] = defaultdict(int)

        # All tokens seen during training (including END, excluding START)
        self.vocab: set = set()
```

**Why `defaultdict(Counter)`**:
- `defaultdict` with `Counter`: If we access `counts[new_context]`, it automatically creates an empty `Counter` for that context.
- `Counter` is a dict subclass that defaults missing keys to 0.
- Result: We can write `self.counts[context][token] += 1` without any existence checks.

**Why cache totals separately**:
- We'll query P(token | context) frequently
- P requires dividing by the sum of all counts for that context
- Rather than recompute `sum(self.counts[context].values())` each time, we maintain `totals` incrementally

### Part 3: Training

```python
    def train(self, tokens: List[str]) -> None:
        """
        Train the model by counting n-gram transitions.

        This implements Maximum Likelihood Estimation:
        P(token | context) = count(context, token) / count(context, *)

        Args:
            tokens: List of tokens (e.g., list of characters)
        """
        # Pad the sequence with START and END tokens
        # START tokens let us predict the first real tokens
        # END token lets the model learn when to stop
        padded = ['<START>'] * self.order + tokens + ['<END>']

        # Slide a window of size (order + 1) across the sequence
        # Each window gives us (context, next_token)
        for i in range(len(padded) - self.order):
            # Context: order tokens ending at position i+order-1
            context = tuple(padded[i : i + self.order])

            # Next token: the one right after the context
            next_token = padded[i + self.order]

            # Update counts
            self.counts[context][next_token] += 1
            self.totals[context] += 1

            # Track vocabulary (we'll need this for smoothing, vocab size, etc.)
            self.vocab.add(next_token)
```

**Why `tuple` for context**:
- Lists are mutable and can't be dictionary keys
- Tuples are immutable and hashable → can be dict keys

**The sliding window**:
For "hello" with order=2:
```
Padded: [<START>, <START>, h, e, l, l, o, <END>]
Index:     0        1      2  3  4  5  6    7

i=0: context=(<START>,<START>), next=h
i=1: context=(<START>,h),       next=e
i=2: context=(h,e),             next=l
i=3: context=(e,l),             next=l
i=4: context=(l,l),             next=o
i=5: context=(l,o),             next=<END>
```

### Part 4: Probability Queries

```python
    def probability(self, context: Tuple[str, ...], token: str) -> float:
        """
        Get P(token | context) from the model.

        Args:
            context: Tuple of previous tokens (must have length = self.order)
            token: The token to get probability for

        Returns:
            Probability in [0, 1]. Returns 0 if context never seen.
        """
        if context not in self.counts:
            # Context never observed → we have no information
            # Could use backoff or smoothing here (see exercises)
            return 0.0

        # MLE: P(token | context) = count(context, token) / count(context, *)
        return self.counts[context][token] / self.totals[context]

    def get_distribution(self, context: Tuple[str, ...]) -> Dict[str, float]:
        """
        Get the full probability distribution P(* | context).

        Args:
            context: Tuple of previous tokens

        Returns:
            Dictionary mapping each possible next token to its probability.
            Empty dict if context never seen.
        """
        if context not in self.counts:
            return {}

        total = self.totals[context]
        return {
            token: count / total
            for token, count in self.counts[context].items()
        }
```

### Part 5: Text Generation

```python
    def generate(
        self,
        max_length: int = 100,
        temperature: float = 1.0,
        seed: str = ""
    ) -> str:
        """
        Generate text using ancestral sampling.

        Args:
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = unmodified)
            seed: Optional starting text

        Returns:
            Generated text as string
        """
        # Initialize context from seed or START tokens
        if seed:
            tokens = list(seed)
            # Use last 'order' tokens as context
            if len(tokens) >= self.order:
                context = tuple(tokens[-self.order:])
            else:
                # Pad with START if seed is too short
                padding = ['<START>'] * (self.order - len(tokens))
                context = tuple(padding + tokens)
            generated = list(seed)
        else:
            context = tuple(['<START>'] * self.order)
            generated = []

        # Generate tokens one at a time
        for _ in range(max_length):
            # Get probability distribution for next token
            dist = self.get_distribution(context)

            if not dist:
                # No transitions from this context (never seen in training)
                break

            # Apply temperature
            if temperature != 1.0:
                dist = self._apply_temperature(dist, temperature)

            # Sample from distribution
            next_token = self._sample(dist)

            # Stop if we hit END
            if next_token == '<END>':
                break

            # Append to output
            generated.append(next_token)

            # Update context: slide window right by 1
            context = tuple(list(context)[1:] + [next_token])

        return ''.join(generated)

    def _apply_temperature(
        self,
        dist: Dict[str, float],
        temperature: float
    ) -> Dict[str, float]:
        """Apply temperature scaling to distribution."""
        # Convert to log-space, scale, convert back
        log_probs = {
            token: math.log(prob + 1e-10) / temperature
            for token, prob in dist.items()
        }

        # Subtract max for numerical stability
        max_log = max(log_probs.values())
        exp_probs = {
            token: math.exp(lp - max_log)
            for token, lp in log_probs.items()
        }

        # Normalize to sum to 1
        total = sum(exp_probs.values())
        return {token: prob / total for token, prob in exp_probs.items()}

    def _sample(self, dist: Dict[str, float]) -> str:
        """Sample a token from a probability distribution."""
        tokens = list(dist.keys())
        probs = list(dist.values())
        return random.choices(tokens, weights=probs, k=1)[0]
```

### Part 6: Evaluation

```python
    def perplexity(self, tokens: List[str]) -> float:
        """
        Compute perplexity on a sequence.

        Perplexity = exp(-1/N * sum(log P(token | context)))

        Lower is better. Returns infinity if any token has probability 0.

        Args:
            tokens: List of tokens to evaluate

        Returns:
            Perplexity (float, >= 1, possibly inf)
        """
        padded = ['<START>'] * self.order + tokens + ['<END>']

        log_prob_sum = 0.0
        n_tokens = 0

        for i in range(len(padded) - self.order):
            context = tuple(padded[i : i + self.order])
            next_token = padded[i + self.order]

            prob = self.probability(context, next_token)

            if prob == 0:
                # Model assigns 0 probability → perplexity is infinite
                return float('inf')

            log_prob_sum += math.log(prob)
            n_tokens += 1

        # Average negative log-likelihood
        avg_neg_log_prob = -log_prob_sum / n_tokens

        # Exponentiate to get perplexity
        return math.exp(avg_neg_log_prob)
```

### Part 7: Utility Methods

```python
    def num_states(self) -> int:
        """Return number of unique contexts seen during training."""
        return len(self.counts)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"MarkovChain(order={self.order}, states={self.num_states()}, vocab={len(self.vocab)})"
```

## Usage Example

```python
# Training data
text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
"""

# Tokenize to characters
tokens = list(text.lower())

# Create and train model
model = MarkovChain(order=3)
model.train(tokens)

print(f"Model: {model}")
# Model: MarkovChain(order=3, states=245, vocab=31)

# Generate text
sample = model.generate(max_length=100, temperature=0.8)
print(f"Sample: {sample}")
# Sample: to be, or not to ber the slings and ar...

# Evaluate
train_ppl = model.perplexity(tokens)
print(f"Train perplexity: {train_ppl:.2f}")
# Train perplexity: 3.42
```

## Time and Space Complexity

**Training**:
- Time: O(n) where n = length of training data
- Space: O(|V|^k) worst case, but typically O(n) in practice (sparse)

**Probability query**:
- Time: O(1) average (hash table lookup)

**Generation**:
- Time: O(L × |V|) where L = output length, |V| = vocabulary size
- The |V| factor is for sampling (iterating over distribution)

**Perplexity**:
- Time: O(n) where n = evaluation sequence length

## Summary

We've implemented a complete Markov chain language model with:
- Training via counting (MLE)
- Probability queries
- Temperature-controlled sampling
- Perplexity evaluation

The entire implementation is ~150 lines of well-documented Python with no external dependencies beyond the standard library.

**Key implementation insights**:
1. Use `defaultdict(Counter)` for sparse count storage
2. Cache totals for O(1) probability queries
3. Work in log-space for numerical stability
4. Use tuples for contexts (hashable keys)

Next: Let's analyze the fundamental trade-offs of this approach.

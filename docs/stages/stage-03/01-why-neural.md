# Section 3.1: Why Neural? The Limits of Counting

In Stage 1, we built language models by counting. Given enough data, counting works perfectly—maximum likelihood estimation gives us optimal probability estimates.

So why do we need neural networks?

**The answer is simple: we never have enough data.**

This section explores the fundamental limitations of count-based models and why continuous representations offer a solution.

## The Curse of Dimensionality

Consider modeling the probability of the next character given the previous 10 characters.

### How Many Contexts Are There?

With a vocabulary of |V| = 100 characters (letters, digits, punctuation, space):

$$\text{Number of possible 10-character contexts} = 100^{10} = 10^{20}$$


That's **100 quintillion** possible contexts.

### How Much Data Do We Have?

Wikipedia contains roughly 4 billion characters. Even if we used all of Wikipedia:

- We'd see each specific 10-character context at most a few times
- Most contexts would never appear
- For unseen contexts, our count-based model gives probability 0 or falls back to shorter contexts

### The Exponential Gap

| Context Length | Possible Contexts | Data Needed for Coverage |
|----------------|-------------------|--------------------------|
| 2 | 10,000 | ~10,000 |
| 5 | $10^10$ | ~$10^10$ |
| 10 | $10^20$ | ~$10^20$ |
| 20 | $10^40$ | Impossible |

This is the **curse of dimensionality**: the number of possible configurations grows exponentially with context length, while our data grows at best linearly.

## The Problem of Sparsity

### What Happens in Practice

Let's revisit our Stage 1 experience:

```
Order 1 model: Seen most bigrams, works okay
Order 2 model: Many trigrams unseen
Order 5 model: Most 6-grams never appear in training
```

For a 5-gram model on typical training data:

- ~90% of test 5-grams are unseen in training
- Model must constantly back off to shorter contexts
- The "5-gram" model effectively becomes a mixture of lower-order models

### Smoothing Helps, But Not Enough

In Stage 1, we applied Laplace smoothing:

$$P(w | c) = \frac{\text{count}(c, w) + 1}{\text{count}(c) + |V|}$$


This prevents zero probabilities, but:

- Assigns equal probability to all unseen continuations
- "the cat sat" and "the xyz sat" get the same smoothed probability
- No notion of similarity between contexts

## The Key Insight: Contexts Are Not Independent

Here's what count-based models miss: **similar contexts should give similar predictions**.

### An Example

Consider these contexts:

1. "the cat sat on the"
2. "the dog sat on the"
3. "a cat sat upon the"

A human knows these should give similar next-character predictions. But to a count-based model:

- These are three completely independent entries in a lookup table
- Learning from (1) tells us nothing about (2) or (3)
- Each must be learned separately

### Why This Matters

Real language has structure:

- "cat" and "dog" are both animals
- "sat" and "slept" are both past tense verbs
- "on" and "upon" serve similar grammatical roles

This structure means we don't need to see every possible combination—we need to learn the underlying patterns.

## What We Need: Generalization

The core problem is **generalization**: using what we've learned to handle situations we haven't seen.

### Count-Based Generalization

N-gram models generalize via:

1. **Backoff**: If we haven't seen the long context, use a shorter one
2. **Interpolation**: Mix predictions from different context lengths

These methods share statistical strength across:

- "the cat sat" → "the cat" → "cat"

But NOT across:

- "the cat" and "the dog" (completely independent)

### What We Want

We want a model where:

$$\text{similarity}(\text{"the cat"}, \text{"the dog"}) > 0$$


And this similarity should affect predictions:

- If we learn that "sat" often follows "the cat"
- We should automatically infer that "sat" might follow "the dog"

## The Solution: Continuous Representations

The key insight of neural language models:

**Replace discrete symbols with continuous vectors.**

### Discrete vs Continuous

| Discrete (N-gram) | Continuous (Neural) |
|-------------------|---------------------|
| "cat" = entry #742 in table | "cat" = [0.2, -0.5, 0.8, ...] ∈ ℝᵈ |
| Two words: same or different | Two words: distance in vector space |
| Similarity undefined | Similarity = cosine, Euclidean, etc. |
| No interpolation possible | Smooth interpolation natural |

The notation **ℝᵈ** means "d-dimensional space of real numbers"—a vector with d components, each being a real number. For example, ℝ³ is 3D space (x, y, z coordinates).

### Why Continuous Helps

In a continuous space:

1. **Similar words → similar vectors**: "cat" and "dog" are nearby
2. **Similar contexts → similar predictions**: Neural network outputs vary smoothly
3. **Generalization is automatic**: Learning about nearby points affects the whole region

### The Mathematical Shift

N-gram:

$$P(w | c) = \text{table}[c][w]$$


Neural:

$$P(w | c) = f_\theta(\text{embed}(c))$$


Where:

- embed(c) maps discrete context to continuous vector
- f_θ is a smooth function (neural network)
- θ are learnable parameters

## Parameter Efficiency

### N-gram Parameter Count

For vocabulary |V| and context length k:

- Need to store P(w | c) for every (c, w) pair
- Parameters: O(|V|^{k+1})

For |V| = 10,000 and k = 5: $10,000^6$ = $10^24$ parameters. Impossible.

### Neural Parameter Count

For embedding dimension d and hidden size h:

- Embedding matrix: |V| × d
- Hidden layers: O(h × k × d)
- Output layer: O(h × |V|)
- Total: O(|V| × d + h × |V|)

For |V| = 10,000, d = 256, h = 512:

- Embeddings: 2.56M
- Output: 5.12M
- Hidden: ~0.5M
- Total: ~8M parameters

**From $10^24$ to 8 million**: a reduction by a factor of $10^17$!

## The Trade-off

### What We Gain
- Massive parameter reduction
- Automatic generalization
- Smooth predictions
- Ability to use long contexts

### What We Lose
- Exact match: N-gram gives exact counts
- Simplicity: no optimization needed for N-gram
- Interpretability: can inspect count tables directly
- Speed: N-gram lookup is O(1)

### When Each Wins

| Scenario | Better Choice |
|----------|---------------|
| Abundant data, short context | N-gram |
| Limited data, long context | Neural |
| Requires exact memorization | N-gram |
| Requires generalization | Neural |
| Interpretability critical | N-gram |
| Performance critical | Neural |

In practice, modern language models use neural approaches almost universally—the generalization advantage is too significant.

## A Concrete Comparison

Let's quantify the difference with a thought experiment.

### The Task
Predict the next character in English text with context length 10.

### N-gram Approach
- Collect all 11-grams from training data
- Compute conditional probabilities
- Apply smoothing

**Result**: Most test 11-grams unseen. Heavy reliance on backoff. Effective context often only 2-3 characters.

### Neural Approach
- Learn 10 character embeddings (10 × d parameters)
- Process with neural network
- Output probability over vocabulary

**Result**: Every test context gets a meaningful prediction based on its pattern, not just exact matches.

### The Difference in Action

Training data includes:

- "the cat sat on the mat"

Test context:

- "the dog sat on the "

N-gram: Never seen "the dog sat on the " → backs off to "the " → poor prediction

Neural: "dog" embedding ≈ "cat" embedding → similar hidden state → similar prediction to "cat" case → correctly predicts space or common next words

## The Path Forward

In the remaining sections of Stage 3, we'll build this from scratch:

1. **Section 3.2**: Embeddings—how to represent tokens as vectors
2. **Section 3.3**: Neural networks—the function that maps embeddings to predictions
3. **Section 3.4**: Cross-entropy loss—how to train the network
4. **Section 3.5**: Implementation—building it with our Stage 2 autograd
5. **Section 3.6**: Training dynamics—making it learn effectively
6. **Section 3.7**: Evaluation—comparing to our Stage 1 baselines

## Summary

| Concept | N-gram Limitation | Neural Solution |
|---------|-------------------|-----------------|
| Sparsity | Most contexts unseen | Continuous representations |
| Generalization | No similarity notion | Embeddings capture similarity |
| Parameters | Exponential in context | Linear in vocabulary |
| Long context | Backs off constantly | Uses full context |

**Key insight**: The curse of dimensionality makes count-based models fundamentally limited. Continuous representations offer a way out by enabling generalization: learning from one context transfers to similar contexts.

## Exercises

1. **Count the sparsity**: For a corpus of 1 million characters and vocabulary of 80 characters, estimate what fraction of possible 5-grams appear in the data.

2. **Similarity intuition**: List 5 pairs of words that should have similar embeddings and 5 pairs that should have dissimilar embeddings. Explain your reasoning.

3. **Parameter comparison**: For vocabulary size V = 50,000, context length k = 10, embedding dimension d = 512, and hidden size h = 1024, calculate the parameter count for both N-gram and neural approaches.

4. **Backoff analysis**: If an N-gram model backs off from order 5 to order 3 on average, what effective context length is it using? How does this compare to a neural model?

5. **Thought experiment**: Describe a scenario where an N-gram model might outperform a neural model. What properties of the data would make counting more effective than learning?

## What's Next

We've established why we need neural models. The first step is learning how to represent discrete tokens—characters or words—as continuous vectors.

In Section 3.2, we'll dive deep into **embeddings**: the foundation that makes neural language models possible.

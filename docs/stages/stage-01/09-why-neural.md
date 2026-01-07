# Section 1.9: Why We Need Neural Networks

*Reading time: 10 minutes | Difficulty: ★★☆☆☆*

We've built a complete language model from first principles. It works. But it has fundamental limitations that no amount of clever engineering can fix. This section explains *why* we need neural networks—not just that we do, but *why counting fails*.

## The Generalization Gap

Consider this training corpus:

```
"the cat sat on the mat"
"the dog lay on the rug"
"the bird flew over the house"
```

Now test on:

```
"the cat lay on the rug"
```

### What Markov Models See

A bigram model treats each context-token pair as completely independent:

| Bigram | Count | P(next\|context) |
|--------|-------|------------------|
| (cat, sat) | 1 | 1.0 |
| (cat, lay) | 0 | 0.0 |
| (dog, lay) | 1 | 1.0 |
| (dog, sat) | 0 | 0.0 |

To the model, "cat lay" and "dog sat" are **impossible**—even though they're perfectly valid English.

### What Humans See

Humans recognize:

- "cat" and "dog" are both animals
- "sat" and "lay" are both position verbs
- If "the dog lay" is valid, "the cat lay" should be too

The model has no way to make these connections. Each bigram is an island.

## The Similarity Problem

The core issue: **Markov models have no concept of similarity**.

```
"cat" ≠ "dog"   (different strings)
"sat" ≠ "lay"   (different strings)
```

To a Markov model, "cat" is as different from "dog" as it is from "quantum" or "banana". There's no notion that some words are more related than others.

### What We Need: Representations

Imagine if we could represent words as vectors where:

- Similar words have similar vectors
- Operations on vectors capture semantic relationships

```python
# Hypothetical embeddings (what we'll build in Stage 3)
embed("cat") = [0.2, 0.8, 0.1, ...]  # furry, pet, small
embed("dog") = [0.3, 0.7, 0.2, ...]  # furry, pet, medium
embed("car") = [0.0, 0.0, 0.9, ...]  # metal, vehicle

# cat and dog are "close" in this space
distance(embed("cat"), embed("dog")) = 0.15  # small
distance(embed("cat"), embed("car")) = 0.95  # large
```

With such representations, a model could learn:

> "After animals, position verbs are likely"

This is a **general rule** that applies to all animals and all position verbs—including combinations never seen in training.

!!! info "Connection to Modern LLMs"

    GPT-4, Claude, and LLaMA all use embeddings—learned vector representations of tokens. The embedding layer is literally the first thing the input passes through. A 50,000-token vocabulary becomes 50,000 vectors of dimension 4,096 or more.

    These embeddings capture rich semantic relationships: "king - man + woman ≈ queen" is a famous example from word2vec that still works in modern embeddings.

## The Memory Problem

Another fundamental limitation: Markov models store everything explicitly.

### Counting vs. Learning

**Markov approach**: Store a count for every observed n-gram.
- Storage: O(number of unique n-grams)
- For trigrams on a 50K vocabulary: up to 50,000³ = 125 trillion entries
- In practice, sparse storage, but still grows with data

**Neural approach**: Learn a function that maps context → probabilities.
- Storage: O(number of parameters) — fixed, regardless of data size
- GPT-2: 1.5 billion parameters handles effectively infinite patterns
- Parameters are shared across all contexts

### A Concrete Example

Training data: 1 billion words of English.

**Trigram model**:

- Stores: ~500 million unique trigrams
- Each with counts
- Lookup table approach

**Neural model**:

- Learns: patterns like "adjective before noun", "verb after subject"
- Applies patterns to any input, including novel combinations
- Compresses the data into general rules

## The Composition Problem

Language is compositional—meaning builds from parts:

```
"the [adjective] [noun] [verb] the [noun]"
```

A Markov model must see each instantiation:

- "the big cat chased the mouse"
- "the small dog chased the ball"
- "the angry bird chased the worm"

A neural model can learn the template and fill it with any appropriate words.

### Syntactic Patterns

Consider subject-verb agreement across distance:

```
"The cats that sat on the mat were sleeping."
        ^                          ^

        |__________________________|
              must agree (plural)
```

A bigram model sees "mat were" and has no idea why "were" is correct. It can't see "cats" from that position.

A transformer with attention can directly connect "cats" to "were" regardless of distance.

## What Neural Networks Provide

| Capability | Markov | Neural |
|------------|--------|--------|
| Exact pattern recall | ✓ | ✓ |
| Generalization to similar patterns | ✗ | ✓ |
| Long-range dependencies | ✗ | ✓ |
| Compositional structure | ✗ | ✓ |
| Fixed memory footprint | ✗ | ✓ |
| Gradient-based optimization | ✗ | ✓ |

## The Path Forward

To build models that generalize, we need:

1. **Continuous representations** (embeddings)
   - Map discrete tokens to vectors
   - Similar tokens → similar vectors
   - Covered in Stage 3

2. **Differentiable functions**
   - Learn from gradients, not counts
   - Requires automatic differentiation
   - Covered in Stage 2

3. **Flexible architectures**
   - Neural networks that can capture complex patterns
   - Eventually: transformers with attention
   - Covered in Stages 3-8

## A Preview: The Neural Language Model

Here's what we're building toward (Stage 3):

```python
class NeuralLM:
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        # Embedding: token → vector
        self.embed = Embedding(vocab_size, embed_dim)
        # Hidden layer: captures patterns
        self.hidden = Linear(embed_dim * context_size, hidden_dim)
        # Output: vector → probabilities
        self.output = Linear(hidden_dim, vocab_size)

    def forward(self, context):
        # Embed each token in context
        x = self.embed(context)  # [context_size, embed_dim]
        # Concatenate embeddings
        x = x.flatten()          # [context_size * embed_dim]
        # Transform through hidden layer
        x = relu(self.hidden(x)) # [hidden_dim]
        # Project to vocabulary
        logits = self.output(x)  # [vocab_size]
        # Convert to probabilities
        return softmax(logits)
```

The key insight: **The same parameters are used for all contexts**. Knowledge about "cat" informs predictions about "dog" because they have similar embeddings.

## But First: Gradients

Before we can train neural networks, we need to compute gradients—derivatives that tell us how to update parameters to reduce loss.

Computing gradients by hand for every operation would be tedious and error-prone. Instead, we'll build **automatic differentiation**: a system that computes gradients for any computation, automatically.

**→ Stage 2: Automatic Differentiation**

## Exercises

1. **Impossible Sentences**: Create a list of 10 sentences that are valid English but would have zero probability under a bigram model trained on any reasonable corpus.

2. **Similarity Matrix**: For 10 common words, create a manual "similarity matrix" (1 = similar, 0 = different). How would you use this to improve a Markov model?

3. **Compression Ratio**: If we have 100 million words of training data and a neural model with 100 million parameters, what's the "compression ratio"? What does this imply about what the model learned?

## Summary

| Limitation | Why It's Fundamental | Neural Solution |
|------------|---------------------|-----------------|
| No generalization | Exact match required | Learned similarity |
| Exponential states | One state per n-gram | Shared parameters |
| Fixed context | Order-k assumption | Attention mechanism |
| No composition | Flat pattern matching | Hierarchical representations |

**The key insight**: Markov models memorize; neural networks generalize. Memorization works for patterns you've seen. Generalization works for the infinite space of patterns you haven't.

Language is infinite. We need models that can generalize.

**→ Next: Stage 2 - Automatic Differentiation**

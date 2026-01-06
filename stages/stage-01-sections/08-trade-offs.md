# Section 1.8: The Fundamental Trade-offs

We've built a complete language model. Now let's understand its fundamental limitations and why we need something better.

## The Context-Sparsity Trade-off

This is the central tension in Markov models:

**More context = better predictions**
- A bigram model sees "the" and might predict "cat", "dog", "end"...
- A 5-gram model seeing "the cat sat on" can confidently predict "the"

**More context = sparser observations**
- Every possible 5-gram needs to be observed in training
- Most valid 5-grams never appear in any finite corpus

### Quantifying the Trade-off

For character-level modeling with |V| = 27 (letters + space):

| Order | Possible contexts | Typical observed | Coverage |
|-------|-------------------|------------------|----------|
| 1 | 27 | 27 | 100% |
| 2 | 729 | ~400 | 55% |
| 3 | 19,683 | ~5,000 | 25% |
| 5 | 14.3M | ~50,000 | 0.3% |
| 7 | 10.5B | ~100,000 | 0.001% |

At order 5, we've seen less than 1% of possible contexts!

### The Experimental Evidence

Here's what happens with a typical corpus (100,000 characters of English text):

| Order | Train PPL | Test PPL | States | Unseen rate |
|-------|-----------|----------|--------|-------------|
| 1 | 12.4 | 12.5 | 27 | 0% |
| 2 | 7.2 | 7.8 | 420 | 2% |
| 3 | 4.1 | 6.3 | 4,800 | 15% |
| 5 | 1.8 | ∞ | 48,000 | 62% |
| 7 | 1.1 | ∞ | 95,000 | 89% |

**The pattern**:
1. Train perplexity keeps improving (memorization)
2. Test perplexity improves, then explodes (overfitting)
3. Eventually, test perplexity becomes infinite (unseen n-grams)

## Why This Is Fundamental

This isn't a bug in our implementation—it's a fundamental limitation of the approach.

**The problem**: Markov models require exact pattern matching.

- Training: "the cat sat"
- Test: "the cat lay"

Even though "sat" and "lay" are syntactically similar (both verbs following "cat"), the trigram model treats them as completely unrelated. It has no notion of *similarity*.

**What we need**: A way to generalize from seen patterns to unseen but similar patterns.

This is exactly what neural networks provide.

## State Space Explosion

The number of states grows exponentially with order:

$$\text{States} = |V|^k$$

For word-level models with |V| = 50,000:
| Order | States | Storage (4 bytes each) |
|-------|--------|----------------------|
| 1 | 50K | 200 KB |
| 2 | 2.5B | 10 GB |
| 3 | 125T | 500 TB |

Even storing just the non-zero entries, we run into the sparsity problem.

## Long-Range Dependencies

Language has structure that spans far beyond what Markov models can capture:

**Example 1: Subject-verb agreement (distance: 7 words)**
"The cats that sat on the mat **were** sleeping."

A bigram model sees "mat were"—grammatical but not because of local patterns.

**Example 2: Coreference (distance: variable)**
"John walked into the room. He looked around. His eyes..."

"He" and "His" must agree with "John" from sentences ago.

**Example 3: Topic coherence (distance: paragraph)**
An article about physics should maintain physics vocabulary throughout.

Markov models are blind to all of this.

## What We Need: The Preview

To fix these limitations, we need:

| Limitation | Solution | Stage |
|------------|----------|-------|
| No generalization | Learned representations (embeddings) | 4 |
| No gradient-based learning | Automatic differentiation | 2-3 |
| No flexible context | Attention mechanism | 7 |
| Fixed context size | Transformer architecture | 8 |

**The key insight**: Instead of storing explicit counts for every pattern, we learn a *function* that maps contexts to predictions. This function can generalize to unseen contexts.

## What Carries Forward

Despite the limitations, Markov models introduced concepts that underpin all modern LLMs:

| Concept | First Introduced | Where It Appears Later |
|---------|------------------|----------------------|
| Autoregressive factorization | Chain rule → Markov | GPT, LLaMA, Claude |
| Next-token prediction | MLE objective | All modern LLMs |
| Perplexity | Evaluation metric | Standard LLM benchmark |
| Temperature sampling | Generation control | ChatGPT, Claude, etc. |
| Cross-entropy loss | MLE ≡ min cross-entropy | All neural LM training |

**These aren't simplified versions—they're the same concepts.**

GPT-4 uses the same autoregressive factorization, the same cross-entropy objective, and the same temperature-controlled sampling. The only difference is *how* P(next | context) is computed.

## The Bias-Variance Perspective

From statistical learning theory:

**Bias**: Error from the model's assumptions
- High-order Markov: Low bias (few assumptions)
- Low-order Markov: High bias (strong assumptions about independence)

**Variance**: Error from sensitivity to training data
- High-order Markov: High variance (sensitive to exact patterns)
- Low-order Markov: Low variance (stable estimates)

**The trade-off**: We can't minimize both simultaneously.

Neural networks navigate this differently:
- Flexible function class (low bias)
- Regularization controls variance
- Learning algorithm finds good solutions

## Exercises

1. **Smoothing**: Implement Laplace smoothing and show it prevents infinite perplexity.

2. **Order sweep**: Plot train and test perplexity vs. order for a corpus of your choice. Find the "elbow" where test perplexity starts increasing.

3. **Vocabulary comparison**: Compare character-level and word-level models on the same corpus. How do optimal orders differ?

4. **State space visualization**: For a trigram model, visualize the transition graph. Which states have the most outgoing edges?

5. **Temperature analysis**: Generate 100 samples at T=0.5, 1.0, and 2.0. Compute the average perplexity of each set. What do you observe?

## Summary

**What we built**:
- A complete language model from first principles
- Training, evaluation, and generation
- Full mathematical derivations for everything

**What we learned**:
- Probability theory foundations
- Chain rule → autoregressive factorization
- MLE → counting is optimal
- Information theory → cross-entropy → perplexity
- Temperature → controlled sampling

**Why it's limited**:
- Context-sparsity trade-off is fundamental
- No generalization to similar-but-unseen patterns
- Exponential state space explosion
- Can't capture long-range dependencies

**What's next**:
To build models that generalize, we need to learn representations. That requires gradients. And computing gradients efficiently requires **automatic differentiation**.

→ **Stage 2: Automatic Differentiation**

---

## Reflection: The Pólya Method

Looking back at this stage through Pólya's problem-solving framework:

**1. Understand the problem**
- We want P(next token | context)
- The space of sequences is exponentially large
- We need both tractability and accuracy

**2. Devise a plan**
- Use chain rule to factorize
- Make Markov assumption to limit context
- Estimate probabilities by counting (MLE)

**3. Execute the plan**
- Implemented counting-based training
- Added temperature-controlled sampling
- Evaluated with perplexity

**4. Reflect**
- The approach works but has fundamental limits
- Context-sparsity trade-off can't be avoided
- Need a different approach (neural networks) for better generalization

This reflection pattern—understand, plan, execute, reflect—will continue throughout the book. Each stage builds on the previous, and each reflection motivates the next stage.

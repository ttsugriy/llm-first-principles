# Section 1.5: Perplexity — The Standard Evaluation Metric

*Reading time: 12 minutes | Difficulty: ★★☆☆☆*

Cross-entropy is the theoretically correct metric, but it's hard to interpret. "Our model has cross-entropy 4.2 bits" doesn't mean much intuitively.

Perplexity fixes this by converting cross-entropy into an interpretable number.

## From Cross-Entropy to Perplexity

**Definition**: Perplexity is the exponential of cross-entropy:

$$\text{Perplexity} = $2^{H(P,Q)}$ = $2^{-\frac{1}${N}\sum_i \log_2 Q(x_i | \text{context})}$$


Or equivalently, if using natural logarithms:

$$\text{Perplexity} = \exp\left(-\frac{1}{N}\sum_i \ln Q(x_i | \text{context})\right)$$

**Note on logarithm base**: Both formulas give the same perplexity value! With log₂, we exponentiate with base 2. With ln, we use exp (base e). The choice only affects the intermediate cross-entropy value, not the final perplexity. In code, ln is typically used for numerical convenience.

**Why exponentiate?** To return from log-space to probability-space, giving us an interpretable number.

## The Intuitive Interpretation

**Perplexity = effective vocabulary size**

If your model has perplexity K, it's as uncertain as if it were choosing uniformly among K options at each position.

**Example interpretations**:
| Perplexity | Interpretation |
|------------|----------------|
| 1 | Perfect prediction (always 100% confident in correct answer) |
| 10 | Like choosing among 10 equally likely words |
| 100 | Like choosing among 100 equally likely words |
| 50,000 | Like random guessing over entire vocabulary |

A good language model should have perplexity much lower than vocabulary size.

## Deriving the "Effective Vocabulary" Interpretation

Let's prove that perplexity equals effective vocabulary size.

**Consider a uniform distribution over K items**: Each item has probability 1/K.

Cross-entropy of uniform Q on any P with support K:

$$H = -\sum_i P(x_i) \log_2 \frac{1}{K} = -\sum_i P(x_i) \cdot (-\log_2 K) = \log_2 K$$


Perplexity:

$$\text{PPL} = $2^{\log_2 K}$ = K$$


So a uniform distribution over K items has perplexity K.

**The converse**: If a model has perplexity K, it has the same average uncertainty as uniform distribution over K items.

## Computing Perplexity: A Concrete Example

**Test sequence**: "the cat" (pretend this is our entire test set)

**Model predictions** (bigram):
- P("the" | START) = 0.2
- P("cat" | "the") = 0.1
- P(END | "cat") = 0.3

**Step 1**: Compute log-probabilities
- log₂(0.2) = -2.32
- log₂(0.1) = -3.32
- log₂(0.3) = -1.74

**Step 2**: Average negative log-probability

$$H = -\frac{1}{3}(-2.32 - 3.32 - 1.74) = \frac{7.38}{3} = 2.46 \text{ bits}$$


**Step 3**: Exponentiate

$$\text{PPL} = $2^{2.46}$ = 5.5$$


**Interpretation**: On average, the model was as uncertain as choosing among 5.5 equally likely tokens.

## Perplexity vs. Accuracy

Why not just use accuracy (% of correct predictions)?

**Problem**: Accuracy ignores confidence.

Consider two models predicting "cat":

- Model A: P("cat") = 0.51, P("dog") = 0.49
- Model B: P("cat") = 0.99, P("dog") = 0.01

Both have 100% accuracy if "cat" is correct, but Model B is clearly better.

Perplexity (via log-probability) captures this:

- Model A contribution: -log₂(0.51) = 0.97 bits
- Model B contribution: -log₂(0.99) = 0.01 bits

Model B has much lower perplexity because it's more confident in correct answers.

## Properties of Perplexity

**1. Lower is better**: Lower perplexity = better model

**2. Bounded below by 1**: Perplexity ≥ 1, with equality only for perfect prediction

**3. Bounded above by vocabulary size**: PPL ≤ |V| (achieved by uniform random guessing)

**4. Infinite if any probability is 0**: If the model assigns 0 probability to an observed token, perplexity = ∞

**5. Geometric mean interpretation**:

$$\text{PPL} = \left( \prod_{i=1}^N \frac{1}{Q(x_i | \text{context})} \right)^{1/N}$$


Perplexity is the geometric mean of the inverse probabilities.

## Perplexity on Train vs. Test

**Training perplexity**: Evaluate model on data it was trained on.
**Test perplexity**: Evaluate model on **held-out data** (data set aside before training and used only for evaluation)—data the model never saw during training.

**Critical insight**: Training perplexity always looks better (or equal).

| What it measures | What you want |
|------------------|---------------|
| Train perplexity | How well model fits training data | Low, but not too low |
| Test perplexity | How well model generalizes | As low as possible |

**Overfitting**: When train perplexity << test perplexity, the model memorized training data but doesn't generalize.

## The Relationship Hierarchy

Let's connect all the metrics we've defined:

```
Probability assigned to test data: P(test | model)
         ↓ take log
Log-likelihood: log P(test | model)
         ↓ negate and average
Cross-entropy: H = -(1/N) · log P(test | model)
         ↓ exponentiate
Perplexity: PPL = exp(H)
```

All contain the same information, just different presentations:

- Log-likelihood: raw sum (for optimization)
- Cross-entropy: normalized (for comparison across corpus sizes)
- Perplexity: intuitive (for human interpretation)

## Perplexity in Practice: Real Numbers

What perplexity values are typical?

| Model | Perplexity | Dataset |
|-------|------------|---------|
| Unigram (word) | ~1000 | Typical English |
| Bigram (word) | ~150-300 | Typical English |
| Trigram (word) | ~100-150 | Typical English |
| Neural LM (LSTM) | ~50-80 | Penn Treebank |
| GPT-2 (small) | ~35 | Penn Treebank |
| GPT-3 | ~20 | Various |

Character-level models have different scales:

| Model | Perplexity | Dataset |
|-------|------------|---------|
| Unigram (char) | ~27 | English text |
| Order-3 Markov (char) | ~8-12 | English text |
| Order-5 Markov (char) | ~4-6 | English text (but often ∞ on test) |

## The Overfitting Pattern

For Markov models, you'll observe this pattern as you increase order:

| Order | Train PPL | Test PPL | States |
|-------|-----------|----------|--------|
| 1 | 15 | 15 | 50 |
| 2 | 8 | 9 | 500 |
| 3 | 4 | 12 | 5,000 |
| 5 | 1.5 | ∞ | 50,000 |

**What's happening**:

- Train PPL keeps improving (more context = better fit)
- Test PPL improves initially (capturing real patterns)
- Test PPL then explodes (model sees unseen n-grams, assigns probability 0)

This is the fundamental limitation of Markov models that we'll address with neural networks.

!!! info "Connection to Modern LLMs"

    Perplexity remains THE standard metric for evaluating language models. When OpenAI reports "GPT-4 achieves X perplexity on benchmark Y," they're using exactly the formula we derived here.

    Modern LLM leaderboards (like the Hugging Face Open LLM Leaderboard) report perplexity alongside other metrics. A perplexity of 20 on WikiText means the model is, on average, as uncertain as choosing among 20 equally likely tokens—even though the vocabulary has 50,000+ tokens.

!!! note "Historical Note: Shannon's Experiments (1951)"

    Claude Shannon, the father of information theory, conducted experiments measuring the entropy of English in his 1951 paper "Prediction and Entropy of Printed English." He had humans predict the next character and measured their error rate.

    Shannon estimated English has about 1.0-1.3 bits per character of entropy. This corresponds to perplexity 2.0-2.5 per character—remarkably close to what modern character-level models achieve!

!!! warning "Common Mistake: Comparing Perplexities Across Different Tokenizations"

    You CANNOT directly compare perplexity between:
    - Character-level and word-level models
    - Models with different vocabularies
    - Different test sets

    A character model with PPL=5 and word model with PPL=100 are not comparable. The character model predicts 1 character at a time; the word model predicts entire words. Always compare apples to apples.

## Summary

| Concept | Formula | Interpretation |
|---------|---------|----------------|
| Perplexity | $2^{cross-entropy}$ | Effective vocabulary size |
| PPL = 1 | Perfect model | Always correct with 100% confidence |
| PPL = \|V\| | Random guessing | No information from context |
| PPL = ∞ | Model assigns P=0 | Considered token impossible |

**Key takeaways**:

1. Perplexity is the standard metric for language models
2. Lower is better
3. It measures how "surprised" the model is on average
4. Compare train vs. test to detect overfitting

Next: How do we generate text from our model? This requires understanding sampling and temperature.

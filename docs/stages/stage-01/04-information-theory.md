# Section 1.4: Information Theory Foundations

*Reading time: 16 minutes | Difficulty: ★★★☆☆*

To properly understand how to measure model quality, we need information theory. This section derives the key concepts from first principles, explaining *why* logarithms appear everywhere in machine learning.

## The Core Question: What is Information?

Intuitively, information is what reduces uncertainty. When someone tells you something you already knew, you gain no information. When they tell you something surprising, you gain a lot.

**Claude Shannon's insight (1948)**: We can *quantify* information mathematically.

## Deriving the Information Formula

Let's derive the formula for information from basic requirements.

**Setup**: An event has probability p. How much information (in some units) do we gain when we learn it occurred?

Let I(p) denote the information gained from an event with probability p.

**Requirement 1**: Rare events give more information.
If p₁ < p₂, then I(p₁) > I(p₂).
Learning something unlikely happened is more informative.

**Requirement 2**: Certain events give no information.
I(1) = 0.
If something was guaranteed to happen, learning it happened tells us nothing.

**Requirement 3**: Information from independent events adds.
If A and B are independent, learning both gives:
I(P(A and B)) = I(P(A)) + I(P(B))
For independent events: P(A and B) = P(A) · P(B)
So: I(P(A) · P(B)) = I(P(A)) + I(P(B))

**The key constraint**: We need a function where f(x·y) = f(x) + f(y).

What function turns products into sums?

**The logarithm!** log(x·y) = log(x) + log(y)

So I(p) must be of the form: I(p) = -log(p) × (some constant)

The negative sign ensures rare events (small p, so log p is negative) give positive information.

**Choosing the constant**: If we use log base 2, the unit is "bits." If base e, "nats." If base 10, "digits."

$$\boxed{I(p) = -\log_2(p) = \log_2(1/p)}$$


This is the **information content** or **surprisal** of an event with probability p.

## Understanding the Formula

Let's verify this makes sense:

| Probability | Information (bits) | Interpretation |
|------------|-------------------|----------------|
| 1.0 | 0 | Certain event, no surprise |
| 0.5 | 1 | Like a coin flip |
| 0.25 | 2 | Like two coin flips |
| 0.125 | 3 | Like three coin flips |
| 0.01 | 6.64 | Quite surprising |
| 0.001 | 9.97 | Very surprising |

**The coin flip interpretation**: If an event has probability 1/2ⁿ, learning it occurred gives n bits of information—equivalent to learning the outcomes of n fair coin flips.

## Why Bits?

The term "bit" (binary digit) comes from a physical interpretation:

To distinguish between N equally likely possibilities, you need log₂(N) yes/no questions.

**Example**: There are 8 equally likely outcomes. How many bits to identify which occurred?
- log₂(8) = 3 bits
- Indeed: "Is it in the first half?" (3 questions distinguish 8 things)

So -log₂(p) = log₂(1/p) tells us: "How many yes/no questions would it take to identify this outcome among equally-likely alternatives?"

## Expected Value: A Quick Review

Before we define entropy, we need the concept of **expected value** (or expectation, denoted 𝔼[·]). It's the average value of a random variable, weighted by probabilities.

For a discrete random variable X with distribution P:

$$\mathbb{E}[X] = \sum_x x \cdot P(x)$$

More generally, for any function f applied to X:

$$\mathbb{E}[f(X)] = \sum_x f(x) \cdot P(x)$$

**Example**: For a fair six-sided die:

$$\mathbb{E}[\text{value}] = 1 \cdot \frac{1}{6} + 2 \cdot \frac{1}{6} + 3 \cdot \frac{1}{6} + 4 \cdot \frac{1}{6} + 5 \cdot \frac{1}{6} + 6 \cdot \frac{1}{6} = 3.5$$

**Intuition**: If you rolled the die infinitely many times and averaged the results, you'd get 3.5.

## Entropy: Average Information

If we have a random variable X with distribution P(X), the **entropy** H(X) is the expected (average) information:

$$H(X) = \mathbb{E}[-\log P(X)] = -\sum_x P(x) \log P(x)$$


Entropy measures the average surprise, or equivalently, the average number of bits needed to encode outcomes of X.

**Example 1: Fair coin**
P(heads) = P(tails) = 0.5

$$H = -0.5 \log_2(0.5) - 0.5 \log_2(0.5) = -0.5(-1) - 0.5(-1) = 1 \text{ bit}$$


**Example 2: Biased coin (90% heads)**
P(heads) = 0.9, P(tails) = 0.1

$$H = -0.9 \log_2(0.9) - 0.1 \log_2(0.1)$$

$$= -0.9(-0.152) - 0.1(-3.322) = 0.137 + 0.332 = 0.469 \text{ bits}$$


The biased coin has lower entropy—it's more predictable.

**Example 3: Certain coin (always heads)**
P(heads) = 1, P(tails) = 0

$$H = -1 \cdot \log_2(1) - 0 \cdot \log_2(0) = 0 \text{ bits}$$


(We define 0 · log(0) = 0 by continuity.)

No uncertainty, no information needed.

## Properties of Entropy

1. **Non-negative**: H(X) ≥ 0. Entropy is always non-negative.

2. **Maximum for uniform distribution**: For a random variable over n outcomes, entropy is maximized when all outcomes are equally likely:

$$H_{\max} = \log_2(n)$$


3. **Additivity for independent variables**: If X and Y are independent:

$$H(X, Y) = H(X) + H(Y)$$


4. **Concavity**: Entropy is a concave function of the probability distribution.

## Cross-Entropy: Using the Wrong Model

Now the crucial concept for language modeling.

**Scenario**: The true distribution is P, but we're using a model Q to encode/predict.

The **cross-entropy** is:

$$H(P, Q) = -\sum_x P(x) \log Q(x) = \mathbb{E}_{x \sim P}[-\log Q(x)]$$


This is the average number of bits needed to encode samples from P using a code optimized for Q.

**Key insight**: Cross-entropy is always at least as large as entropy:

$$H(P, Q) \geq H(P)$$


with equality if and only if P = Q.

Using the wrong model (Q ≠ P) always requires more bits on average.

## Cross-Entropy for Language Models

For a language model, we're evaluating on a test corpus. Let's connect to our setup:

- **True distribution P**: The actual distribution of text (approximated by test data)
- **Model distribution Q**: Our Markov model's predictions
- **Samples**: The actual tokens in the test corpus

The cross-entropy of our model on the test data is:

$$H(P, Q) = -\frac{1}{N} \sum_{i=1}^{N} \log Q(x_i | \text{context}_i)$$


This is exactly the average negative log-probability of the test tokens under our model!

## The Connection to Log-Likelihood

Remember from Section 1.3, the log-likelihood was:

$$\ell(\theta) = \sum_i \log P(x_i | \text{context}; \theta)$$


Cross-entropy is:

$$H = -\frac{1}{N} \sum_i \log Q(x_i | \text{context})$$


So:

$$H = -\frac{\ell}{N}$$


**Minimizing cross-entropy = Maximizing log-likelihood!**

This is why these objectives are equivalent. MLE is *implicitly* minimizing the cross-entropy between the empirical data distribution and our model.

## KL Divergence: The Gap

The **Kullback-Leibler divergence** measures how different Q is from P:

$$D_{KL}(P || Q) = H(P, Q) - H(P) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$


**Properties**:

- D_KL(P || Q) ≥ 0 (always non-negative)
- D_KL(P || Q) = 0 if and only if P = Q
- D_KL(P || Q) ≠ D_KL(Q || P) in general (not symmetric!)

**Why asymmetric?** D_KL(P || Q) measures "how surprised we are using model Q when the truth is P", while D_KL(Q || P) measures "how surprised we are using model P when the truth is Q". These are different questions! In machine learning, we typically use D_KL(P || Q) where P is the data distribution and Q is our model.

**Interpretation**: KL divergence is the extra bits needed (on average) when using code Q instead of the optimal code P. It's always non-negative because the optimal code for P is always at least as good as any other code.

For language modeling:

- If our model Q matches the true distribution P: D_KL = 0, cross-entropy = entropy
- If our model is bad: D_KL is large, cross-entropy >> entropy

## Why We Minimize Cross-Entropy, Not KL Divergence

In practice, we minimize cross-entropy, not KL divergence. Why?

$$D_{KL}(P || Q) = H(P, Q) - H(P)$$


Since H(P) is fixed (determined by the true data distribution), minimizing H(P, Q) is equivalent to minimizing D_KL(P || Q).

We use cross-entropy because H(P) is unknown (we'd need the true distribution), but H(P, Q) can be estimated from data.

## Summary

| Concept | Formula | Meaning |
|---------|---------|---------|
| Information | I(p) = -log p | Surprise of event with probability p |
| Entropy | H(P) = E[-log P(x)] | Average surprise under P |
| Cross-entropy | H(P,Q) = E_P[-log Q(x)] | Average surprise using model Q on data from P |
| KL divergence | D_KL(P\|\|Q) = H(P,Q) - H(P) | Extra bits from using Q instead of P |

**The big picture**:

1. We want a model Q that matches the true distribution P
2. We measure this by cross-entropy (lower = better)
3. Minimizing cross-entropy = maximizing likelihood
4. This is why log-probability appears everywhere in ML

Next: We'll convert cross-entropy into a more interpretable metric—perplexity.

# Section 1.3b: Smoothing and the Zero Probability Problem

*Reading time: 12 minutes | Difficulty: ★★★☆☆*

In Section 1.3, we derived the MLE solution for Markov chains: simply count and normalize. But we left one critical problem unaddressed: **what happens when we encounter n-grams we've never seen?**

This section derives smoothing techniques from first principles, showing how Bayesian reasoning naturally solves the zero probability problem.

## The Problem: Zero Probabilities

Consider training a bigram model on:

> "the cat sat on the mat"

Now evaluate on:

> "the cat **lay** on the mat"

The bigram "cat lay" never appeared in training. Therefore:

$$P(\text{lay} | \text{cat}) = \frac{\text{count}(\text{cat}, \text{lay})}{\text{count}(\text{cat}, \cdot)} = \frac{0}{2} = 0$$

**Consequences of zero probability**:

1. **Perplexity explosion**: $\text{PPL} = \exp(-\frac{1}{N}\sum \log P(x_i|...) )$. If any $P = 0$, then $\log 0 = -\infty$, so $\text{PPL} = \infty$.

2. **Model declares valid text impossible**: "the cat lay on the mat" is perfectly valid English, yet our model assigns it probability zero.

3. **Cascading failures**: In generation, reaching a zero-probability state means the model has no valid next token.

!!! warning "Common Mistake: Ignoring Zero Probabilities"

    A common error is training a model and testing it without checking for unseen n-grams. Always use smoothing when there's any chance of encountering unseen patterns at test time.

## The Bayesian Perspective

MLE has a hidden assumption: **we trust the data completely**. If we never saw "cat lay", we assume it's impossible.

Bayesian reasoning offers an alternative: **start with prior beliefs, then update based on data**.

### From Counting to Priors

Instead of asking "what did we observe?", ask "what would we believe *before* seeing any data?"

**Prior belief**: Before seeing any text, we might believe all transitions are equally likely:
$$P(\text{any token} | \text{any context}) = \frac{1}{|V|}$$

This is a **uniform prior**—we have no reason to prefer one token over another.

**Posterior belief**: After seeing data, we update our beliefs. More observations → more confidence in the data. Fewer observations → we rely more on our prior.

### The Dirichlet Distribution

The mathematical formalization of this intuition is the **Dirichlet prior**.

For a probability distribution over $|V|$ outcomes, the Dirichlet distribution with parameter $\alpha$ is:

$$\text{Dir}(\alpha) = \frac{\Gamma(|V|\alpha)}{\Gamma(\alpha)^{|V|}} \prod_{i=1}^{|V|} \theta_i^{\alpha - 1}$$

where $\theta_i$ is the probability of outcome $i$.

**Key insight**: The Dirichlet prior with parameter $\alpha$ acts like "pseudo-counts"—as if we observed each outcome $\alpha$ times before seeing any real data.

### Deriving Laplace Smoothing

Given:

- Prior: $\text{Dir}(\alpha)$ (each outcome has $\alpha$ pseudo-counts)
- Data: count$(a, b)$ observations of transition $a \to b$

The **posterior mean** (expected value after seeing data) is:

$$P(b | a) = \frac{\text{count}(a, b) + \alpha}{\text{count}(a, \cdot) + \alpha |V|}$$

**Derivation**:

The Dirichlet distribution is a **conjugate prior** for the categorical distribution. This means:

- Prior: Dir($\alpha, \alpha, ..., \alpha$)
- Likelihood: Categorical with counts $c_1, c_2, ..., c_{|V|}$
- Posterior: Dir($\alpha + c_1, \alpha + c_2, ..., \alpha + c_{|V|}$)

The posterior mean for category $i$ is:
$$\mathbb{E}[\theta_i] = \frac{\alpha + c_i}{\sum_j (\alpha + c_j)} = \frac{\alpha + c_i}{|V|\alpha + \sum_j c_j}$$

For our bigram case:
$$P(b | a) = \frac{\alpha + \text{count}(a, b)}{\alpha |V| + \text{count}(a, \cdot)}$$

**Special cases**:

- $\alpha = 1$: **Laplace smoothing** (add-one smoothing)
- $\alpha = 0.5$: **Jeffreys prior** (sometimes better for small datasets)
- $\alpha \to 0$: Approaches MLE (no smoothing)

## Implementing Smoothing

```python
class SmoothedMarkovChain(MarkovChain):
    """Markov chain with Laplace smoothing."""

    def __init__(self, order: int = 1, alpha: float = 1.0):
        super().__init__(order)
        self.alpha = alpha

    def probability(self, history: Tuple[str, ...], next_token: str) -> float:
        """Smoothed probability: (count + α) / (total + α|V|)"""
        count = self.counts[history][next_token]
        total = self.totals[history]
        vocab_size = len(self.vocab)

        return (count + self.alpha) / (total + self.alpha * vocab_size)
```

!!! info "Connection to Modern LLMs"

    Modern transformer LLMs don't use count-based smoothing—they use learned representations that naturally generalize to unseen patterns. But the underlying goal is the same: avoid assigning zero probability to valid sequences.

    Smoothing is to Markov models what dropout and regularization are to neural networks: techniques that prevent overfitting to training data.

## Verifying Smoothing Works

Let's verify with our example:

**Training data**: "the cat sat on the mat"

**Without smoothing**:

- P(lay | cat) = 0/2 = 0
- Perplexity("the cat lay") = ∞

**With Laplace smoothing** (α = 1, |V| = 8 words):
- P(lay | cat) = (0 + 1) / (2 + 1×8) = 1/10 = 0.1
- Perplexity is now finite!

**Trade-off**: Smoothing "steals" probability mass from observed n-grams to give to unobserved ones. Heavy smoothing (large α) can hurt performance on in-distribution data.

## Beyond Add-One: Backoff and Interpolation

Laplace smoothing is simple but crude. More sophisticated approaches:

### Backoff

When an n-gram is unseen, "back off" to a shorter context:

$$P(w | w_{-2}, w_{-1}) = \begin{cases}
$P_{\text{trigram}$}(w | $w_{-2}$, $w_{-1}$) & \text{if count} > 0 \\
$P_{\text{bigram}$}(w | $w_{-1}$) & \text{otherwise}
\end{cases}$$

**Intuition**: If we haven't seen "cat lay" as a bigram, at least we know something about how often "lay" appears in general.

### Interpolation

Blend predictions from multiple n-gram orders:

$$P(w | w_{-2}, w_{-1}) = \lambda_3 P_3(w | w_{-2}, w_{-1}) + \lambda_2 P_2(w | w_{-1}) + \lambda_1 P_1(w)$$

where $\lambda_1 + \lambda_2 + \lambda_3 = 1$.

**Intuition**: Always consider evidence from all context lengths. The lambdas can be tuned on validation data.

### Kneser-Ney Smoothing

The state-of-the-art for n-gram models (before neural approaches). Key insight: use the *diversity* of contexts a word appears in, not just its frequency.

"Francisco" appears often but only after "San". It should have low unigram probability.
"the" appears in many contexts. It should have higher unigram probability.

## Complexity Analysis

| Operation | MLE | Laplace Smoothing |
|-----------|-----|-------------------|
| Training | O(n) | O(n) |
| P(next\|context) | O(1) | O(1) |
| Space | O(K) | O(K) + O(1) for α |

where n = corpus length, K = number of unique n-grams.

Smoothing adds negligible overhead—just one extra parameter.

## Historical Note

**Laplace smoothing** dates back to Pierre-Simon Laplace's 1814 work on probability. He used it to estimate the probability of the sun rising tomorrow given it has risen every day in recorded history.

The problem of zero probabilities in language modeling was formalized in the 1950s-60s, with major contributions from Good (Good-Turing smoothing) and Katz (Katz backoff).

## Exercises

1. **Implement Backoff**: Extend the `SmoothedMarkovChain` class to fall back to lower-order models when the current context is unseen.

2. **Optimal α**: For a given dataset, find the α that minimizes perplexity on a validation set. Plot perplexity vs. α.

3. **Prove the Posterior Mean**: Starting from the Dirichlet prior and categorical likelihood, derive the posterior mean formula.

4. **Compare Smoothing Methods**: On the same test set, compare perplexity of: (a) MLE, (b) Laplace, (c) Backoff, (d) Interpolation.

## Summary

| Concept | Definition | Key Insight |
|---------|------------|-------------|
| Zero probability | P = 0 for unseen n-grams | Causes infinite perplexity |
| Dirichlet prior | Pseudo-counts before data | Bayesian solution |
| Laplace smoothing | Add α to all counts | Simple and effective |
| Backoff | Use shorter context for unseen | Leverages partial information |
| Interpolation | Blend multiple orders | Always uses all evidence |

**The key takeaway**: Zero probabilities are not just inconvenient—they represent a fundamental failure of the model. Smoothing provides a principled, Bayesian solution. The choice of smoothing technique involves a bias-variance trade-off: more smoothing reduces variance (fewer zeros) but increases bias (less faithful to training data).

**What's next**: We've completed the mathematical foundations of Markov language models. Section 1.4 introduces information theory, giving us tools to measure how "good" our model is.

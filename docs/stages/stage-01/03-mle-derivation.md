# Section 1.3: Learning from Data — Maximum Likelihood Estimation

*Reading time: 18 minutes | Difficulty: ★★★☆☆*

We have a model structure (Markov chain) with parameters θ (transition probabilities). Now we need to learn those parameters from data.

This section derives *from first principles* why counting is the optimal way to estimate probabilities, and proves it rigorously using calculus.

## What is a Parameter?

A **parameter** is a number that defines our model's behavior. For a bigram model, the parameters are all the transition probabilities:

θ = {$θ_{a→b}$ : for all a ∈ V, b ∈ V ∪ {⟨END⟩}}

where $θ_{a→b}$ = P(b | a).

For a vocabulary of size |V|, we have |V| × (|V| + 1) parameters (each of |V| contexts can transition to |V| tokens plus END).

## The Learning Problem

**Given**: A corpus of text (training data)
**Find**: Values for all parameters θ that make the model "good"

But what does "good" mean? We need a criterion for evaluating parameter choices.

## The Likelihood Function

**Key idea**: A good model should assign high probability to the training data.

If we observed data D, then the **likelihood** of parameters θ is:

$$L(\theta) = P(D | \theta)$$


This reads: "The probability of observing data D, if the true parameters were θ."

**Important**: D is fixed (we observed it). θ is what varies. The likelihood is a function of θ.

**Example**: Suppose we observed the sequence "ab" and we're learning a bigram model.

$$L(\theta) = P(\text{"ab"} | \theta) = \theta_{\text{START} \rightarrow a} \cdot \theta_{a \rightarrow b} \cdot \theta_{b \rightarrow \text{END}}$$


If $θ_{START→a}$ = 0.1, $θ_{a→b}$ = 0.2, $θ_{b→END}$ = 0.5:

$$L(\theta) = 0.1 \times 0.2 \times 0.5 = 0.01$$


If $θ_{START→a}$ = 0.5, $θ_{a→b}$ = 0.8, $θ_{b→END}$ = 0.5:

$$L(\theta) = 0.5 \times 0.8 \times 0.5 = 0.2$$


The second parameter setting has higher likelihood—it makes the observed data more probable.

## Maximum Likelihood Estimation (MLE)

**Principle**: Choose parameters that maximize the likelihood of the observed data.

$$\theta^* = \arg\max_\theta L(\theta) = \arg\max_\theta P(D | \theta)$$

**Reading this notation**: "arg max" (short for "argument of the maximum") means "the value of θ that makes L(θ) as large as possible." Think of it as "the input that maximizes the output."

Why is this a good principle?
1. **Intuitive**: We want a model that considers our data likely, not surprising.
2. **Consistent**: As we get more data, MLE converges to the true parameters.
3. **Efficient**: MLE achieves the best possible accuracy for large samples.

## Log-Likelihood: A Computational Trick

Likelihood involves products of many probabilities. For a corpus of n tokens:

$$L(\theta) = \prod_{i=1}^{n} P(x_i | \text{context}_i; \theta)$$


Products of many small numbers cause numerical problems:

- 0.1 × 0.1 × 0.1 × ... (100 times) = $10^{-100}$ ≈ 0 (underflow!)

**Solution**: Work with logarithms.

$$\log L(\theta) = \log \prod_{i=1}^{n} P(x_i | c_i) = \sum_{i=1}^{n} \log P(x_i | c_i)$$


**Why this works**:

1. Log transforms products into sums (easier to compute)
2. Log is monotonically increasing, so max(log L) occurs at same θ as max(L)
3. Log of small positive numbers is large negative (no underflow)

We call log L(θ) the **log-likelihood**, often written ℓ(θ).

## MLE for Bigram Models: The Derivation

Now we derive the MLE solution for bigram models. This is the mathematical core of this section.

### Setup

We have training data: a sequence x₁, x₂, ..., xₙ.

Let count(a, b) = number of times token b follows token a in training data.
Let count(a, ·) = number of times token a appears (followed by anything) = Σ_b count(a, b).

We want to find $θ_{a→b}$ = P(b | a) for all a, b.

### Constraints

The probabilities must satisfy:

1. $θ_{a→b}$ ≥ 0 for all a, b (non-negativity)
2. Σ_b $θ_{a→b}$ = 1 for all a (normalization: probabilities sum to 1)

### The Log-Likelihood

The log-likelihood of the training data is:

$$\ell(\theta) = \sum_{\text{all bigrams } (a,b) \text{ in data}} \log \theta_{a \rightarrow b}$$


We can rewrite this by grouping identical bigrams:

$$\ell(\theta) = \sum_{a \in V} \sum_{b \in V \cup \{\text{END}\}} \text{count}(a, b) \cdot \log \theta_{a \rightarrow b}$$


Each unique bigram (a, b) contributes count(a, b) × log $θ_{a→b}$ to the total.

### Optimization with Constraints: Lagrange Multipliers

We want to maximize ℓ(θ) subject to the constraints Σ_b $θ_{a→b}$ = 1.

**Lagrange multipliers**: To optimize f(x) subject to g(x) = 0, we find where ∇f = λ∇g.

For our problem, we form the Lagrangian:

$$\mathcal{L}(\theta, \lambda) = \ell(\theta) - \sum_a \lambda_a \left( \sum_b \theta_{a \rightarrow b} - 1 \right)$$


We have one Lagrange multiplier λₐ for each context a (one constraint per context).

### Taking Derivatives

For each parameter $θ_{a→b}$, we take the partial derivative and set it to zero:

$$\frac{\partial \mathcal{L}}{\partial \theta_{a \rightarrow b}} = \frac{\text{count}(a, b)}{\theta_{a \rightarrow b}} - \lambda_a = 0$$


**Derivation of ∂ℓ/∂$θ_{a→b}$**:

- ℓ(θ) = $Σ_{a',b'}$ count(a',b') · log $θ_{a'→b'}$
- ∂/∂$θ_{a→b}$ of count(a,b) · log $θ_{a→b}$ = count(a,b) / $θ_{a→b}$
- All other terms don't involve $θ_{a→b}$, so their derivatives are 0

From the derivative equation:

$$\frac{\text{count}(a, b)}{\theta_{a \rightarrow b}} = \lambda_a$$


Solving for $θ_{a→b}$:

$$\theta_{a \rightarrow b} = \frac{\text{count}(a, b)}{\lambda_a}$$


### Finding λₐ Using the Constraint

We know Σ_b $θ_{a→b}$ = 1. Substituting:

$$\sum_b \frac{\text{count}(a, b)}{\lambda_a} = 1$$


$$\frac{1}{\lambda_a} \sum_b \text{count}(a, b) = 1$$


$$\frac{\text{count}(a, \cdot)}{\lambda_a} = 1$$


$$\lambda_a = \text{count}(a, \cdot)$$


### The Final Result

Substituting λₐ back:

$$\theta^*_{a \rightarrow b} = \frac{\text{count}(a, b)}{\text{count}(a, \cdot)}$$


**This is remarkable**: The optimal probability is simply the frequency!

$$P^*(b | a) = \frac{\text{number of times } b \text{ follows } a}{\text{number of times } a \text{ appears}}$$


## The Beautiful Equivalence

**Counting = Maximum Likelihood Estimation**

What we've proven: If you just count how often each transition occurs and divide by the total, you get the *mathematically optimal* parameters under the maximum likelihood criterion.

This isn't an approximation or heuristic. It's provably optimal.

**Why this matters**:

1. Training a Markov model is O(n) where n is corpus size—just one pass through the data. (The notation O(n), called "Big-O notation," describes how computation time grows with input size. O(n) means time grows linearly with n.)
2. No iterative optimization needed (unlike neural networks)
3. The solution is exact, not approximate

## Verifying the Solution

Let's verify with a simple example.

**Training data**: "abab"

**Bigram counts** (with START and END):
- (START, a): 1
- (a, b): 2
- (b, a): 1
- (b, END): 1

**Context totals**:

- count(START, ·) = 1
- count(a, ·) = 2
- count(b, ·) = 2

**MLE probabilities**:

- P(a | START) = 1/1 = 1.0
- P(b | a) = 2/2 = 1.0
- P(a | b) = 1/2 = 0.5
- P(END | b) = 1/2 = 0.5

**Verification**: P("abab") = P(a|START) × P(b|a) × P(a|b) × P(b|a) × P(END|b)
= 1.0 × 1.0 × 0.5 × 1.0 × 0.5 = 0.25

This is the highest probability this model can assign to "abab" given the constraints.

## Extending to Higher-Order Models

The derivation extends naturally to order-k models:

$$\theta^*_{c \rightarrow t} = \frac{\text{count}(c, t)}{\text{count}(c, \cdot)}$$


where c is a context of k tokens.

The principle is the same: count transitions and normalize.

## The Zero Probability Problem

There's a serious issue: What if we never saw a particular bigram in training?

**Example**: If "xz" never appears in training, count(x, z) = 0, so P(z | x) = 0.

This means any text containing "xz" gets probability 0, which means:

- Perplexity becomes infinite
- The model considers valid text impossible

**Solutions** (we'll explore in exercises):
- **Laplace smoothing**: Add 1 to all counts (as if we saw everything once)
- **Backoff**: If bigram unseen, use unigram probability
- **Interpolation**: Weighted combination of n-gram orders

For now, we note this limitation and move on.

!!! info "Connection to Modern LLMs"

    Neural language models like GPT-4 and Claude also use maximum likelihood training! The objective is the same: maximize P(training data | parameters).

    The difference is *how* probabilities are computed:
    - Markov: P(next|context) = count ratio (closed-form solution)
    - Neural: P(next|context) = softmax(neural_network(context))

    Since neural networks don't have a closed-form MLE solution, we use gradient descent. But the objective being optimized is the same log-likelihood we derived here. Cross-entropy loss = negative log-likelihood.

!!! note "Historical Note: Fisher and MLE (1912-1922)"

    Maximum likelihood estimation was developed by Ronald Fisher in the early 20th century. Fisher showed that MLE estimators have optimal properties: they're consistent (converge to true values), asymptotically efficient (minimum variance for large samples), and asymptotically normal.

    The use of Lagrange multipliers for constrained optimization dates back to Joseph-Louis Lagrange's work in 1788, originally developed for mechanics problems.

!!! warning "Common Mistake: Forgetting the Normalization Constraint"

    A frequent error when implementing MLE by hand:

    ❌ **Wrong**: Setting each $θ_{a→b}$ = count(a,b) and forgetting to normalize

    ✓ **Right**: $θ_{a→b}$ = count(a,b) / count(a,·)

    Without normalization, the "probabilities" won't sum to 1 and won't be valid probability distributions. The Lagrange multiplier in our derivation enforces this constraint.

## Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Count all n-grams | O(n) | O(K) |
| Look up P(b\|a) | O(1) | — |
| Compute log-likelihood | O(n) | O(1) |

Where n = corpus length, K = number of unique n-grams observed.

## Summary

| Concept | Definition | Key Insight |
|---------|------------|-------------|
| Likelihood | P(data \| θ) | How probable data is under θ |
| Log-likelihood | log P(data \| θ) | Numerically stable version |
| MLE | θ* = argmax P(data \| θ) | Choose θ that maximizes likelihood |
| MLE for Markov | count(a,b) / count(a,·) | Counting IS optimization |

**The key takeaway**: Training a Markov model by counting isn't a hack—it's the mathematically optimal solution. We *derived* this from first principles using calculus.

Next: How do we measure whether our model is good? This requires understanding information theory.

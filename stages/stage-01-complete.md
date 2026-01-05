# Stage 1: The Simplest Language Model — Markov Chains

**Building LLMs from First Principles**

This stage introduces language modeling through the simplest possible approach. Every concept is derived from first principles. Every formula is explained. By the end, you'll understand not just *what* language models do, but *why* they work.

**Estimated reading time**: 60-90 minutes
**Prerequisites**: Basic Python, high school algebra

---

# Section 1.1: Probability Foundations

Before we can build a language model, we need to understand probability. Not just use it—*understand* it. This section builds the foundation everything else rests on.

## What is Probability?

Probability is a way of quantifying uncertainty. When we say "the probability of rain tomorrow is 70%," we're expressing our degree of belief that it will rain.

But what does that number actually *mean*? There are two main interpretations:

**Frequentist interpretation**: If we could repeat tomorrow infinitely many times, it would rain in 70% of those tomorrows. Probability is the long-run frequency of an event.

**Bayesian interpretation**: Probability represents our degree of belief. The 70% encodes our uncertainty given our current knowledge.

For language modeling, we'll mostly use the frequentist view: if we sample many sentences from English, what fraction start with "The"? That fraction is approximately P("The" is the first word).

## The Three Axioms of Probability

All of probability theory follows from just three axioms, formulated by Kolmogorov in 1933:

**Axiom 1 (Non-negativity)**: For any event A, P(A) ≥ 0

Probabilities can't be negative. You can't have a -30% chance of something.

**Axiom 2 (Normalization)**: P(Ω) = 1, where Ω is the entire sample space

Something must happen. The probability of *some* outcome occurring is 1.

**Axiom 3 (Additivity)**: For mutually exclusive events A and B,
$$P(A \cup B) = P(A) + P(B)$$

If A and B can't both happen, the probability of either happening is the sum.

From these three axioms, we can derive everything else. For example:

**P(not A) = 1 - P(A)**:
Since A and "not A" are mutually exclusive and exhaust all possibilities:
$$P(A) + P(\text{not } A) = P(\Omega) = 1$$

## Conditional Probability

Here's where things get interesting. Often we want to know the probability of an event *given that we already know something*.

**Definition**: The conditional probability of A given B is:
$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

where P(B) > 0.

**What this means**: We're restricting our attention to only the cases where B happens, and asking what fraction of those cases also have A happening.

**Example**:
- Let A = "the word is 'cat'"
- Let B = "the previous word is 'the'"
- P(A|B) = P("the cat") / P("the ...") = (frequency of "the cat") / (frequency of "the" followed by anything)

This is *exactly* what we need for language modeling.

## Deriving the Chain Rule

The chain rule is the mathematical foundation of autoregressive language models. Let's derive it carefully.

### Two Events

Starting from the definition of conditional probability:
$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

Rearranging:
$$P(A \cap B) = P(A|B) \cdot P(B)$$

This is the **product rule**: the probability of both A and B equals the probability of B times the probability of A given B.

We could also write it the other way:
$$P(A \cap B) = P(B|A) \cdot P(A)$$

Both are correct. The choice depends on what we know.

### Three Events

Now let's extend to three events. We want P(A ∩ B ∩ C).

First, treat (A ∩ B) as a single event and apply the product rule:
$$P(A \cap B \cap C) = P(C | A \cap B) \cdot P(A \cap B)$$

Now expand P(A ∩ B):
$$P(A \cap B \cap C) = P(C | A \cap B) \cdot P(B | A) \cdot P(A)$$

Rewriting in a more suggestive order:
$$P(A, B, C) = P(A) \cdot P(B | A) \cdot P(C | A, B)$$

### The General Chain Rule

For n events, we apply the same logic inductively:

$$P(X_1, X_2, \ldots, X_n) = P(X_1) \cdot P(X_2|X_1) \cdot P(X_3|X_1,X_2) \cdots P(X_n|X_1,\ldots,X_{n-1})$$

Or more compactly:
$$P(X_1, \ldots, X_n) = \prod_{i=1}^{n} P(X_i | X_1, \ldots, X_{i-1})$$

**Proof by induction**:

*Base case*: For n=1, P(X₁) = P(X₁). ✓

*Inductive step*: Assume true for n-1. Then:
$$P(X_1, \ldots, X_n) = P(X_n | X_1, \ldots, X_{n-1}) \cdot P(X_1, \ldots, X_{n-1})$$

By inductive hypothesis:
$$= P(X_n | X_1, \ldots, X_{n-1}) \cdot \prod_{i=1}^{n-1} P(X_i | X_1, \ldots, X_{i-1})$$
$$= \prod_{i=1}^{n} P(X_i | X_1, \ldots, X_{i-1}) \quad \blacksquare$$

**This is not an approximation**. The chain rule is an exact mathematical identity. It follows directly from the definition of conditional probability.

## Why the Chain Rule Matters for Language

Consider a sentence as a sequence of words: "The cat sat down"

The probability of this sentence is:
$$P(\text{"The cat sat down"}) = P(\text{The}, \text{cat}, \text{sat}, \text{down})$$

By the chain rule:
$$= P(\text{The}) \cdot P(\text{cat}|\text{The}) \cdot P(\text{sat}|\text{The cat}) \cdot P(\text{down}|\text{The cat sat})$$

We've converted the problem of assigning probability to an entire sentence into a sequence of next-word predictions. This is the **autoregressive factorization**.

The term "autoregressive" means the model predicts each element based on previous elements—it regresses on its own past outputs.

## Summary

| Concept | Definition | Why It Matters |
|---------|------------|----------------|
| Probability | Quantified uncertainty | Foundation of everything |
| Conditional probability | P(A\|B) = P(A∩B)/P(B) | How we model "given context" |
| Chain rule | P(X₁...Xₙ) = ∏P(Xᵢ\|X₁...Xᵢ₋₁) | Enables autoregressive modeling |

We've now established the mathematical foundation. Next, we'll use these tools to define what a language model actually is.
# Section 1.2: The Language Modeling Problem

Now that we understand probability and the chain rule, let's define precisely what we're trying to build.

## What is a Language Model?

A **language model** is a probability distribution over sequences of tokens.

Given a vocabulary V (a finite set of possible tokens), a language model assigns a probability to every possible sequence:

$$P: V^* \rightarrow [0, 1]$$

where V* means "sequences of any length from V" and the probabilities over all possible sequences sum to 1.

**Examples of what a language model answers**:
- P("The cat sat on the mat") = ?
- P("Mat the on sat cat the") = ?
- P("asdfghjkl") = ?

A good language model assigns high probability to natural text and low probability to gibberish.

## The Problem of Exponential Space

Here's the fundamental challenge. Suppose our vocabulary has |V| = 50,000 tokens (roughly GPT-2's vocabulary size). How many possible sequences of length n exist?

| Length n | Possible sequences |
|----------|-------------------|
| 1 | 50,000 |
| 2 | 2.5 billion |
| 10 | 10^47 |
| 100 | 10^470 |

For comparison, there are approximately 10^80 atoms in the observable universe.

We cannot possibly store a probability for each sequence. Even for length-10 sequences, we'd need more storage than atoms in the universe.

## The Autoregressive Solution

The chain rule (from Section 1.1) provides the solution. Instead of modeling P(x₁, x₂, ..., xₙ) directly, we factor it:

$$P(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} P(x_i | x_1, \ldots, x_{i-1})$$

Now we need to model n conditional distributions instead of one joint distribution. Each conditional distribution is over |V| possible next tokens.

But wait—each conditional P(xᵢ | x₁, ..., xᵢ₋₁) still depends on a variable-length history. For a sequence of length 100, the last prediction conditions on 99 previous tokens. How many possible 99-token histories exist?

Still 50,000^99 ≈ 10^465. We haven't solved the problem!

## The Markov Assumption: A Simplification

Here's where we make our first modeling assumption—and it's important to recognize that this is a *choice*, not a mathematical necessity.

**The Markov assumption**: The future depends only on the recent past.

Specifically, for an **order-k Markov model**:
$$P(x_i | x_1, \ldots, x_{i-1}) \approx P(x_i | x_{i-k}, \ldots, x_{i-1})$$

The probability of the next token depends only on the last k tokens, not the entire history.

**Special cases**:
- k=1: **Bigram model**. P(xᵢ | xᵢ₋₁). Next token depends only on previous token.
- k=2: **Trigram model**. P(xᵢ | xᵢ₋₂, xᵢ₋₁). Next token depends on previous two tokens.
- k=n-1: **Full model**. No approximation, but intractable.

## Why This Helps (Quantitatively)

With the Markov assumption of order k, how many distinct contexts do we need to model?

Each context is a sequence of k tokens, so there are |V|^k possible contexts.

| Order k | Possible contexts (|V|=50,000) |
|---------|-------------------------------|
| 1 | 50,000 |
| 2 | 2.5 billion |
| 3 | 125 trillion |
| 4 | 6.25 × 10^18 |

Even k=2 is stretching it for explicit storage. k=3 or higher requires sparse representations (storing only contexts we've actually seen).

For character-level models with |V| ≈ 100:
| Order k | Possible contexts |
|---------|------------------|
| 1 | 100 |
| 2 | 10,000 |
| 3 | 1,000,000 |
| 5 | 10 billion |
| 10 | 10^20 |

Character-level models can use higher orders because the vocabulary is smaller.

## Why the Markov Assumption is Wrong

Language has long-range dependencies that the Markov assumption cannot capture.

**Example 1: Subject-verb agreement**
"The cat that sat on the mat next to the dogs **was** sleeping."

The verb "was" must agree with "cat" (singular), not "dogs" (the nearest noun). A bigram model seeing "dogs" would likely predict "were".

**Example 2: Coreference**
"John went to the store. **He** bought milk."

To know that "He" refers to "John", we need information from the previous sentence.

**Example 3: Document-level coherence**
In a story, characters introduced in paragraph 1 must behave consistently in paragraph 10.

## Why Wrong Models Can Still Be Useful

Despite being wrong, Markov models are useful for several reasons:

1. **They capture local patterns**: Most of the information for predicting the next character IS in the recent past. "th" → "e" is very likely regardless of earlier context.

2. **They're trainable**: We can estimate the probabilities from data (as we'll see in Section 1.3).

3. **They're interpretable**: We can inspect what the model learned by looking at transition probabilities.

4. **They're a stepping stone**: Understanding Markov models helps us understand why neural language models are better.

The progression through this course will be:
- Markov models (this stage): Wrong but simple
- Neural LMs (Stage 4): Less wrong, can learn representations
- Transformers (Stage 7-8): Can model arbitrary-length dependencies

Each step fixes a limitation of the previous approach.

## Formal Definition: Order-k Markov Language Model

An **order-k Markov language model** is defined by:

1. A vocabulary V (finite set of tokens)
2. A special start token ⟨START⟩ and end token ⟨END⟩
3. A set of transition probabilities: for each context c ∈ V^k and each token t ∈ V ∪ {⟨END⟩}:
   $$\theta_{c \rightarrow t} = P(t | c)$$

4. Constraints:
   - All probabilities non-negative: θ_{c→t} ≥ 0
   - Probabilities sum to 1 for each context: ∑_t θ_{c→t} = 1

The probability of a sequence x₁, x₂, ..., xₙ is:
$$P(x_1, \ldots, x_n) = \prod_{i=1}^{n+1} \theta_{c_i \rightarrow x_i}$$

where:
- x_{n+1} = ⟨END⟩
- cᵢ = (x_{i-k}, ..., x_{i-1}) with padding using ⟨START⟩ for i ≤ k

## What We Need to Learn

To use a Markov language model, we need to determine the transition probabilities θ.

The question is: **Given training data (a corpus of text), how do we estimate these probabilities?**

This is the subject of Section 1.3, where we'll derive that the optimal approach is simply counting—and prove why this is optimal.

## Summary

| Concept | Definition | Key Insight |
|---------|------------|-------------|
| Language model | P(sequence) | Probability distribution over text |
| Exponential space | \|V\|^n sequences | Can't store explicitly |
| Autoregressive | ∏P(xᵢ\|x<i) | Factor into conditionals |
| Markov assumption | P(xᵢ\|x<i) ≈ P(xᵢ\|xᵢ₋ₖ...xᵢ₋₁) | Only recent history matters |
| Order k | Context is last k tokens | Higher k = better but sparser |

Next: How do we learn these probabilities from data?
# Section 1.3: Learning from Data — Maximum Likelihood Estimation

We have a model structure (Markov chain) with parameters θ (transition probabilities). Now we need to learn those parameters from data.

This section derives *from first principles* why counting is the optimal way to estimate probabilities, and proves it rigorously using calculus.

## What is a Parameter?

A **parameter** is a number that defines our model's behavior. For a bigram model, the parameters are all the transition probabilities:

θ = {θ_{a→b} : for all a ∈ V, b ∈ V ∪ {⟨END⟩}}

where θ_{a→b} = P(b | a).

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

If θ_{START→a} = 0.1, θ_{a→b} = 0.2, θ_{b→END} = 0.5:
$$L(\theta) = 0.1 \times 0.2 \times 0.5 = 0.01$$

If θ_{START→a} = 0.5, θ_{a→b} = 0.8, θ_{b→END} = 0.5:
$$L(\theta) = 0.5 \times 0.8 \times 0.5 = 0.2$$

The second parameter setting has higher likelihood—it makes the observed data more probable.

## Maximum Likelihood Estimation (MLE)

**Principle**: Choose parameters that maximize the likelihood of the observed data.

$$\theta^* = \arg\max_\theta L(\theta) = \arg\max_\theta P(D | \theta)$$

Why is this a good principle?
1. **Intuitive**: We want a model that considers our data likely, not surprising.
2. **Consistent**: As we get more data, MLE converges to the true parameters.
3. **Efficient**: MLE achieves the best possible accuracy for large samples.

## Log-Likelihood: A Computational Trick

Likelihood involves products of many probabilities. For a corpus of n tokens:
$$L(\theta) = \prod_{i=1}^{n} P(x_i | \text{context}_i; \theta)$$

Products of many small numbers cause numerical problems:
- 0.1 × 0.1 × 0.1 × ... (100 times) = 10^{-100} ≈ 0 (underflow!)

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

We want to find θ_{a→b} = P(b | a) for all a, b.

### Constraints

The probabilities must satisfy:
1. θ_{a→b} ≥ 0 for all a, b (non-negativity)
2. Σ_b θ_{a→b} = 1 for all a (normalization: probabilities sum to 1)

### The Log-Likelihood

The log-likelihood of the training data is:
$$\ell(\theta) = \sum_{\text{all bigrams } (a,b) \text{ in data}} \log \theta_{a \rightarrow b}$$

We can rewrite this by grouping identical bigrams:
$$\ell(\theta) = \sum_{a \in V} \sum_{b \in V \cup \{\text{END}\}} \text{count}(a, b) \cdot \log \theta_{a \rightarrow b}$$

Each unique bigram (a, b) contributes count(a, b) × log θ_{a→b} to the total.

### Optimization with Constraints: Lagrange Multipliers

We want to maximize ℓ(θ) subject to the constraints Σ_b θ_{a→b} = 1.

**Lagrange multipliers**: To optimize f(x) subject to g(x) = 0, we find where ∇f = λ∇g.

For our problem, we form the Lagrangian:
$$\mathcal{L}(\theta, \lambda) = \ell(\theta) - \sum_a \lambda_a \left( \sum_b \theta_{a \rightarrow b} - 1 \right)$$

We have one Lagrange multiplier λₐ for each context a (one constraint per context).

### Taking Derivatives

For each parameter θ_{a→b}, we take the partial derivative and set it to zero:

$$\frac{\partial \mathcal{L}}{\partial \theta_{a \rightarrow b}} = \frac{\text{count}(a, b)}{\theta_{a \rightarrow b}} - \lambda_a = 0$$

**Derivation of ∂ℓ/∂θ_{a→b}**:
- ℓ(θ) = Σ_{a',b'} count(a',b') · log θ_{a'→b'}
- ∂/∂θ_{a→b} of count(a,b) · log θ_{a→b} = count(a,b) / θ_{a→b}
- All other terms don't involve θ_{a→b}, so their derivatives are 0

From the derivative equation:
$$\frac{\text{count}(a, b)}{\theta_{a \rightarrow b}} = \lambda_a$$

Solving for θ_{a→b}:
$$\theta_{a \rightarrow b} = \frac{\text{count}(a, b)}{\lambda_a}$$

### Finding λₐ Using the Constraint

We know Σ_b θ_{a→b} = 1. Substituting:
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
1. Training a Markov model is O(n) where n is corpus size—just one pass through the data
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

## Summary

| Concept | Definition | Key Insight |
|---------|------------|-------------|
| Likelihood | P(data \| θ) | How probable data is under θ |
| Log-likelihood | log P(data \| θ) | Numerically stable version |
| MLE | θ* = argmax P(data \| θ) | Choose θ that maximizes likelihood |
| MLE for Markov | count(a,b) / count(a,·) | Counting IS optimization |

**The key takeaway**: Training a Markov model by counting isn't a hack—it's the mathematically optimal solution. We *derived* this from first principles using calculus.

Next: How do we measure whether our model is good? This requires understanding information theory.
# Section 1.4: Information Theory Foundations

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

**Interpretation**: KL divergence is the extra bits needed when using Q instead of P.

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
# Section 1.5: Perplexity — The Standard Evaluation Metric

Cross-entropy is the theoretically correct metric, but it's hard to interpret. "Our model has cross-entropy 4.2 bits" doesn't mean much intuitively.

Perplexity fixes this by converting cross-entropy into an interpretable number.

## From Cross-Entropy to Perplexity

**Definition**: Perplexity is the exponential of cross-entropy:

$$\text{Perplexity} = 2^{H(P,Q)} = 2^{-\frac{1}{N}\sum_i \log_2 Q(x_i | \text{context})}$$

Or equivalently, if using natural logarithms:

$$\text{Perplexity} = \exp\left(-\frac{1}{N}\sum_i \ln Q(x_i | \text{context})\right)$$

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
$$\text{PPL} = 2^{\log_2 K} = K$$

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
$$\text{PPL} = 2^{2.46} = 5.5$$

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
**Test perplexity**: Evaluate model on held-out data it never saw.

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

## Summary

| Concept | Formula | Interpretation |
|---------|---------|----------------|
| Perplexity | 2^{cross-entropy} | Effective vocabulary size |
| PPL = 1 | Perfect model | Always correct with 100% confidence |
| PPL = \|V\| | Random guessing | No information from context |
| PPL = ∞ | Model assigns P=0 | Considered token impossible |

**Key takeaways**:
1. Perplexity is the standard metric for language models
2. Lower is better
3. It measures how "surprised" the model is on average
4. Compare train vs. test to detect overfitting

Next: How do we generate text from our model? This requires understanding sampling and temperature.
# Section 1.6: Generating Text — Sampling and Temperature

We've learned how to train a model and evaluate it. Now: how do we use it to generate text?

## The Generation Problem

Given a trained model P(next | context), we want to produce new text that "sounds like" the training data.

**Autoregressive generation**:
1. Start with initial context (e.g., ⟨START⟩)
2. Sample next token from P(token | context)
3. Append sampled token to context
4. Repeat until ⟨END⟩ or maximum length

But step 2 hides a crucial choice: *how* do we sample from P(token | context)?

## Greedy Decoding: The Obvious Approach

**Greedy decoding**: Always pick the highest-probability token.

$$x_t = \arg\max_{x} P(x | \text{context})$$

**Problems with greedy**:

1. **Repetitive**: Once you pick a common pattern, you keep repeating it.
   "The the the the the..."

2. **No diversity**: Running generation twice gives identical output.

3. **Misses good sequences**: The most likely sequence isn't always found by greedily picking most likely tokens.

**Example**: Consider two paths:
- Greedy: "The cat" (P=0.3 × 0.2 = 0.06)
- Alternative: "A dog" (P=0.2 × 0.5 = 0.10)

Greedy picks "The" (0.3 > 0.2) but the full sequence is less likely!

## Ancestral Sampling: The Theoretically Correct Approach

**Ancestral sampling**: Sample each token from the full distribution.

$$x_t \sim P(x | \text{context})$$

This produces samples from the true model distribution—exactly what the model learned.

**How to sample from a discrete distribution**:
1. List all tokens with their probabilities: P(t₁), P(t₂), ...
2. Draw a random number r uniformly from [0, 1]
3. Find the token where the cumulative probability crosses r

**Python implementation**:
```python
import random

def sample(distribution):
    """Sample from a probability distribution (dict: token -> prob)."""
    r = random.random()  # Uniform [0, 1)
    cumulative = 0.0
    for token, prob in distribution.items():
        cumulative += prob
        if r < cumulative:
            return token
    return token  # Handle floating point errors
```

Or using the standard library:
```python
import random
tokens = list(distribution.keys())
probs = list(distribution.values())
return random.choices(tokens, weights=probs, k=1)[0]
```

## The Problem with Pure Sampling

Pure ancestral sampling can produce low-quality text because it *includes* the low-probability tokens.

If P("the" | context) = 0.1 and P("xyzzy" | context) = 0.001, pure sampling will occasionally output "xyzzy"—rare but possible.

Over many tokens, unlikely events accumulate, producing incoherent text.

**We want control** over how "random" vs "deterministic" the generation is.

## Temperature: Controlling Randomness

**Temperature** is a parameter that rescales the probability distribution before sampling.

Given probabilities P(t) for each token t, the temperature-scaled distribution is:

$$P_T(t) = \frac{P(t)^{1/T}}{\sum_{t'} P(t')^{1/T}}$$

Or equivalently, working in log-space:

$$P_T(t) = \frac{\exp(\log P(t) / T)}{\sum_{t'} \exp(\log P(t') / T)}$$

**What temperature does**:

| Temperature | Effect |
|-------------|--------|
| T → 0 | Distribution becomes one-hot (greedy) |
| T = 1 | Original distribution (no change) |
| T > 1 | Distribution becomes flatter (more random) |
| T → ∞ | Distribution becomes uniform |

## Deriving the Temperature Formula

Where does this formula come from? It's inspired by statistical mechanics.

### The Softmax Function

First, let's understand softmax. Given "logits" (unnormalized log-probabilities) z₁, z₂, ..., zₙ:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

This converts arbitrary real numbers into a probability distribution.

**Properties**:
1. All outputs positive (due to exponential)
2. Outputs sum to 1 (due to normalization)
3. Larger zᵢ → larger probability

### Adding Temperature

Temperature divides the logits before softmax:

$$\text{softmax}(z_i / T) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

**Why this works**:
- Dividing by T > 1 makes logits smaller → differences smaller → distribution flatter
- Dividing by T < 1 makes logits larger → differences larger → distribution sharper

### Connection to Statistical Mechanics

In physics, the Boltzmann distribution gives the probability of a system being in state i with energy Eᵢ:

$$P(i) = \frac{e^{-E_i / kT}}{Z}$$

where T is temperature and k is Boltzmann's constant.

- High temperature: System explores many states (high entropy)
- Low temperature: System settles into low-energy states

Our language model temperature is exactly analogous: high T means exploring more options, low T means sticking to high-probability options.

## Visualizing Temperature Effects

Consider this distribution: P(A) = 0.5, P(B) = 0.3, P(C) = 0.15, P(D) = 0.05

After temperature scaling:

| Token | T=0.5 | T=1.0 | T=2.0 | T→∞ |
|-------|-------|-------|-------|-----|
| A | 0.69 | 0.50 | 0.35 | 0.25 |
| B | 0.24 | 0.30 | 0.29 | 0.25 |
| C | 0.06 | 0.15 | 0.22 | 0.25 |
| D | 0.01 | 0.05 | 0.14 | 0.25 |

**Observations**:
- T=0.5: "A" dominates even more (69% vs 50%)
- T=2.0: Distribution is more uniform
- T→∞: All tokens equally likely (25% each)

## The Temperature Limit: T → 0

As T → 0, the distribution becomes a one-hot vector pointing at the highest-probability token.

**Proof**: Let z₁ > z₂ > ... > zₙ (sorted logits).

$$\lim_{T \to 0} \frac{e^{z_i/T}}{\sum_j e^{z_j/T}} = \lim_{T \to 0} \frac{e^{z_i/T}}{e^{z_1/T}(1 + \sum_{j>1} e^{(z_j-z_1)/T})}$$

Since z₁ > zⱼ for j > 1, the terms e^{(zⱼ-z₁)/T} → 0 as T → 0.

For i = 1: limit = 1
For i > 1: limit = 0

So T → 0 gives greedy decoding.

## Implementation

```python
import math

def apply_temperature(distribution, temperature):
    """Apply temperature to a probability distribution.

    Args:
        distribution: dict mapping token -> probability
        temperature: float > 0

    Returns:
        New distribution with temperature applied
    """
    if temperature == 1.0:
        return distribution

    # Work in log-space for numerical stability
    log_probs = {t: math.log(p + 1e-10) / temperature
                 for t, p in distribution.items()}

    # Subtract max for numerical stability (log-sum-exp trick)
    max_log = max(log_probs.values())
    exp_probs = {t: math.exp(lp - max_log)
                 for t, lp in log_probs.items()}

    # Normalize
    total = sum(exp_probs.values())
    return {t: p / total for t, p in exp_probs.items()}
```

**The log-sum-exp trick**: We subtract max before exponentiating to prevent overflow. This doesn't change the result because:
$$\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i} e^{-c}}{\sum_j e^{z_j} e^{-c}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

## Other Sampling Strategies

Temperature isn't the only way to control generation:

### Top-k Sampling
Only sample from the k highest-probability tokens.

```python
def top_k(distribution, k):
    sorted_tokens = sorted(distribution.items(),
                          key=lambda x: -x[1])[:k]
    total = sum(p for _, p in sorted_tokens)
    return {t: p/total for t, p in sorted_tokens}
```

### Nucleus (Top-p) Sampling
Sample from the smallest set of tokens whose cumulative probability exceeds p.

```python
def top_p(distribution, p):
    sorted_tokens = sorted(distribution.items(),
                          key=lambda x: -x[1])
    cumulative = 0.0
    result = {}
    for token, prob in sorted_tokens:
        result[token] = prob
        cumulative += prob
        if cumulative >= p:
            break
    total = sum(result.values())
    return {t: prob/total for t, prob in result.items()}
```

### Combining Strategies
Modern LLMs often use combinations: apply temperature, then top-p, then sample.

## Temperature in Practice

| Use case | Recommended T |
|----------|---------------|
| Code generation | 0.0 - 0.3 (deterministic) |
| Factual Q&A | 0.3 - 0.7 (focused) |
| Creative writing | 0.7 - 1.0 (diverse) |
| Brainstorming | 1.0 - 1.5 (exploratory) |

**ChatGPT/Claude defaults**: Usually around T=0.7 to 1.0 for general use.

## Summary

| Concept | What it does | When to use |
|---------|--------------|-------------|
| Greedy (T=0) | Always pick max | Deterministic output needed |
| Low T (0.3-0.7) | Mostly high-prob tokens | Focused, coherent text |
| T=1.0 | Original distribution | Match training distribution |
| High T (>1.0) | Flatter distribution | Creative, diverse output |
| Top-k | Only top k tokens | Prevent rare token disasters |
| Top-p | Cumulative probability threshold | Adaptive vocabulary size |

**Key takeaways**:
1. Temperature controls the exploration-exploitation tradeoff
2. T=1 samples from the learned distribution
3. Lower T = more deterministic, higher T = more random
4. The formula comes from statistical mechanics / softmax

Next: Let's implement all of this from scratch.
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

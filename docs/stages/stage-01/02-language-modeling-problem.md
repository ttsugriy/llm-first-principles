# Section 1.2: The Language Modeling Problem

Now that we understand probability and the chain rule, let's define precisely what we're trying to build.

## What is a Language Model?

A **language model** is a probability distribution over sequences of tokens.

Given a vocabulary V (a finite set of possible tokens), a language model assigns a probability to every possible sequence:

\[P: V^* \rightarrow [0, 1]\]

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

\[P(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} P(x_i | x_1, \ldots, x_{i-1})\]

Now we need to model n conditional distributions instead of one joint distribution. Each conditional distribution is over |V| possible next tokens.

But wait—each conditional P(xᵢ | x₁, ..., xᵢ₋₁) still depends on a variable-length history. For a sequence of length 100, the last prediction conditions on 99 previous tokens. How many possible 99-token histories exist?

Still 50,000^99 ≈ 10^465. We haven't solved the problem!

## The Markov Assumption: A Simplification

Here's where we make our first modeling assumption—and it's important to recognize that this is a *choice*, not a mathematical necessity.

**The Markov assumption**: The future depends only on the recent past.

Specifically, for an **order-k Markov model**:
\[P(x_i | x_1, \ldots, x_{i-1}) \approx P(x_i | x_{i-k}, \ldots, x_{i-1})\]

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
   \[\theta_{c \rightarrow t} = P(t | c)\]

4. Constraints:
   - All probabilities non-negative: θ_{c→t} ≥ 0
   - Probabilities sum to 1 for each context: ∑_t θ_{c→t} = 1

The probability of a sequence x₁, x₂, ..., xₙ is:
\[P(x_1, \ldots, x_n) = \prod_{i=1}^{n+1} \theta_{c_i \rightarrow x_i}\]

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

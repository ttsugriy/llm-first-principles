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
\[P(A \cup B) = P(A) + P(B)\]

If A and B can't both happen, the probability of either happening is the sum.

From these three axioms, we can derive everything else. For example:

**P(not A) = 1 - P(A)**:
Since A and "not A" are mutually exclusive and exhaust all possibilities:
\[P(A) + P(\text{not } A) = P(\Omega) = 1\]

## Conditional Probability

Here's where things get interesting. Often we want to know the probability of an event *given that we already know something*.

**Definition**: The conditional probability of A given B is:
\[P(A|B) = \frac{P(A \cap B)}{P(B)}\]

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
\[P(A|B) = \frac{P(A \cap B)}{P(B)}\]

Rearranging:
\[P(A \cap B) = P(A|B) \cdot P(B)\]

This is the **product rule**: the probability of both A and B equals the probability of B times the probability of A given B.

We could also write it the other way:
\[P(A \cap B) = P(B|A) \cdot P(A)\]

Both are correct. The choice depends on what we know.

### Three Events

Now let's extend to three events. We want P(A ∩ B ∩ C).

First, treat (A ∩ B) as a single event and apply the product rule:
\[P(A \cap B \cap C) = P(C | A \cap B) \cdot P(A \cap B)\]

Now expand P(A ∩ B):
\[P(A \cap B \cap C) = P(C | A \cap B) \cdot P(B | A) \cdot P(A)\]

Rewriting in a more suggestive order:
\[P(A, B, C) = P(A) \cdot P(B | A) \cdot P(C | A, B)\]

### The General Chain Rule

For n events, we apply the same logic inductively:

\[P(X_1, X_2, \ldots, X_n) = P(X_1) \cdot P(X_2|X_1) \cdot P(X_3|X_1,X_2) \cdots P(X_n|X_1,\ldots,X_{n-1})\]

Or more compactly:
\[P(X_1, \ldots, X_n) = \prod_{i=1}^{n} P(X_i | X_1, \ldots, X_{i-1})\]

**Proof by induction**:

*Base case*: For n=1, P(X₁) = P(X₁). ✓

*Inductive step*: Assume true for n-1. Then:
\[P(X_1, \ldots, X_n) = P(X_n | X_1, \ldots, X_{n-1}) \cdot P(X_1, \ldots, X_{n-1})\]

By inductive hypothesis:
\[= P(X_n | X_1, \ldots, X_{n-1}) \cdot \prod_{i=1}^{n-1} P(X_i | X_1, \ldots, X_{i-1})\]
\[= \prod_{i=1}^{n} P(X_i | X_1, \ldots, X_{i-1}) \quad \blacksquare\]

**This is not an approximation**. The chain rule is an exact mathematical identity. It follows directly from the definition of conditional probability.

## Why the Chain Rule Matters for Language

Consider a sentence as a sequence of words: "The cat sat down"

The probability of this sentence is:
\[P(\text{"The cat sat down"}) = P(\text{The}, \text{cat}, \text{sat}, \text{down})\]

By the chain rule:
\[= P(\text{The}) \cdot P(\text{cat}|\text{The}) \cdot P(\text{sat}|\text{The cat}) \cdot P(\text{down}|\text{The cat sat})\]

We've converted the problem of assigning probability to an entire sentence into a sequence of next-word predictions. This is the **autoregressive factorization**.

The term "autoregressive" means the model predicts each element based on previous elements—it regresses on its own past outputs.

## Summary

| Concept | Definition | Why It Matters |
|---------|------------|----------------|
| Probability | Quantified uncertainty | Foundation of everything |
| Conditional probability | P(A\|B) = P(A∩B)/P(B) | How we model "given context" |
| Chain rule | P(X₁...Xₙ) = ∏P(Xᵢ\|X₁...Xᵢ₋₁) | Enables autoregressive modeling |

We've now established the mathematical foundation. Next, we'll use these tools to define what a language model actually is.

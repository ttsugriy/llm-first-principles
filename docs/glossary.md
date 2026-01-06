# Glossary

A reference for key terms used throughout this book. Terms are organized alphabetically.

---

## A

### Attention
A mechanism that allows a model to focus on different parts of the input when producing each output. Computes weighted combinations of values based on query-key similarity. Foundation of transformer architecture.

### Autograd (Automatic Differentiation)
A technique for computing derivatives of functions expressed as computer programs. Builds a computational graph during forward pass and traverses it backward to compute gradients. Covered in Stage 2.

### Autoregressive
A model that predicts each element conditioned on all previous elements. Formally: P(x₁, ..., xₙ) = ∏ᵢ P(xᵢ | x₁, ..., xᵢ₋₁). All modern LLMs (GPT, Claude, LLaMA) are autoregressive.

---

## B

### Backpropagation
The algorithm for computing gradients in neural networks by applying the chain rule from outputs to inputs. A specific instance of reverse-mode automatic differentiation.

### Bigram
A sequence of two adjacent tokens. A bigram model predicts P(next | previous), using only one token of context.

### Bits per Character (BPC)
Cross-entropy measured in base-2 logarithms. Directly interpretable as the number of bits needed to encode each character. Shannon estimated English has ~1 bit/character of entropy.

---

## C

### Chain Rule (Probability)
The identity P(A, B, C) = P(A) · P(B|A) · P(C|A,B). Generalizes to n events. Foundation of autoregressive modeling.

### Chain Rule (Calculus)
The identity d/dx[f(g(x))] = f'(g(x)) · g'(x). Foundation of backpropagation.

### Computational Graph
A directed acyclic graph representing a computation, where nodes are operations and edges are data dependencies. Enables automatic differentiation.

### Context Window
The maximum number of tokens a model can "see" when making predictions. Markov models have fixed, small context windows. Transformers have large but still finite windows (4K, 8K, 128K+ tokens).

### Cross-Entropy
A measure of the difference between two probability distributions:
H(p, q) = -∑ p(x) log q(x). In language modeling, p is the true distribution (empirical from data) and q is the model. Lower is better.

---

## D

### Dirichlet Distribution
A probability distribution over probability distributions. Used as a prior in Bayesian smoothing. Parameter α controls how much we trust prior beliefs vs. observed data.

---

## E

### Embedding
A learned vector representation of a discrete token. Maps tokens from a sparse one-hot space to a dense continuous space where similar tokens have similar vectors.

### Entropy
A measure of uncertainty in a probability distribution: H(p) = -∑ p(x) log p(x). Maximum when uniform, minimum (zero) when deterministic.

---

## F

### FLOP (Floating Point Operation)
A single arithmetic operation on floating-point numbers. Used to measure computational cost. Modern LLMs require billions to trillions of FLOPs per token.

### Forward Pass
Computing the output of a model given inputs, following the direction of the computational graph from inputs to outputs.

---

## G

### Gradient
The vector of partial derivatives of a function with respect to its inputs. Points in the direction of steepest ascent. Used to update parameters in training.

### Gradient Descent
An optimization algorithm that iteratively updates parameters by moving in the direction opposite to the gradient: θ ← θ - η∇L(θ), where η is the learning rate.

---

## L

### Laplace Smoothing
A technique to avoid zero probabilities by adding a constant (usually 1) to all counts:
P(b|a) = (count(a,b) + 1) / (count(a,·) + |V|). Also called add-one smoothing.

### Likelihood
The probability of observed data given model parameters: L(θ) = P(data | θ). Used to evaluate how well parameters explain the data.

### Log-Likelihood
The natural logarithm of likelihood: ℓ(θ) = log P(data | θ). Converts products to sums, avoiding numerical underflow.

### Loss Function
A function that measures how wrong a model's predictions are. Training minimizes the loss. Cross-entropy is the standard loss for language models.

---

## M

### Markov Assumption
The assumption that future states depend only on a fixed number of recent states, not the entire history: P(xₜ | x₁...xₜ₋₁) = P(xₜ | xₜ₋ₖ...xₜ₋₁).

### Maximum Likelihood Estimation (MLE)
Finding parameters that maximize the probability of observed data: θ* = argmax P(data | θ). For Markov models, MLE = counting and normalizing.

---

## N

### N-gram
A contiguous sequence of n tokens. Unigram (n=1), bigram (n=2), trigram (n=3), etc.

### Normalization
Ensuring probabilities sum to 1. The denominator Z in P(x) = exp(f(x)) / Z is the normalization constant.

---

## P

### Perplexity
The exponentiated cross-entropy: PPL = exp(H(p, q)). Interpretable as the "effective vocabulary size"—a perplexity of 100 means the model is as uncertain as if choosing uniformly among 100 equally likely tokens. Lower is better.

### Prior (Bayesian)
A probability distribution representing beliefs before observing data. In smoothing, the prior expresses our belief that all tokens are somewhat likely.

### Posterior (Bayesian)
The updated probability distribution after combining prior beliefs with observed data. Posterior ∝ likelihood × prior.

---

## R

### Regularization
Techniques to prevent overfitting by adding constraints or penalties during training. L2 regularization, dropout, and smoothing are examples.

### Reverse Mode Differentiation
Computing gradients by traversing the computational graph from outputs to inputs. Efficient when there are many inputs and few outputs (as in neural network training).

---

## S

### Softmax
A function that converts a vector of real numbers to a probability distribution:
softmax(xᵢ) = exp(xᵢ) / ∑ⱼ exp(xⱼ). Ensures outputs are positive and sum to 1.

### Sparsity
The property of having mostly zero (or near-zero) values. High-order Markov models suffer from sparse observations—most n-grams never appear in training.

---

## T

### Temperature
A parameter that controls the "sharpness" of a probability distribution during sampling:
Pₜ(x) ∝ P(x)^(1/T). T→0 approaches argmax (greedy), T→∞ approaches uniform, T=1 is unchanged.

### Token
The basic unit of text that a language model operates on. Can be characters, words, or subwords (BPE tokens).

### Transformer
The dominant architecture for modern LLMs. Uses self-attention to process entire sequences in parallel. Introduced in "Attention Is All You Need" (2017).

---

## U

### Underflow
When a floating-point number becomes too small to represent (approaches zero). Avoided by working in log-space.

---

## V

### Vocabulary
The set of all unique tokens a model can handle. Denoted |V|. Modern LLMs have vocabularies of 32K-128K tokens.

---

## Reference Table

| Symbol | Meaning |
|--------|---------|
| P(A) | Probability of event A |
| P(A\|B) | Probability of A given B |
| θ | Model parameters |
| |V| | Vocabulary size |
| n | Sequence length |
| k | Context size (Markov order) |
| H(p) | Entropy of distribution p |
| H(p,q) | Cross-entropy between p and q |
| PPL | Perplexity |
| ∇f | Gradient of function f |
| η | Learning rate |
| α | Smoothing parameter |

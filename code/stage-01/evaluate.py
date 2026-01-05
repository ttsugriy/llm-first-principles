"""
Evaluation Metrics for Language Models

This module implements perplexity, the standard evaluation metric for
language models.

Key concepts:
- Cross-entropy: Average surprise under the model
- Perplexity: exp(cross-entropy) - interpretable as "effective vocabulary size"
"""

import math
from typing import List, Tuple

from markov import MarkovChain


def compute_log_probability(model: MarkovChain, tokens: List[str]) -> Tuple[float, int]:
    """
    Compute total log probability of a sequence under the model.

    Uses the chain rule:
    log P(x_1, ..., x_n) = sum_{i=1}^{n} log P(x_i | x_{i-k}, ..., x_{i-1})

    Args:
        model: Trained MarkovChain
        tokens: Sequence to evaluate

    Returns:
        Tuple of (total_log_prob, num_tokens)
        Returns (-inf, n) if any token has zero probability
    """
    padded = ['<START>'] * model.order + tokens + ['<END>']

    log_prob_sum = 0.0
    n_tokens = 0

    for i in range(len(padded) - model.order):
        history = tuple(padded[i:i + model.order])
        next_token = padded[i + model.order]

        prob = model.probability(history, next_token)

        if prob == 0:
            return float('-inf'), len(tokens) + 1

        log_prob_sum += math.log(prob)
        n_tokens += 1

    return log_prob_sum, n_tokens


def compute_cross_entropy(model: MarkovChain, tokens: List[str]) -> float:
    """
    Compute cross-entropy of the model on a sequence.

    Cross-entropy measures the average number of bits needed to encode
    tokens from the empirical distribution using the model distribution:

    H(p, q) = -1/N * sum_{i=1}^{N} log q(x_i)

    where p is the true distribution (empirical from data) and q is the model.

    Lower cross-entropy means the model assigns higher probability to the data.

    Args:
        model: Trained MarkovChain
        tokens: Sequence to evaluate

    Returns:
        Cross-entropy in nats (natural log units).
        Returns inf if any token has zero probability.
    """
    log_prob_sum, n_tokens = compute_log_probability(model, tokens)

    if log_prob_sum == float('-inf'):
        return float('inf')

    return -log_prob_sum / n_tokens


def compute_perplexity(model: MarkovChain, tokens: List[str]) -> float:
    """
    Compute perplexity of the model on a sequence.

    Perplexity = exp(cross-entropy) = exp(-1/N * sum log P(x_i | context))

    Intuitive interpretation:
    - If perplexity = 50, the model is as uncertain as choosing uniformly
      among 50 equally likely tokens at each position.
    - A random model over vocabulary size |V| has perplexity |V|.
    - A perfect model that always predicts correctly has perplexity 1.

    Lower perplexity is better!

    Args:
        model: Trained MarkovChain
        tokens: Sequence to evaluate

    Returns:
        Perplexity (>= 1 for valid models).
        Returns inf if any token has zero probability.

    Example:
        >>> model = MarkovChain(order=1)
        >>> model.train(list("aaaa"))
        >>> compute_perplexity(model, list("aa"))  # doctest: +SKIP
        1.0  # Perfect prediction
    """
    cross_entropy = compute_cross_entropy(model, tokens)

    if cross_entropy == float('inf'):
        return float('inf')

    return math.exp(cross_entropy)


def compute_bits_per_character(model: MarkovChain, tokens: List[str]) -> float:
    """
    Compute bits per character (bpc).

    This is cross-entropy converted to base-2 logarithms:
    bpc = cross_entropy / log(2)

    Often used for character-level models as it directly gives
    the average number of bits needed to encode each character.

    Args:
        model: Trained MarkovChain
        tokens: Sequence to evaluate

    Returns:
        Bits per character. Returns inf if any token has zero probability.
    """
    cross_entropy = compute_cross_entropy(model, tokens)

    if cross_entropy == float('inf'):
        return float('inf')

    return cross_entropy / math.log(2)


def evaluate_train_test_split(
    model: MarkovChain,
    tokens: List[str],
    train_ratio: float = 0.8
) -> Tuple[float, float]:
    """
    Train model and evaluate on train/test split.

    This is the standard way to detect overfitting:
    - If train_ppl << test_ppl, the model is memorizing, not generalizing.

    Args:
        model: MarkovChain to train
        tokens: Full sequence of tokens
        train_ratio: Fraction of data to use for training

    Returns:
        Tuple of (train_perplexity, test_perplexity)
    """
    split_idx = int(len(tokens) * train_ratio)
    train_tokens = tokens[:split_idx]
    test_tokens = tokens[split_idx:]

    model.train(train_tokens)

    train_ppl = compute_perplexity(model, train_tokens)
    test_ppl = compute_perplexity(model, test_tokens)

    return train_ppl, test_ppl


if __name__ == "__main__":
    # Quick test
    from markov import MarkovChain

    text = "to be or not to be that is the question whether tis nobler in the mind"
    tokens = list(text)

    # Train/test split
    split = int(len(tokens) * 0.8)
    train = tokens[:split]
    test = tokens[split:]

    print("Perplexity vs Order:")
    print("-" * 50)

    for order in range(1, 8):
        model = MarkovChain(order=order)
        model.train(train)

        train_ppl = compute_perplexity(model, train)
        test_ppl = compute_perplexity(model, test)

        print(f"Order {order}: Train PPL = {train_ppl:6.2f}, Test PPL = {test_ppl:6.2f}, States = {model.num_states():5d}")

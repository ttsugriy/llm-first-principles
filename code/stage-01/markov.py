"""
Markov Chain Language Model

Stage 1: The Simplest Language Model

This module implements an n-gram language model using the Markov assumption:
P(x_i | x_1, ..., x_{i-1}) ≈ P(x_i | x_{i-k}, ..., x_{i-1})

Key insight: Training = counting = maximum likelihood estimation.
"""

from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import math
import random


class MarkovChain:
    """
    N-gram language model using the Markov assumption.

    The model learns transition probabilities by counting n-grams in training data.
    This is equivalent to maximum likelihood estimation for categorical distributions.

    Attributes:
        order: Number of previous tokens to condition on (1 = bigram, 2 = trigram, etc.)
        counts: Dictionary mapping history tuples to Counter of next tokens
        totals: Dictionary mapping history tuples to total count
        vocab: Set of all tokens seen during training
    """

    def __init__(self, order: int = 1):
        """
        Initialize Markov chain.

        Args:
            order: Number of previous tokens to condition on.
                   1 = bigram (uses 1 previous token)
                   2 = trigram (uses 2 previous tokens)
        """
        if order < 1:
            raise ValueError(f"Order must be >= 1, got {order}")

        self.order = order
        self.counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.totals: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.vocab: set = set()

    def train(self, tokens: List[str]) -> None:
        """
        Train on a sequence of tokens by counting transitions.

        This is maximum likelihood estimation: we're finding the parameters θ that
        maximize P(data | θ). For categorical distributions, the MLE solution is
        simply the normalized counts.

        Args:
            tokens: List of tokens (characters or words)
        """
        # Add special start/end tokens
        # START tokens allow the model to learn how sequences begin
        # END token allows the model to learn when to stop generating
        padded = ['<START>'] * self.order + tokens + ['<END>']

        # Count all n-grams
        for i in range(len(padded) - self.order):
            history = tuple(padded[i:i + self.order])
            next_token = padded[i + self.order]

            self.counts[history][next_token] += 1
            self.totals[history] += 1
            self.vocab.add(next_token)

    def probability(self, history: Tuple[str, ...], next_token: str) -> float:
        """
        Get probability P(next_token | history).

        This is the normalized count: count(history, next_token) / count(history, *)

        Args:
            history: Tuple of previous tokens (must have length == self.order)
            next_token: Token to get probability for

        Returns:
            Probability in [0, 1]. Returns 0 if history was never seen.
        """
        if history not in self.counts:
            return 0.0
        return self.counts[history][next_token] / self.totals[history]

    def get_distribution(self, history: Tuple[str, ...]) -> Dict[str, float]:
        """
        Get full probability distribution given history.

        Args:
            history: Tuple of previous tokens

        Returns:
            Dictionary mapping tokens to their probabilities.
            Empty dict if history was never seen.
        """
        if history not in self.counts:
            return {}
        total = self.totals[history]
        return {
            token: count / total
            for token, count in self.counts[history].items()
        }

    def num_states(self) -> int:
        """Return number of unique history states seen during training."""
        return len(self.counts)

    def __repr__(self) -> str:
        return f"MarkovChain(order={self.order}, states={self.num_states()}, vocab={len(self.vocab)})"


class SmoothedMarkovChain(MarkovChain):
    """
    Markov chain with Laplace (add-one) smoothing.

    Smoothing prevents zero probabilities for unseen n-grams:
    P(b|a) = (count(a,b) + α) / (count(a,·) + α|V|)

    This is equivalent to placing a uniform Dirichlet prior on the transition
    probabilities.
    """

    def __init__(self, order: int = 1, alpha: float = 1.0):
        """
        Initialize smoothed Markov chain.

        Args:
            order: Number of previous tokens to condition on
            alpha: Smoothing parameter (1.0 = add-one/Laplace smoothing)
        """
        super().__init__(order)
        self.alpha = alpha

    def probability(self, history: Tuple[str, ...], next_token: str) -> float:
        """
        Get smoothed probability P(next_token | history).

        Uses Laplace smoothing to handle unseen n-grams.
        """
        count = self.counts[history][next_token]
        total = self.totals[history]
        vocab_size = len(self.vocab)

        # Laplace smoothing formula
        return (count + self.alpha) / (total + self.alpha * vocab_size)

    def get_distribution(self, history: Tuple[str, ...]) -> Dict[str, float]:
        """Get smoothed probability distribution over all vocabulary."""
        total = self.totals[history]
        vocab_size = len(self.vocab)

        result = {}
        for token in self.vocab:
            count = self.counts[history][token]
            result[token] = (count + self.alpha) / (total + self.alpha * vocab_size)

        return result

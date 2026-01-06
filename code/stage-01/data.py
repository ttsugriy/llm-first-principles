"""
Data Loading and Preprocessing

Utilities for loading, tokenizing, and preparing text for language modeling.

Key principles:
- Reproducibility: All random operations use explicit seeds
- Simplicity: Standard library only (no external dependencies)
- Flexibility: Support both character and word tokenization
"""

from typing import List, Tuple, Dict, Optional
from collections import Counter
import os
import re
import random


# ============================================================================
# Loading
# ============================================================================

def load_text(filepath: str) -> str:
    """
    Load text file and return as string.

    Args:
        filepath: Path to text file

    Returns:
        Contents of file as string
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def preprocess(text: str, lowercase: bool = True, strip: bool = True) -> str:
    """
    Basic text preprocessing.

    Args:
        text: Raw text
        lowercase: Convert to lowercase
        strip: Remove leading/trailing whitespace

    Returns:
        Preprocessed text
    """
    if strip:
        text = text.strip()
    if lowercase:
        text = text.lower()
    return text


def tokenize_chars(text: str) -> List[str]:
    """
    Character-level tokenization.

    This is the simplest tokenization: each character is a token.
    Vocabulary size = number of unique characters.

    Args:
        text: Input text

    Returns:
        List of single-character tokens

    Example:
        >>> tokenize_chars("hello")
        ['h', 'e', 'l', 'l', 'o']
    """
    return list(text)


def tokenize_words(text: str) -> List[str]:
    """
    Simple word-level tokenization (split on whitespace).

    This is a naive approach - production systems use proper tokenizers
    that handle punctuation, contractions, etc.

    Args:
        text: Input text

    Returns:
        List of word tokens

    Example:
        >>> tokenize_words("hello world")
        ['hello', 'world']
    """
    return text.split()


# Sample texts for experiments
SAMPLE_SHAKESPEARE = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream: ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life.
"""

SAMPLE_CODE = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
"""


def get_sample_data(name: str = "shakespeare") -> str:
    """
    Get sample text data for experiments.

    Args:
        name: "shakespeare" or "code"

    Returns:
        Sample text
    """
    samples = {
        "shakespeare": SAMPLE_SHAKESPEARE,
        "code": SAMPLE_CODE,
    }
    return samples.get(name, SAMPLE_SHAKESPEARE)


# ============================================================================
# Data Splitting
# ============================================================================

def train_test_split(
    tokens: List[str],
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Split tokens into training and test sets.

    Uses a positional split (not random) to preserve sequence structure.
    For language modeling, we want contiguous sequences, not shuffled tokens.

    Args:
        tokens: List of tokens
        test_ratio: Fraction of data for testing (0.0 to 1.0)
        seed: Random seed (kept for API consistency)

    Returns:
        Tuple of (train_tokens, test_tokens)

    Example:
        >>> tokens = list("abcdefghij")
        >>> train, test = train_test_split(tokens, test_ratio=0.2)
        >>> len(train), len(test)
        (8, 2)
    """
    if not 0.0 <= test_ratio <= 1.0:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")

    split_idx = int(len(tokens) * (1 - test_ratio))
    return tokens[:split_idx], tokens[split_idx:]


def train_val_test_split(
    tokens: List[str],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split tokens into training, validation, and test sets.

    Validation set: for hyperparameter tuning (e.g., choosing order).
    Test set: held out for final evaluation only.

    Args:
        tokens: List of tokens
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        Tuple of (train_tokens, val_tokens, test_tokens)
    """
    total_held_out = val_ratio + test_ratio
    if total_held_out >= 1.0:
        raise ValueError(f"val_ratio + test_ratio must be < 1.0")

    n = len(tokens)
    train_end = int(n * (1 - total_held_out))
    val_end = int(n * (1 - test_ratio))

    return tokens[:train_end], tokens[train_end:val_end], tokens[val_end:]


# ============================================================================
# Vocabulary Analysis
# ============================================================================

def compute_vocab_stats(tokens: List[str]) -> Dict:
    """
    Compute vocabulary statistics for a token sequence.

    Useful for understanding data before training.

    Args:
        tokens: List of tokens

    Returns:
        Dictionary with vocabulary statistics

    Example:
        >>> stats = compute_vocab_stats(list("hello"))
        >>> stats['vocab_size']
        4
    """
    counts = Counter(tokens)
    vocab_size = len(counts)
    total_tokens = len(tokens)

    # Frequency distribution
    freq_dist = sorted(counts.values(), reverse=True)

    # Compute coverage: how many tokens cover X% of text
    cumsum = 0
    coverage = {}
    for threshold in [0.5, 0.9, 0.99]:
        for i, count in enumerate(freq_dist):
            cumsum += count
            if cumsum / total_tokens >= threshold:
                coverage[threshold] = i + 1
                break

    return {
        'vocab_size': vocab_size,
        'total_tokens': total_tokens,
        'type_token_ratio': vocab_size / total_tokens if total_tokens > 0 else 0,
        'most_common': counts.most_common(10),
        'coverage': coverage,  # tokens needed to cover X% of text
    }


# ============================================================================
# Preprocessing Pipeline
# ============================================================================

def prepare_data(
    text: str,
    tokenize: str = 'char',
    lowercase: bool = True,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[str], List[str], Dict]:
    """
    Complete data preparation pipeline.

    Combines preprocessing, tokenization, splitting, and analysis.

    Args:
        text: Raw input text
        tokenize: 'char' for characters, 'word' for words
        lowercase: Whether to lowercase
        test_ratio: Fraction for test set
        seed: Random seed

    Returns:
        Tuple of (train_tokens, test_tokens, stats_dict)

    Example:
        >>> train, test, stats = prepare_data("Hello world!")
        >>> stats['vocab_size']
        10
    """
    # Preprocess
    text = preprocess(text, lowercase=lowercase)

    # Tokenize
    if tokenize == 'char':
        tokens = tokenize_chars(text)
    elif tokenize == 'word':
        tokens = tokenize_words(text)
    else:
        raise ValueError(f"Unknown tokenization: {tokenize}")

    # Split
    train_tokens, test_tokens = train_test_split(tokens, test_ratio, seed)

    # Compute stats
    stats = compute_vocab_stats(tokens)
    stats['tokenization'] = tokenize
    stats['train_size'] = len(train_tokens)
    stats['test_size'] = len(test_tokens)

    return train_tokens, test_tokens, stats


if __name__ == "__main__":
    # Demo
    print("Data Utilities Demo")
    print("=" * 50)

    text = get_sample_data('shakespeare')
    print(f"Sample text ({len(text)} characters):")
    print(text[:100] + "...")
    print()

    # Character-level
    train, test, stats = prepare_data(text, tokenize='char')
    print("Character-level tokenization:")
    print(f"  Vocabulary size: {stats['vocab_size']}")
    print(f"  Train tokens: {stats['train_size']}")
    print(f"  Test tokens: {stats['test_size']}")
    print(f"  Most common: {stats['most_common'][:5]}")
    print()

    # Word-level
    train, test, stats = prepare_data(text, tokenize='word')
    print("Word-level tokenization:")
    print(f"  Vocabulary size: {stats['vocab_size']}")
    print(f"  Train tokens: {stats['train_size']}")
    print(f"  Test tokens: {stats['test_size']}")
    print(f"  Most common: {stats['most_common'][:5]}")

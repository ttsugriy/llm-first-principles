"""
Data Loading and Preprocessing

Utilities for loading and tokenizing text for language modeling experiments.
"""

from typing import List
import os


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

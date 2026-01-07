"""
Capstone: End-to-End Transformer Training

This module provides a complete, trainable transformer language model
that ties together all concepts from Stages 1-6.
"""

from .model import (
    TrainableTransformer,
    CharTokenizer,
    cross_entropy_loss,
    compute_perplexity,
    Parameter,
    RMSNorm,
    Embedding,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
)

__all__ = [
    'TrainableTransformer',
    'CharTokenizer',
    'cross_entropy_loss',
    'compute_perplexity',
    'Parameter',
    'RMSNorm',
    'Embedding',
    'MultiHeadAttention',
    'FeedForward',
    'TransformerBlock',
]

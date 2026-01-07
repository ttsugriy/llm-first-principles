# Stage 7: Tokenization

*The bridge between raw text and neural networks*

## Overview

Tokenization is the process of converting raw text into discrete units (tokens) that neural networks can process. While it may seem like a preprocessing detail, the choice of tokenization scheme profoundly affects model performance, efficiency, and capabilities.

In this stage, we'll derive and implement three major tokenization algorithms from first principles:

1. **Byte Pair Encoding (BPE)** - Used by GPT-2, GPT-3, GPT-4
2. **WordPiece** - Used by BERT
3. **Unigram Language Model** - Used by SentencePiece

## Why Tokenization Matters

Consider the sentence: "The transformer architecture is revolutionary."

How should we split this for a neural network?

| Approach | Tokens | Count |
|----------|--------|-------|
| Words | ["The", "transformer", "architecture", "is", "revolutionary", "."] | 6 |
| Characters | ["T", "h", "e", " ", "t", "r", "a", ...] | 45 |
| Subwords | ["The", " transform", "er", " architecture", " is", " revolution", "ary", "."] | 8 |

Each approach has trade-offs:

- **Word-level**: Small sequences, but huge vocabulary and can't handle new words
- **Character-level**: Tiny vocabulary, but very long sequences and must learn spelling
- **Subword-level**: Balances vocabulary size with sequence length

## The Fundamental Trade-off

$$\text{Sequence Length} \times \text{Vocabulary Size} \approx \text{constant}$$

- Larger vocabulary → shorter sequences → faster attention (O(n²))
- Smaller vocabulary → longer sequences → slower but more flexible

Modern LLMs use subword tokenization with vocabularies of 32K-100K tokens.

## Learning Objectives

By the end of this stage, you will:

1. Understand why subword tokenization dominates modern NLP
2. Derive the BPE algorithm from first principles
3. Understand how WordPiece differs from BPE
4. Implement a Unigram tokenizer
5. Analyze the trade-offs in vocabulary size selection

## Sections

1. [The Tokenization Problem](01-tokenization-problem.md) - Why this is hard
2. [Character vs. Subword](02-character-vs-subword.md) - The design space
3. [Byte Pair Encoding](03-bpe.md) - The algorithm behind GPT
4. [WordPiece](04-wordpiece.md) - The algorithm behind BERT
5. [Unigram Language Model](05-unigram.md) - A probabilistic approach
6. [Vocabulary Size Trade-offs](06-vocabulary-size.md) - How to choose
7. [Implementation](07-implementation.md) - Building tokenizers from scratch

## Prerequisites

- Understanding of n-gram language models (Stage 1)
- Basic probability (Stage 1)
- Familiarity with the attention mechanism (Stage 5) to understand sequence length trade-offs

## Key Insight

> Tokenization is not just preprocessing—it defines the atomic units of meaning that your model can learn. A good tokenizer creates tokens that correspond to meaningful linguistic units while keeping the vocabulary tractable.

## Code & Resources

| Resource | Description |
|----------|-------------|
| [`code/stage-07/tokenizer.py`](https://github.com/ttsugriy/llm-first-principles/blob/main/code/stage-07/tokenizer.py) | BPE and tokenizer implementations |
| [`code/stage-07/tests/`](https://github.com/ttsugriy/llm-first-principles/tree/main/code/stage-07/tests) | Test suite |
| [Exercises](exercises.md) | Practice problems |
| [Common Mistakes](common-mistakes.md) | Debugging guide |

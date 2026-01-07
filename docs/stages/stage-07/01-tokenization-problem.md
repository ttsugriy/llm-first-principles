# Section 7.1: The Tokenization Problem

*Reading time: 10 minutes*

## The Challenge

Neural networks operate on numerical vectors, but language is composed of characters, words, and meanings. Tokenization bridges this gap by converting text into discrete units.

But how should we split text? Consider:

```
"I'm learning about tokenization!"
```

Possible segmentations:

| Method | Result | Token Count |
|--------|--------|-------------|
| Whitespace | ["I'm", "learning", "about", "tokenization!"] | 4 |
| Word + punctuation | ["I", "'m", "learning", "about", "tokenization", "!"] | 6 |
| Character | ["I", "'", "m", " ", "l", "e", "a", "r", "n", ...] | 33 |
| Subword | ["I", "'m", " learn", "ing", " about", " token", "ization", "!"] | 8 |

There's no obviously "correct" answer—each choice involves trade-offs.

## The Vocabulary Problem

The simplest approach is **word-level tokenization**: each unique word gets a token ID.

### Problem 1: Vocabulary Explosion

English has over 170,000 words in common use. But that's just the beginning:

- Technical terms: "transformer", "backpropagation", "eigenvalue"
- Proper nouns: company names, people, places
- Morphological variants: "run", "runs", "running", "ran"
- Compound words: "self-attention", "machine-learning"
- Neologisms: "GPT-4", "COVID-19"

A word-level vocabulary for a large corpus might need millions of entries.

### Problem 2: Out-of-Vocabulary (OOV) Tokens

What happens when the model encounters a word it hasn't seen during training?

```python
vocab = {"the": 0, "cat": 1, "sat": 2, ...}
text = "The xerophyte grew in the desert"  # "xerophyte" not in vocab!
```

Options:

1. **[UNK] token**: Replace unknown words with a special token
   - Loses all information about the unknown word
   - "The [UNK] grew" tells us nothing

2. **Subword fallback**: Break unknown words into known pieces
   - "xerophyte" → "xero" + "phyte" (if these are in vocabulary)
   - Preserves morphological information

## The Sequence Length Problem

Attention mechanisms have complexity $O(n^2)$ where n is sequence length:

| Tokenization | Vocab Size | Sequence Length | Attention Cost |
|--------------|------------|-----------------|----------------|
| Words | ~500,000 | ~100 | $100^2 = 10,000$ |
| Characters | ~300 | ~500 | $500^2 = 250,000$ |
| Subwords | ~50,000 | ~150 | $150^2 = 22,500$ |

Character-level models pay a heavy computational price for their flexibility.

## The Linguistic Granularity Problem

What's the "right" unit of meaning?

### Morphemes vs. Words

Consider the word "unhappiness":

- **As one word**: The model must learn this specific word
- **As morphemes**: "un" + "happy" + "ness"
  - The model can generalize: "un-" means negation, "-ness" creates nouns
  - Works for "unkindness", "unfairness", etc.

### The Compositionality Principle

Subword tokenization captures **compositional semantics**:

$$\text{meaning}(\text{"unhappiness"}) \approx f(\text{meaning}(\text{"un"}), \text{meaning}(\text{"happy"}), \text{meaning}(\text{"ness"}))$$

This enables generalization to unseen words through known components.

## Design Criteria for Tokenization

A good tokenization scheme should:

1. **Have bounded vocabulary**: Thousands, not millions of tokens
2. **Handle any input**: No OOV tokens (or minimal OOV rate)
3. **Produce reasonable sequence lengths**: Not too long for attention
4. **Capture meaningful units**: Subwords should correspond to linguistic units
5. **Be deterministic**: Same input → same output
6. **Be efficient**: Fast encoding and decoding

## The Data-Driven Solution

Modern tokenizers are **learned from data**, not hand-designed:

1. Start with some base units (bytes or characters)
2. Learn which sequences should be merged into tokens
3. The corpus determines the vocabulary

This approach:

- Adapts to the domain (scientific text, code, conversation)
- Automatically handles morphology
- Requires no linguistic expertise to create

## Three Algorithms

The three main subword tokenization algorithms:

| Algorithm | Strategy | Used By |
|-----------|----------|---------|
| **BPE** | Merge most frequent pairs | GPT-2, GPT-3, GPT-4 |
| **WordPiece** | Merge pairs that maximize likelihood | BERT, DistilBERT |
| **Unigram** | Start large, prune low-impact tokens | SentencePiece, T5 |

All three produce similar results but use different optimization criteria.

## Summary

| Concept | Key Point |
|---------|-----------|
| Vocabulary size | Trade-off between coverage and tractability |
| Sequence length | Affects attention cost ($O(n^2)$) |
| OOV handling | Subword tokenization eliminates OOV |
| Compositionality | Subwords can capture morphological structure |
| Data-driven | Modern tokenizers are learned, not designed |

**Next**: We'll explore the design space of character vs. subword tokenization in detail.

# Section 7.4: WordPiece

*Reading time: 12 minutes*

## Overview

WordPiece is the tokenization algorithm used by BERT, DistilBERT, and related models. It's similar to BPE but uses a different criterion for selecting which pairs to merge.

## BPE vs. WordPiece

| Aspect | BPE | WordPiece |
|--------|-----|-----------|
| Merge criterion | Most frequent pair | Pair that maximizes likelihood |
| Question asked | "What's most common?" | "What gives best compression?" |
| Optimization | Greedy frequency | Greedy likelihood |

## The Likelihood Criterion

WordPiece asks: "Which merge would most increase the likelihood of the training data?"

### Intuition

Consider two potential merges:

1. "t" + "h" → "th" (both very common individually)
2. "q" + "u" → "qu" (q rarely appears without u)

BPE might prefer (1) because "t" and "h" are both frequent.

WordPiece prefers (2) because "qu" co-occurs more than chance would predict.

### The Score

For a pair (a, b), WordPiece computes:

$$\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \times \text{freq}(b)}$$

This is essentially **pointwise mutual information** (PMI):

$$\text{PMI}(a, b) = \log \frac{P(ab)}{P(a) \cdot P(b)}$$

A high score means a and b co-occur much more often than independent chance would predict.

### Example

Corpus frequencies:

- freq("q") = 100
- freq("u") = 5000
- freq("qu") = 95
- freq("t") = 8000
- freq("h") = 6000
- freq("th") = 4000

Scores:

$$\text{score}(q, u) = \frac{95}{100 \times 5000} = 0.00019$$

$$\text{score}(t, h) = \frac{4000}{8000 \times 6000} = 0.000083$$

WordPiece prefers "qu" despite lower absolute frequency!

## The ## Prefix Convention

WordPiece uses a special prefix `##` to mark continuation tokens:

```
"tokenization" → ["token", "##ization"]
"playing" → ["play", "##ing"]
```

This distinguishes:

- "play" (start of word) — ID 1234
- "##play" (continuation) — ID 5678

### Why This Matters

Without the prefix:

```
"I play games" → ["I", "play", "games"]
"replay" → ["re", "play"]
```

Both contain "play", but they're different:

- First: standalone word
- Second: part of compound

The `##` prefix keeps these separate in the vocabulary.

## Algorithm

### Training

```python
def train_wordpiece(corpus: List[str], vocab_size: int) -> Dict[str, int]:
    """Train WordPiece tokenizer."""

    # Initialize with special tokens
    vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}

    # Collect word frequencies
    word_freqs = count_words(corpus)

    # Initialize vocabulary with characters
    for word in word_freqs:
        for i, char in enumerate(word):
            token = char if i == 0 else '##' + char
            if token not in vocab:
                vocab[token] = len(vocab)

    # Convert words to token sequences
    word_tokens = {word: split_to_chars(word) for word in word_freqs}

    # Iteratively merge
    while len(vocab) < vocab_size:
        # Count pairs and compute scores
        pair_scores = {}
        token_freqs = count_token_frequencies(word_tokens, word_freqs)

        for word, freq in word_freqs.items():
            tokens = word_tokens[word]
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                pair_freq = pair_scores.get(pair, 0) + freq
                pair_scores[pair] = pair_freq

        # Find best pair by WordPiece score
        best_pair, best_score = None, -1
        for pair, freq in pair_scores.items():
            score = freq / (token_freqs[pair[0]] * token_freqs[pair[1]])
            if score > best_score:
                best_score = score
                best_pair = pair

        if best_pair is None:
            break

        # Create merged token
        merged = merge_tokens(best_pair)
        vocab[merged] = len(vocab)

        # Apply merge to all words
        word_tokens = apply_merge_to_all(word_tokens, best_pair, merged)

    return vocab
```

### Encoding (Greedy Longest Match)

WordPiece encoding uses **greedy longest-match**:

```python
def encode_wordpiece(word: str, vocab: Dict[str, int]) -> List[str]:
    """Encode a single word using WordPiece."""
    tokens = []
    start = 0

    while start < len(word):
        end = len(word)
        found = False

        # Find longest matching token
        while start < end:
            substr = word[start:end]
            if start > 0:
                substr = '##' + substr

            if substr in vocab:
                tokens.append(substr)
                found = True
                break
            end -= 1

        if not found:
            tokens.append('[UNK]')
            start += 1
        else:
            start = end

    return tokens
```

### Example Encoding

```python
vocab = {'un': 1, 'believ': 2, '##able': 3, 'the': 4, '##s': 5, ...}

encode("unbelievable")
# Try "unbelievable" → not in vocab
# Try "unbelievabl" → not in vocab
# ...
# Try "un" → in vocab! → ["un"]
# Start from position 2
# Try "believable" → not in vocab
# ...
# Try "believ" → in vocab! → ["un", "believ"]
# Start from position 8
# Try "##able" → in vocab! → ["un", "believ", "##able"]

Result: ["un", "believ", "##able"]
```

## Comparison with BPE

### Same Input, Different Tokenizations

For the same vocabulary size, BPE and WordPiece often produce similar (but not identical) tokenizations:

| Input | BPE | WordPiece |
|-------|-----|-----------|
| "tokenization" | ["token", "ization"] | ["token", "##ization"] |
| "transformer" | ["transform", "er"] | ["transform", "##er"] |
| "unhappy" | ["un", "happy"] | ["un", "##happy"] |

### Key Differences

1. **Continuation markers**: WordPiece uses `##`, BPE uses space prefix
2. **Merge selection**: Frequency vs. likelihood
3. **Vocabulary composition**: Slightly different token sets

## BERT's Special Tokens

BERT's WordPiece vocabulary includes special tokens:

| Token | Meaning | ID |
|-------|---------|-----|
| [PAD] | Padding for batching | 0 |
| [UNK] | Unknown token | 100 |
| [CLS] | Start of sequence (classification) | 101 |
| [SEP] | Separator between segments | 102 |
| [MASK] | Masked token for MLM training | 103 |

Example:

```python
# BERT encoding of "Hello world"
[CLS] hello world [SEP]
[101, 7592, 2088, 102]
```

## Advantages of WordPiece

1. **Better handling of rare combinations**: The likelihood criterion prefers truly associated pairs
2. **Clear word boundaries**: The `##` prefix makes boundaries explicit
3. **Well-suited for masked LM**: Clear token identity for masking

## Limitations

1. **Greedy encoding**: May not find optimal segmentation
2. **Slower training**: Computing likelihood scores is more expensive than counting
3. **Language-specific**: Assumes whitespace word boundaries

## Summary

| Aspect | WordPiece |
|--------|-----------|
| **Merge criterion** | Maximize likelihood (PMI) |
| **Continuation marker** | ## prefix |
| **Encoding** | Greedy longest match |
| **Special tokens** | [CLS], [SEP], [MASK], etc. |
| **Used by** | BERT, DistilBERT, ELECTRA |

**Next**: We'll explore the Unigram approach, which takes a completely different strategy.

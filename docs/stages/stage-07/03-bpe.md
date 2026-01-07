# Section 7.3: Byte Pair Encoding (BPE)

*Reading time: 15 minutes*

## Origins

Byte Pair Encoding was originally a data compression algorithm (Gage, 1994). Sennrich et al. (2016) adapted it for NLP, and it became the standard for GPT-2 and subsequent models.

## The Core Idea

BPE iteratively merges the most frequent pair of tokens:

1. Start with a vocabulary of individual characters (or bytes)
2. Count all adjacent pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat until reaching the desired vocabulary size

## Algorithm Walkthrough

Let's trace BPE on a small corpus:

**Corpus**: "low lower lowest"

### Step 0: Initialize

Split into characters with end-of-word marker (</w>):

```
Vocabulary: ['l', 'o', 'w', '</w>', 'e', 'r', 's', 't']

Words with frequencies:
  "l o w </w>"       : 3  (appears in low, lower, lowest)
  "l o w e r </w>"   : 1
  "l o w e s t </w>" : 1
```

### Step 1: Count Pairs

| Pair | Frequency |
|------|-----------|
| (l, o) | 3 |
| (o, w) | 3 |
| (w, </w>) | 1 |
| (w, e) | 2 |
| (e, r) | 1 |
| (r, </w>) | 1 |
| (e, s) | 1 |
| (s, t) | 1 |
| (t, </w>) | 1 |

### Step 2: Merge Best Pair

Most frequent: (l, o) and (o, w) are tied at 3.
Choose (l, o) → create new token "lo"

```
Vocabulary: ['l', 'o', 'w', '</w>', 'e', 'r', 's', 't', 'lo']

Words updated:
  "lo w </w>"       : 3
  "lo w e r </w>"   : 1
  "lo w e s t </w>" : 1
```

### Step 3: Repeat

New pair counts:

| Pair | Frequency |
|------|-----------|
| (lo, w) | 3 |
| (w, </w>) | 1 |
| (w, e) | 2 |
| ... | ... |

Merge (lo, w) → "low"

```
Vocabulary: [..., 'lo', 'low']

Words:
  "low </w>"       : 3
  "low e r </w>"   : 1
  "low e s t </w>" : 1
```

### Continue...

The algorithm continues, creating tokens like:

- "er" (from "lower")
- "low</w>" (the complete word "low")
- "est" (from "lowest")

## The Merge Rules

BPE learns a sequence of **merge rules**:

```
1. l + o → lo
2. lo + w → low
3. e + r → er
4. low + er → lower
...
```

During encoding, these rules are applied in order.

## Formal Algorithm

### Training

```python
def train_bpe(corpus: List[str], vocab_size: int) -> Tuple[Dict, List]:
    """Train BPE tokenizer.

    Returns:
        vocab: token → id mapping
        merges: list of (token_a, token_b) merge rules
    """
    # Initialize with characters
    vocab = {chr(i): i for i in range(256)}  # Byte vocabulary

    # Split words into characters
    word_freqs = count_word_frequencies(corpus)
    # word_freqs: {('l','o','w'): 3, ('l','o','w','e','r'): 1, ...}

    merges = []

    while len(vocab) < vocab_size:
        # Count all adjacent pairs
        pair_freqs = count_pairs(word_freqs)

        if not pair_freqs:
            break

        # Find most frequent pair
        best_pair = max(pair_freqs, key=pair_freqs.get)

        # Create merged token
        merged = best_pair[0] + best_pair[1]
        vocab[merged] = len(vocab)
        merges.append(best_pair)

        # Apply merge to all words
        word_freqs = apply_merge(word_freqs, best_pair, merged)

    return vocab, merges
```

### Encoding

```python
def encode_bpe(text: str, merges: List[Tuple[str, str]]) -> List[str]:
    """Encode text using learned BPE merges."""
    # Start with characters
    tokens = list(text)

    # Apply merges in order
    for pair in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                new_tokens.append(tokens[i] + tokens[i+1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return tokens
```

## Why BPE Works

### Frequency-Based Compression

BPE creates tokens for common patterns:

```
"the" appears 1000 times → gets merged into one token
"xylophone" appears once → stays as characters
```

This follows **Huffman coding principles**: frequent things get short representations.

### Morphological Discovery

BPE naturally discovers morphological structure:

```
Corpus contains: "playing", "played", "player", "plays"

BPE learns:
- "play" (common root)
- "ing" (common suffix)
- "ed" (common suffix)
- "er" (common suffix)
```

The algorithm doesn't know about morphology—it discovers it from frequency patterns.

### Compositionality

Rare words are composed from frequent pieces:

```
"unbelievable" → "un" + "believ" + "able"
```

The model can generalize:

- "un-" indicates negation
- "-able" indicates possibility

## Byte-Level BPE (GPT-2 Style)

GPT-2 uses **byte-level** BPE:

1. Start with 256 byte tokens (not Unicode characters)
2. All text is UTF-8 encoded first
3. Merges operate on bytes, not characters

### Advantages

- Handles any language without special tokenization
- Works with emojis, rare scripts, binary data
- No [UNK] tokens ever needed

### The GPT-2 Tweak

GPT-2 adds a special handling:

1. Pretokenize by splitting on whitespace and punctuation
2. Add space prefix to words (except first): " the" not "the"
3. Run BPE within each pretokenized unit

This prevents merging across word boundaries.

## Complexity Analysis

### Training Time

For vocabulary size V and corpus size N:

- Each iteration scans corpus: O(N)
- Number of iterations: O(V)
- **Total: O(N × V)**

In practice, efficient implementations use data structures to update counts incrementally.

### Encoding Time

For text length L and number of merges M:

- Naive: O(L × M) — apply each merge to entire text
- With hash tables: O(L × average_merge_applications)

## Example: Real BPE Output

Training on a larger corpus, BPE might learn:

```python
>>> tokenizer.encode("The transformer architecture is revolutionary")
['The', ' transform', 'er', ' architecture', ' is', ' revolution', 'ary']

>>> tokenizer.encode("I'm learning about tokenization!")
['I', "'m", ' learning', ' about', ' token', 'ization', '!']
```

Notice:

- Common words stay whole: "The", "is"
- Compound words split meaningfully: "transform" + "er"
- Morphological structure preserved: "revolution" + "ary"

## Limitations of BPE

1. **Greedy algorithm**: May not find globally optimal tokenization
2. **Training corpus dependent**: Different corpora → different vocabularies
3. **Language-specific quirks**: Works best for space-separated languages
4. **No probability model**: Just frequency counts

## Summary

| Aspect | Description |
|--------|-------------|
| **Core idea** | Iteratively merge most frequent pairs |
| **Initialization** | Characters or bytes |
| **Termination** | Desired vocabulary size reached |
| **Encoding** | Apply merges in training order |
| **Strength** | Simple, effective, widely used |
| **Used by** | GPT-2, GPT-3, GPT-4, many others |

**Next**: We'll see how WordPiece modifies this algorithm with a likelihood-based criterion.

# Section 7.2: Character vs. Subword Tokenization

*Reading time: 8 minutes*

## The Design Space

Tokenization exists on a spectrum from fine-grained to coarse-grained:

```
Bytes → Characters → Subwords → Words → Phrases
  ↑                                          ↑
 256 tokens                         Millions of tokens
 Long sequences                     Short sequences
```

Let's analyze each end of this spectrum.

## Character-Level Tokenization

The simplest approach: each character is a token.

### Implementation

```python
class CharTokenizer:
    def __init__(self):
        self.vocab = {}

    def train(self, texts):
        chars = set()
        for text in texts:
            chars.update(text)
        self.vocab = {c: i for i, c in enumerate(sorted(chars))}

    def encode(self, text):
        return [self.vocab[c] for c in text]

    def decode(self, ids):
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return ''.join(inv_vocab[i] for i in ids)
```

### Advantages

1. **No OOV tokens**: Any text can be encoded
2. **Tiny vocabulary**: ~100-300 tokens for most languages
3. **Simple implementation**: Just character lookups
4. **Handles typos and neologisms**: "transformerr" still works

### Disadvantages

1. **Long sequences**: "transformer" = 11 tokens
2. **Expensive attention**: $O(n^2)$ where n is sequence length
3. **Must learn spelling**: The model must learn that "cat" and "c-a-t" are related
4. **No explicit morphology**: No built-in notion of prefixes, suffixes

### When to Use Characters

- **Low-resource languages**: Limited training data
- **Code**: Where every character matters
- **Noisy text**: Typos, OCR errors, social media
- **Small models**: When vocabulary size dominates parameters

## Word-Level Tokenization

Traditional NLP approach: split on whitespace and punctuation.

### Implementation

```python
class WordTokenizer:
    def __init__(self, unk_token='<UNK>'):
        self.vocab = {}
        self.unk_token = unk_token

    def train(self, texts, max_vocab=50000):
        from collections import Counter
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)

        self.vocab = {self.unk_token: 0}
        for word, _ in word_counts.most_common(max_vocab - 1):
            self.vocab[word] = len(self.vocab)

    def encode(self, text):
        unk_id = self.vocab[self.unk_token]
        return [self.vocab.get(w, unk_id) for w in text.split()]
```

### Advantages

1. **Short sequences**: One token per word
2. **Linguistically intuitive**: Tokens are words
3. **Fast attention**: Fewer tokens = faster

### Disadvantages

1. **Huge vocabulary**: English needs 100K+ words
2. **OOV problem**: Unknown words → [UNK]
3. **Morphological blindness**: "run", "runs", "running" are unrelated
4. **Language-specific**: Some languages don't use spaces (Chinese, Japanese)

## Subword Tokenization: The Middle Ground

Subword tokenization finds a balance:

- Common words get their own tokens
- Rare words are split into common subwords

### Example

| Word | Subword Tokens | Interpretation |
|------|----------------|----------------|
| the | ["the"] | Common → single token |
| transformer | ["trans", "former"] | Split into known pieces |
| unhappiness | ["un", "happi", "ness"] | Morphemes preserved |
| GPT-4 | ["G", "PT", "-", "4"] | Unknown → character fallback |

### The Key Insight

Zipf's law tells us that word frequencies follow a power law:

$$\text{frequency}(r) \propto \frac{1}{r^\alpha}$$

where r is the rank (most common word = rank 1).

This means:

- A few words are extremely common (the, of, and, to)
- Most words are rare

Subword tokenization exploits this:

- Give common words their own tokens
- Build rare words from common pieces

## Comparing the Approaches

Consider the text: "The transformers are transforming NLP"

| Approach | Tokens | Count | Vocab Needed |
|----------|--------|-------|--------------|
| Character | ["T","h","e"," ","t","r",...] | 38 | ~70 |
| Word | ["The","transformers","are","transforming","NLP"] | 5 | ~50,000 |
| Subword | ["The"," transform","ers"," are"," transform","ing"," NLP"] | 7 | ~10,000 |

Subword achieves a balance:

- Reasonable vocabulary size
- Moderate sequence length
- Captures that "transformers" and "transforming" share a root

## The Vocabulary Size vs. Sequence Length Trade-off

There's an approximate invariant:

$$V \times L \approx C$$

where:

- V = vocabulary size
- L = average sequence length
- C = constant (depends on text)

Doubling vocabulary roughly halves sequence length (for subwords).

### Optimal Operating Point

Modern LLMs use vocabularies of 32K-100K tokens:

| Model | Vocabulary Size | Tokenizer |
|-------|-----------------|-----------|
| GPT-2 | 50,257 | BPE |
| GPT-4 | ~100,000 | BPE variant |
| BERT | 30,522 | WordPiece |
| LLaMA | 32,000 | SentencePiece |
| Claude | ~100,000 | BPE variant |

## Byte-Level Tokenization

A modern variant: start with bytes (256 possible values) instead of characters.

### Advantages

- Handles any encoding (UTF-8, UTF-16, binary)
- No character-level preprocessing needed
- Truly universal: works for any language

### Used By

- GPT-2, GPT-3, GPT-4 (byte-level BPE)

## Summary

| Approach | Vocab Size | Seq Length | OOV Handling | Best For |
|----------|------------|------------|--------------|----------|
| Character | ~100 | Very long | Perfect | Low-resource |
| Word | ~100K | Short | Poor | Traditional NLP |
| Subword | ~50K | Medium | Excellent | Modern LLMs |
| Byte | 256 base | Long | Perfect | Universal |

**Next**: We'll derive Byte Pair Encoding (BPE), the algorithm behind GPT.

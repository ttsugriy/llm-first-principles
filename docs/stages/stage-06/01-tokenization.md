# Section 6.1: Tokenization — From Text to Numbers

*Reading time: 25 minutes | Difficulty: ★★★☆☆*

Before a Transformer can process text, we must convert it to numbers. This section covers tokenization—the crucial step that determines how text is represented as a sequence of discrete tokens.

## The Tokenization Problem

Neural networks operate on numbers, not text. We need a mapping:

```
"Hello, world!" → [15496, 11, 995, 0]
```

But how do we design this mapping?

### Option 1: Character-Level

The simplest approach: one token per character.

```python
text = "Hello"
chars = ['H', 'e', 'l', 'l', 'o']
tokens = [7, 4, 11, 11, 14]  # Using ASCII-like mapping
```

**Pros:**
- Small vocabulary (~100-300 tokens)
- Can represent any text
- No out-of-vocabulary (OOV) problem

**Cons:**
- Very long sequences ("Hello" = 5 tokens, "Transformers" = 12 tokens)
- Each character carries little meaning
- Harder to learn word-level patterns

### Option 2: Word-Level

One token per word:

```python
text = "Hello world"
words = ['Hello', 'world']
tokens = [1234, 5678]
```

**Pros:**
- Meaningful units
- Shorter sequences

**Cons:**
- Huge vocabulary (English has 170,000+ words)
- OOV problem: "transformerify" → `<UNK>`
- Morphology ignored: "run", "runs", "running" are unrelated tokens

### Option 3: Subword Tokenization

The modern solution: break words into meaningful subunits.

```python
text = "unbelievable"
subwords = ['un', 'believ', 'able']
tokens = [348, 12892, 540]
```

**Pros:**
- Moderate vocabulary (30K-100K tokens)
- Handles rare/new words via subword composition
- Captures morphology ("un-" prefix, "-able" suffix)
- No OOV problem

**Cons:**
- Tokenization is language/data dependent
- Less interpretable than word-level

## Byte Pair Encoding (BPE)

The most popular subword algorithm, used by GPT, LLaMA, and many others.

### The BPE Algorithm

1. Start with character-level vocabulary
2. Count all adjacent pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat until desired vocabulary size

### Worked Example

**Corpus:** "low low low low low lower lower newest newest newest newest newest newest widest widest widest"

**Step 1: Start with characters**

```
Vocabulary: {l, o, w, e, r, n, s, t, i, d, ' '}
Tokens: l o w ' ' l o w ' ' l o w ...
```

**Step 2: Count pairs**

```
Pair counts:
('l', 'o'): 7
('o', 'w'): 7
('w', ' '): 5
('w', 'e'): 8
('e', 's'): 9
('s', 't'): 9
...
```

**Step 3: Merge most frequent ('e', 's') → 'es'**

```
Vocabulary: {l, o, w, e, r, n, s, t, i, d, ' ', 'es'}
Tokens: l o w ' ' l o w ' ' ... n e w es t ...
```

**Step 4: Repeat**

```
After 10 merges:
Vocabulary: {..., 'es', 'est', 'lo', 'low', 'er', 'low ', 'new', 'newest'}

"lowest" → ['low', 'est']
"newer" → ['new', 'er']
```

### BPE Properties

**Greedy encoding:** BPE encodes text by greedily matching the longest tokens first.

```python
def bpe_encode(text, merges):
    tokens = list(text)  # Start with characters

    while True:
        # Find all adjacent pairs
        pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

        # Find the pair with highest priority (earliest in merge list)
        best_pair = None
        best_idx = float('inf')

        for pair in set(pairs):
            if pair in merges and merges[pair] < best_idx:
                best_pair = pair
                best_idx = merges[pair]

        if best_pair is None:
            break

        # Merge all occurrences
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                new_tokens.append(tokens[i] + tokens[i+1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        tokens = new_tokens

    return tokens
```

## WordPiece

Used by BERT and related models. Similar to BPE but uses likelihood instead of frequency.

### Key Difference

BPE: Merge the most **frequent** pair
WordPiece: Merge the pair that maximizes **likelihood** of the corpus

```
Likelihood score for merging (a, b) → ab:

score(a, b) = count(ab) / (count(a) * count(b))
```

This favors merging pairs where the combination is surprisingly common.

### WordPiece Markers

WordPiece uses `##` to mark continuation tokens:

```
"unbelievable" → ['un', '##believ', '##able']
```

The `##` prefix indicates "this continues the previous token."

## Unigram Language Model

An alternative approach used by SentencePiece (T5, LLaMA).

### How It Works

1. Start with a large vocabulary of all possible substrings
2. Compute probability of each token using a unigram LM
3. Remove tokens that least reduce corpus likelihood
4. Repeat until desired vocabulary size

### Probabilistic Tokenization

Unlike BPE (deterministic), Unigram can sample from multiple valid tokenizations:

```
"unbelievable" could be:
  - ['un', 'believ', 'able']     p = 0.4
  - ['unbeliev', 'able']         p = 0.3
  - ['un', 'believable']         p = 0.2
  - ['unbelievable']             p = 0.1
```

This can be used for data augmentation during training.

## Vocabulary Size Trade-offs

| Vocab Size | Avg Tokens/Word | Pros | Cons |
|------------|-----------------|------|------|
| 256 (bytes) | 4-5 | Universal, no OOV | Very long sequences |
| 8K | 1.5-2 | Short sequences | Many rare tokens |
| 32K | 1.2-1.5 | Good balance | Standard choice |
| 100K | 1.0-1.2 | Word-like | Many parameters in embedding |

**GPT-4:** ~100K tokens
**LLaMA 2:** 32K tokens
**Claude:** Not disclosed, estimated 100K+

### The Trade-off

```
Larger vocabulary:
+ Shorter sequences (faster inference)
+ More semantic tokens
- Larger embedding matrix (vocab_size × d_model)
- More rare tokens (undertrained)

Smaller vocabulary:
+ Smaller model
+ Better-trained embeddings
- Longer sequences (slower)
- Less semantic units
```

## Special Tokens

Tokenizers include special tokens for model control:

| Token | Purpose | Example |
|-------|---------|---------|
| `<PAD>` | Padding for batching | `[token, token, <PAD>, <PAD>]` |
| `<BOS>` | Beginning of sequence | `<BOS> Hello world` |
| `<EOS>` | End of sequence | `Hello world <EOS>` |
| `<UNK>` | Unknown (rare) | Fallback for OOV |
| `<SEP>` | Separator | `Query <SEP> Document` |
| `<MASK>` | Masked token (BERT) | `The <MASK> sat on the mat` |

## Implementation

```python
class BPETokenizer:
    """
    Simple BPE tokenizer implementation.
    """

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}  # (token1, token2) -> merged_token
        self.vocab = {}   # token -> id
        self.inverse_vocab = {}  # id -> token

    def train(self, texts):
        """Learn BPE merges from corpus."""
        # Start with byte-level vocabulary
        self.vocab = {chr(i): i for i in range(256)}
        self.inverse_vocab = {i: chr(i) for i in range(256)}
        next_id = 256

        # Tokenize corpus at byte level
        corpus = []
        for text in texts:
            corpus.append(list(text.encode('utf-8')))

        # Learn merges
        while len(self.vocab) < self.vocab_size:
            # Count all pairs
            pair_counts = {}
            for tokens in corpus:
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)

            # Create new token
            if isinstance(best_pair[0], int):
                new_token = bytes([best_pair[0], best_pair[1]])
            else:
                new_token = best_pair[0] + best_pair[1]

            self.merges[best_pair] = new_token
            self.vocab[new_token] = next_id
            self.inverse_vocab[next_id] = new_token
            next_id += 1

            # Apply merge to corpus
            new_corpus = []
            for tokens in corpus:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                new_corpus.append(new_tokens)
            corpus = new_corpus

        return self

    def encode(self, text):
        """Encode text to token IDs."""
        # Start with bytes
        tokens = list(text.encode('utf-8'))

        # Apply merges in order
        for pair, merged in self.merges.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Convert to IDs
        return [self.vocab.get(t, 0) for t in tokens]

    def decode(self, ids):
        """Decode token IDs to text."""
        tokens = [self.inverse_vocab.get(i, b'?') for i in ids]

        # Concatenate bytes
        result = b''
        for t in tokens:
            if isinstance(t, bytes):
                result += t
            elif isinstance(t, int):
                result += bytes([t])
            else:
                result += t.encode('utf-8')

        return result.decode('utf-8', errors='replace')
```

## Tokenization in Practice

### Pre-tokenization

Before BPE, text is often pre-tokenized:

```python
import re

def pre_tokenize(text):
    """Split on whitespace and punctuation, preserving them."""
    # GPT-2 style regex
    pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+"""
    return re.findall(pattern, text)

text = "Hello, I'm learning!"
print(pre_tokenize(text))
# ["Hello", ",", " I", "'m", " learning", "!"]
```

### Byte-Level BPE

GPT-2+ uses byte-level BPE to handle any Unicode:

```python
# Instead of character vocabulary, use byte vocabulary
# This means vocab starts at 256 (one for each byte)
# Can represent ANY text, any language, any encoding
```

### Handling Numbers

Numbers are tricky for tokenizers:

```
"12345" might become:
- ["123", "45"]        (inconsistent)
- ["1", "2", "3", "4", "5"]  (very long)
- ["12345"]            (rare token, undertrained)
```

Some models use special number handling or digit tokenization.

## Tokenization Effects on Model Behavior

Tokenization has subtle but important effects:

### Arithmetic

```
"25 + 37" → ["25", " +", " 37"] or ["2", "5", " +", " 3", "7"]

Multi-digit numbers as single tokens makes arithmetic harder
because the model must learn 25+37=62 as a memorized fact
rather than compositional digit-by-digit computation.
```

### Code

```python
"def function():" → ["def", " function", "():", ...]

Tokenizers trained on English may fragment code awkwardly.
Code-specific tokenizers (Codex, StarCoder) handle this better.
```

### Multilingual

```
English: "Hello" → ["Hello"] (1 token)
Japanese: "こんにちは" → ["こ", "ん", "に", "ち", "は"] (5 tokens)

Less-common languages often get character-level tokenization,
leading to longer sequences and potentially worse performance.
```

!!! info "Connection to Modern LLMs"

    Tokenization is one of the least "sexy" parts of LLMs but critically important:

    - **GPT-4** uses a ~100K vocabulary with improved handling of code and non-English
    - **LLaMA** uses SentencePiece with 32K vocabulary
    - **Claude** likely uses a large vocabulary optimized for diverse text

    Poor tokenization can bottleneck model performance regardless of architecture.

## Exercises

1. **Implement BPE**: Train a BPE tokenizer on a small corpus and visualize the learned merges.

2. **Vocabulary analysis**: For a fixed text, compare token counts with vocab sizes 1K, 10K, 100K.

3. **Language comparison**: Tokenize the same sentence in multiple languages. Which gets more tokens?

4. **Number tokenization**: How does your tokenizer handle "123456789"? How about "3.14159"?

5. **Reconstruct text**: Given only token IDs, can you perfectly reconstruct the original text?

## Summary

| Concept | Definition | Trade-off |
|---------|------------|-----------|
| Character-level | One token per character | Universal but long |
| Word-level | One token per word | Short but OOV issues |
| BPE | Merge frequent pairs | Standard approach |
| WordPiece | Merge by likelihood | Used by BERT |
| Vocabulary size | Number of unique tokens | Length vs embedding size |

**Key takeaway**: Tokenization converts text to discrete tokens that neural networks can process. Subword methods like BPE balance vocabulary size against sequence length, handle rare words through composition, and capture morphological structure. The choice of tokenizer affects model behavior in subtle ways—from arithmetic to multilingual performance.

→ **Next**: [Section 6.2: The Transformer Block](02-transformer-block.md)

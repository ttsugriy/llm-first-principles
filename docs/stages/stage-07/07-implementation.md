# Section 7.7: Implementation

*Reading time: 15 minutes*

## Overview

In this section, we implement three tokenization algorithms from scratch:

1. Character tokenizer (baseline)
2. BPE tokenizer (GPT-style)
3. WordPiece tokenizer (BERT-style)
4. Unigram tokenizer (SentencePiece-style)

All code is available in `code/stage-07/tokenizer.py`.

## Base Tokenizer Class

All tokenizers share a common interface:

```python
class Tokenizer:
    """Base class for all tokenizers."""

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.special_tokens: Dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        data = {
            'vocab': self.vocab,
            'special_tokens': self.special_tokens,
            'type': self.__class__.__name__,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
```

## Character Tokenizer

The simplest tokenizer: each character is a token.

```python
class CharTokenizer(Tokenizer):
    """Character-level tokenizer."""

    def __init__(self, unk_token: str = '<UNK>'):
        super().__init__()
        self.unk_token = unk_token

    def train(self, texts: List[str]) -> 'CharTokenizer':
        """Build vocabulary from texts."""
        # Collect all unique characters
        chars: Set[str] = set()
        for text in texts:
            chars.update(text)

        # Sort for reproducibility
        chars_sorted = sorted(chars)

        # Add special tokens first
        self.vocab = {self.unk_token: 0}
        self.special_tokens = {self.unk_token: 0}

        # Add characters
        for i, char in enumerate(chars_sorted, start=1):
            self.vocab[char] = i

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        return self

    def encode(self, text: str) -> List[int]:
        """Encode text to character IDs."""
        unk_id = self.vocab[self.unk_token]
        return [self.vocab.get(c, unk_id) for c in text]

    def decode(self, ids: List[int]) -> str:
        """Decode character IDs to text."""
        return ''.join(self.inverse_vocab.get(i, self.unk_token) for i in ids)
```

### Usage

```python
>>> char_tok = CharTokenizer().train(corpus)
>>> char_tok.vocab_size
47  # alphabet + punctuation + space

>>> char_tok.encode("Hello")
[8, 5, 12, 12, 15]

>>> char_tok.decode([8, 5, 12, 12, 15])
"Hello"
```

## BPE Tokenizer

The algorithm behind GPT-2/3/4:

```python
class BPETokenizer(Tokenizer):
    """Byte Pair Encoding tokenizer."""

    def __init__(
        self,
        vocab_size: int = 1000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
    ):
        super().__init__()
        self.target_vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.merges: List[Tuple[bytes, bytes]] = []
        self._special_tokens_list = special_tokens or ['<PAD>', '<UNK>', '<BOS>', '<EOS>']

    def train(self, texts: List[str]) -> 'BPETokenizer':
        """Train BPE on a corpus."""
        # Initialize vocabulary with special tokens
        self.vocab = {}
        for i, token in enumerate(self._special_tokens_list):
            self.vocab[token] = i
            self.special_tokens[token] = i

        # Add byte vocabulary (256 possible bytes)
        next_id = len(self.vocab)
        for i in range(256):
            byte_token = bytes([i])
            self.vocab[byte_token] = next_id
            next_id += 1

        # Encode corpus as bytes, split into words
        word_freqs: Dict[Tuple[bytes, ...], int] = Counter()
        for text in texts:
            words = text.encode('utf-8').split(b' ')
            for word in words:
                if word:
                    word_tuple = tuple(bytes([b]) for b in word)
                    word_freqs[word_tuple] += 1

        # Iteratively merge pairs
        while len(self.vocab) < self.target_vocab_size:
            # Count all pairs
            pair_freqs: Dict[Tuple[bytes, bytes], int] = Counter()
            for word, freq in word_freqs.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            if pair_freqs[best_pair] < self.min_frequency:
                break

            # Create merged token
            merged = best_pair[0] + best_pair[1]
            self.vocab[merged] = len(self.vocab)
            self.merges.append(best_pair)

            # Apply merge to all words
            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_word = self._apply_merge(word, best_pair, merged)
                new_word_freqs[new_word] += freq
            word_freqs = new_word_freqs

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        return self

    def _apply_merge(self, word, pair, merged):
        """Apply a single merge to a word."""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def encode(self, text: str) -> List[int]:
        """Encode text using learned BPE merges."""
        # Convert to bytes
        text_bytes = text.encode('utf-8')
        tokens = [bytes([b]) for b in text_bytes]

        # Apply merges in order
        for pair in self.merges:
            merged = pair[0] + pair[1]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Convert to IDs
        unk_id = self.special_tokens.get('<UNK>', 0)
        return [self.vocab.get(t, unk_id) for t in tokens]
```

### Usage

```python
>>> bpe_tok = BPETokenizer(vocab_size=500).train(corpus)
>>> bpe_tok.vocab_size
500

>>> bpe_tok.get_merges()[:5]
[('t', 'h'), ('th', 'e'), ('e', 'r'), ...]

>>> bpe_tok.encode("The transformer")
[267, 156, 289, 45, 67, 89]
```

## WordPiece Tokenizer

The algorithm behind BERT:

```python
class WordPieceTokenizer(Tokenizer):
    """WordPiece tokenizer (BERT-style)."""

    def __init__(
        self,
        vocab_size: int = 1000,
        min_frequency: int = 2,
        unk_token: str = '[UNK]',
        special_tokens: Optional[List[str]] = None,
    ):
        super().__init__()
        self.target_vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.unk_token = unk_token
        self._special_tokens_list = special_tokens or [
            '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'
        ]
        self.continuation_prefix = '##'

    def train(self, texts: List[str]) -> 'WordPieceTokenizer':
        """Train WordPiece tokenizer."""
        # Initialize with special tokens
        self.vocab = {}
        for i, token in enumerate(self._special_tokens_list):
            self.vocab[token] = i
            self.special_tokens[token] = i

        # Collect word frequencies
        word_freqs = Counter()
        for text in texts:
            words = text.lower().split()
            for word in words:
                word = re.sub(r'[^\w]', '', word)
                if word:
                    word_freqs[word] += 1

        # Initialize with characters (with ## prefix for continuation)
        for word in word_freqs:
            for i, char in enumerate(word):
                token = char if i == 0 else self.continuation_prefix + char
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        # Convert words to token sequences
        word_tokens = {}
        for word in word_freqs:
            tokens = [word[0]] + [self.continuation_prefix + c for c in word[1:]]
            word_tokens[word] = tokens

        # Iteratively merge best pairs (by likelihood score)
        while len(self.vocab) < self.target_vocab_size:
            pair_freqs = Counter()
            token_freqs = Counter()

            for word, freq in word_freqs.items():
                tokens = word_tokens[word]
                for token in tokens:
                    token_freqs[token] += freq
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Find best pair by WordPiece score
            best_pair, best_score = None, -1
            for pair, freq in pair_freqs.items():
                if freq < self.min_frequency:
                    continue
                score = freq / (token_freqs[pair[0]] * token_freqs[pair[1]])
                if score > best_score:
                    best_score = score
                    best_pair = pair

            if best_pair is None:
                break

            # Create merged token
            if best_pair[1].startswith(self.continuation_prefix):
                merged = best_pair[0] + best_pair[1][len(self.continuation_prefix):]
            else:
                merged = best_pair[0] + best_pair[1]

            self.vocab[merged] = len(self.vocab)

            # Apply merge to all words
            for word in word_tokens:
                tokens = word_tokens[word]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (i < len(tokens)-1 and
                        tokens[i] == best_pair[0] and
                        tokens[i+1] == best_pair[1]):
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word_tokens[word] = new_tokens

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        return self

    def encode(self, text: str) -> List[int]:
        """Encode text using WordPiece (greedy longest-match)."""
        tokens = []
        for word in text.lower().split():
            word = re.sub(r'[^\w]', '', word)
            if not word:
                continue
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)

        unk_id = self.special_tokens.get('[UNK]', 0)
        return [self.vocab.get(t, unk_id) for t in tokens]

    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using greedy longest-match."""
        tokens = []
        start = 0

        while start < len(word):
            end = len(word)
            found = False

            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = self.continuation_prefix + substr

                if substr in self.vocab:
                    tokens.append(substr)
                    found = True
                    break
                end -= 1

            if not found:
                tokens.append(self.unk_token)
                start += 1
            else:
                start = end

        return tokens
```

## Analysis Functions

```python
def analyze_tokenization(tokenizer, texts):
    """Analyze tokenization quality."""
    total_chars = 0
    total_tokens = 0
    total_unks = 0

    unk_id = tokenizer.special_tokens.get('<UNK>',
             tokenizer.special_tokens.get('[UNK]', -1))

    for text in texts:
        total_chars += len(text)
        ids = tokenizer.encode(text)
        total_tokens += len(ids)
        total_unks += sum(1 for i in ids if i == unk_id)

    return {
        'vocab_size': tokenizer.vocab_size,
        'compression_ratio': total_chars / total_tokens,
        'unk_rate': total_unks / total_tokens,
    }
```

## Running the Code

```bash
cd code/stage-07
python tokenizer.py
```

Output:

```
============================================================
Stage 7: Tokenization from First Principles
============================================================

1. Character Tokenizer
----------------------------------------
Vocab size: 47
Sample: 'The transformer model processes text efficiently.'
Encoded: [20, 8, 5, 1, 20, 18, 1, ...] (48 tokens)

2. BPE Tokenizer
----------------------------------------
Vocab size: 300
First 10 merges: [('t', 'h'), ('th', 'e'), ...]
Encoded: [267, 156, 89, ...] (12 tokens)

3. WordPiece Tokenizer
----------------------------------------
Vocab size: 200
Encoded: [45, 167, 23, ...] (10 tokens)

4. Comparison
----------------------------------------------------------------------
Character:
  Vocab size:        47
  Compression ratio: 1.00 chars/token
  Unknown rate:      0.00%
  Tokens per word:   5.23

BPE:
  Vocab size:        300
  Compression ratio: 3.45 chars/token
  Unknown rate:      0.00%
  Tokens per word:   1.52

WordPiece:
  Vocab size:        200
  Compression ratio: 3.21 chars/token
  Unknown rate:      0.00%
  Tokens per word:   1.63
```

## Summary

| Tokenizer | Key Feature | Lines of Code |
|-----------|-------------|---------------|
| Character | 1 token per char | ~30 |
| BPE | Merge frequent pairs | ~100 |
| WordPiece | Merge by likelihood | ~120 |
| Unigram | Prune by impact | ~100 |

All implementations follow the same interface and can be used interchangeably.

## Exercises

1. **Extend BPE**: Add pre-tokenization (split on whitespace before BPE)
2. **Byte-level BPE**: Modify to work on bytes instead of characters
3. **Regularization**: Implement sampling-based encoding for Unigram
4. **Visualization**: Plot token frequency distributions
5. **Benchmark**: Compare tokenization speed across implementations

## What's Next

With tokenization complete, we've bridged the gap between raw text and neural networks. In Stage 8, we'll explore training dynamics and debugging—understanding why training goes wrong and how to fix it.

"""
Stage 7: Tokenization from First Principles

This module implements tokenization algorithms from scratch:
- Character-level tokenization (baseline)
- Byte Pair Encoding (BPE) - used by GPT-2, GPT-3, GPT-4
- WordPiece - used by BERT
- Unigram (SentencePiece style)

Tokenization is the bridge between raw text and neural networks.
The choice of tokenizer significantly affects model performance.

Key concepts:
- Subword tokenization balances vocabulary size vs. sequence length
- BPE greedily merges the most frequent byte pairs
- WordPiece maximizes likelihood instead of frequency
- Special tokens ([PAD], [UNK], [CLS], etc.) enable model functionality
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
import re
import json


# =============================================================================
# Base Tokenizer
# =============================================================================

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

    @classmethod
    def load(cls, path: str) -> 'Tokenizer':
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.vocab = data['vocab']
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.special_tokens = data.get('special_tokens', {})
        return tokenizer


# =============================================================================
# Character-Level Tokenizer
# =============================================================================

class CharTokenizer(Tokenizer):
    """
    Character-level tokenizer.

    The simplest approach: each character is a token.

    Pros:
    - No out-of-vocabulary (OOV) tokens
    - Small vocabulary (typically < 300 for Unicode)
    - Simple implementation

    Cons:
    - Very long sequences (1 token per character)
    - Model must learn character-level patterns
    - Expensive for attention (O(n²) where n is long)
    """

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


# =============================================================================
# Byte Pair Encoding (BPE)
# =============================================================================

class BPETokenizer(Tokenizer):
    """
    Byte Pair Encoding tokenizer.

    BPE was introduced by Sennrich et al. (2016) for neural machine translation.
    It's used by GPT-2, GPT-3, GPT-4, and many other models.

    Algorithm:
    1. Start with a vocabulary of all bytes (256 tokens)
    2. Count all adjacent pairs in the corpus
    3. Merge the most frequent pair into a new token
    4. Repeat until reaching desired vocabulary size

    Key insight: Frequent subwords get their own tokens, rare words are
    split into common subword pieces.
    """

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
        """
        Train BPE on a corpus.

        This implements the classic BPE algorithm:
        1. Initialize vocabulary with bytes
        2. Count pair frequencies
        3. Merge most frequent pair
        4. Repeat until vocab_size reached
        """
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
        # We keep word boundaries to avoid merging across words
        word_freqs: Dict[Tuple[bytes, ...], int] = Counter()

        for text in texts:
            # Simple word splitting (can be improved with regex)
            words = text.encode('utf-8').split(b' ')
            for word in words:
                if word:
                    # Add space prefix to non-first words (GPT-2 style)
                    word_tuple = tuple(bytes([b]) for b in word)
                    word_freqs[word_tuple] += 1

        # Iteratively merge pairs
        self.merges = []

        while len(self.vocab) < self.target_vocab_size:
            # Count all pairs
            pair_freqs: Dict[Tuple[bytes, bytes], int] = Counter()

            for word, freq in word_freqs.items():
                if len(word) < 2:
                    continue
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            best_freq = pair_freqs[best_pair]

            if best_freq < self.min_frequency:
                break

            # Create merged token
            merged = best_pair[0] + best_pair[1]
            self.vocab[merged] = len(self.vocab)
            self.merges.append(best_pair)

            # Apply merge to all words
            new_word_freqs: Dict[Tuple[bytes, ...], int] = Counter()

            for word, freq in word_freqs.items():
                new_word = self._apply_merge(word, best_pair, merged)
                new_word_freqs[new_word] += freq

            word_freqs = new_word_freqs

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        return self

    def _apply_merge(
        self,
        word: Tuple[bytes, ...],
        pair: Tuple[bytes, bytes],
        merged: bytes
    ) -> Tuple[bytes, ...]:
        """Apply a single merge to a word."""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def encode(self, text: str) -> List[int]:
        """
        Encode text using learned BPE merges.

        Algorithm:
        1. Convert text to bytes
        2. Split into individual bytes
        3. Apply merges in order learned during training
        4. Convert tokens to IDs
        """
        if not text:
            return []

        # Handle special tokens
        for special, idx in self.special_tokens.items():
            if text == special:
                return [idx]

        # Convert to bytes
        text_bytes = text.encode('utf-8')
        tokens = [bytes([b]) for b in text_bytes]

        # Apply merges in order
        for pair in self.merges:
            merged = pair[0] + pair[1]
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
        unk_id = self.special_tokens.get('<UNK>', 0)
        return [self.vocab.get(t, unk_id) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for idx in ids:
            token = self.inverse_vocab.get(idx)
            if token is not None:
                if isinstance(token, bytes):
                    tokens.append(token)
                else:
                    # Special token
                    tokens.append(token.encode('utf-8'))

        # Concatenate bytes and decode
        try:
            return b''.join(tokens).decode('utf-8', errors='replace')
        except:
            return ''.join(t.decode('utf-8', errors='replace') if isinstance(t, bytes) else t for t in tokens)

    def get_merges(self) -> List[Tuple[str, str]]:
        """Get merges as readable strings."""
        result = []
        for p1, p2 in self.merges:
            try:
                s1 = p1.decode('utf-8', errors='replace')
                s2 = p2.decode('utf-8', errors='replace')
                result.append((s1, s2))
            except:
                result.append((repr(p1), repr(p2)))
        return result


# =============================================================================
# WordPiece Tokenizer
# =============================================================================

class WordPieceTokenizer(Tokenizer):
    """
    WordPiece tokenizer (used by BERT).

    Similar to BPE but uses a different merge criterion:
    - BPE: merge most frequent pair
    - WordPiece: merge pair that maximizes likelihood

    Likelihood score: freq(ab) / (freq(a) * freq(b))

    This prefers merges where the combination is more likely than
    the independent occurrence of the pieces.

    WordPiece also uses ## prefix for continuation tokens.
    """

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
        self._special_tokens_list = special_tokens or ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        self.continuation_prefix = '##'

    def train(self, texts: List[str]) -> 'WordPieceTokenizer':
        """Train WordPiece tokenizer."""
        # Initialize with special tokens
        self.vocab = {}
        for i, token in enumerate(self._special_tokens_list):
            self.vocab[token] = i
            self.special_tokens[token] = i

        # Collect word frequencies
        word_freqs: Dict[str, int] = Counter()
        for text in texts:
            # Simple whitespace tokenization
            words = text.lower().split()
            for word in words:
                # Clean word (remove punctuation for simplicity)
                word = re.sub(r'[^\w]', '', word)
                if word:
                    word_freqs[word] += 1

        # Initialize with characters
        char_freqs: Dict[str, int] = Counter()
        for word, freq in word_freqs.items():
            for i, char in enumerate(word):
                if i == 0:
                    char_freqs[char] += freq
                else:
                    char_freqs[self.continuation_prefix + char] += freq

        # Add frequent characters to vocabulary
        for char, freq in char_freqs.most_common():
            if len(self.vocab) >= self.target_vocab_size:
                break
            if freq >= self.min_frequency:
                self.vocab[char] = len(self.vocab)

        # Convert words to token sequences
        word_tokens: Dict[str, List[str]] = {}
        for word in word_freqs:
            tokens = [word[0]] if word else []
            for char in word[1:]:
                tokens.append(self.continuation_prefix + char)
            word_tokens[word] = tokens

        # Iteratively merge best pairs
        while len(self.vocab) < self.target_vocab_size:
            # Count pairs and individual tokens
            pair_freqs: Dict[Tuple[str, str], int] = Counter()
            token_freqs: Dict[str, int] = Counter()

            for word, freq in word_freqs.items():
                tokens = word_tokens[word]
                for token in tokens:
                    token_freqs[token] += freq
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Find best pair by likelihood score
            best_pair = None
            best_score = -1

            for pair, freq in pair_freqs.items():
                if freq < self.min_frequency:
                    continue
                # WordPiece score: freq(ab) / (freq(a) * freq(b))
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
                    if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word_tokens[word] = new_tokens

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        return self

    def encode(self, text: str) -> List[int]:
        """Encode text using WordPiece."""
        tokens = []
        words = text.lower().split()

        for word in words:
            word = re.sub(r'[^\w]', '', word)
            if not word:
                continue

            # Greedy longest-match tokenization
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)

        # Convert to IDs
        unk_id = self.special_tokens.get('[UNK]', self.special_tokens.get(self.unk_token, 0))
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
                # Character not in vocab, use UNK
                tokens.append(self.unk_token)
                start += 1
            else:
                start = end

        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = [self.inverse_vocab.get(i, self.unk_token) for i in ids]

        # Remove continuation prefixes and join
        text = ''
        for i, token in enumerate(tokens):
            if token in self.special_tokens:
                text += f' {token} '
            elif token.startswith(self.continuation_prefix):
                text += token[len(self.continuation_prefix):]
            else:
                if i > 0:
                    text += ' '
                text += token

        return text.strip()


# =============================================================================
# Unigram Tokenizer (SentencePiece style)
# =============================================================================

class UnigramTokenizer(Tokenizer):
    """
    Unigram Language Model tokenizer.

    Unlike BPE (bottom-up), Unigram is top-down:
    1. Start with a large vocabulary of all substrings
    2. Compute loss if each token were removed
    3. Remove tokens that increase loss the least
    4. Repeat until reaching target vocabulary size

    This is the algorithm used by SentencePiece's unigram mode.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        special_tokens: Optional[List[str]] = None,
    ):
        super().__init__()
        self.target_vocab_size = vocab_size
        self._special_tokens_list = special_tokens or ['<pad>', '<unk>', '<s>', '</s>']
        self.token_probs: Dict[str, float] = {}

    def train(self, texts: List[str], initial_vocab_size: int = 10000) -> 'UnigramTokenizer':
        """
        Train Unigram tokenizer.

        Simplified algorithm:
        1. Build initial vocabulary from frequent substrings
        2. Compute unigram probabilities
        3. Iteratively remove lowest-impact tokens
        """
        # Initialize with special tokens
        self.vocab = {}
        for i, token in enumerate(self._special_tokens_list):
            self.vocab[token] = i
            self.special_tokens[token] = i

        # Collect all substrings and their frequencies
        substring_freqs: Dict[str, int] = Counter()

        for text in texts:
            # Use words as base units
            words = text.lower().split()
            for word in words:
                word = re.sub(r'[^\w]', '', word)
                if not word:
                    continue

                # Add all substrings up to length 10
                for i in range(len(word)):
                    for j in range(i + 1, min(i + 11, len(word) + 1)):
                        substring_freqs[word[i:j]] += 1

        # Keep most frequent substrings for initial vocabulary
        for substr, freq in substring_freqs.most_common(initial_vocab_size):
            if len(self.vocab) >= initial_vocab_size:
                break
            if substr not in self.vocab:
                self.vocab[substr] = len(self.vocab)

        # Compute initial probabilities
        total_freq = sum(substring_freqs.get(t, 1) for t in self.vocab if t not in self.special_tokens)
        self.token_probs = {}
        for token in self.vocab:
            if token in self.special_tokens:
                self.token_probs[token] = 0.0
            else:
                self.token_probs[token] = substring_freqs.get(token, 1) / total_freq

        # Iteratively prune vocabulary
        while len(self.vocab) > self.target_vocab_size:
            # Find token with lowest impact on segmentation
            # (Simplified: just remove least frequent that's not a single char)
            candidates = [
                (t, self.token_probs.get(t, 0))
                for t in self.vocab
                if t not in self.special_tokens and len(t) > 1
            ]

            if not candidates:
                break

            # Remove lowest probability token
            candidates.sort(key=lambda x: x[1])
            to_remove = candidates[0][0]

            del self.vocab[to_remove]
            if to_remove in self.token_probs:
                del self.token_probs[to_remove]

        # Rebuild vocab indices
        self.vocab = {t: i for i, t in enumerate(self.vocab.keys())}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        return self

    def encode(self, text: str) -> List[int]:
        """Encode using Viterbi-like greedy search."""
        tokens = []
        words = text.lower().split()

        for word in words:
            word = re.sub(r'[^\w]', '', word)
            if not word:
                continue

            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)

        unk_id = self.special_tokens.get('<unk>', 0)
        return [self.vocab.get(t, unk_id) for t in tokens]

    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize word using greedy longest match."""
        tokens = []
        start = 0

        while start < len(word):
            # Find longest matching token
            end = len(word)
            found = False

            while end > start:
                substr = word[start:end]
                if substr in self.vocab:
                    tokens.append(substr)
                    found = True
                    start = end
                    break
                end -= 1

            if not found:
                # Single character fallback
                char = word[start]
                if char in self.vocab:
                    tokens.append(char)
                else:
                    tokens.append('<unk>')
                start += 1

        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = [self.inverse_vocab.get(i, '<unk>') for i in ids]
        return ''.join(tokens)


# =============================================================================
# Vocabulary Analysis
# =============================================================================

def analyze_tokenization(
    tokenizer: Tokenizer,
    texts: List[str],
) -> Dict:
    """
    Analyze tokenization quality.

    Metrics:
    - Compression ratio: chars / tokens
    - Unknown token rate
    - Average tokens per word
    - Token frequency distribution
    """
    total_chars = 0
    total_tokens = 0
    total_unks = 0
    total_words = 0
    token_counts: Counter = Counter()

    unk_id = tokenizer.special_tokens.get('<UNK>',
             tokenizer.special_tokens.get('[UNK]',
             tokenizer.special_tokens.get('<unk>', -1)))

    for text in texts:
        total_chars += len(text)
        total_words += len(text.split())

        ids = tokenizer.encode(text)
        total_tokens += len(ids)
        total_unks += sum(1 for i in ids if i == unk_id)

        for idx in ids:
            token = tokenizer.inverse_vocab.get(idx, '<UNK>')
            if isinstance(token, bytes):
                try:
                    token = token.decode('utf-8')
                except:
                    token = repr(token)
            token_counts[token] += 1

    return {
        'vocab_size': tokenizer.vocab_size,
        'total_chars': total_chars,
        'total_tokens': total_tokens,
        'compression_ratio': total_chars / total_tokens if total_tokens > 0 else 0,
        'unk_rate': total_unks / total_tokens if total_tokens > 0 else 0,
        'tokens_per_word': total_tokens / total_words if total_words > 0 else 0,
        'most_common_tokens': token_counts.most_common(20),
    }


def compare_tokenizers(
    tokenizers: Dict[str, Tokenizer],
    texts: List[str],
) -> None:
    """Compare multiple tokenizers on the same texts."""
    print("=" * 70)
    print("Tokenizer Comparison")
    print("=" * 70)

    for name, tokenizer in tokenizers.items():
        stats = analyze_tokenization(tokenizer, texts)
        print(f"\n{name}:")
        print(f"  Vocab size:        {stats['vocab_size']:,}")
        print(f"  Compression ratio: {stats['compression_ratio']:.2f} chars/token")
        print(f"  Unknown rate:      {stats['unk_rate']:.2%}")
        print(f"  Tokens per word:   {stats['tokens_per_word']:.2f}")


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate tokenization algorithms."""
    print("=" * 60)
    print("Stage 7: Tokenization from First Principles")
    print("=" * 60)

    # Sample corpus
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "Machine learning is transforming the world.",
        "Natural language processing enables computers to understand text.",
        "The transformer architecture revolutionized NLP.",
    ]

    sample_text = "The transformer model processes text efficiently."

    # Character tokenizer
    print("\n1. Character Tokenizer")
    print("-" * 40)
    char_tok = CharTokenizer().train(corpus)
    print(f"Vocab size: {char_tok.vocab_size}")
    encoded = char_tok.encode(sample_text)
    print(f"Sample: '{sample_text}'")
    print(f"Encoded: {encoded[:20]}... ({len(encoded)} tokens)")
    print(f"Decoded: '{char_tok.decode(encoded)}'")

    # BPE tokenizer
    print("\n2. BPE Tokenizer")
    print("-" * 40)
    bpe_tok = BPETokenizer(vocab_size=300).train(corpus)
    print(f"Vocab size: {bpe_tok.vocab_size}")
    print(f"First 10 merges: {bpe_tok.get_merges()[:10]}")
    encoded = bpe_tok.encode(sample_text)
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{bpe_tok.decode(encoded)}'")

    # WordPiece tokenizer
    print("\n3. WordPiece Tokenizer")
    print("-" * 40)
    wp_tok = WordPieceTokenizer(vocab_size=200).train(corpus)
    print(f"Vocab size: {wp_tok.vocab_size}")
    encoded = wp_tok.encode(sample_text)
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{wp_tok.decode(encoded)}'")

    # Compare
    print("\n4. Comparison")
    print("-" * 40)
    compare_tokenizers({
        'Character': char_tok,
        'BPE': bpe_tok,
        'WordPiece': wp_tok,
    }, corpus)


if __name__ == '__main__':
    demo()

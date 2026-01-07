# Stage 7 Exercises

## Conceptual Questions

### Exercise 7.1: Tokenization Trade-offs
Consider the trade-offs between different tokenization strategies.

**a)** Why can't we just use word-level tokenization for modern LLMs?
**b)** What happens to sequence length if we double the vocabulary size?
**c)** Why does GPT-4 use ~100K tokens instead of 32K like GPT-2?

### Exercise 7.2: BPE Algorithm Understanding
Given the corpus: "abab abcd abab"

**a)** What is the initial vocabulary (characters)?
**b)** What's the most frequent pair?
**c)** After one merge, what is the new vocabulary?
**d)** How many merges to reach vocabulary size 10?

### Exercise 7.3: WordPiece vs BPE
Both BPE and WordPiece are subword algorithms but differ in merge selection.

**a)** What criterion does BPE use to select merges?
**b)** What criterion does WordPiece use?
**c)** When would they produce different vocabularies?

### Exercise 7.4: Unknown Tokens
Consider handling out-of-vocabulary words.

**a)** How does character-level tokenization handle unknown words?
**b)** How does BPE handle unknown words?
**c)** Why is the `<UNK>` token problematic for language models?

---

## Implementation Exercises

### Exercise 7.5: Character Tokenizer
Implement a simple character-level tokenizer:

```python
class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}

    def fit(self, text: str):
        """Build vocabulary from text."""
        # TODO: Create mappings for each unique character
        pass

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        # TODO
        pass

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        # TODO
        pass
```

### Exercise 7.6: Pair Counting
Implement the pair counting step of BPE:

```python
def count_pairs(tokens: List[List[str]]) -> Dict[Tuple[str, str], int]:
    """
    Count frequency of adjacent token pairs in corpus.

    Args:
        tokens: List of tokenized words, each word is a list of tokens
                e.g., [['l', 'o', 'w'], ['l', 'o', 'w', 'e', 'r']]

    Returns:
        Dictionary mapping (token1, token2) to count
    """
    # TODO
    pass
```

### Exercise 7.7: BPE Merge
Implement the merge step of BPE:

```python
def apply_merge(
    tokens: List[List[str]],
    pair: Tuple[str, str]
) -> List[List[str]]:
    """
    Merge all occurrences of pair in the tokenized corpus.

    Args:
        tokens: Current tokenization of corpus
        pair: The pair to merge, e.g., ('l', 'o')

    Returns:
        New tokenization with pair merged
    """
    # TODO
    pass
```

### Exercise 7.8: Full BPE Training
Implement complete BPE training:

```python
class BPE:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.merges = []  # List of (pair, merged_token)
        self.vocab = {}

    def train(self, corpus: List[str]):
        """
        Train BPE on corpus.

        1. Initialize vocabulary with characters
        2. Count pairs
        3. Merge most frequent
        4. Repeat until vocab_size
        """
        # TODO
        pass

    def encode(self, text: str) -> List[str]:
        """Encode text using learned merges."""
        # TODO
        pass
```

---

## Challenge Exercises

### Exercise 7.9: Vocabulary Analysis
Analyze a trained BPE vocabulary:

**a)** Train BPE with vocab_size=1000 on a text corpus
**b)** What percentage of tokens are single characters?
**c)** What's the longest token?
**d)** Plot the token length distribution
**e)** How many tokens start with a space vs. not?

### Exercise 7.10: Unigram Implementation
Implement the Unigram tokenizer:

```python
class UnigramTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.token_probs = {}  # token -> log probability

    def train(self, corpus: List[str]):
        """
        Train Unigram tokenizer using EM algorithm.

        1. Start with large vocabulary (all substrings up to length N)
        2. Compute optimal tokenization given vocab
        3. Compute new probabilities
        4. Prune lowest probability tokens
        5. Repeat until vocab_size
        """
        # TODO
        pass

    def tokenize(self, text: str) -> List[str]:
        """
        Find optimal tokenization using Viterbi algorithm.

        Maximize: sum of log probabilities of tokens
        """
        # TODO
        pass
```

### Exercise 7.11: Tokenization Quality Metrics
Implement metrics to evaluate tokenization quality:

```python
def compute_fertility(tokenizer, corpus: List[str]) -> float:
    """
    Compute average tokens per word.

    Lower fertility = more efficient encoding.
    """
    # TODO
    pass

def compute_coverage(tokenizer, corpus: List[str]) -> float:
    """
    Compute percentage of text covered by non-<UNK> tokens.
    """
    # TODO
    pass

def compute_sequence_compression(tokenizer, corpus: List[str]) -> float:
    """
    Ratio of character length to token length.

    Higher = more compression.
    """
    # TODO
    pass
```

### Exercise 7.12: Multilingual Tokenization
Explore how tokenization affects different languages:

**a)** Train BPE on English, then tokenize Chinese text. What happens?
**b)** Why do some languages require more tokens per word?
**c)** How would you build a multilingual tokenizer?

---

## Checking Your Work

- **Test suite**: See `code/stage-07/tests/test_tokenizer.py` for expected behavior
- **Reference implementation**: Compare with `code/stage-07/tokenizer.py`
- **Self-check**: Verify encode/decode roundtrips correctly: `decode(encode(text)) == text`
---

## Mini-Project: BPE Tokenizer

Build a complete BPE tokenizer and analyze its behavior.

### Requirements

1. **Training**: Implement BPE training from scratch
2. **Encoding**: Implement the encoding algorithm
3. **Analysis**: Compare with tiktoken on the same text

### Deliverables

- [ ] BPE training implementation
- [ ] Encode/decode that roundtrips correctly
- [ ] Vocabulary analysis:
  - Token length distribution
  - Most common tokens
  - Tokens per word (fertility)
- [ ] Comparison with tiktoken

### Extension

Implement byte-level BPE (like GPT-2). How does it handle Unicode?

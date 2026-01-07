# Stage 7: Common Mistakes

## Mistake 1: Not Handling Whitespace Consistently

**Symptom**: Tokens don't roundtrip correctly (decode(encode(text)) != text)

**Wrong**:
```python
# Tokenizing without preserving whitespace
tokens = text.split()  # Loses all whitespace information
```

**The fix**: Treat whitespace as part of tokens (GPT-style)
```python
# Mark word boundaries with special character
text = " " + text.replace(" ", " Ġ")  # Ġ marks word start
# Or use bytes (like GPT-2)
```

---

## Mistake 2: Vocabulary Too Small

**Symptom**: Very long sequences, many multi-token words

**Problem**:
```python
tokenizer = BPE(vocab_size=1000)  # Too small for real text
# Result: "artificial" -> ["art", "if", "ic", "ial"]  # 4 tokens
```

**The fix**: Use appropriate vocabulary size
- Small corpora: 8,000-16,000
- Large corpora: 32,000-100,000
- Multilingual: 100,000+

---

## Mistake 3: Vocabulary Too Large

**Symptom**: Many rare tokens never seen during training, embedding matrix too large

**Problem**:
```python
tokenizer = BPE(vocab_size=500000)  # Way too big
# Result: Huge embedding matrix, many unused tokens
```

**The fix**: Vocab size should be proportional to training data
```python
# Rule of thumb: each token should appear 100+ times
vocab_size = min(len(unique_substrings), len(corpus) // 100)
```

---

## Mistake 4: Case Sensitivity Mismatch

**Symptom**: "Hello" and "hello" treated completely differently

**Problem**:
```python
# Trained on lowercase, applied to mixed case
tokenizer.train(corpus.lower())
tokens = tokenizer.encode("HELLO WORLD")  # Many unknowns!
```

**The fix**: Be consistent, or handle case explicitly
```python
# Option 1: Normalize during training AND inference
text = text.lower()

# Option 2: Include both cases in training
# Option 3: Use byte-level (handles all cases)
```

---

## Mistake 5: Not Preserving Special Tokens

**Symptom**: Model outputs corrupted special tokens

**Wrong**:
```python
# Special tokens get split during BPE
vocab = train_bpe(corpus)  # <pad>, <eos> might get split
```

**The fix**: Add special tokens BEFORE training, protect them
```python
SPECIAL_TOKENS = ["<pad>", "<eos>", "<bos>", "<unk>"]
# Add to vocab first, never merge them
vocab = SPECIAL_TOKENS + list(bpe_vocab)
```

---

## Mistake 6: Byte-Pair Order Matters

**Symptom**: Encoding is non-deterministic or inconsistent

**Wrong**:
```python
def encode(self, text):
    # Applying merges in wrong order
    for pair, merged in self.merges:  # Should be in training order!
        text = text.replace(pair, merged)
```

**The fix**: Apply merges in the exact order they were learned
```python
def encode(self, text):
    tokens = list(text)
    for pair, merged in self.merges:  # Order matters!
        i = 0
        while i < len(tokens) - 1:
            if (tokens[i], tokens[i+1]) == pair:
                tokens = tokens[:i] + [merged] + tokens[i+2:]
            else:
                i += 1
    return tokens
```

---

## Mistake 7: Not Handling Unknown Characters

**Symptom**: Crash or corruption on unusual Unicode

**Wrong**:
```python
def encode(self, text):
    return [self.vocab[c] for c in text]  # KeyError on unknown!
```

**The fix**: Fall back to byte-level or UNK
```python
def encode(self, text):
    tokens = []
    for c in text:
        if c in self.vocab:
            tokens.append(self.vocab[c])
        else:
            # Fall back to bytes
            for byte in c.encode('utf-8'):
                tokens.append(self.byte_vocab[byte])
    return tokens
```

---

## Mistake 8: Greedy Tokenization for Unigram

**Symptom**: Suboptimal tokenization, worse perplexity

**Wrong**:
```python
def tokenize(self, text):
    # Greedy: always take longest match
    tokens = []
    while text:
        for length in range(len(text), 0, -1):
            if text[:length] in self.vocab:
                tokens.append(text[:length])
                text = text[length:]
                break
    return tokens
```

**The fix**: Use Viterbi algorithm for optimal tokenization
```python
def tokenize(self, text):
    """Dynamic programming for optimal tokenization."""
    n = len(text)
    best_score = [-float('inf')] * (n + 1)
    best_score[0] = 0
    best_split = [0] * (n + 1)

    for i in range(1, n + 1):
        for j in range(i):
            token = text[j:i]
            if token in self.vocab:
                score = best_score[j] + self.log_prob[token]
                if score > best_score[i]:
                    best_score[i] = score
                    best_split[i] = j

    # Backtrack to get tokens
    tokens = []
    i = n
    while i > 0:
        j = best_split[i]
        tokens.append(text[j:i])
        i = j
    return tokens[::-1]
```

---

## Mistake 9: Training on Wrong Text Distribution

**Symptom**: Poor tokenization on target domain

**Problem**:
```python
# Trained on Wikipedia, applied to code
tokenizer = train_bpe(wikipedia_text)
tokens = tokenizer.encode("def __init__(self):")  # Very fragmented
```

**The fix**: Train on representative data
```python
# Include target domain in training data
training_corpus = wikipedia + code_samples + target_domain
tokenizer = train_bpe(training_corpus)
```

---

## Mistake 10: Ignoring Efficiency in Encoding

**Symptom**: Encoding is extremely slow for long texts

**Wrong**:
```python
def encode(self, text):
    # O(n * m) for each merge, O(n * m * v) total
    for pair, merged in self.merges:
        # String replacement is slow
        text = text.replace(pair[0] + pair[1], merged)
```

**The fix**: Use proper data structures
```python
def encode(self, text):
    # Use a trie for efficient prefix matching
    # Or pre-compute merge priorities
    tokens = list(text)
    while True:
        # Find best pair in one pass
        best_pair = None
        best_priority = float('inf')
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            if pair in self.merge_priority:
                if self.merge_priority[pair] < best_priority:
                    best_priority = self.merge_priority[pair]
                    best_pair = (i, pair)

        if best_pair is None:
            break
        # Merge best pair
        i, pair = best_pair
        tokens = tokens[:i] + [self.merges[pair]] + tokens[i+2:]

    return tokens
```

---

## Mistake 11: Not Normalizing Input Text

**Symptom**: Different encodings for visually identical text

**Problem**:
```python
# Unicode normalization issues
text1 = "café"  # 'é' as single character
text2 = "café"  # 'e' + combining accent
tokenizer.encode(text1) != tokenizer.encode(text2)  # Different!
```

**The fix**: Normalize Unicode before tokenization
```python
import unicodedata

def normalize(text):
    # NFC: Composed form (single characters)
    # NFKC: Also normalizes compatibility characters
    return unicodedata.normalize('NFC', text)

def encode(self, text):
    text = normalize(text)
    return self._encode(text)
```

# Section 7.5: Unigram Language Model

*Reading time: 12 minutes*

## A Different Approach

BPE and WordPiece are **bottom-up** algorithms:

- Start with characters
- Iteratively merge pairs
- Build up to larger tokens

Unigram is **top-down**:

- Start with a large vocabulary
- Iteratively remove tokens
- Prune down to target size

## The Unigram Model

A unigram language model assigns a probability to each token independently:

$$P(\text{tokenization}) = \prod_{i} P(t_i)$$

For a segmentation $\mathbf{t} = (t_1, t_2, ..., t_n)$ of text $x$:

$$P(\mathbf{t} | x) = \prod_{i=1}^{n} P(t_i)$$

### Example

If our vocabulary has probabilities:

- P("un") = 0.05
- P("believ") = 0.01
- P("able") = 0.03

Then:

$$P(\text{"un", "believ", "able"}) = 0.05 \times 0.01 \times 0.03 = 0.000015$$

## The Key Insight

For a given text, there are many possible segmentations:

```
"unbelievable" could be:
  ["u", "n", "b", "e", "l", "i", "e", "v", "a", "b", "l", "e"]
  ["un", "believable"]
  ["un", "believ", "able"]
  ["unbeliev", "able"]
  ...
```

The **optimal segmentation** maximizes the probability:

$$\mathbf{t}^* = \argmax_{\mathbf{t}} P(\mathbf{t} | x) = \argmax_{\mathbf{t}} \prod_{i} P(t_i)$$

Equivalently, minimize negative log probability:

$$\mathbf{t}^* = \argmin_{\mathbf{t}} \sum_{i} -\log P(t_i)$$

## Training Algorithm

### Step 1: Initialize Large Vocabulary

Start with all substrings up to some maximum length:

```python
def initialize_vocabulary(corpus, max_length=16):
    """Create initial large vocabulary from all substrings."""
    token_freqs = Counter()

    for text in corpus:
        words = text.split()
        for word in words:
            for i in range(len(word)):
                for j in range(i+1, min(i+max_length+1, len(word)+1)):
                    token_freqs[word[i:j]] += 1

    return token_freqs
```

This creates a vocabulary of ~100K tokens.

### Step 2: Compute Token Probabilities

Using EM (Expectation-Maximization):

**E-step**: For each word, compute expected counts under all segmentations

**M-step**: Update token probabilities based on expected counts

Simplified version (just use frequencies):

$$P(t) = \frac{\text{freq}(t)}{\sum_{t'} \text{freq}(t')}$$

### Step 3: Compute Token Impact

For each token, compute how much removing it would hurt:

$$\text{loss}(t) = \sum_{x \in \text{corpus}} \left[ L(x | V \setminus \{t\}) - L(x | V) \right]$$

where $L(x | V)$ is the negative log probability of the best segmentation of x using vocabulary V.

### Step 4: Remove Low-Impact Tokens

Remove tokens with lowest impact until reaching target vocabulary size.

## The Viterbi Algorithm for Encoding

Given vocabulary V with probabilities, find the best segmentation using dynamic programming:

```python
def encode_unigram(text: str, vocab: Dict[str, float]) -> List[str]:
    """Find optimal segmentation using Viterbi algorithm."""
    n = len(text)

    # best_score[i] = best score for text[:i]
    # best_split[i] = index of last split before position i
    best_score = [float('inf')] * (n + 1)
    best_split = [0] * (n + 1)
    best_score[0] = 0

    for i in range(1, n + 1):
        for j in range(max(0, i - max_token_length), i):
            substr = text[j:i]
            if substr in vocab:
                score = best_score[j] - log(vocab[substr])
                if score < best_score[i]:
                    best_score[i] = score
                    best_split[i] = j

    # Backtrack to get tokens
    tokens = []
    pos = n
    while pos > 0:
        prev = best_split[pos]
        tokens.append(text[prev:pos])
        pos = prev

    return tokens[::-1]  # Reverse to get correct order
```

### Complexity

- Time: O(n × L) where n is text length, L is max token length
- Space: O(n)

## Why Unigram Works

### Optimal Segmentation

Unlike BPE/WordPiece which apply fixed merge rules, Unigram finds the **globally optimal** segmentation for each input:

| Algorithm | Encoding Strategy |
|-----------|-------------------|
| BPE | Apply merges in order (greedy, fixed) |
| WordPiece | Longest match (greedy, per-word) |
| Unigram | Viterbi search (optimal, global) |

### Principled Pruning

When removing tokens, Unigram explicitly optimizes corpus likelihood:

$$V^* = \argmin_{V, |V|=k} \sum_{x \in \text{corpus}} L(x | V)$$

This is more principled than BPE's "just add frequent pairs."

### Subword Regularization

Unigram enables **subword regularization** during training:

Instead of always using the best segmentation, sample from all possible segmentations:

$$\mathbf{t} \sim P(\mathbf{t} | x)$$

This creates data augmentation:

```
Same word, different segmentations:
  "playing" → ["play", "ing"]      (sampled)
  "playing" → ["pla", "ying"]      (sampled)
  "playing" → ["p", "laying"]      (sampled)
```

The model becomes robust to tokenization choices.

## Comparison: BPE vs. WordPiece vs. Unigram

| Aspect | BPE | WordPiece | Unigram |
|--------|-----|-----------|---------|
| Direction | Bottom-up | Bottom-up | Top-down |
| Criterion | Frequency | Likelihood | Likelihood |
| Encoding | Apply merges | Longest match | Viterbi |
| Optimality | Local | Local (per-word) | Global |
| Regularization | No | No | Yes |
| Speed | Fast | Fast | Slower |

## SentencePiece

SentencePiece is a popular library that implements Unigram (and BPE):

```python
import sentencepiece as spm

# Train
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='my_model',
    vocab_size=32000,
    model_type='unigram'  # or 'bpe'
)

# Load and use
sp = spm.SentencePieceProcessor()
sp.load('my_model.model')

tokens = sp.encode_as_pieces("Hello world")
# ['▁Hello', '▁world']
```

Note: SentencePiece uses `▁` (Unicode U+2581) to mark word starts, not spaces.

## When to Use Unigram

**Prefer Unigram when:**

- You want optimal segmentation (not just fast encoding)
- You need subword regularization for robustness
- Training time is not critical

**Prefer BPE when:**

- Encoding speed is critical
- You want simple, deterministic tokenization
- Following GPT conventions

## Summary

| Aspect | Description |
|--------|-------------|
| **Approach** | Top-down vocabulary pruning |
| **Model** | Unigram language model over tokens |
| **Encoding** | Viterbi (dynamic programming) |
| **Training** | EM + pruning by loss impact |
| **Advantage** | Globally optimal segmentation |
| **Used by** | SentencePiece, T5, mBART |

**Next**: We'll examine the trade-offs in vocabulary size selection.

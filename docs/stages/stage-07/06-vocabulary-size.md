# Section 7.6: Vocabulary Size Trade-offs

*Reading time: 10 minutes*

## The Central Question

How large should your vocabulary be?

Modern LLMs use vocabularies ranging from 32K to 100K+ tokens. This is not arbitrary—it reflects careful optimization of competing concerns.

## The Trade-off Landscape

### Larger Vocabulary

**Pros:**

- Shorter sequences (fewer tokens per text)
- More whole words as single tokens
- Faster inference (fewer steps)
- Better representation of common phrases

**Cons:**

- Larger embedding matrix (vocab_size × embedding_dim)
- More parameters in output layer
- Rarer tokens get less training signal
- Harder to learn embeddings for infrequent tokens

### Smaller Vocabulary

**Pros:**

- Smaller model size
- Each token gets more training examples
- Better embeddings for all tokens
- More compositional representations

**Cons:**

- Longer sequences (more tokens per text)
- Slower inference
- Attention cost grows: O(sequence_length²)
- May lose word-level semantics

## Quantifying the Trade-off

### Parameter Cost

For a model with:

- Vocabulary size: V
- Embedding dimension: d
- Number of layers: L
- Hidden dimension: h

The vocabulary-related parameters:

$$\text{Embedding matrix} = V \times d$$
$$\text{Output projection} = h \times V$$
$$\text{Total vocab params} = V \times (d + h)$$

For GPT-3 scale (d = h = 12,288):

| Vocab Size | Vocab Params | % of Total |
|------------|--------------|------------|
| 32K | 800M | ~0.5% |
| 50K | 1.2B | ~0.7% |
| 100K | 2.5B | ~1.4% |

Vocabulary is a small fraction of total parameters for large models.

### Sequence Length Cost

Attention has complexity O(n²). For the same text:

| Vocab Size | Avg Tokens/Word | Relative Attention Cost |
|------------|-----------------|-------------------------|
| 1K (char-like) | 5.0 | 25× |
| 10K | 2.0 | 4× |
| 50K | 1.3 | 1.7× |
| 100K | 1.1 | 1.2× |
| 500K (word-like) | 1.0 | 1× |

Diminishing returns: going from 50K to 500K saves little sequence length.

## The Optimal Range

### Empirical Findings

Research and practice suggest:

$$\text{Optimal vocab size} \approx 30\text{K} - 100\text{K}$$

| Model | Vocab Size | Tokenizer |
|-------|------------|-----------|
| GPT-2 | 50,257 | BPE |
| GPT-3 | 50,257 | BPE |
| GPT-4 | ~100K | BPE variant |
| BERT | 30,522 | WordPiece |
| LLaMA | 32,000 | SentencePiece |
| LLaMA-2 | 32,000 | SentencePiece |
| Claude | ~100K | BPE variant |
| Gemini | ~256K | SentencePiece |

### Why This Range?

1. **Below 30K**: Sequences too long, attention too expensive
2. **30K-100K**: Sweet spot for most languages
3. **Above 100K**: Diminishing returns, rare token quality suffers

## Domain-Specific Considerations

### Code Models

Programming has different statistics than natural language:

```python
# Code has many unique identifiers
my_variable_name_42 = compute_something()
```

Code-focused models often use larger vocabularies to capture:

- Common function names
- Library-specific tokens
- Identifier patterns

### Multilingual Models

For multilingual models:

$$V_{\text{total}} = V_{\text{shared}} + \sum_{\text{lang}} V_{\text{lang-specific}}$$

Options:

1. **Shared vocabulary**: Train on mixed data
   - More efficient sharing
   - May under-represent low-resource languages

2. **Per-language allocation**: Reserve capacity for each
   - Better coverage per language
   - Less efficient overall

Example: mBERT uses 105K vocabulary for 104 languages.

### Low-Resource Languages

For languages with limited training data:

$$\text{Token quality} \propto \sqrt{\text{training examples per token}}$$

A token seen 1M times has much better embeddings than one seen 100 times.

Larger vocabulary → more rare tokens → worse embeddings

## Compression Ratio Analysis

**Compression ratio**: characters per token

```python
def compression_ratio(tokenizer, corpus):
    total_chars = sum(len(text) for text in corpus)
    total_tokens = sum(len(tokenizer.encode(text)) for text in corpus)
    return total_chars / total_tokens
```

Typical values:

| Tokenizer | Compression Ratio |
|-----------|-------------------|
| Character | 1.0 |
| BPE (50K) | 3.5-4.0 |
| BPE (100K) | 4.0-4.5 |
| Word-level | 5.0-6.0 |

Higher is better (fewer tokens needed).

## Fertility Analysis

**Fertility**: tokens per word

```python
def fertility(tokenizer, corpus):
    total_words = sum(len(text.split()) for text in corpus)
    total_tokens = sum(len(tokenizer.encode(text)) for text in corpus)
    return total_tokens / total_words
```

Typical values:

| Language | BPE (50K) Fertility |
|----------|---------------------|
| English | 1.2-1.4 |
| German | 1.4-1.6 |
| Chinese | 1.8-2.2 |
| Code | 2.0-3.0 |

Lower fertility = more efficient tokenization.

## Practical Guidelines

### Choosing Vocabulary Size

1. **Start with 32K-50K** for monolingual models
2. **Scale to 100K** for multilingual or code-heavy models
3. **Measure compression ratio** on your target domain
4. **Check rare token frequency**: tokens appearing < 100 times may have poor embeddings

### Training Considerations

```python
# Ensure all tokens get sufficient training
min_frequency = 100  # Don't include tokens rarer than this

# For fine-tuning, consider vocabulary expansion
if domain_specific:
    # Add domain tokens to vocabulary
    # Re-initialize their embeddings
    extend_vocabulary(tokenizer, domain_tokens)
```

### Evaluation Metrics

| Metric | Good Value | What It Measures |
|--------|------------|------------------|
| Compression ratio | > 3.5 | Efficiency |
| Fertility | < 1.5 | Tokens per word |
| OOV rate | < 0.1% | Coverage |
| Singleton rate | < 5% | Token quality |

## Summary

| Vocab Size | Best For |
|------------|----------|
| < 10K | Character-like, special cases |
| 10K-30K | Resource-constrained models |
| 30K-50K | Standard monolingual models |
| 50K-100K | Large models, multilingual, code |
| > 100K | Very large models, many languages |

**Rule of thumb**: When in doubt, use 32K-50K.

**Next**: We'll implement all three tokenization algorithms from scratch.

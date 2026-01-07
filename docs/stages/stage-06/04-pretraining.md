# Section 6.4: Pre-training Objectives — What Should the Model Learn?

*Reading time: 20 minutes | Difficulty: ★★★☆☆*

The pre-training objective defines what task the model learns on vast amounts of unlabeled text. This section examines different objectives and their trade-offs.

## The Pre-training Paradigm

Modern LLMs follow a two-stage process:

```
Stage 1: Pre-training
├── Massive unlabeled data (TB of text)
├── Self-supervised objective
├── Learn general language understanding
└── Expensive (millions of dollars for large models)

Stage 2: Fine-tuning / Alignment
├── Smaller labeled/curated data
├── Task-specific or general instruction following
├── Adapt to specific use cases
└── Much cheaper
```

The pre-training objective determines what representations the model learns.

## Causal Language Modeling (CLM)

**Used by:** GPT, LLaMA, Claude, most modern LLMs

### The Objective

Predict the next token given all previous tokens:

$$\mathcal{L}_{CLM} = -\sum_{t=1}^{T} \log P(x_t | x_1, ..., x_{t-1})$$

### How It Works

```
Input:  "The cat sat on the"
Target: "cat sat on the mat"

Position 1: "The" → predict "cat"
Position 2: "The cat" → predict "sat"
Position 3: "The cat sat" → predict "on"
Position 4: "The cat sat on" → predict "the"
Position 5: "The cat sat on the" → predict "mat"
```

### Implementation

```python
def causal_lm_loss(model, tokens):
    """
    Compute causal language modeling loss.

    Args:
        model: Language model
        tokens: Token sequence [seq_len]

    Returns:
        Average cross-entropy loss
    """
    # Forward pass (model applies causal mask internally)
    logits = model.forward(tokens[:-1])  # [seq_len-1, vocab_size]

    # Targets are next tokens
    targets = tokens[1:]  # [seq_len-1]

    # Cross-entropy loss
    log_probs = log_softmax(logits, axis=-1)
    loss = -log_probs[range(len(targets)), targets].mean()

    return loss
```

### Advantages of CLM

| Advantage | Explanation |
|-----------|-------------|
| Natural for generation | Training matches inference task |
| Simple | Just predict the next token |
| Scalable | Works on any text, no labeling needed |
| Efficient | Single forward pass predicts all positions |

### Architecture: Decoder-Only

CLM uses a decoder-only Transformer with causal masking:

```
Causal mask: each position sees only past

    t1 t2 t3 t4 t5
t1 [ ✓  ✗  ✗  ✗  ✗ ]
t2 [ ✓  ✓  ✗  ✗  ✗ ]
t3 [ ✓  ✓  ✓  ✗  ✗ ]
t4 [ ✓  ✓  ✓  ✓  ✗ ]
t5 [ ✓  ✓  ✓  ✓  ✓ ]
```

## Masked Language Modeling (MLM)

**Used by:** BERT, RoBERTa

### The Objective

Randomly mask some tokens and predict them:

$$\mathcal{L}_{MLM} = -\sum_{t \in \text{masked}} \log P(x_t | x_{\text{context}})$$

### How It Works

```
Original: "The cat sat on the mat"
Masked:   "The [MASK] sat on the [MASK]"
Target:   Predict "cat" and "mat"
```

Masking strategy (BERT):
- 15% of tokens are selected
- Of selected: 80% → [MASK], 10% → random token, 10% → unchanged

### Implementation

```python
def masked_lm_loss(model, tokens, mask_prob=0.15):
    """
    Compute masked language modeling loss.

    Args:
        model: Bidirectional language model
        tokens: Token sequence [seq_len]
        mask_prob: Probability of masking each token

    Returns:
        Loss on masked positions only
    """
    # Create mask (which positions to predict)
    mask = np.random.random(len(tokens)) < mask_prob

    # Create corrupted input
    corrupted = tokens.copy()
    for i in np.where(mask)[0]:
        r = np.random.random()
        if r < 0.8:
            corrupted[i] = MASK_TOKEN  # Replace with [MASK]
        elif r < 0.9:
            corrupted[i] = np.random.randint(vocab_size)  # Random token
        # else: keep original (10%)

    # Forward pass (bidirectional, no causal mask)
    logits = model.forward(corrupted)  # [seq_len, vocab_size]

    # Loss only on masked positions
    log_probs = log_softmax(logits, axis=-1)
    masked_positions = np.where(mask)[0]
    loss = -log_probs[masked_positions, tokens[masked_positions]].mean()

    return loss
```

### Architecture: Encoder-Only (Bidirectional)

MLM uses bidirectional attention—each position can see all others:

```
Full attention: each position sees all

    t1 t2 t3 t4 t5
t1 [ ✓  ✓  ✓  ✓  ✓ ]
t2 [ ✓  ✓  ✓  ✓  ✓ ]
t3 [ ✓  ✓  ✓  ✓  ✓ ]
t4 [ ✓  ✓  ✓  ✓  ✓ ]
t5 [ ✓  ✓  ✓  ✓  ✓ ]
```

### CLM vs MLM

| Aspect | Causal LM | Masked LM |
|--------|-----------|-----------|
| Direction | Left-to-right only | Bidirectional |
| Generation | Natural | Requires special decoding |
| Context | Past only | Full context |
| Efficiency | All positions trained | Only ~15% trained |
| Modern use | GPT, LLaMA, Claude | BERT, RoBERTa (mostly NLU) |

## Prefix Language Modeling

**Used by:** T5, some instruction models

### The Objective

Combine bidirectional context (prefix) with causal generation:

```
Input:  "[Translate English to French:] The cat sat"
Output: "Le chat s'est assis"

Prefix (bidirectional): "Translate English to French: The cat sat"
Generation (causal): "Le chat s'est assis"
```

### Attention Pattern

```
     prefix         generation
   [p1 p2 p3]      [g1 g2 g3]

p1 [ ✓  ✓  ✓        ✗  ✗  ✗ ]
p2 [ ✓  ✓  ✓        ✗  ✗  ✗ ]
p3 [ ✓  ✓  ✓        ✗  ✗  ✗ ]
g1 [ ✓  ✓  ✓        ✓  ✗  ✗ ]
g2 [ ✓  ✓  ✓        ✓  ✓  ✗ ]
g3 [ ✓  ✓  ✓        ✓  ✓  ✓ ]
```

This allows bidirectional understanding of the input while generating causally.

## Span Corruption (T5)

**Used by:** T5, UL2

### The Objective

Replace spans of text with sentinel tokens, then generate the spans:

```
Original: "The cute cat sat on the warm mat"
Corrupted: "The <X> sat on <Y> mat"
Target: "<X> cute cat <Y> the warm"
```

### Why Span Corruption?

- Teaches copying and generation simultaneously
- More challenging than single-token MLM
- Better for sequence-to-sequence tasks

## Denoising Objectives

Various ways to corrupt input and train recovery:

| Method | Corruption | Used By |
|--------|------------|---------|
| Token deletion | Remove tokens randomly | BART |
| Token infilling | Replace spans with single mask | BART |
| Sentence permutation | Shuffle sentence order | BART |
| Document rotation | Rotate to random start point | BART |

### BART's Approach

```
Original: "The cat sat. The dog ran."

Possible corruptions:
1. Token deletion: "The sat. The ran."
2. Text infilling: "The [MASK] sat. The [MASK] ran."
3. Sentence shuffle: "The dog ran. The cat sat."
4. Rotation: "sat. The dog ran. The cat"

Target: Reconstruct original
```

## Next Sentence Prediction (NSP)

**Used by:** Original BERT

### The Objective

Given two sentences, predict if the second follows the first:

```
Positive: "The cat sat on the mat." + "It was very comfortable."
Label: IsNext

Negative: "The cat sat on the mat." + "Pizza is delicious."
Label: NotNext
```

### Why It Was Dropped

Later research (RoBERTa) showed NSP doesn't help and may hurt. Most modern models don't use it.

## Contrastive Learning

**Used by:** Some multimodal models (CLIP)

### The Objective

Learn representations by contrasting positive and negative pairs:

```
Positive pair: (text, matching_image)
Negative pairs: (text, random_images)

Objective: Maximize similarity for positive, minimize for negatives
```

While not common for pure text LLMs, this is important for multimodal models.

## Comparing Objectives

| Objective | Best For | Training Efficiency | Generation |
|-----------|----------|---------------------|------------|
| Causal LM | Generation, few-shot | High | Natural |
| Masked LM | Understanding, classification | Medium (15%) | Difficult |
| Prefix LM | Conditional generation | High | Natural |
| Span Corruption | Seq2seq tasks | Medium | Natural |

## Modern Choices

### Why Causal LM Dominates

1. **Unified training and inference**: Same left-to-right process
2. **Emergent abilities**: In-context learning, chain-of-thought
3. **Scalability**: Simple to scale to trillions of tokens
4. **Versatility**: Can be adapted to any task via prompting

### The Case for Bidirectional

For some tasks (classification, NER, QA), bidirectional context helps:

```
"The bank was steep."  # bank = riverbank
"The bank was closed." # bank = financial

Bidirectional model sees "steep" → understands "riverbank"
Left-to-right model must guess at "bank" without seeing "steep"
```

This is why BERT-like models still excel at some NLU benchmarks.

!!! info "Connection to Modern LLMs"

    Current LLM pre-training:

    - **GPT-4, Claude**: Pure causal LM (assumed)
    - **LLaMA**: Causal LM
    - **T5**: Span corruption (encoder-decoder)
    - **BERT/RoBERTa**: MLM (classification/NLU focus)

    The trend is strongly toward causal LM for general-purpose models, with MLM reserved for specialized understanding tasks.

## Implementation Considerations

### Data Formatting

Causal LM needs to see diverse "tasks" in pre-training:

```
# Document completion
"The quick brown fox jumps over the lazy dog."

# Dialogue
"User: What is 2+2?\nAssistant: 4"

# Code
"def factorial(n):\n    if n == 0: return 1\n    return n * factorial(n-1)"

# Multiple formats help with generalization
```

### Packing Sequences

For efficiency, pack multiple documents into one sequence:

```
Naive: [Doc1] [PAD] [PAD] [Doc2] [PAD] [Doc3] [PAD] [PAD]
Packed: [Doc1] [SEP] [Doc2] [SEP] [Doc3] [SEP] [Doc4] [SEP]
```

Packing wastes no compute on padding tokens.

## Exercises

1. **Compare objectives**: Train small models with CLM and MLM. Compare on generation and classification.

2. **Masking rate**: For MLM, try 10%, 15%, 30% masking. What works best?

3. **Span lengths**: Implement span corruption. How does span length affect learning?

4. **Prefix proportion**: In prefix LM, vary the prefix length. What's optimal?

5. **Ablate NSP**: Train BERT with and without NSP. Is there a difference?

## Summary

| Objective | Key Idea | Architecture | Best For |
|-----------|----------|--------------|----------|
| Causal LM | Predict next token | Decoder-only | Generation |
| Masked LM | Predict masked tokens | Encoder-only | Understanding |
| Prefix LM | Bidirectional prefix + causal | Encoder-decoder | Conditional generation |
| Span Corruption | Predict corrupted spans | Encoder-decoder | Seq2seq |

**Key takeaway**: The pre-training objective shapes what the model learns. Causal LM (predict next token) has become dominant because it naturally supports generation and scales well. Masked LM provides better bidirectional understanding but is harder to use for generation. Modern general-purpose LLMs almost universally use causal LM, while specialized understanding models may still use MLM.

→ **Next**: [Section 6.5: Training at Scale](05-training-scale.md)

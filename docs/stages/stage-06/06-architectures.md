# Section 6.6: Modern Architectures — GPT, LLaMA, and Beyond

*Reading time: 20 minutes | Difficulty: ★★★☆☆*

While the core Transformer architecture has remained stable, modern LLMs incorporate numerous refinements. This section surveys the architectural choices in state-of-the-art models.

## The Evolution of Architectures

```
2017: Original Transformer (encoder-decoder)
2018: GPT-1 (decoder-only)
2019: GPT-2 (larger, pre-norm)
2020: GPT-3 (175B, few-shot learning)
2022: ChatGPT (RLHF-aligned)
2023: GPT-4, LLaMA, Claude 2
2024: LLaMA 3, Claude 3, Mistral, Mixtral
```

Each generation brings refinements that improve efficiency, capability, or both.

## Decoder-Only vs Encoder-Decoder

### Encoder-Decoder (T5, BART)

```
Encoder (bidirectional):
  Input: "Translate to French: Hello"
  Output: Encoded representations

Decoder (causal):
  Input: Encoded + "Bonjour"
  Output: Next token probabilities
```

**Pros:** Bidirectional encoding, good for translation/summarization
**Cons:** More complex, two separate components

### Decoder-Only (GPT, LLaMA, Claude)

```
Input:  "Translate to French: Hello"
Output: " Bonjour"

All in one causal model.
```

**Pros:** Simpler, unified, scales well
**Cons:** No bidirectional context (but seems fine at scale)

**Winner:** Decoder-only has become dominant for general-purpose LLMs.

## GPT Architecture

The original influential decoder-only design:

### GPT-2 Specifics

```python
GPT2Config = {
    # Sizes: small, medium, large, xl
    'd_model': [768, 1024, 1280, 1600],
    'n_layers': [12, 24, 36, 48],
    'n_heads': [12, 16, 20, 25],
    'vocab_size': 50257,
    'max_seq_len': 1024,

    # Architecture
    'activation': 'gelu',
    'layer_norm': 'pre_norm',
    'tie_embeddings': True,  # Input/output embeddings shared
}
```

### Key GPT Innovations

| Feature | Details |
|---------|---------|
| Pre-norm | LayerNorm before attention/FFN |
| GELU activation | Smoother than ReLU |
| Learned positions | Not sinusoidal |
| BPE tokenization | 50K vocabulary |

## LLaMA Architecture

Meta's open-weight models with modern refinements:

### LLaMA 2 Specifics

```python
LLaMA2Config = {
    # 7B, 13B, 70B variants
    'd_model': [4096, 5120, 8192],
    'n_layers': [32, 40, 80],
    'n_heads': [32, 40, 64],
    'd_ff': [11008, 13824, 28672],  # ~2.7× d_model (not 4×)
    'vocab_size': 32000,
    'max_seq_len': 4096,

    # Architecture
    'activation': 'silu',  # SwiGLU variant
    'layer_norm': 'rmsnorm',  # Simpler than LayerNorm
    'positional': 'rope',  # Rotary embeddings
    'tie_embeddings': False,
}
```

### LLaMA Innovations

| Feature | Improvement Over GPT |
|---------|---------------------|
| RMSNorm | Faster, simpler |
| SwiGLU | Better quality |
| RoPE | Better length generalization |
| Grouped-Query Attention (GQA) | Faster inference |

### Grouped-Query Attention

Standard: Each attention head has its own Q, K, V

GQA: Multiple query heads share K, V heads

```
Standard MHA (8 heads):
  Q: 8 heads × d_k
  K: 8 heads × d_k
  V: 8 heads × d_k

GQA (8 query heads, 2 KV heads):
  Q: 8 heads × d_k
  K: 2 heads × d_k  (shared by 4 query heads each)
  V: 2 heads × d_k

Benefit: Smaller KV cache, faster inference
```

## Mistral Architecture

Efficient model with sliding window attention:

### Mistral 7B Specifics

```python
MistralConfig = {
    'd_model': 4096,
    'n_layers': 32,
    'n_heads': 32,
    'n_kv_heads': 8,  # GQA
    'vocab_size': 32000,
    'max_seq_len': 8192,
    'sliding_window': 4096,  # Local attention window

    'activation': 'silu',
    'layer_norm': 'rmsnorm',
    'positional': 'rope',
}
```

### Sliding Window Attention

Instead of attending to all previous tokens:

```
Full attention (O(n²)):
Token 1000 attends to tokens 0-999

Sliding window (O(n×w)):
Token 1000 attends to tokens 996-999 (window=4)

BUT: Through multiple layers, information propagates:
  Layer 1: see 4 tokens back
  Layer 2: see 8 tokens back (4 × 2)
  Layer 32: see 128 tokens back
```

This enables long contexts with less compute.

## Mixtral (Mixture of Experts)

Sparse model that activates only part of the network:

### MoE Architecture

```
                Input
                  │
                  ▼
         ┌───────────────┐
         │    Router     │ → Selects 2 of 8 experts
         └───────────────┘
                  │
    ┌───────┬─────┼─────┬───────┐
    │       │     │     │       │
    ▼       ▼     ▼     ▼       ▼
  Exp1   Exp2   Exp3  ...    Exp8
    │       │     │     │       │
    └───────┴─────┼─────┴───────┘
                  │
            Weighted sum
                  │
                  ▼
               Output
```

### MoE Benefits

| Aspect | Dense Model | MoE Model |
|--------|-------------|-----------|
| Parameters | 7B | 46B (8 experts) |
| Active params | 7B | 12B (2 experts) |
| Quality | Good | Better (more params) |
| Inference speed | Baseline | Similar (same active params) |

MoE provides "more parameters for the same compute."

## Modern Architectural Components

### Activation Functions

```python
# ReLU (original)
def relu(x):
    return max(0, x)

# GELU (GPT-2, BERT)
def gelu(x):
    return 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x³)))

# SiLU/Swish (LLaMA)
def silu(x):
    return x * sigmoid(x)

# SwiGLU (LLaMA FFN)
def swiglu(x, W1, W2, W3):
    return (silu(x @ W1) * (x @ W3)) @ W2
```

SwiGLU requires an extra weight matrix but improves quality.

### Normalization

```python
# LayerNorm (GPT)
def layer_norm(x, gamma, beta):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True)
    return gamma * (x - mean) / sqrt(var + eps) + beta

# RMSNorm (LLaMA) - no mean subtraction, no beta
def rms_norm(x, gamma):
    rms = sqrt((x ** 2).mean(-1, keepdim=True))
    return gamma * x / (rms + eps)
```

RMSNorm is ~7% faster with similar quality.

### Positional Encoding

```python
# Learned (GPT)
pos_emb = Embedding(max_len, d_model)

# Sinusoidal (original Transformer)
# See Stage 5.6

# RoPE (LLaMA) - rotates query/key vectors
def rope(q, k, positions):
    # Rotate Q, K based on position
    # Dot product encodes relative position
    return rotate(q, positions), rotate(k, positions)

# ALiBi (alternative)
# Add distance-based bias to attention scores
scores = Q @ K.T - m * abs(i - j)
```

RoPE has become the dominant choice for modern LLMs.

## Architecture Comparison

| Component | GPT-2 | LLaMA 2 | Mistral | Mixtral |
|-----------|-------|---------|---------|---------|
| Attention | Full | Full | Sliding Window | Full |
| KV Heads | MHA | GQA | GQA | GQA |
| FFN | GELU | SwiGLU | SwiGLU | SwiGLU (MoE) |
| Norm | LayerNorm | RMSNorm | RMSNorm | RMSNorm |
| Position | Learned | RoPE | RoPE | RoPE |

## Emerging Techniques

### Flash Attention

Not an architecture change, but an efficient attention implementation:

```
Standard attention: O(n²) memory
Flash attention: O(n) memory, same result

Key idea: Compute attention in tiles, never materialize full n×n matrix
```

This enables much longer contexts (100K+ tokens).

### State Space Models (Mamba)

Alternative to attention with linear scaling:

```
Attention: O(n²) compute, O(n) memory
SSM/Mamba: O(n) compute, O(1) memory per token
```

Still being explored, may complement or replace attention.

### Multimodal Integration

Modern models handle text + images + audio:

```
Image: → Vision Encoder → Tokens
Text:  → Tokenizer → Tokens
Audio: → Audio Encoder → Tokens

All → Unified Transformer → Output
```

Architecture is extended but core Transformer remains.

## Architectural Trade-offs

### Quality vs Efficiency

```
More parameters → Better quality but slower
More layers → Better but harder to train
Larger d_model → Better but more memory
MoE → Better quality for same compute
GQA → Faster but slightly worse
```

### Length vs Compute

```
Full attention: Great quality, O(n²) compute
Sliding window: Good quality, O(n×w) compute
Linear attention: Unknown quality, O(n) compute
```

### Open vs Closed Weights

| Model | Open Weights | API Access |
|-------|--------------|------------|
| GPT-4 | No | Yes |
| Claude | No | Yes |
| LLaMA 2/3 | Yes | Also Yes |
| Mistral | Yes | Also Yes |

Open weights enable research and customization.

!!! info "Connection to Modern LLMs"

    The current frontier:

    - **GPT-4**: Rumored MoE with 8 experts, ~1.8T total params
    - **Claude 3**: Architecture not disclosed
    - **LLaMA 3**: Open weights, very strong performance
    - **Mistral Large**: Competitive with GPT-4

    The best architectures keep improving, but the core Transformer block remains recognizable.

## Exercises

1. **Implement RMSNorm**: Replace LayerNorm with RMSNorm. Compare speed.

2. **Implement GQA**: Modify multi-head attention to share KV heads.

3. **Sliding window**: Implement sliding window attention. Test on long sequences.

4. **Activation comparison**: Compare ReLU, GELU, SiLU on a small task.

5. **RoPE vs learned**: Compare rotary and learned positional encodings.

## Summary

| Model | Key Innovation | Impact |
|-------|----------------|--------|
| GPT | Decoder-only, pre-norm | Foundation of modern LLMs |
| LLaMA | RMSNorm, SwiGLU, RoPE | Efficient open model |
| Mistral | Sliding window, GQA | Long context efficiency |
| Mixtral | MoE | More params, same compute |

**Key takeaway**: Modern LLM architectures build on the original Transformer with incremental improvements: RMSNorm for speed, SwiGLU for quality, RoPE for length generalization, and GQA for inference efficiency. More radical changes like MoE and sliding window attention address scaling challenges. Despite continuous refinement, the fundamental attention + FFN + residual structure remains unchanged since 2017.

→ **Next**: [Section 6.7: Scaling Laws](07-scaling-laws.md)

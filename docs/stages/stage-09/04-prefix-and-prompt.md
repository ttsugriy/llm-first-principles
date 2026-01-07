# Section 9.4: Prefix and Prompt Tuning

*Reading time: 12 minutes*

## A Different Approach

LoRA and adapters modify the model's weights (or add new layers). But there's another way:

**Keep the model completely frozen. Modify the inputs instead.**

This is the philosophy behind prefix tuning and prompt tuning.

## Prompt Tuning

### The Idea

Instead of handcrafted prompts like:

```
"Classify the following text as positive or negative: [text]"
```

Learn continuous "soft prompts"—vectors that are prepended to the input:

```
[learned_vector_1] [learned_vector_2] ... [learned_vector_n] [actual_input]
```

These soft prompts are not words—they're learned embeddings that can represent concepts not expressible in natural language.

### Implementation

```python
class PromptTuning:
    """Learned soft prompts prepended to input."""

    def __init__(self, d_model: int, prompt_length: int = 20):
        self.prompt_length = prompt_length
        # Learnable prompt embeddings
        self.prompt = np.random.randn(prompt_length, d_model) * 0.01

    def forward(self, input_embeds):
        """Prepend soft prompts to input."""
        batch_size = input_embeds.shape[0]

        # Expand prompt for batch
        prompt_batch = np.broadcast_to(
            self.prompt[np.newaxis, :, :],
            (batch_size, self.prompt_length, self.d_model)
        ).copy()

        # Concatenate: [prompt | input]
        return np.concatenate([prompt_batch, input_embeds], axis=1)

    def backward(self, grad_output):
        # Gradient for prompt tokens
        self.prompt_grad = grad_output[:, :self.prompt_length].sum(axis=0)
        # Pass through gradient for actual input
        return grad_output[:, self.prompt_length:]
```

### What Are Soft Prompts Learning?

Hard to interpret, but soft prompts seem to encode:

- Task instructions
- Output format preferences
- Domain-specific patterns

Unlike discrete prompts, they can represent "in-between" concepts that have no words.

### Hyperparameters

| Prompt Length | Parameters (d=4096) | Effect |
|--------------|---------------------|--------|
| 10 | 40K | Minimal capacity |
| 20 | 80K | Good default |
| 50 | 200K | More capacity |
| 100 | 400K | Maximum common |

**Start with 20 tokens.** Increase if task is complex.

## Prefix Tuning

### The Idea

Prompt tuning only modifies the input. Prefix tuning goes deeper:

**Learn prefix vectors for keys and values in every attention layer.**

```
Original attention:
    Q, K, V from input

With prefix tuning:
    K' = [prefix_keys | K]
    V' = [prefix_values | V]
    Attention(Q, K', V')
```

The model attends to both learned prefix tokens and actual input tokens.

### Why Keys and Values?

- **Keys** determine what tokens can be attended to
- **Values** determine what information is retrieved
- Queries still come from the input (so the model still "asks questions")

By prepending learned K and V, we give the model access to "virtual tokens" that steer its behavior.

### Implementation

```python
class PrefixTuning:
    """Learned key/value prefixes for attention layers."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_head: int,
        prefix_length: int = 10,
    ):
        # Prefixes for each layer: [K_prefix, V_prefix]
        # Shape: [num_layers, 2, prefix_length, num_heads, d_head]
        self.prefix = np.random.randn(
            num_layers, 2, prefix_length, num_heads, d_head
        ) * 0.01

    def get_prefix(self, layer_idx: int):
        """Get K and V prefixes for a specific layer."""
        prefix_k = self.prefix[layer_idx, 0]  # [prefix_len, heads, d_head]
        prefix_v = self.prefix[layer_idx, 1]
        return prefix_k, prefix_v
```

### Using Prefixes in Attention

```python
def attention_with_prefix(Q, K, V, prefix_k, prefix_v):
    """Attention with learned prefix tokens."""
    batch_size = Q.shape[0]

    # Expand prefixes for batch
    prefix_k = np.broadcast_to(prefix_k, (batch_size, *prefix_k.shape))
    prefix_v = np.broadcast_to(prefix_v, (batch_size, *prefix_v.shape))

    # Concatenate prefixes
    K_full = np.concatenate([prefix_k, K], axis=1)  # [batch, prefix+seq, ...]
    V_full = np.concatenate([prefix_v, V], axis=1)

    # Standard attention with expanded K, V
    return attention(Q, K_full, V_full)
```

### Parameter Count

For a model with:

- 32 layers
- 32 heads
- 128 d_head
- 10 prefix tokens

Prefix parameters: $32 \times 2 \times 10 \times 32 \times 128 = 2.6M$

That's still much smaller than the full model!

## Comparison: Prompt vs Prefix Tuning

| Aspect | Prompt Tuning | Prefix Tuning |
|--------|---------------|---------------|
| Where | Input only | Every attention layer |
| Parameters | Very few (~80K) | More (~2M) |
| Capacity | Limited | Higher |
| Best for | Simple tasks | Complex tasks |
| Complexity | Very simple | More complex |

## When to Use Each

### Prompt Tuning

**Good for**:

- Simple classification tasks
- When you have very little compute
- Quick experiments
- Large models (scales better)

**Limitations**:

- Limited capacity for complex tasks
- May struggle with generation

### Prefix Tuning

**Good for**:

- Generation tasks
- More complex adaptations
- When prompt tuning underfits

**Limitations**:

- More parameters than prompt tuning
- More complex implementation

## The Spectrum of PEFT Methods

```
Fewest Parameters ←————————————————————————→ Most Parameters

Prompt      Prefix       LoRA        Adapters     Full
Tuning      Tuning                              Fine-tuning
(~80K)      (~2M)        (~4M)       (~50M)      (~7B)

Least                                           Most
Capacity                                       Capacity
```

## Training Tips

### Learning Rate

- Prompt tuning: Higher LR (1e-3 to 1e-2)
- Prefix tuning: Moderate LR (1e-4 to 1e-3)

### Initialization

**Prompt tuning options**:

1. Random initialization (simple)
2. Initialize from actual embeddings (better for some tasks)

```python
# Initialize from vocabulary
vocab_indices = np.random.choice(vocab_size, prompt_length)
prompt = embedding_matrix[vocab_indices].copy()
```

**Prefix tuning**:

- Small random initialization works well
- Some use an MLP to generate prefixes from a smaller embedding (adds capacity)

## Common Mistakes

1. **Too few prompt tokens**: Underfitting
2. **Too many prompt tokens**: Overfitting, slow
3. **Wrong attention masking**: Prefix tokens should attend to each other
4. **Ignoring position embeddings**: Consider how positions interact with prefixes

## Summary

| Method | What's Learned | Where |
|--------|---------------|-------|
| Prompt Tuning | Input embeddings | Before first layer |
| Prefix Tuning | K/V prefixes | Every attention layer |

| Method | Parameters | Capacity | Complexity |
|--------|------------|----------|------------|
| Prompt | ~0.001% | Low | Very simple |
| Prefix | ~0.01% | Medium | Moderate |

**Key insight**: You don't always need to modify weights. Sometimes, modifying the input (or attention context) is enough to steer model behavior.

**Next**: We'll discuss how to choose between these PEFT methods.

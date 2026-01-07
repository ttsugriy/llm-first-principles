# Section 9.5: Choosing a Method

*Reading time: 8 minutes*

## The Decision Framework

With multiple PEFT methods available, how do you choose? Consider these factors:

1. **Task complexity**
2. **Available compute**
3. **Inference requirements**
4. **Data size**

## Quick Reference

| If You Need... | Use |
|---------------|-----|
| Maximum efficiency | Prompt Tuning |
| Balance of quality and efficiency | LoRA |
| Maximum quality (near full fine-tuning) | Adapters or LoRA with high rank |
| Zero inference overhead | LoRA (merged) |
| Quick experiments | Prompt Tuning |
| Production deployment | LoRA (merged) |

## Decision Tree

```
Start
  │
  ├─ Is inference latency critical?
  │    │
  │    Yes ──▶ LoRA (can merge weights)
  │    │
  │    No ──▶ Continue
  │
  ├─ Is compute extremely limited?
  │    │
  │    Yes ──▶ Prompt Tuning
  │    │
  │    No ──▶ Continue
  │
  ├─ Is the task simple (classification, etc.)?
  │    │
  │    Yes ──▶ Prompt Tuning or LoRA (r=4)
  │    │
  │    No ──▶ Continue
  │
  ├─ Do you need to combine multiple tasks?
  │    │
  │    Yes ──▶ Adapters (easy to swap/combine)
  │    │
  │    No ──▶ Continue
  │
  └─ Default recommendation ──▶ LoRA (r=8, α=16)
```

## Detailed Comparison

### Parameter Efficiency

| Method | Params (7B model) | Storage |
|--------|------------------|---------|
| Prompt Tuning (20 tokens) | ~80K | ~300KB |
| Prefix Tuning (10 per layer) | ~2.6M | ~10MB |
| LoRA (r=8, Q+V) | ~4M | ~16MB |
| Adapters (bottleneck=64) | ~50M | ~200MB |
| Full Fine-Tuning | ~7B | ~28GB |

### Quality vs Efficiency Trade-off

```
Quality
    │
100%│        ●───────────●  Full Fine-tuning
    │       ╱  Adapters  │
 98%│   ●──╱             │
    │   │ LoRA           │
 95%│   │                │
    │   │                │
 90%│●  │                │
    │ Prefix             │
 85%│                    │
    │●  Prompt           │
    └────────────────────┴────▶
        0.001%  0.1%  1%  100%
              Parameters (log scale)
```

### Inference Overhead

| Method | Extra Compute | Can Merge? |
|--------|--------------|------------|
| Prompt Tuning | Minimal (longer sequence) | N/A |
| Prefix Tuning | Small (attention overhead) | No |
| LoRA | None if merged | Yes |
| Adapters | Small (extra layers) | No (needs distillation) |

## Task-Specific Recommendations

### Classification

**Best**: Prompt Tuning or LoRA (r=4)

- Simple task, doesn't need many parameters
- Quick training
- Low risk of overfitting

### Instruction Following

**Best**: LoRA (r=8-16)

- Needs moderate capacity
- Benefits from merged deployment
- Well-studied for this use case

### Domain Adaptation

**Best**: LoRA (r=16-32) or Adapters

- Need to learn new knowledge
- May need more capacity
- Consider combining with continued pretraining

### Style/Persona Transfer

**Best**: LoRA (r=8) or Prefix Tuning

- Mainly steering behavior, not adding knowledge
- Works well with fewer parameters

### Code Generation

**Best**: LoRA (r=8-16) on all attention weights

- Complex task
- Benefits from adapting Q, K, V, and O
- May also benefit from FFN adaptation

## Combining Methods

Sometimes one method isn't enough. You can combine:

### LoRA + Prompt Tuning

```python
# Learn soft prompts AND low-rank weight updates
soft_prompt = PromptTuning(d_model=4096, prompt_length=20)
lora_layers = add_lora_to_model(model, rank=8)

# Training: optimize both
optimizer = Adam(
    list(soft_prompt.parameters()) + list(lora_layers.parameters())
)
```

### Multiple LoRAs

Train separate LoRAs for different aspects:

- LoRA A: Style adaptation
- LoRA B: Domain knowledge
- Combine at inference with weighted sum

## Scaling Behavior

As models get larger, PEFT methods become relatively more efficient:

| Model Size | Full FT Memory | LoRA Memory | Savings |
|------------|---------------|-------------|---------|
| 1B | ~4GB | ~1GB | 4x |
| 7B | ~28GB | ~8GB | 3.5x |
| 70B | ~280GB | ~40GB | 7x |

Larger models benefit more from PEFT!

## Practical Recommendations

### Starting Point

```
1. Try LoRA (r=8, α=16) on Q and V matrices
2. Train for a few epochs
3. Evaluate

If underfitting:
  - Increase rank to 16 or 32
  - Add more target modules (K, O, FFN)

If overfitting:
  - Decrease rank to 4
  - Add dropout
  - Reduce training epochs
```

### Production Deployment

1. Train with LoRA
2. Validate performance
3. Merge weights for deployment
4. No inference overhead

### Research/Experimentation

1. Start with Prompt Tuning (fastest)
2. Graduate to LoRA if needed
3. Use Adapters for multi-task scenarios

## Common Pitfalls

| Mistake | Why It's Wrong | Fix |
|---------|---------------|-----|
| Starting with full FT | Wasteful, often unnecessary | Start with LoRA |
| Rank too high | Overfitting, slow | Start low, increase if needed |
| Wrong target modules | Missing important weights | Start with Q+V, expand |
| Ignoring inference | Adapters add latency | Use LoRA for latency-sensitive apps |
| One-size-fits-all | Tasks have different needs | Tune hyperparams per task |

## Summary

| Factor | Recommendation |
|--------|----------------|
| Simple task + low compute | Prompt Tuning |
| General purpose | LoRA (r=8) |
| Complex task | LoRA (r=16+) or Adapters |
| Zero latency overhead | LoRA (merged) |
| Multi-task | Adapters or multiple LoRAs |

**Default recommendation**: Start with LoRA (r=8, α=16) on Q and V projections. Adjust from there.

**Key insight**: There's no single best method—the right choice depends on your specific constraints and requirements.

**Next**: We'll implement all these methods from scratch.

# Section 10.5: Choosing a Method

*Reading time: 8 minutes*

## The Decision Framework

Three main approaches to alignment:

1. **Reward Modeling + Best-of-N**: Simple, no RL
2. **RLHF (PPO)**: Full RL loop
3. **DPO**: Direct optimization

## Quick Reference

| Situation | Recommendation |
|-----------|----------------|
| Getting started | DPO |
| Production system with feedback loop | RLHF |
| Limited compute | DPO |
| Custom reward signals | RLHF |
| Offline preference data | DPO |
| Maximum flexibility | RLHF |

## Method Comparison

### Complexity

```
Simple ←——————————————————————————→ Complex

Best-of-N     DPO           RLHF
  ●————————————●—————————————●
```

### Quality (given enough compute)

```
Lower ←——————————————————————————→ Higher

Best-of-N     DPO           RLHF
  ●————————————●—————————————●
            (often comparable)
```

### Stability

```
Less Stable ←————————————————→ More Stable

RLHF           Best-of-N       DPO
  ●—————————————●———————————————●
```

## Detailed Trade-offs

### Best-of-N Sampling

**How it works**:

1. Train a reward model
2. Generate N responses
3. Select the highest-reward response

**Pros**:

- Very simple
- No policy training needed
- Can use any reward signal

**Cons**:

- Expensive at inference (N forward passes)
- Doesn't improve the policy itself
- Quality limited by sampling

**Best for**: Quick experiments, when inference cost doesn't matter.

### RLHF with PPO

**How it works**:

1. Train reward model
2. Generate responses from policy
3. Score with reward model
4. Update policy with PPO
5. Repeat

**Pros**:

- Online learning (can improve from new feedback)
- Flexible reward signals
- Well-studied algorithm

**Cons**:

- Complex: 3 models to manage
- Unstable: RL training is tricky
- Sample inefficient: needs many generations

**Best for**: Production systems with continuous feedback, custom rewards.

### DPO

**How it works**:

1. Collect preference pairs
2. Compute log probs under policy and reference
3. Apply DPO loss
4. Update policy

**Pros**:

- Simple: supervised learning style
- Stable: no RL instabilities
- Efficient: no generation during training

**Cons**:

- Offline only: can't learn from new preferences during training
- Requires good reference model
- Less flexible than reward models

**Best for**: Most use cases, especially when starting out.

## Decision Tree

```
Start
  │
  ├─ Do you need online learning from new preferences?
  │    │
  │    Yes ──▶ RLHF
  │    │
  │    No ──▶ Continue
  │
  ├─ Do you have custom reward signals (not just preferences)?
  │    │
  │    Yes ──▶ RLHF (or Reward Model + Best-of-N)
  │    │
  │    No ──▶ Continue
  │
  ├─ Is simplicity and stability important?
  │    │
  │    Yes ──▶ DPO
  │    │
  │    No ──▶ Continue
  │
  └─ Default ──▶ DPO (it's almost always a good choice)
```

## Practical Recommendations

### Starting a New Project

1. Start with DPO
2. Get a baseline working
3. Only add RLHF complexity if needed

### Production System

1. DPO for initial alignment
2. Add RLHF if you have continuous feedback
3. Consider Best-of-N for safety-critical applications

### Research

1. DPO for quick experiments
2. RLHF for studying online learning
3. Both for comparing methods

## Combining Methods

You can combine approaches:

### DPO + Online Feedback

1. Train initial policy with DPO
2. Collect new preferences from deployed model
3. Fine-tune with more DPO

### RLHF + DPO Initialization

1. Pre-train policy with DPO
2. Continue with RLHF for online learning

### Multi-Stage Alignment

```
SFT → DPO (general alignment) → RLHF (task-specific refinement)
```

## Common Mistakes

| Mistake | Why It's Wrong | Fix |
|---------|---------------|-----|
| Starting with RLHF | Unnecessary complexity | Start with DPO |
| No reference model in DPO | KL constraint is essential | Always use reference |
| Too low beta in DPO | Model diverges from reference | Start with β=0.1 |
| Poor preference data | Garbage in, garbage out | Invest in data quality |
| Ignoring evaluation | Can't tell if it's working | Measure continuously |

## Evaluation

Whatever method you choose, evaluate on:

1. **Preference accuracy**: Does the model match human preferences?
2. **Safety**: Does it refuse harmful requests?
3. **Helpfulness**: Does it solve user problems?
4. **Capability**: Did it lose pre-training abilities?

## Summary

| Method | Complexity | Stability | Flexibility | Best For |
|--------|------------|-----------|-------------|----------|
| Best-of-N | Low | High | Medium | Quick experiments |
| DPO | Medium | High | Medium | Most use cases |
| RLHF | High | Low | High | Online learning |

**Default recommendation**: Start with DPO. It's simple, stable, and effective.

**Key insight**: You don't need the most complex method—you need the method that works for your situation.

**Next**: We'll implement all these methods from scratch.

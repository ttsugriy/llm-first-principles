# Section 10.1: The Alignment Problem

*Reading time: 10 minutes*

## What Goes Wrong?

A language model trained only on next-token prediction learns to complete text. Not to be helpful. Not to be safe.

### Example 1: Following Harmful Instructions

**User**: "How do I make a bomb?"

**Unaligned model**: [Provides detailed instructions]

The model learned from text that includes such information. It's just completing the pattern.

### Example 2: Confident Falsehoods

**User**: "What year was the Eiffel Tower built?"

**Unaligned model**: "The Eiffel Tower was built in 1892 and stands 350 meters tall."

Wrong on both counts, but stated with complete confidence.

### Example 3: Unhelpful Responses

**User**: "Can you help me write an email?"

**Unaligned model**: "Email is a method of electronic communication..."

Technically relevant, completely useless.

## The Gap

Pre-training learns:

$$P(\text{token} | \text{context})$$

What we want:

$$P(\text{helpful, honest, harmless response} | \text{user intent})$$

These objectives are not the same.

## Why Can't We Just Use Rules?

**Attempt 1**: "Never output harmful content"

Problem: What counts as "harmful"? Is a chemistry textbook harmful? A security research paper?

**Attempt 2**: "Always be helpful"

Problem: Helping with harmful requests is itself harmful.

**Attempt 3**: "Be accurate"

Problem: Models don't know what they don't know.

## The RLHF Insight

Instead of encoding rules, **learn preferences from humans**.

Human preferences capture:

- Context-dependent judgments
- Trade-offs between competing values
- Cultural and situational nuances

Things that are nearly impossible to specify with rules.

## The Three H's

Modern alignment targets three goals:

### 1. Helpful

The model should:

- Understand what the user actually wants
- Provide useful, actionable responses
- Complete tasks effectively

### 2. Harmless

The model should:

- Refuse genuinely harmful requests
- Not produce toxic content
- Consider second-order effects

### 3. Honest

The model should:

- Express uncertainty when appropriate
- Not hallucinate facts
- Acknowledge limitations

## The Training Pipeline

```
1. Pre-training (Stage 1-6)
   ↓
   Raw language model (predicts tokens)

2. Supervised Fine-Tuning (SFT)
   ↓
   Model follows instructions

3. Alignment (RLHF or DPO)
   ↓
   Model is helpful, harmless, honest
```

Each stage builds on the previous.

## What Alignment Actually Changes

### Before Alignment

```
User: Write me a poem about war
Model: [Writes any poem about war, possibly glorifying violence]
```

### After Alignment

```
User: Write me a poem about war
Model: [Writes a thoughtful poem about war's human cost]
```

The model's capabilities are similar, but its choices are different.

## The Role of Human Feedback

Human annotators provide:

**Preference comparisons**: "Response A is better than Response B"

This is easier than:

- Defining "better" mathematically
- Rating responses on absolute scales
- Specifying all possible edge cases

Humans are good at comparison. We leverage that.

## Challenges

### 1. Preference Inconsistency

Different humans have different preferences. Even the same human might be inconsistent.

**Solution**: Use many annotators, average preferences.

### 2. Reward Hacking

Models can find unexpected ways to maximize reward without being actually helpful.

**Solution**: KL penalty, diverse evaluation.

### 3. Specification Gaming

"Be concise" might lead to responses that are too short.

**Solution**: Multi-objective optimization, careful reward design.

### 4. Distributional Shift

Training preferences might not match deployment scenarios.

**Solution**: Diverse training data, robust evaluation.

## Success Story: InstructGPT

OpenAI's InstructGPT (2022) showed:

| Model | Training | Human Preference |
|-------|----------|------------------|
| GPT-3 (175B) | Pre-training only | Baseline |
| InstructGPT (1.3B) | Pre-training + RLHF | 85% preferred |

A 100x smaller model was preferred because it was aligned.

**Key insight**: Alignment > Scale (for user preference)

## Summary

| Problem | Root Cause | Solution |
|---------|------------|----------|
| Harmful outputs | Training on harmful data | Learn to refuse |
| Unhelpful responses | Wrong objective | Optimize for helpfulness |
| Confident errors | No uncertainty signal | Learn to express uncertainty |

**Key insight**: Alignment bridges the gap between "predicts well" and "behaves well."

**Next**: We'll learn how reward models capture human preferences.

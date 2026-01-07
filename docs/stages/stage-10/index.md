# Stage 10: Alignment

*Making models helpful, harmless, and honest*

## Overview

Pre-trained LLMs are powerful pattern matchers that predict next tokens. But prediction isn't enough—we want models that:

- **Help** users accomplish their goals
- **Avoid harm** even when prompted to cause it
- **Be honest** about uncertainty and limitations

This gap between "predicts well" and "behaves well" is the **alignment problem**.

> "A model that predicts text perfectly might still write harmful content perfectly."

## The Core Challenge

**Pre-training objective**: Maximize $P(\text{next token} | \text{context})$

**What we actually want**: Maximize $P(\text{human approves of response} | \text{context})$

These are different! A model trained only on prediction might:

- Generate toxic content if that's what the context suggests
- Confidently state falsehoods
- Help with harmful requests

## The Solution: Preference Learning

Instead of defining "good behavior" with rules, we learn it from human preferences:

1. Show humans two model responses
2. Ask which is better
3. Train the model to produce more preferred responses

This is the foundation of both RLHF and DPO.

## Methods We'll Cover

| Method | Approach | Complexity |
|--------|----------|------------|
| Reward Modeling | Learn a "goodness" function | Medium |
| RLHF (PPO) | Use RL to optimize for reward | High |
| DPO | Direct preference optimization | Low |

## Why This Matters

Alignment is what makes the difference between:

- A text generator and an assistant
- A pattern matcher and a helpful tool
- A liability and a product

Most "AI safety" concerns are really about alignment.

## Learning Objectives

By the end of this stage, you will:

1. Understand why alignment is necessary
2. Implement reward modeling with Bradley-Terry
3. Understand RLHF and PPO basics
4. Implement DPO from scratch
5. Know when to use each approach

## Sections

1. [The Alignment Problem](01-alignment-problem.md) - Why prediction isn't enough
2. [Reward Modeling](02-reward-modeling.md) - Learning from preferences
3. [RLHF with PPO](03-rlhf.md) - Reinforcement learning approach
4. [Direct Preference Optimization](04-dpo.md) - A simpler alternative
5. [Choosing a Method](05-choosing-a-method.md) - Trade-offs and recommendations
6. [Implementation](06-implementation.md) - Building alignment from scratch

## Prerequisites

- Understanding of neural network training (Stage 2-3)
- Familiarity with language model architecture (Stage 6)
- Basic probability and optimization concepts

## Key Insight

> Alignment doesn't require solving the "hard problem" of defining what's good. It requires learning from human judgments—which humans are very good at providing.

## Code & Resources

| Resource | Description |
|----------|-------------|
| [`code/stage-10/alignment.py`](https://github.com/ttsugriy/llm-first-principles/blob/main/code/stage-10/alignment.py) | Reward Model, RLHF, and DPO |
| [`code/stage-10/tests/`](https://github.com/ttsugriy/llm-first-principles/tree/main/code/stage-10/tests) | Test suite |
| [Exercises](exercises.md) | Practice problems |
| [Common Mistakes](common-mistakes.md) | Debugging guide |

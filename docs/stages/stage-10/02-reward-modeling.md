# Section 10.2: Reward Modeling

*Reading time: 12 minutes*

## The Core Idea

We want to optimize for "human preferences," but neural networks need numbers. Reward models bridge this gap:

**Reward model**: A function $r(x, y)$ that predicts how much humans would prefer response $y$ to prompt $x$.

## Preference Data

The fundamental unit of alignment data:

```python
@dataclass
class PreferencePair:
    prompt: str
    chosen: str      # The preferred response
    rejected: str    # The less preferred response
```

Human annotators see both responses and choose which is better.

### Example

**Prompt**: "How do I cook pasta?"

**Response A**: "Boil water, add pasta, cook 8-10 minutes, drain."

**Response B**: "Pasta is a type of Italian food made from wheat."

Human preference: **A > B** (A is more helpful)

## The Bradley-Terry Model

How do we turn preferences into a trainable objective?

The Bradley-Terry model says:

$$P(\text{A beats B}) = \frac{e^{r(A)}}{e^{r(A)} + e^{r(B)}} = \sigma(r(A) - r(B))$$

Where $\sigma$ is the sigmoid function.

**Intuition**: If A has higher reward, it's more likely to be preferred.

## The Loss Function

Given preference pair (chosen, rejected), we want:

$$r(\text{chosen}) > r(\text{rejected})$$

Loss:

$$\mathcal{L} = -\log \sigma(r(\text{chosen}) - r(\text{rejected}))$$

This is just binary cross-entropy where we predict "chosen > rejected."

### Gradient

$$\frac{\partial \mathcal{L}}{\partial r_c} = \sigma(r_c - r_r) - 1$$

$$\frac{\partial \mathcal{L}}{\partial r_r} = -(\sigma(r_c - r_r) - 1)$$

The gradient pushes $r_c$ up and $r_r$ down until the model correctly predicts preferences.

## Architecture

A reward model typically:

1. Takes the same architecture as the language model
2. Replaces the output head with a scalar "reward head"
3. Uses the final token's representation

```
Input: [prompt, response]
   ↓
Transformer (shared with LM)
   ↓
Final token representation
   ↓
Linear(d_model, 1)
   ↓
Scalar reward
```

## Implementation

```python
class RewardModel:
    """Reward model that predicts human preferences."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        # Simple MLP reward head
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros(1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute reward for input representation."""
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return h @ self.W2 + self.b2


def reward_loss(reward_chosen, reward_rejected):
    """Bradley-Terry preference loss."""
    diff = reward_chosen - reward_rejected
    sigmoid = 1 / (1 + np.exp(-diff))
    loss = -np.mean(np.log(sigmoid + 1e-10))
    return loss
```

## Training Procedure

```python
for batch in preference_data:
    # Encode prompt + response pairs
    chosen_repr = encode(batch.prompt, batch.chosen)
    rejected_repr = encode(batch.prompt, batch.rejected)

    # Get rewards
    r_chosen = reward_model(chosen_repr)
    r_rejected = reward_model(rejected_repr)

    # Compute loss
    loss = -log(sigmoid(r_chosen - r_rejected))

    # Update
    loss.backward()
    optimizer.step()
```

## Data Collection

### Human Annotation Process

1. Sample prompts from user queries
2. Generate multiple responses from the model
3. Present pairs to annotators
4. Collect preferences

### Guidelines

Annotators typically receive instructions like:

- "Choose the response that is more helpful"
- "Prefer responses that are accurate and honest"
- "Avoid responses that are harmful or offensive"

### Quality Control

- Multiple annotators per comparison
- Inter-annotator agreement metrics
- Clear guidelines and training

## What Makes a Good Reward Model?

### Generalization

Should work on prompts/responses not seen during training.

### Calibration

High reward should mean high human preference.

### Robustness

Shouldn't be easily fooled by surface features:

- Length (longer ≠ better)
- Confidence (certain ≠ correct)
- Verbosity (more words ≠ more helpful)

## Common Pitfalls

### 1. Length Bias

Models often prefer longer responses. Fix: normalize by length or use length-balanced data.

### 2. Sycophancy

Reward models might prefer responses that agree with the user, even when wrong.

### 3. Reward Hacking

The policy might find ways to maximize reward without being actually helpful.

Example: Adding "I hope this helps!" increases reward without improving quality.

### 4. Distribution Shift

Training on generated responses but evaluating on diverse user queries.

## Reward Model Evaluation

### Accuracy

On held-out preference pairs:

```python
accuracy = mean(reward_model(chosen) > reward_model(rejected))
```

Target: 70%+ (50% is random)

### Calibration

Does high reward actually mean human preference?

### Qualitative Analysis

Manually inspect high-reward and low-reward responses.

## From Reward Model to Policy

Once we have a reward model, we can:

1. **RLHF**: Use RL to optimize policy for reward
2. **Best-of-N**: Generate N responses, pick highest reward
3. **DPO**: Skip the reward model entirely (next section)

## Summary

| Component | Purpose |
|-----------|---------|
| Preference pairs | Training data format |
| Bradley-Terry | Turn preferences into probabilities |
| Reward head | Output scalar reward |
| BCE loss | Train to predict preferences |

**Key insight**: Reward models distill human preferences into a trainable signal.

**Next**: We'll use the reward model to train policies with RLHF.

# Section 10.3: RLHF with PPO

*Reading time: 15 minutes*

## The RLHF Pipeline

RLHF (Reinforcement Learning from Human Feedback) uses the reward model to train the language model:

```
1. Sample prompts
2. Generate responses from policy
3. Score responses with reward model
4. Update policy with RL (PPO)
5. Repeat
```

## Framing LM Generation as RL

| RL Concept | LM Equivalent |
|------------|---------------|
| State | Current tokens |
| Action | Next token |
| Policy | Language model $\pi(a|s)$ |
| Reward | Reward model score |
| Trajectory | Complete response |

## The Objective

We want to maximize expected reward while staying close to a reference policy:

$$\max_\pi \mathbb{E}_{x \sim D, y \sim \pi(y|x)}[r(x, y)] - \beta \cdot \text{KL}(\pi || \pi_{\text{ref}})$$

Where:

- $r(x, y)$: Reward model score
- $\pi_{\text{ref}}$: Reference policy (usually the SFT model)
- $\beta$: KL penalty coefficient

## Why the KL Penalty?

Without it, the policy can:

1. **Collapse**: Produce the same high-reward response for everything
2. **Reward hack**: Find degenerate responses that score high
3. **Drift**: Lose pre-training capabilities

The KL penalty says: "Be good, but don't change too much."

## PPO (Proximal Policy Optimization)

PPO is the most common RL algorithm for RLHF.

### Core Idea

Update the policy, but don't change too much in one step.

### The Clipped Objective

$$L^{\text{CLIP}} = \mathbb{E}\left[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)\right]$$

Where:

- $r_t = \frac{\pi(a|s)}{\pi_{\text{old}}(a|s)}$: Probability ratio
- $A_t$: Advantage (how much better than expected)
- $\epsilon$: Clipping parameter (typically 0.2)

### Intuition

- If action was better than expected ($A > 0$), increase its probability
- But don't increase too much (clipping prevents this)
- If action was worse ($A < 0$), decrease its probability
- But don't decrease too much

## Advantage Estimation

The advantage tells us: "How much better was this action than expected?"

$$A_t = Q(s_t, a_t) - V(s_t)$$

We estimate it with GAE (Generalized Advantage Estimation):

$$A_t = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + ...$$

Where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

## Implementation Components

### PPO Buffer

Stores experience for training:

```python
class PPOBuffer:
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma  # Discount factor
        self.lam = lam      # GAE lambda

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []

    def compute_advantages(self, last_value=0):
        """Compute GAE advantages."""
        rewards = np.array(self.rewards)
        values = np.array(self.values)

        advantages = np.zeros(len(rewards))
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * last_gae

        returns = advantages + values
        return advantages, returns
```

### PPO Loss

```python
def ppo_loss(new_log_probs, old_log_probs, advantages, clip_ratio=0.2):
    """PPO clipped surrogate loss."""
    # Probability ratio
    ratio = np.exp(new_log_probs - old_log_probs)

    # Clipped ratio
    clipped_ratio = np.clip(ratio, 1 - clip_ratio, 1 + clip_ratio)

    # Take minimum (pessimistic bound)
    loss = -np.mean(np.minimum(
        ratio * advantages,
        clipped_ratio * advantages
    ))

    return loss
```

### Value Loss

Train a value function alongside the policy:

```python
def value_loss(predicted_values, returns):
    """Simple MSE value loss."""
    return 0.5 * np.mean((predicted_values - returns) ** 2)
```

## The RLHF Training Loop

```python
for iteration in range(num_iterations):
    # 1. Collect experience
    buffer = PPOBuffer()

    for prompt in sample_prompts():
        # Generate response from policy
        response = policy.generate(prompt)

        # Get reward from reward model
        reward = reward_model(prompt, response)

        # Get value estimate
        value = value_function(prompt)

        # Store
        buffer.add(prompt, response, reward, value, log_prob)

    # 2. Compute advantages
    advantages, returns = buffer.compute_advantages()

    # 3. PPO update (multiple epochs over the data)
    for epoch in range(ppo_epochs):
        for batch in buffer.batches():
            # Recompute log probs under current policy
            new_log_probs = policy.log_prob(batch.response)

            # Policy loss
            policy_loss = ppo_loss(
                new_log_probs,
                batch.old_log_probs,
                batch.advantages
            )

            # Value loss
            new_values = value_function(batch.prompt)
            v_loss = value_loss(new_values, batch.returns)

            # KL penalty
            kl = kl_divergence(policy, ref_policy, batch.prompt)
            kl_loss = beta * kl

            # Total loss
            total_loss = policy_loss + v_loss_coef * v_loss + kl_loss

            # Update
            total_loss.backward()
            optimizer.step()
```

## Practical Considerations

### 1. Model Architecture

In RLHF, you typically have:

- **Policy model**: The LLM being trained
- **Reference model**: Frozen copy of SFT model
- **Reward model**: Trained on preferences
- **Value model**: Often shares backbone with policy

### 2. KL Control

**Adaptive KL**: Adjust $\beta$ based on observed KL:

- If KL too high: Increase $\beta$
- If KL too low: Decrease $\beta$

Target KL is typically 0.01-0.1.

### 3. Reward Normalization

Normalize rewards to have mean 0 and std 1 within each batch.

### 4. Gradient Clipping

Clip gradients to prevent instability.

## RLHF Challenges

### Instability

RL training is notoriously unstable. Careful hyperparameter tuning required.

### Sample Efficiency

Needs many model generations per update.

### Mode Collapse

Policy might produce very similar outputs.

### Reward Hacking

Policy finds loopholes in the reward model.

## Summary

| Component | Purpose |
|-----------|---------|
| PPO | Stable policy updates |
| GAE | Advantage estimation |
| KL penalty | Prevent divergence |
| Value function | Baseline for variance reduction |
| Clipping | Prevent too-large updates |

**Key insight**: RLHF treats language generation as an RL problem, using the reward model as the reward function.

**Next**: We'll see DPO, a simpler alternative that eliminates the RL loop entirely.

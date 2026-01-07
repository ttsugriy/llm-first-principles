# Section 10.6: Implementation

*Reading time: 15 minutes*

## Overview

In this section, we implement the core alignment components:

1. **Preference Data**: Structure for human comparisons
2. **Reward Model**: Learning from preferences
3. **DPO Trainer**: Direct preference optimization
4. **PPO Components**: For RLHF

All code is available in `code/stage-10/alignment.py`.

## Preference Data

The fundamental unit of alignment data:

```python
@dataclass
class PreferencePair:
    """A single preference comparison."""
    prompt: str
    chosen: str      # The preferred response
    rejected: str    # The less preferred response


def create_preference_dataset(
    prompts: List[str],
    chosen_responses: List[str],
    rejected_responses: List[str],
) -> List[PreferencePair]:
    """Create a preference dataset from parallel lists."""
    return [
        PreferencePair(prompt=p, chosen=c, rejected=r)
        for p, c, r in zip(prompts, chosen_responses, rejected_responses)
    ]
```

## Reward Model

```python
class RewardModel:
    """
    Reward model that predicts human preferences.

    Architecture: Takes representation, outputs scalar reward.
    Trained using Bradley-Terry model on preference pairs.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        # Simple MLP reward head
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * scale2
        self.b2 = np.zeros(1)

        self.grads = {}
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute reward for input representation."""
        h = x @ self.W1 + self.b1
        h_relu = np.maximum(0, h)
        reward = h_relu @ self.W2 + self.b2

        self.cache = {'x': x, 'h': h, 'h_relu': h_relu}
        return reward

    def backward(self, grad_reward: np.ndarray) -> np.ndarray:
        """Backward pass through reward model."""
        x = self.cache['x']
        h = self.cache['h']
        h_relu = self.cache['h_relu']

        # Output layer gradients
        self.grads['W2'] = h_relu.T @ grad_reward
        self.grads['b2'] = grad_reward.sum(axis=0)

        grad_h_relu = grad_reward @ self.W2.T

        # ReLU gradient
        grad_h = grad_h_relu * (h > 0).astype(float)

        # Input layer gradients
        self.grads['W1'] = x.T @ grad_h
        self.grads['b1'] = grad_h.sum(axis=0)

        return grad_h @ self.W1.T
```

## Bradley-Terry Loss

```python
def reward_model_loss(
    reward_chosen: np.ndarray,
    reward_rejected: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Bradley-Terry preference loss.

    Loss = -log(sigmoid(r_chosen - r_rejected))
    """
    diff = reward_chosen - reward_rejected

    # Numerically stable sigmoid
    sigmoid = np.where(
        diff >= 0,
        1 / (1 + np.exp(-diff)),
        np.exp(diff) / (1 + np.exp(diff))
    )

    # Loss
    loss = -np.mean(np.log(sigmoid + 1e-10))

    # Gradients
    grad_diff = (sigmoid - 1) / len(diff)
    grad_chosen = grad_diff
    grad_rejected = -grad_diff

    return float(loss), grad_chosen, grad_rejected
```

## DPO Trainer

```python
class DPOTrainer:
    """
    Direct Preference Optimization trainer.

    DPO eliminates the need for a reward model by directly
    optimizing the policy using preference data.
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        self.beta = beta
        self.label_smoothing = label_smoothing

    def compute_loss(
        self,
        policy_chosen_logps: np.ndarray,
        policy_rejected_logps: np.ndarray,
        ref_chosen_logps: np.ndarray,
        ref_rejected_logps: np.ndarray,
    ) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
        """Compute DPO loss."""
        # Log ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        # Implicit rewards
        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        # DPO loss
        logits = chosen_rewards - rejected_rewards

        # Label smoothing
        labels = 1.0 - self.label_smoothing

        # Binary cross-entropy
        sigmoid_logits = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        loss = -np.mean(
            labels * np.log(sigmoid_logits + 1e-10) +
            (1 - labels) * np.log(1 - sigmoid_logits + 1e-10)
        )

        # Metrics
        metrics = {
            'loss': float(loss),
            'accuracy': float(np.mean(logits > 0)),
            'chosen_reward': float(np.mean(chosen_rewards)),
            'rejected_reward': float(np.mean(rejected_rewards)),
            'reward_margin': float(np.mean(chosen_rewards - rejected_rewards)),
        }

        # Gradients
        grad_logits = (sigmoid_logits - labels) / len(logits)
        grad_policy_chosen = self.beta * grad_logits
        grad_policy_rejected = -self.beta * grad_logits

        return loss, metrics, grad_policy_chosen, grad_policy_rejected
```

## Log Probability Computation

```python
def compute_log_probs(
    logits: np.ndarray,
    tokens: np.ndarray,
) -> np.ndarray:
    """
    Compute log probabilities of tokens under logits.

    Args:
        logits: Model output [batch, seq_len, vocab_size]
        tokens: Target tokens [batch, seq_len]

    Returns:
        Sum of log probs per sequence [batch]
    """
    # Log softmax
    logits_max = logits.max(axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(logits - logits_max), axis=-1, keepdims=True))
    log_probs = logits - logits_max - log_sum_exp

    # Select log probs for target tokens
    selected_log_probs = np.take_along_axis(
        log_probs,
        tokens[:, :, np.newaxis],
        axis=-1
    ).squeeze(-1)

    # Sum over sequence
    return selected_log_probs.sum(axis=-1)
```

## PPO Components

### Experience Buffer

```python
class PPOBuffer:
    """Experience buffer for PPO training."""

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, state, action, reward, value, log_prob, done=False):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_advantages(self, last_value=0.0):
        """Compute GAE advantages."""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        n = len(rewards)
        advantages = np.zeros(n)

        last_gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            next_non_terminal = 1.0 - float(dones[t])

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae

        returns = advantages + values
        return advantages, returns
```

### PPO Loss

```python
def ppo_loss(
    new_log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    clip_ratio: float = 0.2,
) -> Tuple[float, np.ndarray]:
    """PPO clipped surrogate loss."""
    # Probability ratio
    ratio = np.exp(new_log_probs - old_log_probs)

    # Clipped ratio
    clipped_ratio = np.clip(ratio, 1 - clip_ratio, 1 + clip_ratio)

    # Take minimum
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    loss = -np.mean(np.minimum(surr1, surr2))

    # Gradient (only where not clipped)
    clipped = (ratio < 1 - clip_ratio) | (ratio > 1 + clip_ratio)
    grad_ratio = np.where(clipped, 0, advantages)
    grad_log_prob = ratio * grad_ratio / len(ratio)

    return float(loss), -grad_log_prob
```

### KL Divergence

```python
def kl_divergence(log_probs_p: np.ndarray, log_probs_q: np.ndarray) -> float:
    """Compute KL(P || Q) from log probabilities."""
    p = np.exp(log_probs_p)
    return float(np.sum(p * (log_probs_p - log_probs_q)))


def kl_penalty_reward(
    policy_log_probs: np.ndarray,
    ref_log_probs: np.ndarray,
    kl_coef: float = 0.1,
) -> np.ndarray:
    """Compute KL penalty for RLHF reward."""
    kl_per_token = np.exp(policy_log_probs) * (policy_log_probs - ref_log_probs)
    return kl_coef * kl_per_token
```

## Running the Demo

```bash
cd code/stage-10
python alignment.py
```

Output:

```
============================================================
Stage 10: Alignment - RLHF and DPO
============================================================

1. Reward Model Training
----------------------------------------
Reward model loss: 0.7234
Chosen rewards mean: 0.0123
Rejected rewards mean: -0.0089

2. Direct Preference Optimization (DPO)
----------------------------------------
DPO loss: 0.6892
Accuracy: 52.50%
Reward margin: 0.0156

3. PPO Components (for RLHF)
----------------------------------------
Collected 10 steps
Advantages shape: (10,)
Returns shape: (10,)
PPO loss: 0.0234

4. RLHF vs DPO Comparison
----------------------------------------

RLHF (PPO):
  Models needed: 3
  Steps: 5
  Pros: Well-studied RL algorithm, Can use any reward signal
  Cons: Complex: 3 models, Unstable: RL training is tricky

DPO:
  Models needed: 2
  Steps: 4
  Pros: Simple: supervised learning style, Stable: no RL instabilities
  Cons: Offline only: can't learn from new preferences, Needs good reference
```

## Summary

| Component | Purpose | Key Functions |
|-----------|---------|---------------|
| PreferencePair | Store preference data | Dataclass |
| RewardModel | Predict human preferences | `forward()`, `backward()` |
| reward_model_loss | Bradley-Terry loss | Returns loss and gradients |
| DPOTrainer | Direct optimization | `compute_loss()` |
| PPOBuffer | Store RL experience | `add()`, `compute_advantages()` |
| ppo_loss | Clipped surrogate loss | Returns loss and gradients |

## Exercises

1. **Train a reward model**: Generate synthetic preferences and train
2. **Implement full DPO training loop**: Integrate with a language model
3. **Add adaptive KL**: Implement KL target for RLHF
4. **Compare methods**: Train with DPO and RLHF on same data
5. **Reward hacking**: Find ways to exploit the reward model
6. **Data quality**: Study effect of noisy preferences

## Conclusion

You've now learned the complete alignment pipeline:

- How to collect and structure preference data
- How to train reward models
- How to use RL (PPO) for optimization
- How to use DPO for simpler training

This completes Stage 10 and the main curriculum.

## What's Next

The Capstone project will tie everything together: build a complete language model from tokenization to alignment.

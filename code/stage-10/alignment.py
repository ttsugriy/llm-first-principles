"""
Stage 10: Alignment - RLHF and DPO from First Principles

This module implements alignment techniques that make LLMs helpful and safe:
- Reward Modeling: Learning human preferences
- RLHF (Reinforcement Learning from Human Feedback): PPO-based fine-tuning
- DPO (Direct Preference Optimization): Simpler, more stable alternative

The alignment problem: Pre-trained LLMs predict next tokens, but we want
them to be helpful, harmless, and honest. Alignment bridges this gap.

Key insight from DPO (Rafailov et al., 2023):
The optimal RLHF policy has a closed-form solution! Instead of training
a reward model + RL, we can directly optimize preferences.

"DPO turns alignment from a complex RL problem into supervised learning."
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass


# =============================================================================
# Preference Data
# =============================================================================

@dataclass
class PreferencePair:
    """
    A single preference comparison.

    Human annotators compare two model responses and indicate which is better.
    This is the fundamental unit of alignment data.
    """
    prompt: str
    chosen: str      # The preferred response
    rejected: str    # The less preferred response

    # Optional metadata
    chosen_score: Optional[float] = None
    rejected_score: Optional[float] = None


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


# =============================================================================
# Reward Modeling
# =============================================================================

class RewardModel:
    """
    Reward model that predicts human preferences.

    Architecture: Takes (prompt, response) and outputs a scalar reward.
    Trained using Bradley-Terry model on preference pairs.

    The Bradley-Terry model says:
        P(chosen > rejected) = sigmoid(r(chosen) - r(rejected))

    Loss: -log(sigmoid(r_chosen - r_rejected))

    This is equivalent to binary cross-entropy where we predict
    which response humans prefer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
    ):
        """
        Initialize reward model.

        In practice, reward models are often initialized from the
        SFT (supervised fine-tuned) model with a new head.

        Args:
            input_dim: Dimension of input representations
            hidden_dim: Hidden layer dimension
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Simple MLP reward head
        # In practice: transformer backbone + linear head
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * scale2
        self.b2 = np.zeros(1)

        # Gradients
        self.grads: Dict[str, np.ndarray] = {}
        self.cache: Dict[str, Any] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute reward for input representation.

        Args:
            x: Input representation [batch, input_dim]

        Returns:
            Reward scores [batch, 1]
        """
        # Hidden layer with ReLU
        h = x @ self.W1 + self.b1
        h_relu = np.maximum(0, h)

        # Output layer
        reward = h_relu @ self.W2 + self.b2

        self.cache = {'x': x, 'h': h, 'h_relu': h_relu}
        return reward

    def backward(self, grad_reward: np.ndarray) -> np.ndarray:
        """Backward pass through reward model."""
        x = self.cache['x']
        h = self.cache['h']
        h_relu = self.cache['h_relu']

        # Gradient through output layer
        self.grads['W2'] = h_relu.T @ grad_reward
        self.grads['b2'] = grad_reward.sum(axis=0)

        grad_h_relu = grad_reward @ self.W2.T

        # Gradient through ReLU
        grad_h = grad_h_relu * (h > 0).astype(float)

        # Gradient through input layer
        self.grads['W1'] = x.T @ grad_h
        self.grads['b1'] = grad_h.sum(axis=0)

        grad_x = grad_h @ self.W1.T
        return grad_x

    def parameters(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def gradients(self) -> List[np.ndarray]:
        return [self.grads.get('W1'), self.grads.get('b1'),
                self.grads.get('W2'), self.grads.get('b2')]


def reward_model_loss(
    reward_chosen: np.ndarray,
    reward_rejected: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Bradley-Terry preference loss.

    Loss = -log(sigmoid(r_chosen - r_rejected))
         = -log(1 / (1 + exp(-(r_chosen - r_rejected))))
         = log(1 + exp(r_rejected - r_chosen))

    Args:
        reward_chosen: Rewards for chosen responses [batch, 1]
        reward_rejected: Rewards for rejected responses [batch, 1]

    Returns:
        (loss, grad_chosen, grad_rejected)
    """
    diff = reward_chosen - reward_rejected  # [batch, 1]

    # Numerically stable sigmoid
    # sigmoid(x) = 1 / (1 + exp(-x))
    # For stability: when x < 0, use exp(x) / (1 + exp(x))
    sigmoid = np.where(
        diff >= 0,
        1 / (1 + np.exp(-diff)),
        np.exp(diff) / (1 + np.exp(diff))
    )

    # Loss = -log(sigmoid(diff))
    # Numerically stable: log(sigmoid(x)) = x - softplus(x) = -softplus(-x)
    loss = -np.mean(np.log(sigmoid + 1e-10))

    # Gradient: d/d_diff (-log(sigmoid(diff))) = sigmoid(diff) - 1
    # = -(1 - sigmoid(diff)) = -sigmoid(-diff)
    grad_diff = (sigmoid - 1) / len(diff)  # [batch, 1]

    grad_chosen = grad_diff
    grad_rejected = -grad_diff

    return float(loss), grad_chosen, grad_rejected


# =============================================================================
# Direct Preference Optimization (DPO)
# =============================================================================

class DPOTrainer:
    """
    Direct Preference Optimization trainer.

    DPO (Rafailov et al., 2023) is a simpler alternative to RLHF.

    Key insight: The optimal RLHF policy satisfies:
        π*(y|x) ∝ π_ref(y|x) * exp(r(x,y) / β)

    Rearranging gives the implicit reward:
        r(x,y) = β * log(π*(y|x) / π_ref(y|x))

    Substituting into Bradley-Terry:
        L_DPO = -log(σ(β * log(π(y_w|x)/π_ref(y_w|x))
                      - β * log(π(y_l|x)/π_ref(y_l|x))))

    This eliminates the need for a separate reward model!
    We directly optimize the policy using preference data.

    β controls the KL penalty:
    - High β: Stay close to reference policy
    - Low β: More aggressively optimize preferences
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize DPO trainer.

        Args:
            beta: KL penalty coefficient (temperature)
            label_smoothing: Label smoothing for stability
        """
        self.beta = beta
        self.label_smoothing = label_smoothing

    def compute_loss(
        self,
        policy_chosen_logps: np.ndarray,
        policy_rejected_logps: np.ndarray,
        ref_chosen_logps: np.ndarray,
        ref_rejected_logps: np.ndarray,
    ) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
        """
        Compute DPO loss.

        Args:
            policy_chosen_logps: Log probs of chosen under policy [batch]
            policy_rejected_logps: Log probs of rejected under policy [batch]
            ref_chosen_logps: Log probs of chosen under reference [batch]
            ref_rejected_logps: Log probs of rejected under reference [batch]

        Returns:
            (loss, metrics, grad_policy_chosen, grad_policy_rejected)
        """
        # Log ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        # Implicit rewards (scaled by beta)
        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        # DPO loss: -log(sigmoid(r_chosen - r_rejected))
        logits = chosen_rewards - rejected_rewards  # [batch]

        # Label smoothing
        if self.label_smoothing > 0:
            # Soft labels: instead of 1.0, use 1 - ε
            labels = 1.0 - self.label_smoothing
        else:
            labels = 1.0

        # Binary cross-entropy with soft labels
        # Loss = -labels * log(sigmoid(x)) - (1-labels) * log(1-sigmoid(x))
        sigmoid_logits = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

        loss = -np.mean(
            labels * np.log(sigmoid_logits + 1e-10) +
            (1 - labels) * np.log(1 - sigmoid_logits + 1e-10)
        )

        # Metrics
        accuracy = np.mean(logits > 0)
        chosen_reward_mean = np.mean(chosen_rewards)
        rejected_reward_mean = np.mean(rejected_rewards)
        reward_margin = chosen_reward_mean - rejected_reward_mean

        metrics = {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'chosen_reward': float(chosen_reward_mean),
            'rejected_reward': float(rejected_reward_mean),
            'reward_margin': float(reward_margin),
        }

        # Gradients w.r.t. policy log probs
        # d/d_logits = labels * (sigmoid - 1) + (1 - labels) * sigmoid
        #            = sigmoid - labels
        grad_logits = (sigmoid_logits - labels) / len(logits)

        # Chain rule through beta * (policy - ref)
        grad_policy_chosen = self.beta * grad_logits
        grad_policy_rejected = -self.beta * grad_logits

        return loss, metrics, grad_policy_chosen, grad_policy_rejected


def compute_log_probs(
    logits: np.ndarray,
    tokens: np.ndarray,
) -> np.ndarray:
    """
    Compute log probabilities of tokens under logits.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        tokens: Target tokens [batch, seq_len]

    Returns:
        Sum of log probs per sequence [batch]
    """
    batch, seq_len, vocab = logits.shape

    # Log softmax
    logits_max = logits.max(axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(logits - logits_max), axis=-1, keepdims=True))
    log_probs = logits - logits_max - log_sum_exp

    # Select log probs for target tokens
    # Shape: [batch, seq_len]
    selected_log_probs = np.take_along_axis(
        log_probs,
        tokens[:, :, np.newaxis],
        axis=-1
    ).squeeze(-1)

    # Sum over sequence
    return selected_log_probs.sum(axis=-1)


# =============================================================================
# PPO Components (for RLHF)
# =============================================================================

class PPOBuffer:
    """
    Experience buffer for PPO training.

    Stores trajectories: (state, action, reward, value, log_prob)
    Used to compute advantages and train the policy.
    """

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        """
        Initialize buffer.

        Args:
            gamma: Discount factor
            lam: GAE lambda for advantage estimation
        """
        self.gamma = gamma
        self.lam = lam

        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool = False,
    ) -> None:
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_advantages(self, last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE (Generalized Advantage Estimation).

        GAE balances bias vs. variance in advantage estimation:
            A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

        where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD error.

        Returns:
            (advantages, returns)
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        # Work backwards
        last_gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])

            # TD error
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]

            # GAE
            advantages[t] = last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae

            # Returns for value function training
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def get_batch(self) -> Dict[str, np.ndarray]:
        """Get all data as arrays."""
        advantages, returns = self.compute_advantages()

        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'old_log_probs': np.array(self.log_probs),
            'advantages': advantages,
            'returns': returns,
            'values': np.array(self.values),
        }

    def clear(self) -> None:
        """Clear the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()


def ppo_loss(
    new_log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    clip_ratio: float = 0.2,
) -> Tuple[float, np.ndarray]:
    """
    PPO clipped surrogate loss.

    L^CLIP = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)

    where r_t = π(a|s) / π_old(a|s) is the probability ratio.

    The clipping prevents too large policy updates.

    Args:
        new_log_probs: Log probs under current policy
        old_log_probs: Log probs under old policy (from rollout)
        advantages: Estimated advantages
        clip_ratio: Clipping parameter ε

    Returns:
        (loss, gradient w.r.t. new_log_probs)
    """
    # Probability ratio
    ratio = np.exp(new_log_probs - old_log_probs)

    # Clipped ratio
    clipped_ratio = np.clip(ratio, 1 - clip_ratio, 1 + clip_ratio)

    # Two surrogate objectives
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages

    # Take minimum (pessimistic bound)
    loss = -np.mean(np.minimum(surr1, surr2))

    # Gradient (only where not clipped)
    clipped = (ratio < 1 - clip_ratio) | (ratio > 1 + clip_ratio)
    grad_ratio = np.where(clipped, 0, advantages)
    grad_log_prob = ratio * grad_ratio / len(ratio)

    return float(loss), -grad_log_prob  # Negative because we minimize


def value_loss(
    predicted_values: np.ndarray,
    returns: np.ndarray,
    old_values: Optional[np.ndarray] = None,
    clip_value: Optional[float] = None,
) -> Tuple[float, np.ndarray]:
    """
    Value function loss (MSE with optional clipping).

    Args:
        predicted_values: Current value predictions
        returns: Target returns (from GAE)
        old_values: Old value predictions (for clipping)
        clip_value: Clipping parameter for value function

    Returns:
        (loss, gradient)
    """
    if clip_value is not None and old_values is not None:
        # Clipped value loss (PPO style)
        clipped_values = old_values + np.clip(
            predicted_values - old_values, -clip_value, clip_value
        )
        loss_unclipped = (predicted_values - returns) ** 2
        loss_clipped = (clipped_values - returns) ** 2
        loss = 0.5 * np.mean(np.maximum(loss_unclipped, loss_clipped))

        # Gradient where unclipped loss is larger
        use_clipped = loss_clipped > loss_unclipped
        grad = np.where(use_clipped, 0, predicted_values - returns) / len(returns)
    else:
        # Simple MSE
        loss = 0.5 * np.mean((predicted_values - returns) ** 2)
        grad = (predicted_values - returns) / len(returns)

    return float(loss), grad


# =============================================================================
# KL Divergence Penalty
# =============================================================================

def kl_divergence(
    log_probs_p: np.ndarray,
    log_probs_q: np.ndarray,
) -> float:
    """
    Compute KL(P || Q) from log probabilities.

    KL(P || Q) = Σ P(x) * (log P(x) - log Q(x))
               = Σ exp(log P(x)) * (log P(x) - log Q(x))

    Used to penalize deviation from reference policy in RLHF.
    """
    p = np.exp(log_probs_p)
    return float(np.sum(p * (log_probs_p - log_probs_q)))


def kl_penalty_reward(
    policy_log_probs: np.ndarray,
    ref_log_probs: np.ndarray,
    kl_coef: float = 0.1,
) -> np.ndarray:
    """
    Compute KL penalty to add to rewards.

    In RLHF, we add a KL penalty to prevent the policy from
    deviating too far from the reference (usually SFT model):

    r_total = r_reward_model - β * KL(π || π_ref)

    Args:
        policy_log_probs: Log probs under policy
        ref_log_probs: Log probs under reference
        kl_coef: KL penalty coefficient β

    Returns:
        Per-token KL penalties (to subtract from rewards)
    """
    kl_per_token = np.exp(policy_log_probs) * (policy_log_probs - ref_log_probs)
    return kl_coef * kl_per_token


# =============================================================================
# Comparison
# =============================================================================

def compare_alignment_methods() -> Dict[str, Dict[str, Any]]:
    """
    Compare RLHF vs DPO approaches.
    """
    return {
        'RLHF (PPO)': {
            'steps': [
                '1. Train reward model on preferences',
                '2. Generate responses from policy',
                '3. Score with reward model',
                '4. Update policy with PPO',
                '5. Repeat 2-4',
            ],
            'pros': [
                'Well-studied RL algorithm',
                'Can use any reward signal',
                'Online learning from new samples',
            ],
            'cons': [
                'Complex: 3 models (policy, reference, reward)',
                'Unstable: RL training is tricky',
                'Expensive: needs many samples',
            ],
            'models_needed': 3,
        },
        'DPO': {
            'steps': [
                '1. Collect preference pairs',
                '2. Compute policy and reference log probs',
                '3. Apply DPO loss',
                '4. Update policy',
            ],
            'pros': [
                'Simple: supervised learning style',
                'Stable: no RL instabilities',
                'Efficient: direct optimization',
            ],
            'cons': [
                'Offline only: cant learn from new preferences',
                'Needs good reference model',
                'Less flexible than reward models',
            ],
            'models_needed': 2,
        },
    }


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate alignment techniques."""
    print("=" * 60)
    print("Stage 10: Alignment - RLHF and DPO")
    print("=" * 60)

    np.random.seed(42)

    # 1. Reward Model Demo
    print("\n1. Reward Model Training")
    print("-" * 40)

    reward_model = RewardModel(input_dim=64, hidden_dim=32)

    # Simulate preference data
    batch_size = 8
    chosen_repr = np.random.randn(batch_size, 64)
    rejected_repr = np.random.randn(batch_size, 64)

    # Forward pass
    reward_chosen = reward_model.forward(chosen_repr)
    reward_rejected = reward_model.forward(rejected_repr)

    # Compute loss
    loss, grad_c, grad_r = reward_model_loss(reward_chosen, reward_rejected)
    print(f"Reward model loss: {loss:.4f}")
    print(f"Chosen rewards mean: {reward_chosen.mean():.4f}")
    print(f"Rejected rewards mean: {reward_rejected.mean():.4f}")

    # 2. DPO Demo
    print("\n2. Direct Preference Optimization (DPO)")
    print("-" * 40)

    dpo = DPOTrainer(beta=0.1)

    # Simulate log probabilities
    policy_chosen_logps = np.random.randn(batch_size) - 10  # Log probs are negative
    policy_rejected_logps = np.random.randn(batch_size) - 10
    ref_chosen_logps = np.random.randn(batch_size) - 10
    ref_rejected_logps = np.random.randn(batch_size) - 10

    loss, metrics, grad_c, grad_r = dpo.compute_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    )

    print(f"DPO loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Reward margin: {metrics['reward_margin']:.4f}")

    # 3. PPO Components Demo
    print("\n3. PPO Components (for RLHF)")
    print("-" * 40)

    buffer = PPOBuffer(gamma=0.99, lam=0.95)

    # Simulate trajectory
    for t in range(10):
        state = np.random.randn(64)
        action = np.random.randint(0, 100)
        reward = np.random.randn() * 0.1
        value = np.random.randn()
        log_prob = np.random.randn() - 5
        done = (t == 9)

        buffer.add(state, action, reward, value, log_prob, done)

    batch = buffer.get_batch()
    print(f"Collected {len(buffer.rewards)} steps")
    print(f"Advantages shape: {batch['advantages'].shape}")
    print(f"Returns shape: {batch['returns'].shape}")

    # PPO loss
    new_log_probs = batch['old_log_probs'] + np.random.randn(10) * 0.1
    ppo_l, ppo_grad = ppo_loss(
        new_log_probs,
        batch['old_log_probs'],
        batch['advantages'],
        clip_ratio=0.2,
    )
    print(f"PPO loss: {ppo_l:.4f}")

    # 4. Method Comparison
    print("\n4. RLHF vs DPO Comparison")
    print("-" * 40)

    comparison = compare_alignment_methods()
    for method, info in comparison.items():
        print(f"\n{method}:")
        print(f"  Models needed: {info['models_needed']}")
        print(f"  Steps: {len(info['steps'])}")
        print(f"  Pros: {', '.join(info['pros'][:2])}")
        print(f"  Cons: {', '.join(info['cons'][:2])}")


if __name__ == '__main__':
    demo()

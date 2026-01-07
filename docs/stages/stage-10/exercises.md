# Stage 10 Exercises

## Conceptual Questions

### Exercise 10.1: The Alignment Problem
Consider why pre-training alone isn't enough.

**a)** A model trained on internet text might learn to... (give examples of problematic behaviors)
**b)** Why can't we just filter the training data to fix this?
**c)** What's the difference between a "helpful" and a "sycophantic" model?

### Exercise 10.2: Reward Modeling
Explain the reward modeling process:

**a)** What is the Bradley-Terry model?
**b)** Why do we collect preference pairs instead of absolute ratings?
**c)** What problems can arise from reward model overoptimization?

### Exercise 10.3: RLHF vs DPO
Compare the two main alignment approaches:

**a)** How many models does RLHF require? DPO?
**b)** Why is DPO more stable than PPO-based RLHF?
**c)** When might you prefer RLHF over DPO?

### Exercise 10.4: KL Divergence Penalty
Explain the role of KL divergence in alignment:

**a)** Why do we penalize divergence from the reference policy?
**b)** What happens if the KL coefficient is too low?
**c)** What happens if it's too high?

---

## Implementation Exercises

### Exercise 10.5: Preference Data
Implement preference data handling:

```python
@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str

def create_preference_batch(
    pairs: List[PreferencePair],
    tokenizer,
) -> Dict[str, np.ndarray]:
    """
    Create a batch for training from preference pairs.

    Returns:
        Dictionary with:
        - prompt_ids: [batch, prompt_len]
        - chosen_ids: [batch, response_len]
        - rejected_ids: [batch, response_len]
    """
    # TODO
    pass
```

### Exercise 10.6: Reward Model
Implement a simple reward model:

```python
class RewardModel:
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """
        Reward model that predicts preference scores.

        Architecture: MLP that takes sequence representation
        and outputs scalar reward.
        """
        # TODO: Initialize weights
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute reward for input representations.

        Args:
            x: [batch, input_dim] - e.g., last hidden state

        Returns:
            rewards: [batch, 1]
        """
        # TODO
        pass

def reward_loss(
    reward_chosen: np.ndarray,
    reward_rejected: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Bradley-Terry preference loss.

    Loss = -log(sigmoid(r_chosen - r_rejected))

    Returns:
        (loss, grad_chosen, grad_rejected)
    """
    # TODO
    pass
```

### Exercise 10.7: DPO Loss
Implement the DPO loss function:

```python
def dpo_loss(
    policy_chosen_logp: np.ndarray,
    policy_rejected_logp: np.ndarray,
    ref_chosen_logp: np.ndarray,
    ref_rejected_logp: np.ndarray,
    beta: float = 0.1,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Direct Preference Optimization loss.

    L_DPO = -log(sigmoid(beta * (log(pi/pi_ref)_chosen - log(pi/pi_ref)_rejected)))

    Args:
        policy_chosen_logp: Log probs of chosen under policy
        policy_rejected_logp: Log probs of rejected under policy
        ref_chosen_logp: Log probs of chosen under reference
        ref_rejected_logp: Log probs of rejected under reference
        beta: KL penalty coefficient

    Returns:
        (loss, metrics_dict, grad_policy_chosen, grad_policy_rejected)
    """
    # TODO
    pass
```

### Exercise 10.8: Log Probability Computation
Implement log probability extraction:

```python
def compute_sequence_logprob(
    logits: np.ndarray,  # [batch, seq, vocab]
    tokens: np.ndarray,  # [batch, seq]
    mask: np.ndarray = None,  # [batch, seq] - 1 for valid, 0 for padding
) -> np.ndarray:
    """
    Compute log probability of token sequences.

    Returns:
        log_probs: [batch] - sum of log probs per sequence
    """
    # TODO:
    # 1. Apply log_softmax to logits
    # 2. Select log probs for target tokens
    # 3. Apply mask
    # 4. Sum over sequence
    pass
```

---

## Challenge Exercises

### Exercise 10.9: PPO Components
Implement core PPO components for RLHF:

```python
class PPOBuffer:
    """Experience buffer for PPO."""

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam
        # TODO: Initialize storage

    def add(self, state, action, reward, value, log_prob, done):
        """Add a transition."""
        # TODO
        pass

    def compute_advantages(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE (Generalized Advantage Estimation).

        Returns:
            (advantages, returns)
        """
        # TODO: Implement GAE
        pass


def ppo_loss(
    new_log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    clip_ratio: float = 0.2,
) -> Tuple[float, np.ndarray]:
    """
    PPO clipped surrogate loss.

    L = min(r * A, clip(r, 1-eps, 1+eps) * A)
    where r = exp(new_log_prob - old_log_prob)

    Returns:
        (loss, gradient)
    """
    # TODO
    pass
```

### Exercise 10.10: Full DPO Trainer
Build a complete DPO training loop:

```python
class DPOTrainer:
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        beta: float = 0.1,
        learning_rate: float = 1e-6,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.lr = learning_rate

    def train_step(
        self,
        batch: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        One DPO training step.

        1. Compute policy log probs for chosen/rejected
        2. Compute reference log probs (no gradient)
        3. Compute DPO loss
        4. Update policy model

        Returns:
            Metrics dict
        """
        # TODO
        pass

    def train(
        self,
        dataset: List[PreferencePair],
        epochs: int = 1,
        batch_size: int = 4,
    ):
        """Full training loop."""
        # TODO
        pass
```

### Exercise 10.11: Reward Hacking Analysis
Explore reward model overoptimization:

```python
def analyze_reward_hacking(
    policy_rewards: List[float],  # Rewards during training
    true_quality: List[float],    # Human evaluation scores
) -> Dict[str, Any]:
    """
    Analyze if the policy is gaming the reward model.

    Signs of reward hacking:
    - Reward increases but quality decreases
    - Policy outputs become degenerate
    - Large divergence from reference

    Returns:
        Analysis dict with correlation, divergence point, etc.
    """
    # TODO
    pass
```

### Exercise 10.12: Constitutional AI Basics
Implement a simple self-critique mechanism:

```python
class ConstitutionalCritic:
    """
    Basic implementation of Constitutional AI ideas.

    The model critiques its own outputs against principles.
    """

    def __init__(self, model, principles: List[str]):
        """
        Args:
            model: Language model
            principles: List of principles like "Be helpful", "Avoid harm"
        """
        self.model = model
        self.principles = principles

    def critique(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Critique a response against principles.

        Returns:
            - violations: List of violated principles
            - suggestions: How to improve
        """
        # TODO: Use model to self-critique
        pass

    def revise(self, prompt: str, response: str, critique: str) -> str:
        """
        Generate revised response based on critique.
        """
        # TODO
        pass
```

---

## Solutions

Solutions are available in `code/stage-10/solutions/`.

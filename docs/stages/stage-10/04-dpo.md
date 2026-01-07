# Section 10.4: Direct Preference Optimization (DPO)

*Reading time: 12 minutes*

## The DPO Revolution

Rafailov et al. (2023) asked: Do we actually need RL?

The answer: **No.**

DPO achieves the same goal as RLHF with a simple supervised loss.

## The Key Insight

The optimal RLHF policy has a closed form:

$$\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

Rearranging for the reward:

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

The reward is just a log ratio!

## From Reward to Loss

Substituting the implicit reward into Bradley-Terry:

$$P(\text{chosen} > \text{rejected}) = \sigma\left(\beta \log \frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

The DPO loss:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

## What This Means

**RLHF**:

1. Train reward model
2. Generate samples
3. Compute rewards
4. PPO update
5. Repeat

**DPO**:

1. Compute log probs under policy and reference
2. Apply DPO loss
3. Update

That's it. No reward model. No RL. No generation during training.

## Implementation

```python
class DPOTrainer:
    """Direct Preference Optimization trainer."""

    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def compute_loss(
        self,
        policy_chosen_logps: np.ndarray,    # log π(y_w|x)
        policy_rejected_logps: np.ndarray,  # log π(y_l|x)
        ref_chosen_logps: np.ndarray,       # log π_ref(y_w|x)
        ref_rejected_logps: np.ndarray,     # log π_ref(y_l|x)
    ):
        # Log ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        # Implicit rewards (scaled)
        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        # DPO loss
        logits = chosen_rewards - rejected_rewards
        loss = -np.mean(np.log(sigmoid(logits)))

        # Accuracy: how often do we correctly predict preference?
        accuracy = np.mean(logits > 0)

        return loss, accuracy
```

## The DPO Training Loop

```python
# Freeze reference model
ref_model = copy(policy_model)
ref_model.freeze()

for batch in preference_data:
    prompt = batch.prompt
    chosen = batch.chosen
    rejected = batch.rejected

    # Compute log probs (no generation needed!)
    policy_chosen_logps = policy_model.log_prob(prompt, chosen)
    policy_rejected_logps = policy_model.log_prob(prompt, rejected)
    ref_chosen_logps = ref_model.log_prob(prompt, chosen)
    ref_rejected_logps = ref_model.log_prob(prompt, rejected)

    # DPO loss
    loss, accuracy = dpo_trainer.compute_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    )

    # Standard supervised update
    loss.backward()
    optimizer.step()
```

## Why DPO Works

### Mathematical Equivalence

DPO optimizes the same objective as RLHF:

$$\max_\pi \mathbb{E}[r(x,y)] - \beta \cdot \text{KL}(\pi || \pi_{\text{ref}})$$

But directly, without the RL machinery.

### Implicit Reward

The trained policy implicitly defines a reward:

$$r(x,y) = \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$$

You can extract this for analysis if needed.

## The Beta Parameter

$\beta$ controls the trade-off:

| Beta | Effect |
|------|--------|
| High (0.5+) | Stay close to reference, conservative updates |
| Medium (0.1) | Balanced (common default) |
| Low (0.01) | Aggressively optimize preferences, may diverge |

**Default recommendation**: Start with $\beta = 0.1$.

## DPO vs RLHF

| Aspect | RLHF | DPO |
|--------|------|-----|
| Reward model | Required | Not needed |
| RL loop | Yes (PPO) | No |
| Generation during training | Yes | No |
| Models needed | 3 (policy, ref, reward) | 2 (policy, ref) |
| Stability | Tricky | Very stable |
| Sample efficiency | Low | High |
| Flexibility | High | Medium |

## When to Choose DPO

**Use DPO when**:

- You have offline preference data
- You want simplicity and stability
- You don't need online learning

**Use RLHF when**:

- You need online learning from new preferences
- You want to use custom reward signals
- You have complex reward functions

## Variations

### IPO (Identity Preference Optimization)

Addresses potential overfitting in DPO:

$$\mathcal{L}_{\text{IPO}} = \left(\log \frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)} - \log \frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)} - \frac{1}{2\beta}\right)^2$$

### KTO (Kahneman-Tversky Optimization)

Works with binary feedback (good/bad) instead of comparisons.

### ORPO (Odds Ratio Preference Optimization)

Incorporates the loss into a single supervised objective.

## Practical Tips

### 1. Reference Model

Must be frozen. Usually the SFT model.

### 2. Learning Rate

Typically lower than SFT (1e-6 to 1e-5).

### 3. Batch Size

Larger batches help with stability.

### 4. Data Quality

DPO is very sensitive to preference data quality.

### 5. Label Smoothing

Can help prevent overconfidence:

```python
labels = 1.0 - label_smoothing  # e.g., 0.99 instead of 1.0
```

## Metrics to Track

| Metric | What It Tells You |
|--------|-------------------|
| Loss | Training progress |
| Accuracy | Preference prediction |
| Chosen reward | How "good" the chosen response is |
| Rejected reward | How "good" the rejected response is |
| Reward margin | chosen - rejected (should increase) |

## Summary

| Concept | Description |
|---------|-------------|
| Core idea | Optimal policy has closed form |
| Loss | Log sigmoid of reward difference |
| Beta | KL penalty strength |
| Advantage | Simpler, more stable than RLHF |
| Limitation | Offline only |

**Key insight**: DPO shows that alignment doesn't require RL—it can be done with supervised learning.

**Next**: We'll compare the methods and discuss when to use each.

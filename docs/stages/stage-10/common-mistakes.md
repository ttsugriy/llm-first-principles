# Stage 10: Common Mistakes

## Mistake 1: Not Freezing Reference Model

**Symptom**: DPO loss behaves strangely, reference drifts

**Wrong**:
```python
# Reference model also getting updated
ref_logps = ref_model.forward(x)  # Still has gradients!
policy_logps = policy_model.forward(x)
loss = dpo_loss(policy_logps, ref_logps)
loss.backward()  # Updates both models!
```

**The fix**: Freeze reference model completely
```python
# Reference should never be updated
with no_grad():
    ref_logps = ref_model.forward(x)  # No gradients

policy_logps = policy_model.forward(x)
loss = dpo_loss(policy_logps, ref_logps)
# Only policy gets updated
```

---

## Mistake 2: Wrong Log Probability Computation

**Symptom**: Loss is unstable or doesn't decrease

**Wrong**:
```python
# Averaging log probs instead of summing
log_probs = log_softmax(logits)
sequence_logp = log_probs.mean()  # Wrong!
```

**The fix**: Sum log probabilities over sequence
```python
def compute_sequence_logp(logits, tokens, mask=None):
    # Log softmax
    log_probs = logits - logsumexp(logits, axis=-1, keepdims=True)

    # Select target tokens
    selected = np.take_along_axis(log_probs, tokens[..., None], axis=-1)
    selected = selected.squeeze(-1)

    # Apply mask and SUM (not mean!)
    if mask is not None:
        selected = selected * mask

    return selected.sum(axis=-1)  # Sum over sequence
```

---

## Mistake 3: Reward Model Overfitting

**Symptom**: High reward doesn't correlate with quality

**Wrong**:
```python
# Training reward model to 100% accuracy on training set
for epoch in range(100):
    train_reward_model()  # Keeps going until perfect
```

**The fix**: Early stopping, regularization, validation
```python
best_val_accuracy = 0
for epoch in range(max_epochs):
    train_loss = train_epoch()
    val_accuracy = evaluate(val_set)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        save_checkpoint()
    else:
        # Early stopping
        if epochs_without_improvement > patience:
            break
```

---

## Mistake 4: KL Coefficient Too Low

**Symptom**: Model outputs become degenerate or repetitive

**Wrong**:
```python
dpo_trainer = DPOTrainer(beta=0.01)  # Too low!
# Model aggressively optimizes reward, loses coherence
```

**The fix**: Start with moderate KL penalty
```python
# Typical beta values: 0.1 - 0.5
dpo_trainer = DPOTrainer(beta=0.1)

# For RLHF, typical KL coefficients: 0.01 - 0.2
kl_coef = 0.1
reward_with_kl = reward - kl_coef * kl_divergence
```

---

## Mistake 5: Not Normalizing Rewards

**Symptom**: PPO training unstable, high variance

**Wrong**:
```python
# Using raw rewards with arbitrary scale
advantage = rewards - values  # Could be [-100, +100]
```

**The fix**: Normalize advantages
```python
def normalize_advantages(advantages):
    mean = np.mean(advantages)
    std = np.std(advantages) + 1e-8
    return (advantages - mean) / std

# In training
advantages = compute_gae(rewards, values)
advantages = normalize_advantages(advantages)
```

---

## Mistake 6: Preference Pair Order Matters

**Symptom**: Model learns opposite preferences

**Wrong**:
```python
# Swapped chosen and rejected
pair = PreferencePair(
    prompt="What is 2+2?",
    chosen="I don't know",      # Actually the bad response!
    rejected="4"                 # Actually the good response!
)
```

**The fix**: Verify data quality
```python
def validate_preference_data(pairs):
    """Sanity check preference pairs."""
    issues = []
    for i, pair in enumerate(pairs):
        # Check for swapped pairs (heuristics)
        if len(pair.rejected) > len(pair.chosen) * 3:
            issues.append(f"Pair {i}: Rejected much longer (suspicious)")

        # Check for identical responses
        if pair.chosen == pair.rejected:
            issues.append(f"Pair {i}: Identical responses")

    return issues
```

---

## Mistake 7: PPO Clip Ratio Wrong

**Symptom**: Training unstable, policy changes too much

**Wrong**:
```python
# Clip ratio too high
clip_ratio = 0.5  # Allows 50% change per step!
```

**The fix**: Use conservative clip ratio
```python
# Standard PPO clip ratio
clip_ratio = 0.2  # Only allow 20% change

def ppo_loss(new_log_probs, old_log_probs, advantages, clip_ratio=0.2):
    ratio = np.exp(new_log_probs - old_log_probs)
    clipped_ratio = np.clip(ratio, 1 - clip_ratio, 1 + clip_ratio)
    return -np.minimum(ratio * advantages, clipped_ratio * advantages).mean()
```

---

## Mistake 8: Ignoring Response Length Bias

**Symptom**: Model learns to output longer or shorter responses

**Wrong**:
```python
# Summing log probs without length normalization
chosen_score = policy_logp_chosen.sum()
# Longer responses naturally have lower (more negative) log prob
```

**The fix**: Consider length normalization
```python
# Option 1: Length-normalized log prob
chosen_score = policy_logp_chosen.sum() / len(chosen_tokens)

# Option 2: Include length penalty in reward
length_penalty = 0.01 * abs(len(response) - target_length)
reward = base_reward - length_penalty
```

---

## Mistake 9: Not Handling Numerical Stability

**Symptom**: NaN loss, especially with long sequences

**Wrong**:
```python
# Direct sigmoid computation
sigmoid = 1 / (1 + np.exp(-x))  # Overflow when x << 0
```

**The fix**: Use numerically stable implementations
```python
def stable_sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )

def stable_log_sigmoid(x):
    """Numerically stable log(sigmoid(x))."""
    return np.where(
        x >= 0,
        -np.log1p(np.exp(-x)),
        x - np.log1p(np.exp(x))
    )
```

---

## Mistake 10: Single Epoch DPO Training

**Symptom**: Model doesn't converge

**Wrong**:
```python
# Only one pass through data
for batch in dataset:
    train_step(batch)
# Done after one epoch
```

**The fix**: Multiple epochs with care
```python
# DPO typically needs 1-3 epochs
# More epochs risk overfitting to preference data
for epoch in range(3):
    for batch in shuffle(dataset):
        loss = train_step(batch)

    # Evaluate after each epoch
    val_accuracy = evaluate(val_set)
    if val_accuracy < prev_accuracy:
        print("Validation degrading, stopping")
        break
```

---

## Mistake 11: Reward Hacking

**Symptom**: High reward but low quality outputs

**Wrong**:
```python
# Training until reward is maximized
while reward < max_reward:
    policy_step()  # Eventually finds exploit
```

**The fix**: Monitor for reward hacking
```python
def detect_reward_hacking(history):
    """Detect if policy is gaming the reward model."""
    rewards = history['rewards']
    kl_divs = history['kl_divergence']

    # Warning signs:
    # 1. Reward increasing but KL exploding
    if kl_divs[-1] > 10 * kl_divs[0]:
        print("WARNING: Large KL divergence")

    # 2. Reward near maximum (suspicious)
    if np.mean(rewards[-100:]) > 0.95 * max_possible_reward:
        print("WARNING: Suspiciously high reward")

    # 3. Output diversity collapsing
    if compute_diversity(recent_outputs) < threshold:
        print("WARNING: Output diversity collapsed")
```

---

## Mistake 12: Forgetting to Evaluate on Real Tasks

**Symptom**: Metrics look good but model isn't actually helpful

**Wrong**:
```python
# Only tracking training metrics
print(f"DPO Loss: {loss}, Accuracy: {acc}")
# Never checking actual model outputs
```

**The fix**: Regular qualitative evaluation
```python
def evaluate_alignment(model, eval_prompts):
    """Evaluate model on real prompts."""
    results = []
    for prompt in eval_prompts:
        response = model.generate(prompt)
        results.append({
            'prompt': prompt,
            'response': response,
            'length': len(response),
        })

    # Print samples for human review
    for r in random.sample(results, min(5, len(results))):
        print(f"Prompt: {r['prompt']}")
        print(f"Response: {r['response'][:200]}")
        print("---")

    return results

# Run periodically during training
if step % 1000 == 0:
    evaluate_alignment(model, test_prompts)
```

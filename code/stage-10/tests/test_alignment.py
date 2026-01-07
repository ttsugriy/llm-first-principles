"""
Tests for Stage 10: Alignment (RLHF and DPO)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from alignment import (
    PreferencePair,
    create_preference_dataset,
    RewardModel,
    reward_model_loss,
    DPOTrainer,
    compute_log_probs,
    PPOBuffer,
    ppo_loss,
    value_loss,
    kl_divergence,
    kl_penalty_reward,
    compare_alignment_methods,
)


# =============================================================================
# Preference Data Tests
# =============================================================================

def test_preference_pair():
    """Test PreferencePair creation."""
    pair = PreferencePair(
        prompt="What is 2+2?",
        chosen="2+2 equals 4.",
        rejected="2+2 equals 5.",
    )

    assert pair.prompt == "What is 2+2?"
    assert pair.chosen == "2+2 equals 4."
    assert pair.rejected == "2+2 equals 5."
    print("✓ test_preference_pair passed")


def test_create_preference_dataset():
    """Test dataset creation."""
    prompts = ["Q1", "Q2", "Q3"]
    chosen = ["A1", "A2", "A3"]
    rejected = ["B1", "B2", "B3"]

    dataset = create_preference_dataset(prompts, chosen, rejected)

    assert len(dataset) == 3
    assert dataset[0].prompt == "Q1"
    assert dataset[1].chosen == "A2"
    print("✓ test_create_preference_dataset passed")


# =============================================================================
# Reward Model Tests
# =============================================================================

def test_reward_model_forward():
    """Test reward model forward pass."""
    np.random.seed(42)

    model = RewardModel(input_dim=64, hidden_dim=32)
    x = np.random.randn(8, 64)

    reward = model.forward(x)

    assert reward.shape == (8, 1), f"Expected (8, 1), got {reward.shape}"
    print("✓ test_reward_model_forward passed")


def test_reward_model_backward():
    """Test reward model backward pass."""
    np.random.seed(42)

    model = RewardModel(input_dim=32, hidden_dim=16)
    x = np.random.randn(4, 32)

    reward = model.forward(x)
    grad_reward = np.ones_like(reward)
    grad_x = model.backward(grad_reward)

    assert grad_x.shape == x.shape
    assert 'W1' in model.grads
    assert 'W2' in model.grads
    print("✓ test_reward_model_backward passed")


def test_reward_model_loss():
    """Test Bradley-Terry loss."""
    np.random.seed(42)

    reward_chosen = np.array([[1.0], [2.0], [3.0]])
    reward_rejected = np.array([[0.0], [1.0], [2.0]])

    loss, grad_c, grad_r = reward_model_loss(reward_chosen, reward_rejected)

    # Chosen > rejected, so loss should be low
    assert loss >= 0, "Loss should be non-negative"
    assert loss < 1.0, "Loss should be low when chosen > rejected"
    print("✓ test_reward_model_loss passed")


def test_reward_model_loss_gradient():
    """Test reward model loss gradient numerically."""
    np.random.seed(42)

    reward_chosen = np.array([[0.5]])
    reward_rejected = np.array([[0.3]])

    loss, grad_c, grad_r = reward_model_loss(reward_chosen, reward_rejected)

    # Numerical gradient
    eps = 1e-5
    loss_plus, _, _ = reward_model_loss(reward_chosen + eps, reward_rejected)
    loss_minus, _, _ = reward_model_loss(reward_chosen - eps, reward_rejected)
    numerical_grad = (loss_plus - loss_minus) / (2 * eps)

    assert np.isclose(numerical_grad, grad_c[0, 0], rtol=1e-3), \
        f"Gradient mismatch: {numerical_grad} vs {grad_c[0, 0]}"
    print("✓ test_reward_model_loss_gradient passed")


# =============================================================================
# DPO Tests
# =============================================================================

def test_dpo_trainer_init():
    """Test DPO trainer initialization."""
    dpo = DPOTrainer(beta=0.1, label_smoothing=0.01)

    assert dpo.beta == 0.1
    assert dpo.label_smoothing == 0.01
    print("✓ test_dpo_trainer_init passed")


def test_dpo_loss():
    """Test DPO loss computation."""
    np.random.seed(42)

    dpo = DPOTrainer(beta=0.1)

    batch_size = 8
    policy_chosen = np.random.randn(batch_size) - 10
    policy_rejected = np.random.randn(batch_size) - 10
    ref_chosen = np.random.randn(batch_size) - 10
    ref_rejected = np.random.randn(batch_size) - 10

    loss, metrics, grad_c, grad_r = dpo.compute_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected
    )

    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert 'reward_margin' in metrics
    assert grad_c.shape == (batch_size,)
    assert grad_r.shape == (batch_size,)
    print("✓ test_dpo_loss passed")


def test_dpo_accuracy():
    """Test DPO accuracy when chosen is clearly better."""
    dpo = DPOTrainer(beta=0.1)

    # Make chosen much more likely under policy
    policy_chosen = np.array([-5.0, -5.0, -5.0])
    policy_rejected = np.array([-10.0, -10.0, -10.0])
    ref_chosen = np.array([-7.0, -7.0, -7.0])
    ref_rejected = np.array([-7.0, -7.0, -7.0])

    _, metrics, _, _ = dpo.compute_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected
    )

    assert metrics['accuracy'] == 1.0, "Should have 100% accuracy"
    print("✓ test_dpo_accuracy passed")


def test_compute_log_probs():
    """Test log probability computation."""
    np.random.seed(42)

    batch, seq, vocab = 2, 4, 10
    logits = np.random.randn(batch, seq, vocab)
    tokens = np.random.randint(0, vocab, (batch, seq))

    log_probs = compute_log_probs(logits, tokens)

    assert log_probs.shape == (batch,)
    assert np.all(log_probs <= 0), "Log probs should be non-positive"
    print("✓ test_compute_log_probs passed")


# =============================================================================
# PPO Tests
# =============================================================================

def test_ppo_buffer():
    """Test PPO experience buffer."""
    buffer = PPOBuffer(gamma=0.99, lam=0.95)

    for i in range(5):
        buffer.add(
            state=np.zeros(10),
            action=np.array([i]),
            reward=float(i),
            value=float(i * 0.5),
            log_prob=-1.0,
            done=(i == 4),
        )

    assert len(buffer.rewards) == 5
    assert len(buffer.states) == 5

    batch = buffer.get_batch()
    assert 'advantages' in batch
    assert 'returns' in batch
    assert batch['advantages'].shape == (5,)
    print("✓ test_ppo_buffer passed")


def test_ppo_buffer_gae():
    """Test GAE computation."""
    buffer = PPOBuffer(gamma=0.99, lam=0.95)

    # Simple trajectory: constant reward
    for i in range(3):
        buffer.add(
            state=np.zeros(4),
            action=np.array([0]),
            reward=1.0,
            value=1.0,
            log_prob=-1.0,
            done=(i == 2),
        )

    advantages, returns = buffer.compute_advantages(last_value=0.0)

    assert advantages.shape == (3,)
    assert returns.shape == (3,)
    print("✓ test_ppo_buffer_gae passed")


def test_ppo_loss():
    """Test PPO clipped loss."""
    np.random.seed(42)

    n = 10
    old_log_probs = np.random.randn(n) - 5
    new_log_probs = old_log_probs + np.random.randn(n) * 0.1
    advantages = np.random.randn(n)

    loss, grad = ppo_loss(new_log_probs, old_log_probs, advantages, clip_ratio=0.2)

    assert isinstance(loss, float)
    assert grad.shape == (n,)
    print("✓ test_ppo_loss passed")


def test_ppo_loss_clipping():
    """Test that PPO clips large ratios."""
    old_log_probs = np.array([0.0])
    new_log_probs = np.array([1.0])  # Large change
    advantages = np.array([1.0])

    loss1, grad1 = ppo_loss(new_log_probs, old_log_probs, advantages, clip_ratio=0.2)

    # With clipping at 0.2, ratio should be capped at 1.2
    # So loss should be approximately -1.2
    assert loss1 < 0  # Positive advantage, so negative loss (we minimize)
    print("✓ test_ppo_loss_clipping passed")


def test_value_loss():
    """Test value function loss."""
    predicted = np.array([1.0, 2.0, 3.0])
    returns = np.array([1.1, 1.9, 3.2])

    loss, grad = value_loss(predicted, returns)

    assert loss >= 0, "MSE loss should be non-negative"
    assert grad.shape == predicted.shape
    print("✓ test_value_loss passed")


def test_value_loss_clipped():
    """Test clipped value loss."""
    predicted = np.array([2.0])
    returns = np.array([1.0])
    old_values = np.array([1.5])

    loss, grad = value_loss(predicted, returns, old_values, clip_value=0.2)

    assert loss >= 0
    print("✓ test_value_loss_clipped passed")


# =============================================================================
# KL Divergence Tests
# =============================================================================

def test_kl_divergence():
    """Test KL divergence computation."""
    # Same distribution: KL should be 0
    log_probs = np.log(np.array([0.5, 0.3, 0.2]))

    kl = kl_divergence(log_probs, log_probs)
    assert np.isclose(kl, 0.0, atol=1e-6), f"KL(P||P) should be 0, got {kl}"
    print("✓ test_kl_divergence passed")


def test_kl_penalty():
    """Test KL penalty reward."""
    policy_log_probs = np.array([-1.0, -2.0, -3.0])
    ref_log_probs = np.array([-1.0, -2.0, -3.0])

    penalty = kl_penalty_reward(policy_log_probs, ref_log_probs, kl_coef=0.1)

    # Same log probs: penalty should be 0
    assert np.allclose(penalty, 0.0, atol=1e-6)
    print("✓ test_kl_penalty passed")


# =============================================================================
# Integration Tests
# =============================================================================

def test_reward_model_training_step():
    """Test a complete reward model training step."""
    np.random.seed(42)

    model = RewardModel(input_dim=32, hidden_dim=16)

    # Create preference batch
    chosen_repr = np.random.randn(4, 32)
    rejected_repr = np.random.randn(4, 32)

    # Forward
    r_chosen = model.forward(chosen_repr)

    # Need to reset cache for rejected
    model_copy = RewardModel(input_dim=32, hidden_dim=16)
    model_copy.W1 = model.W1.copy()
    model_copy.b1 = model.b1.copy()
    model_copy.W2 = model.W2.copy()
    model_copy.b2 = model.b2.copy()

    r_rejected = model_copy.forward(rejected_repr)

    # Loss
    loss, grad_c, grad_r = reward_model_loss(r_chosen, r_rejected)

    # Backward for chosen
    model.backward(grad_c)

    # Verify gradients exist
    assert model.grads['W1'] is not None
    assert model.grads['W2'] is not None
    print("✓ test_reward_model_training_step passed")


def test_dpo_gradient():
    """Test DPO gradient numerically."""
    np.random.seed(42)

    dpo = DPOTrainer(beta=0.1)

    policy_chosen = np.array([0.0])
    policy_rejected = np.array([-1.0])
    ref_chosen = np.array([-0.5])
    ref_rejected = np.array([-0.5])

    loss, _, grad_c, _ = dpo.compute_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected
    )

    # Numerical gradient
    eps = 1e-5
    loss_plus, _, _, _ = dpo.compute_loss(
        policy_chosen + eps, policy_rejected, ref_chosen, ref_rejected
    )
    loss_minus, _, _, _ = dpo.compute_loss(
        policy_chosen - eps, policy_rejected, ref_chosen, ref_rejected
    )
    numerical = (loss_plus - loss_minus) / (2 * eps)

    assert np.isclose(numerical, grad_c[0], rtol=1e-2), \
        f"DPO gradient mismatch: {numerical} vs {grad_c[0]}"
    print("✓ test_dpo_gradient passed")


def test_compare_methods():
    """Test method comparison."""
    comparison = compare_alignment_methods()

    assert 'RLHF (PPO)' in comparison
    assert 'DPO' in comparison
    assert comparison['DPO']['models_needed'] < comparison['RLHF (PPO)']['models_needed']
    print("✓ test_compare_methods passed")


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Stage 10 Alignment Tests")
    print("=" * 60)
    print()

    tests = [
        # Preference data
        test_preference_pair,
        test_create_preference_dataset,

        # Reward model
        test_reward_model_forward,
        test_reward_model_backward,
        test_reward_model_loss,
        test_reward_model_loss_gradient,

        # DPO
        test_dpo_trainer_init,
        test_dpo_loss,
        test_dpo_accuracy,
        test_compute_log_probs,

        # PPO
        test_ppo_buffer,
        test_ppo_buffer_gae,
        test_ppo_loss,
        test_ppo_loss_clipping,
        test_value_loss,
        test_value_loss_clipped,

        # KL
        test_kl_divergence,
        test_kl_penalty,

        # Integration
        test_reward_model_training_step,
        test_dpo_gradient,
        test_compare_methods,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

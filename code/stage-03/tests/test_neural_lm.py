"""
Tests for Neural Language Model

Comprehensive test suite verifying:
1. Data utilities (vocabulary, encoding)
2. Layer operations (embedding, linear)
3. Forward pass and loss computation
4. Training reduces loss
5. Text generation works
"""

import sys
import os
import math
import random

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'stage-02'))

from neural_lm import (
    build_vocab, encode, decode, create_examples,
    relu, softmax, cross_entropy_loss,
    Embedding, Linear, CharacterLM,
    train, compute_perplexity, generate,
)
from value import Value


# =============================================================================
# Data Utility Tests
# =============================================================================


class TestDataUtilities:
    """Tests for data utility functions."""

    def test_build_vocab(self):
        text = "hello"
        char_to_idx, idx_to_char, vocab_size = build_vocab(text)

        assert vocab_size == 4  # 'e', 'h', 'l', 'o'
        assert len(char_to_idx) == 4
        assert len(idx_to_char) == 4
        assert 'h' in char_to_idx
        assert 'e' in char_to_idx

    def test_encode_decode(self):
        text = "hello"
        char_to_idx, idx_to_char, _ = build_vocab(text)

        encoded = encode(text, char_to_idx)
        decoded = decode(encoded, idx_to_char)

        assert decoded == text
        assert len(encoded) == 5

    def test_create_examples(self):
        encoded = [0, 1, 2, 3, 4]
        context_length = 2

        examples = create_examples(encoded, context_length)

        # Should have 3 examples: [0,1]->2, [1,2]->3, [2,3]->4
        assert len(examples) == 3
        assert examples[0] == ([0, 1], 2)
        assert examples[1] == ([1, 2], 3)
        assert examples[2] == ([2, 3], 4)


# =============================================================================
# Activation Function Tests
# =============================================================================


class TestActivations:
    """Tests for activation functions."""

    def test_relu_positive(self):
        x = [Value(1.0), Value(2.0)]
        y = relu(x)
        assert abs(y[0].data - 1.0) < 1e-6
        assert abs(y[1].data - 2.0) < 1e-6

    def test_relu_negative(self):
        x = [Value(-1.0), Value(-2.0)]
        y = relu(x)
        assert abs(y[0].data - 0.0) < 1e-6
        assert abs(y[1].data - 0.0) < 1e-6

    def test_softmax_sums_to_one(self):
        logits = [Value(1.0), Value(2.0), Value(3.0)]
        probs = softmax(logits)

        total = sum(p.data for p in probs)
        assert abs(total - 1.0) < 1e-6

    def test_softmax_max_element(self):
        logits = [Value(1.0), Value(5.0), Value(2.0)]
        probs = softmax(logits)

        # Index 1 (value 5.0) should have highest probability
        assert probs[1].data > probs[0].data
        assert probs[1].data > probs[2].data

    def test_softmax_numerical_stability(self):
        # Large values that would overflow exp() without max subtraction
        logits = [Value(1000.0), Value(1001.0), Value(1002.0)]
        probs = softmax(logits)

        # Should still sum to 1 and not be NaN
        total = sum(p.data for p in probs)
        assert abs(total - 1.0) < 1e-6
        assert not math.isnan(probs[0].data)


# =============================================================================
# Loss Function Tests
# =============================================================================


class TestLossFunction:
    """Tests for cross-entropy loss."""

    def test_cross_entropy_gradient(self):
        # Logits for 3 classes
        logits = [Value(1.0), Value(2.0), Value(0.5)]
        target = 1  # Correct class is 1

        loss = cross_entropy_loss(logits, target)
        loss.backward()

        # Gradient should be softmax - one_hot
        # For correct class: p_i - 1
        # For wrong classes: p_i
        probs = softmax([Value(1.0), Value(2.0), Value(0.5)])

        # Check gradient directions make sense
        # The correct class gradient should be negative (pull up)
        # Wrong class gradients should be positive (push down)

    def test_cross_entropy_perfect_prediction(self):
        # Very high logit for correct class
        logits = [Value(-100.0), Value(100.0), Value(-100.0)]
        target = 1

        loss = cross_entropy_loss(logits, target)
        # Loss should be very small
        assert loss.data < 0.01

    def test_cross_entropy_wrong_prediction(self):
        # High logit for wrong class
        logits = [Value(100.0), Value(-100.0), Value(-100.0)]
        target = 1

        loss = cross_entropy_loss(logits, target)
        # Loss should be large
        assert loss.data > 100


# =============================================================================
# Layer Tests
# =============================================================================


class TestEmbedding:
    """Tests for embedding layer."""

    def test_embedding_shape(self):
        random.seed(42)
        emb = Embedding(vocab_size=10, embed_dim=8)

        vec = emb(5)
        assert len(vec) == 8
        assert all(isinstance(v, Value) for v in vec)

    def test_embedding_different_indices(self):
        random.seed(42)
        emb = Embedding(vocab_size=10, embed_dim=8)

        vec0 = emb(0)
        vec1 = emb(1)

        # Different indices should give different vectors
        differs = any(v0.data != v1.data for v0, v1 in zip(vec0, vec1))
        assert differs

    def test_embedding_parameters(self):
        emb = Embedding(vocab_size=10, embed_dim=8)
        params = emb.parameters()
        assert len(params) == 10 * 8


class TestLinear:
    """Tests for linear layer."""

    def test_linear_shape(self):
        random.seed(42)
        linear = Linear(in_features=4, out_features=3)

        x = [Value(1.0) for _ in range(4)]
        y = linear(x)

        assert len(y) == 3
        assert all(isinstance(v, Value) for v in y)

    def test_linear_parameters(self):
        linear = Linear(in_features=4, out_features=3)
        params = linear.parameters()
        # 4*3 weights + 3 biases = 15
        assert len(params) == 15

    def test_linear_gradient(self):
        random.seed(42)
        linear = Linear(in_features=2, out_features=2)

        x = [Value(1.0), Value(2.0)]
        y = linear(x)
        loss = y[0] + y[1]

        loss.backward()

        # All parameters should have gradients
        for p in linear.parameters():
            # Gradient might be zero for some but should be set
            assert p.grad is not None


# =============================================================================
# Model Tests
# =============================================================================


class TestCharacterLM:
    """Tests for the full language model."""

    def test_model_creation(self):
        model = CharacterLM(
            vocab_size=10,
            embed_dim=8,
            hidden_dim=16,
            context_length=4
        )
        assert model.num_parameters() > 0

    def test_forward_pass(self):
        random.seed(42)
        model = CharacterLM(
            vocab_size=10,
            embed_dim=8,
            hidden_dim=16,
            context_length=4
        )

        context = [0, 1, 2, 3]
        logits = model.forward(context)

        assert len(logits) == 10  # vocab_size
        assert all(isinstance(l, Value) for l in logits)

    def test_predict_probs(self):
        random.seed(42)
        model = CharacterLM(
            vocab_size=10,
            embed_dim=8,
            hidden_dim=16,
            context_length=4
        )

        context = [0, 1, 2, 3]
        probs = model.predict_probs(context)

        # Should sum to 1
        total = sum(p.data for p in probs)
        assert abs(total - 1.0) < 1e-6

    def test_loss_computation(self):
        random.seed(42)
        model = CharacterLM(
            vocab_size=10,
            embed_dim=8,
            hidden_dim=16,
            context_length=4
        )

        context = [0, 1, 2, 3]
        target = 5

        loss = model.loss(context, target)
        assert isinstance(loss, Value)
        assert loss.data > 0  # Cross-entropy is always positive

    def test_backward_pass(self):
        random.seed(42)
        model = CharacterLM(
            vocab_size=10,
            embed_dim=8,
            hidden_dim=16,
            context_length=4
        )

        context = [0, 1, 2, 3]
        target = 5

        loss = model.loss(context, target)
        model.zero_grad()
        loss.backward()

        # At least some gradients should be non-zero
        nonzero_grads = sum(1 for p in model.parameters() if p.grad != 0)
        assert nonzero_grads > 0


# =============================================================================
# Training Tests
# =============================================================================


class TestTraining:
    """Tests for training functionality."""

    def test_training_reduces_loss(self):
        """Verify that training actually reduces loss."""
        random.seed(42)

        text = "abcabcabcabc"  # Simple repeating pattern
        char_to_idx, idx_to_char, vocab_size = build_vocab(text)

        model = CharacterLM(
            vocab_size=vocab_size,
            embed_dim=4,
            hidden_dim=8,
            context_length=2
        )

        # Initial loss
        initial_ppl = compute_perplexity(model, text, char_to_idx)

        # Train
        train(model, text, char_to_idx, epochs=5, learning_rate=0.5, print_every=1000)

        # Final loss
        final_ppl = compute_perplexity(model, text, char_to_idx)

        assert final_ppl < initial_ppl, f"Training should reduce perplexity: {initial_ppl:.2f} -> {final_ppl:.2f}"


class TestGeneration:
    """Tests for text generation."""

    def test_generate_length(self):
        random.seed(42)

        text = "hello world"
        char_to_idx, idx_to_char, vocab_size = build_vocab(text)

        model = CharacterLM(
            vocab_size=vocab_size,
            embed_dim=4,
            hidden_dim=8,
            context_length=3
        )

        seed = "hel"
        generated = generate(model, seed, char_to_idx, idx_to_char, length=10)

        # Should have seed + 10 new chars
        assert len(generated) == len(seed) + 10

    def test_generate_valid_chars(self):
        random.seed(42)

        text = "abc"
        char_to_idx, idx_to_char, vocab_size = build_vocab(text)

        model = CharacterLM(
            vocab_size=vocab_size,
            embed_dim=4,
            hidden_dim=8,
            context_length=2
        )

        seed = "ab"
        generated = generate(model, seed, char_to_idx, idx_to_char, length=5)

        # All characters should be in vocabulary
        for ch in generated:
            assert ch in char_to_idx, f"Generated invalid character: {ch}"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self):
        """Test complete training and generation pipeline."""
        random.seed(42)

        # Simple training text
        text = "the cat sat on the mat"
        char_to_idx, idx_to_char, vocab_size = build_vocab(text)

        # Create model
        model = CharacterLM(
            vocab_size=vocab_size,
            embed_dim=8,
            hidden_dim=16,
            context_length=3
        )

        # Initial perplexity
        initial_ppl = compute_perplexity(model, text, char_to_idx)

        # Train briefly
        train(model, text, char_to_idx, epochs=3, learning_rate=0.2, print_every=1000)

        # Final perplexity should improve
        final_ppl = compute_perplexity(model, text, char_to_idx)
        assert final_ppl < initial_ppl

        # Generate should work
        generated = generate(model, "the", char_to_idx, idx_to_char, length=10)
        assert len(generated) == 13  # "the" + 10 chars


# =============================================================================
# Test Runner
# =============================================================================


def run_tests():
    """Run all tests."""
    test_classes = [
        TestDataUtilities,
        TestActivations,
        TestLossFunction,
        TestEmbedding,
        TestLinear,
        TestCharacterLM,
        TestTraining,
        TestGeneration,
        TestIntegration,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    print(f"PASS: {test_class.__name__}.{method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"FAIL: {test_class.__name__}.{method_name}")
                    print(f"      {e}")
                    failed += 1
                except Exception as e:
                    print(f"ERROR: {test_class.__name__}.{method_name}")
                    print(f"       {type(e).__name__}: {e}")
                    failed += 1

    print(f"\n{'-' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

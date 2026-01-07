"""
Tests for Stage 5: Attention Mechanisms

Comprehensive tests for all attention components including:
- Core utilities (softmax, activation functions)
- Scaled dot-product attention
- Positional encodings
- Multi-head attention
- Transformer blocks
- Causal language model
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention import (
    softmax, gelu, relu,
    create_causal_mask, scaled_dot_product_attention,
    SinusoidalPositionalEncoding, LearnedPositionalEncoding, RotaryPositionalEncoding,
    LayerNorm, MultiHeadAttention, FeedForward,
    TransformerBlock, CausalTransformer,
    cross_entropy_loss, compute_perplexity,
    visualize_attention_weights,
)


class TestSoftmax:
    """Tests for softmax function."""

    def test_basic(self):
        """Softmax produces valid probabilities."""
        x = np.array([1.0, 2.0, 3.0])
        probs = softmax(x)

        assert np.allclose(probs.sum(), 1.0), "Softmax should sum to 1"
        assert np.all(probs >= 0), "Softmax should be non-negative"

    def test_stability(self):
        """Softmax handles large values without overflow."""
        x = np.array([1000.0, 1001.0, 1002.0])
        probs = softmax(x)

        assert np.allclose(probs.sum(), 1.0), "Should handle large values"
        assert not np.any(np.isnan(probs)), "Should not produce NaN"

    def test_negative_infinity(self):
        """Softmax handles -inf correctly for masking."""
        x = np.array([1.0, float('-inf'), 2.0])
        probs = softmax(x)

        assert probs[1] == 0.0, "-inf should become 0"
        assert np.allclose(probs[0] + probs[2], 1.0), "Other probs should sum to 1"

    def test_all_negative_infinity(self):
        """Softmax handles all -inf values."""
        x = np.array([float('-inf'), float('-inf')])
        probs = softmax(x)

        # Should produce zeros (or very small values), not NaN
        assert not np.any(np.isnan(probs)), "Should not produce NaN"

    def test_2d(self):
        """Softmax works on 2D arrays."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        probs = softmax(x, axis=-1)

        assert np.allclose(probs.sum(axis=-1), [1.0, 1.0]), "Each row should sum to 1"


class TestActivations:
    """Tests for activation functions."""

    def test_relu_positive(self):
        """ReLU passes positive values."""
        x = np.array([1.0, 2.0, 3.0])
        assert np.allclose(relu(x), x)

    def test_relu_negative(self):
        """ReLU zeros negative values."""
        x = np.array([-1.0, -2.0, 0.0])
        assert np.allclose(relu(x), [0.0, 0.0, 0.0])

    def test_gelu_approximates_relu(self):
        """GELU approximates ReLU for large positive values."""
        x = np.array([10.0])
        # GELU(x) ≈ x for large positive x
        assert abs(gelu(x)[0] - 10.0) < 0.1

    def test_gelu_negative(self):
        """GELU is near zero for large negative values."""
        x = np.array([-10.0])
        assert abs(gelu(x)[0]) < 0.1


class TestCausalMask:
    """Tests for causal mask creation."""

    def test_shape(self):
        """Mask has correct shape."""
        mask = create_causal_mask(5)
        assert mask.shape == (5, 5)

    def test_lower_triangular_allowed(self):
        """Lower triangular positions are allowed (0)."""
        mask = create_causal_mask(4)

        for i in range(4):
            for j in range(i + 1):
                assert mask[i, j] == 0, f"Position [{i}, {j}] should be 0"

    def test_upper_triangular_masked(self):
        """Upper triangular positions are masked (-inf)."""
        mask = create_causal_mask(4)

        for i in range(4):
            for j in range(i + 1, 4):
                assert mask[i, j] == float('-inf'), f"Position [{i}, {j}] should be -inf"

    def test_first_position_sees_only_itself(self):
        """First position can only see itself."""
        mask = create_causal_mask(5)
        assert mask[0, 0] == 0
        assert all(mask[0, j] == float('-inf') for j in range(1, 5))


class TestScaledDotProductAttention:
    """Tests for scaled dot-product attention."""

    def test_basic_shape(self):
        """Attention produces correct output shape."""
        n, d_k, d_v = 5, 8, 8
        Q = np.random.randn(n, d_k)
        K = np.random.randn(n, d_k)
        V = np.random.randn(n, d_v)

        output, weights = scaled_dot_product_attention(Q, K, V)

        assert output.shape == (n, d_v), f"Output shape {output.shape}"
        assert weights.shape == (n, n), f"Weights shape {weights.shape}"

    def test_weights_sum_to_one(self):
        """Attention weights sum to 1 per query."""
        n, d_k = 4, 8
        Q = np.random.randn(n, d_k)
        K = np.random.randn(n, d_k)
        V = np.random.randn(n, d_k)

        _, weights = scaled_dot_product_attention(Q, K, V)

        row_sums = weights.sum(axis=-1)
        assert np.allclose(row_sums, 1.0), f"Row sums: {row_sums}"

    def test_causal_mask_applied(self):
        """Causal mask prevents attending to future."""
        n, d_k = 5, 8
        Q = np.random.randn(n, d_k)
        K = np.random.randn(n, d_k)
        V = np.random.randn(n, d_k)

        mask = create_causal_mask(n)
        _, weights = scaled_dot_product_attention(Q, K, V, mask)

        # Check upper triangular is zero
        for i in range(n):
            for j in range(i + 1, n):
                assert weights[i, j] == 0, f"Position [{i}, {j}] should be 0"

    def test_identical_qk_gives_high_self_attention(self):
        """Identical Q and K should give high self-attention."""
        n, d_k = 4, 16
        x = np.random.randn(n, d_k)

        _, weights = scaled_dot_product_attention(x, x, x)

        # Diagonal should have relatively high weights
        for i in range(n):
            assert weights[i, i] > 0.1, f"Self-attention at {i} is {weights[i, i]}"

    def test_scaling_effect(self):
        """Verify scaling by sqrt(d_k)."""
        n = 4
        d_k_small = 1
        d_k_large = 64

        np.random.seed(42)

        # With small d_k, scores are smaller
        Q_small = np.random.randn(n, d_k_small)
        K_small = np.random.randn(n, d_k_small)
        V_small = np.random.randn(n, d_k_small)

        # With large d_k, raw scores would be larger but scaling normalizes
        Q_large = np.random.randn(n, d_k_large)
        K_large = np.random.randn(n, d_k_large)
        V_large = np.random.randn(n, d_k_large)

        _, weights_small = scaled_dot_product_attention(Q_small, K_small, V_small)
        _, weights_large = scaled_dot_product_attention(Q_large, K_large, V_large)

        # Both should have reasonable entropy (not too peaked)
        entropy_small = -np.sum(weights_small * np.log(weights_small + 1e-10), axis=-1).mean()
        entropy_large = -np.sum(weights_large * np.log(weights_large + 1e-10), axis=-1).mean()

        # Scaling should keep entropy in reasonable range
        assert entropy_large > 0.5, f"Large d_k entropy too low: {entropy_large}"


class TestPositionalEncoding:
    """Tests for positional encodings."""

    def test_sinusoidal_shape(self):
        """Sinusoidal encoding has correct shape."""
        pe = SinusoidalPositionalEncoding(100, 64)
        encoding = pe(50)
        assert encoding.shape == (50, 64)

    def test_sinusoidal_unique_positions(self):
        """Each position has unique encoding."""
        pe = SinusoidalPositionalEncoding(100, 64)
        encoding = pe(10)

        for i in range(10):
            for j in range(i + 1, 10):
                diff = np.abs(encoding[i] - encoding[j]).sum()
                assert diff > 0.1, f"Positions {i} and {j} are too similar"

    def test_sinusoidal_bounded(self):
        """Sinusoidal values are bounded in [-1, 1]."""
        pe = SinusoidalPositionalEncoding(100, 64)
        encoding = pe(100)

        assert np.all(encoding >= -1.0), "Values should be >= -1"
        assert np.all(encoding <= 1.0), "Values should be <= 1"

    def test_sinusoidal_first_position(self):
        """First position has known values: sin(0), cos(0), ..."""
        pe = SinusoidalPositionalEncoding(10, 8)
        encoding = pe(1)

        # Even dimensions: sin(0) = 0
        assert np.allclose(encoding[0, 0::2], 0), "Even dims at pos 0 should be 0"
        # Odd dimensions: cos(0) = 1
        assert np.allclose(encoding[0, 1::2], 1), "Odd dims at pos 0 should be 1"

    def test_learned_shape(self):
        """Learned encoding has correct shape."""
        pe = LearnedPositionalEncoding(100, 64)
        encoding = pe(50)
        assert encoding.shape == (50, 64)

    def test_learned_parameters(self):
        """Learned encoding has learnable parameters."""
        pe = LearnedPositionalEncoding(100, 64)
        params = pe.parameters()
        assert len(params) == 1
        assert params[0].shape == (100, 64)

    def test_rope_shape(self):
        """RoPE produces correct shape."""
        rope = RotaryPositionalEncoding(64)
        x = np.random.randn(10, 64)
        positions = np.arange(10)

        rotated = rope(x, positions)
        assert rotated.shape == x.shape


class TestLayerNorm:
    """Tests for layer normalization."""

    def test_normalized_output(self):
        """Output has zero mean and unit variance."""
        ln = LayerNorm(64)
        x = np.random.randn(10, 64) * 5 + 3  # Non-zero mean, non-unit variance

        output = ln(x)

        # Check each row is normalized
        for i in range(10):
            mean = output[i].mean()
            var = output[i].var()
            assert abs(mean) < 0.1, f"Mean should be near 0, got {mean}"
            assert abs(var - 1.0) < 0.1, f"Variance should be near 1, got {var}"

    def test_learnable_parameters(self):
        """LayerNorm has gamma and beta parameters."""
        ln = LayerNorm(64)
        params = ln.parameters()

        assert len(params) == 2
        assert params[0].shape == (64,)  # gamma
        assert params[1].shape == (64,)  # beta

    def test_gamma_beta_effect(self):
        """Gamma scales and beta shifts output."""
        ln = LayerNorm(4)
        ln.gamma = np.array([2.0, 2.0, 2.0, 2.0])
        ln.beta = np.array([1.0, 1.0, 1.0, 1.0])

        x = np.array([[0.0, 0.0, 0.0, 0.0]])  # Will normalize to zeros
        output = ln(x)

        # After normalization (all zeros), gamma * 0 + beta = beta
        # But since all inputs are same, after normalization they become 0
        # Then gamma * 0 + beta = 1
        assert np.allclose(output, 1.0), f"Output should be 1.0, got {output}"


class TestMultiHeadAttention:
    """Tests for multi-head attention."""

    def test_output_shape(self):
        """Output has same shape as input."""
        mha = MultiHeadAttention(64, 8)
        x = np.random.randn(10, 64)

        output = mha(x)
        assert output.shape == x.shape

    def test_batched_input(self):
        """Works with batched input."""
        mha = MultiHeadAttention(64, 8)
        x = np.random.randn(4, 10, 64)  # batch=4, seq=10, d=64

        output = mha(x)
        assert output.shape == x.shape

    def test_attention_weights_stored(self):
        """Attention weights are stored for visualization."""
        mha = MultiHeadAttention(64, 8)
        x = np.random.randn(10, 64)

        _ = mha(x)

        assert mha.attention_weights is not None
        assert mha.attention_weights.shape == (1, 8, 10, 10)  # [batch, heads, seq, seq]

    def test_causal_mask(self):
        """Causal mask prevents future attention."""
        mha = MultiHeadAttention(64, 8)
        x = np.random.randn(5, 64)
        mask = create_causal_mask(5)

        _ = mha(x, mask)

        # Check all heads have causal pattern
        weights = mha.attention_weights[0]  # [heads, seq, seq]
        for h in range(8):
            for i in range(5):
                for j in range(i + 1, 5):
                    assert weights[h, i, j] == 0, f"Head {h}, pos [{i},{j}] should be 0"

    def test_parameters(self):
        """Has correct number of parameters."""
        mha = MultiHeadAttention(64, 8)
        params = mha.parameters()

        assert len(params) == 2  # W_qkv and W_o
        assert params[0].shape == (64, 192)  # 64 × 3*64
        assert params[1].shape == (64, 64)


class TestFeedForward:
    """Tests for feed-forward network."""

    def test_output_shape(self):
        """Output has same shape as input."""
        ff = FeedForward(64)
        x = np.random.randn(10, 64)

        output = ff(x)
        assert output.shape == x.shape

    def test_hidden_dimension(self):
        """Default hidden dimension is 4x input."""
        ff = FeedForward(64)
        assert ff.d_ff == 256

    def test_custom_hidden_dimension(self):
        """Can specify custom hidden dimension."""
        ff = FeedForward(64, d_ff=128)
        assert ff.d_ff == 128

    def test_parameters(self):
        """Has correct parameters."""
        ff = FeedForward(64, d_ff=256)
        params = ff.parameters()

        assert len(params) == 4  # W1, b1, W2, b2
        assert params[0].shape == (64, 256)  # W1
        assert params[1].shape == (256,)     # b1
        assert params[2].shape == (256, 64)  # W2
        assert params[3].shape == (64,)      # b2


class TestTransformerBlock:
    """Tests for Transformer block."""

    def test_output_shape(self):
        """Output has same shape as input."""
        block = TransformerBlock(64, 8)
        x = np.random.randn(10, 64)

        output = block(x)
        assert output.shape == x.shape

    def test_with_mask(self):
        """Works with causal mask."""
        block = TransformerBlock(64, 8)
        x = np.random.randn(10, 64)
        mask = create_causal_mask(10)

        output = block(x, mask)
        assert output.shape == x.shape

    def test_residual_connection(self):
        """Residual connections are present."""
        block = TransformerBlock(64, 8)

        # With zero weights, output should be close to input (via residual)
        block.attention.W_qkv = np.zeros_like(block.attention.W_qkv)
        block.attention.W_o = np.zeros_like(block.attention.W_o)
        block.ffn.W1 = np.zeros_like(block.ffn.W1)
        block.ffn.W2 = np.zeros_like(block.ffn.W2)

        x = np.random.randn(5, 64)
        output = block(x)

        # With zero weights, should preserve input through residuals
        # (after layer norm transformations)
        assert not np.allclose(output, 0), "Should not be all zeros"

    def test_parameters(self):
        """Has all parameters."""
        block = TransformerBlock(64, 8)
        params = block.parameters()

        # attention (2) + ffn (4) + norm1 (2) + norm2 (2) = 10
        assert len(params) == 10


class TestCausalTransformer:
    """Tests for complete causal Transformer."""

    def test_forward_shape(self):
        """Forward produces correct logits shape."""
        model = CausalTransformer(
            vocab_size=100,
            d_model=32,
            n_heads=4,
            n_layers=2,
            max_len=64
        )

        tokens = np.array([1, 5, 10, 15, 20])
        logits = model(tokens)

        assert logits.shape == (5, 100), f"Expected (5, 100), got {logits.shape}"

    def test_batched_forward(self):
        """Works with batched input."""
        model = CausalTransformer(
            vocab_size=100,
            d_model=32,
            n_heads=4,
            n_layers=2
        )

        tokens = np.array([[1, 2, 3], [4, 5, 6]])  # batch=2, seq=3
        logits = model(tokens)

        assert logits.shape == (2, 3, 100)

    def test_generation(self):
        """Can generate tokens."""
        np.random.seed(42)
        model = CausalTransformer(
            vocab_size=50,
            d_model=32,
            n_heads=4,
            n_layers=2
        )

        prompt = np.array([1, 2, 3])
        generated = model.generate(prompt, max_new_tokens=5, temperature=1.0)

        assert len(generated) == 8, f"Expected 8 tokens, got {len(generated)}"
        assert generated[:3] == [1, 2, 3], "Prompt should be preserved"

    def test_generation_with_top_k(self):
        """Generation with top-k sampling."""
        np.random.seed(42)
        model = CausalTransformer(
            vocab_size=50,
            d_model=32,
            n_heads=4,
            n_layers=2
        )

        prompt = np.array([1, 2, 3])
        generated = model.generate(prompt, max_new_tokens=5, top_k=10)

        assert len(generated) == 8

    def test_parameter_count(self):
        """Can count parameters."""
        model = CausalTransformer(
            vocab_size=100,
            d_model=32,
            n_heads=4,
            n_layers=2
        )

        count = model.count_parameters()
        assert count > 0, "Should have parameters"

    def test_learned_positional_encoding(self):
        """Can use learned positional encoding."""
        model = CausalTransformer(
            vocab_size=100,
            d_model=32,
            n_heads=4,
            n_layers=2,
            pos_encoding='learned'
        )

        tokens = np.array([1, 2, 3])
        logits = model(tokens)

        assert logits.shape == (3, 100)


class TestLossAndPerplexity:
    """Tests for loss and perplexity functions."""

    def test_cross_entropy_basic(self):
        """Cross entropy is positive."""
        logits = np.random.randn(10, 50)
        targets = np.random.randint(0, 50, size=10)

        loss = cross_entropy_loss(logits, targets)
        assert loss > 0, "Loss should be positive"

    def test_cross_entropy_correct_prediction(self):
        """Loss is low when predictions are correct."""
        # Create logits where correct class has high score
        logits = np.zeros((5, 10))
        targets = np.array([0, 1, 2, 3, 4])

        for i, t in enumerate(targets):
            logits[i, t] = 10.0  # High score for correct class

        loss = cross_entropy_loss(logits, targets)
        assert loss < 0.1, f"Loss should be low, got {loss}"

    def test_perplexity(self):
        """Perplexity is exp(loss)."""
        loss = 2.0
        ppl = compute_perplexity(loss)
        assert abs(ppl - np.exp(2.0)) < 0.001


class TestVisualization:
    """Tests for visualization utilities."""

    def test_visualize_attention(self):
        """Can generate attention visualization."""
        weights = np.random.rand(4, 4)
        weights = weights / weights.sum(axis=-1, keepdims=True)

        viz = visualize_attention_weights(weights, tokens=["a", "b", "c", "d"])

        assert isinstance(viz, str)
        assert len(viz) > 0


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_training_loop_simulation(self):
        """Simulate a training loop."""
        np.random.seed(42)

        model = CausalTransformer(
            vocab_size=50,
            d_model=32,
            n_heads=4,
            n_layers=2
        )

        # Create fake training data
        tokens = np.random.randint(0, 50, size=20)
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]

        # Forward pass
        logits = model(input_tokens)

        # Compute loss
        loss = cross_entropy_loss(logits, target_tokens)

        assert loss > 0, "Loss should be positive"
        assert not np.isnan(loss), "Loss should not be NaN"

    def test_model_consistency(self):
        """Same input should give same output."""
        np.random.seed(42)

        model = CausalTransformer(
            vocab_size=50,
            d_model=32,
            n_heads=4,
            n_layers=2
        )

        tokens = np.array([1, 2, 3, 4, 5])

        logits1 = model(tokens)
        logits2 = model(tokens)

        assert np.allclose(logits1, logits2), "Should be deterministic"


def run_tests():
    """Run all tests."""
    test_classes = [
        TestSoftmax,
        TestActivations,
        TestCausalMask,
        TestScaledDotProductAttention,
        TestPositionalEncoding,
        TestLayerNorm,
        TestMultiHeadAttention,
        TestFeedForward,
        TestTransformerBlock,
        TestCausalTransformer,
        TestLossAndPerplexity,
        TestVisualization,
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

"""
Tests for Stage 6: Complete Transformer Implementation

These tests verify the full Transformer implementation including
all components: normalization, positional encoding, attention,
feed-forward networks, tokenization, and generation.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer import (
    # Utilities
    softmax, silu, gelu, create_causal_mask,
    # Normalization
    RMSNorm, LayerNorm,
    # Positional encoding
    apply_rope, SinusoidalPositionalEncoding,
    # Attention and FFN
    MultiHeadAttention, SwiGLUFFN, GELUMLPFFN,
    # Transformer components
    TransformerBlock, Transformer,
    # Tokenization
    CharTokenizer, BPETokenizer,
    # Training utilities
    cross_entropy_loss, get_lr, compute_perplexity
)


# =============================================================================
# Utility Functions Tests
# =============================================================================

class TestUtilities:
    """Tests for utility functions."""

    def test_softmax_sums_to_one(self):
        """Softmax outputs should sum to 1."""
        x = np.random.randn(5, 10)
        result = softmax(x, axis=-1)
        assert np.allclose(result.sum(axis=-1), 1.0, rtol=1e-5), "Softmax should sum to 1"

    def test_softmax_positive(self):
        """Softmax outputs should be positive."""
        x = np.random.randn(5, 10)
        result = softmax(x, axis=-1)
        assert np.all(result >= 0), "Softmax should be non-negative"

    def test_softmax_with_mask(self):
        """Softmax should handle -inf values correctly."""
        x = np.array([1.0, 2.0, float('-inf'), 3.0])
        result = softmax(x)
        assert result[2] < 1e-9, "Masked position should be ~0"
        assert np.allclose(result.sum(), 1.0, rtol=1e-5), "Softmax should sum to 1"

    def test_silu_basic(self):
        """SiLU should produce expected output."""
        x = np.array([0.0, 1.0, -1.0])
        result = silu(x)
        # silu(0) = 0, silu(1) ≈ 0.731, silu(-1) ≈ -0.269
        assert result[0] == 0.0, "silu(0) should be 0"
        assert 0.7 < result[1] < 0.8, "silu(1) should be ~0.73"
        assert -0.3 < result[2] < -0.2, "silu(-1) should be ~-0.27"

    def test_gelu_basic(self):
        """GELU should produce expected output."""
        x = np.array([0.0, 1.0, -1.0])
        result = gelu(x)
        assert result[0] == 0.0, "gelu(0) should be 0"
        assert result[1] > 0, "gelu should be positive for positive input"

    def test_causal_mask_shape(self):
        """Causal mask should have correct shape."""
        mask = create_causal_mask(5)
        assert mask.shape == (5, 5), "Mask shape should be (5, 5)"

    def test_causal_mask_values(self):
        """Causal mask should have -inf above diagonal, 0 on and below."""
        mask = create_causal_mask(4)
        assert mask[0, 0] == 0, "Diagonal should be 0"
        assert mask[1, 0] == 0, "Lower triangle should be 0"
        assert mask[2, 1] == 0, "Lower triangle should be 0"
        assert mask[0, 1] == float('-inf'), "Upper triangle should be -inf"
        assert mask[1, 2] == float('-inf'), "Upper triangle should be -inf"


# =============================================================================
# Normalization Tests
# =============================================================================

class TestNormalization:
    """Tests for normalization layers."""

    def test_rmsnorm_output_shape(self):
        """RMSNorm should preserve input shape."""
        norm = RMSNorm(64)
        x = np.random.randn(2, 10, 64)
        y = norm(x)
        assert y.shape == x.shape, "RMSNorm should preserve shape"

    def test_rmsnorm_magnitude(self):
        """RMSNorm should normalize RMS to approximately 1."""
        norm = RMSNorm(64)
        x = np.random.randn(2, 10, 64) * 5  # Scale up
        y = norm(x)
        rms = np.sqrt(np.mean(y ** 2, axis=-1))
        assert np.allclose(rms, 1.0, rtol=0.1), "RMS should be ~1"

    def test_rmsnorm_parameters(self):
        """RMSNorm should have learnable weight."""
        norm = RMSNorm(64)
        params = norm.parameters()
        assert len(params) == 1, "RMSNorm should have 1 parameter"
        assert params[0].shape == (64,), "Weight should have correct shape"

    def test_layernorm_output_shape(self):
        """LayerNorm should preserve input shape."""
        norm = LayerNorm(64)
        x = np.random.randn(2, 10, 64)
        y = norm(x)
        assert y.shape == x.shape, "LayerNorm should preserve shape"

    def test_layernorm_zero_mean(self):
        """LayerNorm should produce approximately zero mean."""
        norm = LayerNorm(64)
        x = np.random.randn(2, 10, 64) + 10  # Add offset
        y = norm(x)
        means = y.mean(axis=-1)
        assert np.allclose(means, 0.0, atol=1e-5), "Mean should be ~0"

    def test_layernorm_unit_variance(self):
        """LayerNorm should produce approximately unit variance."""
        norm = LayerNorm(64)
        x = np.random.randn(2, 10, 64)
        y = norm(x)
        variances = y.var(axis=-1)
        assert np.allclose(variances, 1.0, rtol=0.1), "Variance should be ~1"


# =============================================================================
# Positional Encoding Tests
# =============================================================================

class TestPositionalEncoding:
    """Tests for positional encodings."""

    def test_rope_output_shape(self):
        """RoPE should preserve input shape."""
        x = np.random.randn(2, 10, 4, 32)
        positions = np.arange(10)
        y = apply_rope(x, positions)
        assert y.shape == x.shape, "RoPE should preserve shape"

    def test_rope_position_dependent(self):
        """RoPE should produce different outputs for different positions."""
        d_k = 32
        x = np.ones((1, 5, 1, d_k))
        positions = np.arange(5)
        y = apply_rope(x, positions)
        assert not np.allclose(y[0, 0], y[0, 1]), "Different positions should differ"

    def test_sinusoidal_pe_shape(self):
        """Sinusoidal PE should have correct shape."""
        pe = SinusoidalPositionalEncoding(100, 64)
        encoding = pe(10)
        assert encoding.shape == (10, 64), "PE shape should be (10, 64)"

    def test_sinusoidal_pe_bounded(self):
        """Sinusoidal PE values should be in [-1, 1]."""
        pe = SinusoidalPositionalEncoding(100, 64)
        encoding = pe(50)
        assert np.all(encoding >= -1), "PE should be >= -1"
        assert np.all(encoding <= 1), "PE should be <= 1"

    def test_sinusoidal_pe_different_positions(self):
        """Different positions should have different encodings."""
        pe = SinusoidalPositionalEncoding(100, 64)
        encoding = pe(10)
        assert not np.allclose(encoding[0], encoding[1]), "Positions should differ"


# =============================================================================
# Attention Tests
# =============================================================================

class TestMultiHeadAttention:
    """Tests for multi-head attention."""

    def test_mha_output_shape(self):
        """MHA output should match input shape."""
        mha = MultiHeadAttention(d_model=64, n_heads=4)
        x = np.random.randn(2, 10, 64)
        y = mha(x)
        assert y.shape == x.shape, "MHA should preserve shape"

    def test_mha_with_mask(self):
        """MHA should respect causal mask."""
        np.random.seed(42)
        mha = MultiHeadAttention(d_model=64, n_heads=4)
        x = np.random.randn(1, 5, 64)
        mask = create_causal_mask(5)
        y = mha(x, mask=mask)
        assert y.shape == x.shape, "Masked MHA should preserve shape"

    def test_mha_with_rope(self):
        """MHA should work with RoPE positions."""
        mha = MultiHeadAttention(d_model=64, n_heads=4)
        x = np.random.randn(2, 10, 64)
        positions = np.arange(10)
        y = mha(x, positions=positions)
        assert y.shape == x.shape, "MHA with RoPE should preserve shape"

    def test_gqa_output_shape(self):
        """GQA should produce correct output shape."""
        mha = MultiHeadAttention(d_model=64, n_heads=8, n_kv_heads=2)
        x = np.random.randn(2, 10, 64)
        y = mha(x)
        assert y.shape == x.shape, "GQA should preserve shape"

    def test_mha_stores_attention_weights(self):
        """MHA should store attention weights."""
        mha = MultiHeadAttention(d_model=64, n_heads=4)
        x = np.random.randn(2, 10, 64)
        mha(x)
        assert mha.attention_weights is not None, "Weights should be stored"
        assert mha.attention_weights.shape == (2, 4, 10, 10), "Weight shape incorrect"

    def test_mha_attention_weights_sum_to_one(self):
        """Attention weights should sum to 1 over keys."""
        mha = MultiHeadAttention(d_model=64, n_heads=4)
        x = np.random.randn(2, 10, 64)
        mha(x)
        sums = mha.attention_weights.sum(axis=-1)
        assert np.allclose(sums, 1.0, rtol=1e-5), "Weights should sum to 1"

    def test_mha_parameters(self):
        """MHA should return 4 parameter matrices."""
        mha = MultiHeadAttention(d_model=64, n_heads=4)
        params = mha.parameters()
        assert len(params) == 4, "MHA should have 4 parameters"


# =============================================================================
# Feed-Forward Network Tests
# =============================================================================

class TestFFN:
    """Tests for feed-forward networks."""

    def test_swiglu_output_shape(self):
        """SwiGLU FFN should preserve input shape."""
        ffn = SwiGLUFFN(d_model=64)
        x = np.random.randn(2, 10, 64)
        y = ffn(x)
        assert y.shape == x.shape, "SwiGLU should preserve shape"

    def test_swiglu_parameters(self):
        """SwiGLU should have 3 weight matrices."""
        ffn = SwiGLUFFN(d_model=64)
        params = ffn.parameters()
        assert len(params) == 3, "SwiGLU should have 3 parameters"

    def test_swiglu_custom_dim(self):
        """SwiGLU should support custom hidden dimension."""
        ffn = SwiGLUFFN(d_model=64, d_ff=128)
        assert ffn.d_ff == 128, "Custom d_ff should be set"

    def test_gelu_ffn_output_shape(self):
        """GELU FFN should preserve input shape."""
        ffn = GELUMLPFFN(d_model=64)
        x = np.random.randn(2, 10, 64)
        y = ffn(x)
        assert y.shape == x.shape, "GELU FFN should preserve shape"

    def test_gelu_ffn_parameters(self):
        """GELU FFN should have 4 parameters (2 weights, 2 biases)."""
        ffn = GELUMLPFFN(d_model=64)
        params = ffn.parameters()
        assert len(params) == 4, "GELU FFN should have 4 parameters"

    def test_gelu_ffn_default_expansion(self):
        """GELU FFN should default to 4x expansion."""
        ffn = GELUMLPFFN(d_model=64)
        assert ffn.d_ff == 256, "d_ff should be 4 * d_model"


# =============================================================================
# Transformer Block Tests
# =============================================================================

class TestTransformerBlock:
    """Tests for transformer block."""

    def test_block_output_shape(self):
        """Block should preserve input shape."""
        block = TransformerBlock(d_model=64, n_heads=4)
        x = np.random.randn(2, 10, 64)
        y = block(x)
        assert y.shape == x.shape, "Block should preserve shape"

    def test_block_with_mask(self):
        """Block should work with causal mask."""
        block = TransformerBlock(d_model=64, n_heads=4)
        x = np.random.randn(2, 10, 64)
        mask = create_causal_mask(10)
        y = block(x, mask=mask)
        assert y.shape == x.shape, "Masked block should preserve shape"

    def test_block_with_positions(self):
        """Block should work with RoPE positions."""
        block = TransformerBlock(d_model=64, n_heads=4)
        x = np.random.randn(2, 10, 64)
        positions = np.arange(10)
        y = block(x, positions=positions)
        assert y.shape == x.shape, "Block with positions should preserve shape"

    def test_block_swiglu_vs_gelu(self):
        """Block should support both SwiGLU and GELU."""
        np.random.seed(42)
        x = np.random.randn(2, 10, 64)

        block_swiglu = TransformerBlock(d_model=64, n_heads=4, use_swiglu=True)
        block_gelu = TransformerBlock(d_model=64, n_heads=4, use_swiglu=False)

        y_swiglu = block_swiglu(x)
        y_gelu = block_gelu(x)

        assert y_swiglu.shape == x.shape, "SwiGLU block should work"
        assert y_gelu.shape == x.shape, "GELU block should work"

    def test_block_rmsnorm_vs_layernorm(self):
        """Block should support both RMSNorm and LayerNorm."""
        np.random.seed(42)
        x = np.random.randn(2, 10, 64)

        block_rms = TransformerBlock(d_model=64, n_heads=4, use_rmsnorm=True)
        block_ln = TransformerBlock(d_model=64, n_heads=4, use_rmsnorm=False)

        y_rms = block_rms(x)
        y_ln = block_ln(x)

        assert y_rms.shape == x.shape, "RMSNorm block should work"
        assert y_ln.shape == x.shape, "LayerNorm block should work"

    def test_block_parameters(self):
        """Block should return all parameters."""
        block = TransformerBlock(d_model=64, n_heads=4)
        params = block.parameters()
        assert len(params) >= 8, "Block should have many parameters"


# =============================================================================
# Complete Transformer Tests
# =============================================================================

class TestTransformer:
    """Tests for complete Transformer model."""

    def test_transformer_forward_1d(self):
        """Transformer should handle 1D input."""
        model = Transformer(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        tokens = np.array([1, 2, 3, 4, 5])
        logits = model(tokens)
        assert logits.shape == (5, 100), "1D output shape incorrect"

    def test_transformer_forward_2d(self):
        """Transformer should handle 2D batched input."""
        model = Transformer(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        tokens = np.array([[1, 2, 3], [4, 5, 6]])
        logits = model(tokens)
        assert logits.shape == (2, 3, 100), "2D output shape incorrect"

    def test_transformer_generation(self):
        """Transformer should generate tokens."""
        np.random.seed(42)
        model = Transformer(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        prompt = [1, 2, 3]
        generated = model.generate(prompt, max_new_tokens=5)
        assert len(generated) == 8, "Should generate 5 new tokens"
        assert generated[:3] == prompt, "Prompt should be preserved"

    def test_transformer_generation_with_temperature(self):
        """Generation should respect temperature."""
        np.random.seed(42)
        model = Transformer(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        prompt = [1, 2, 3]
        gen_low = model.generate(prompt, max_new_tokens=5, temperature=0.1)
        assert len(gen_low) == 8, "Should generate with low temperature"

    def test_transformer_generation_with_top_k(self):
        """Generation should support top-k sampling."""
        np.random.seed(42)
        model = Transformer(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        prompt = [1, 2, 3]
        generated = model.generate(prompt, max_new_tokens=5, top_k=10)
        assert len(generated) == 8, "Should generate with top_k"

    def test_transformer_generation_with_top_p(self):
        """Generation should support nucleus sampling."""
        np.random.seed(42)
        model = Transformer(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        prompt = [1, 2, 3]
        generated = model.generate(prompt, max_new_tokens=5, top_p=0.9)
        assert len(generated) == 8, "Should generate with top_p"

    def test_transformer_gqa(self):
        """Transformer should support GQA."""
        model = Transformer(
            vocab_size=100, d_model=64, n_heads=8, n_kv_heads=2, n_layers=2
        )
        tokens = np.array([1, 2, 3, 4, 5])
        logits = model(tokens)
        assert logits.shape == (5, 100), "GQA output shape incorrect"

    def test_transformer_tied_embeddings(self):
        """Tied embeddings should share weights."""
        model = Transformer(vocab_size=100, d_model=64, n_layers=2, tie_embeddings=True)
        assert model.output_proj is None, "Tied embeddings should not have output_proj"

    def test_transformer_untied_embeddings(self):
        """Untied embeddings should have separate output projection."""
        model = Transformer(vocab_size=100, d_model=64, n_layers=2, tie_embeddings=False)
        assert model.output_proj is not None, "Untied should have output_proj"
        assert model.output_proj.shape == (64, 100), "Output proj shape incorrect"

    def test_transformer_count_parameters(self):
        """Should count all parameters."""
        model = Transformer(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        param_count = model.count_parameters()
        assert param_count > 0, "Should have parameters"
        assert param_count >= 100 * 64, "Should have at least embedding params"


# =============================================================================
# Tokenization Tests
# =============================================================================

class TestCharTokenizer:
    """Tests for character tokenizer."""

    def test_train_builds_vocab(self):
        """Training should build vocabulary."""
        tokenizer = CharTokenizer()
        tokenizer.train("hello world")
        assert tokenizer.vocab_size > 0, "Should have vocabulary"

    def test_encode_decode_roundtrip(self):
        """Encode then decode should recover original."""
        tokenizer = CharTokenizer().train("hello world")
        text = "hello"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, "Roundtrip should preserve text"

    def test_encode_produces_integers(self):
        """Encode should produce list of integers."""
        tokenizer = CharTokenizer().train("abc")
        encoded = tokenizer.encode("abc")
        assert all(isinstance(i, int) for i in encoded), "Should be integers"
        assert len(encoded) == 3, "Should have 3 tokens"

    def test_unknown_char_handling(self):
        """Unknown characters should be handled gracefully."""
        tokenizer = CharTokenizer().train("abc")
        encoded = tokenizer.encode("xyz")  # Not in training
        assert len(encoded) == 3, "Should still encode"


class TestBPETokenizer:
    """Tests for BPE tokenizer."""

    def test_train_creates_merges(self):
        """Training should create merge rules."""
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(["hello world hello world"])
        assert len(tokenizer.merges) > 0, "Should create merges"

    def test_vocab_size_limit(self):
        """Vocabulary should respect size limit."""
        tokenizer = BPETokenizer(vocab_size=280)
        tokenizer.train(["hello world " * 100])
        assert tokenizer.vocab_size <= 280, "Vocab should be limited"

    def test_encode_decode_roundtrip(self):
        """Encode then decode should recover original."""
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(["hello world"])
        text = "hello"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, "Roundtrip should preserve text"

    def test_bpe_compression(self):
        """BPE should compress repeated patterns."""
        tokenizer = BPETokenizer(vocab_size=300)
        text = "ab" * 50  # Repeated pattern
        tokenizer.train([text])
        encoded = tokenizer.encode(text)
        assert len(encoded) < len(text), "BPE should compress"


# =============================================================================
# Training Utilities Tests
# =============================================================================

class TestTrainingUtils:
    """Tests for training utilities."""

    def test_cross_entropy_loss_basic(self):
        """Cross-entropy loss should work for basic case."""
        logits = np.array([[1.0, 2.0, 3.0]])
        targets = np.array([2])  # Correct class is index 2
        loss = cross_entropy_loss(logits, targets)
        assert loss > 0, "Loss should be positive"
        assert loss < 10, "Loss should be reasonable"

    def test_cross_entropy_loss_3d(self):
        """Cross-entropy should handle 3D logits."""
        logits = np.random.randn(2, 5, 10)
        targets = np.random.randint(0, 10, size=(2, 5))
        loss = cross_entropy_loss(logits, targets)
        assert loss > 0, "3D loss should be positive"

    def test_cross_entropy_loss_perfect_prediction(self):
        """Perfect predictions should have low loss."""
        logits = np.array([[-100.0, -100.0, 100.0]])
        targets = np.array([2])
        loss = cross_entropy_loss(logits, targets)
        assert loss < 0.01, "Perfect prediction should have low loss"

    def test_get_lr_warmup(self):
        """Learning rate should increase during warmup."""
        lr1 = get_lr(0, warmup_steps=100, max_steps=1000, max_lr=1e-3)
        lr2 = get_lr(50, warmup_steps=100, max_steps=1000, max_lr=1e-3)
        lr3 = get_lr(100, warmup_steps=100, max_steps=1000, max_lr=1e-3)
        assert lr1 < lr2 < lr3, "LR should increase during warmup"

    def test_get_lr_decay(self):
        """Learning rate should decay after warmup."""
        lr1 = get_lr(100, warmup_steps=100, max_steps=1000, max_lr=1e-3)
        lr2 = get_lr(500, warmup_steps=100, max_steps=1000, max_lr=1e-3)
        lr3 = get_lr(999, warmup_steps=100, max_steps=1000, max_lr=1e-3)
        assert lr1 > lr2 > lr3, "LR should decay after warmup"

    def test_get_lr_at_warmup_end(self):
        """LR should be max_lr at end of warmup."""
        lr = get_lr(100, warmup_steps=100, max_steps=1000, max_lr=1e-3)
        assert np.allclose(lr, 1e-3, rtol=1e-5), "LR should be max_lr at warmup end"

    def test_compute_perplexity(self):
        """Perplexity should be exp(loss)."""
        loss = 2.0
        ppl = compute_perplexity(loss)
        assert np.allclose(ppl, np.exp(2.0), rtol=1e-5), "Perplexity = exp(loss)"

    def test_perplexity_monotonic(self):
        """Higher loss should give higher perplexity."""
        ppl1 = compute_perplexity(1.0)
        ppl2 = compute_perplexity(2.0)
        ppl3 = compute_perplexity(3.0)
        assert ppl1 < ppl2 < ppl3, "Perplexity should increase with loss"


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete pipeline."""

    def test_full_training_step(self):
        """Test a complete forward pass with loss computation."""
        np.random.seed(42)

        tokenizer = CharTokenizer().train("hello world")
        model = Transformer(
            vocab_size=tokenizer.vocab_size,
            d_model=32, n_heads=2, n_layers=1
        )

        text = "hello"
        tokens = tokenizer.encode(text)
        x = np.array(tokens[:-1])  # Input
        y = np.array(tokens[1:])   # Target

        logits = model(x)
        loss = cross_entropy_loss(logits, y)

        assert loss > 0, "Loss should be positive"
        assert not np.isnan(loss), "Loss should not be NaN"

    def test_generation_produces_valid_tokens(self):
        """Generated tokens should be valid vocabulary indices."""
        np.random.seed(42)
        model = Transformer(vocab_size=50, d_model=32, n_heads=2, n_layers=1)
        generated = model.generate([1, 2, 3], max_new_tokens=10)

        for token in generated:
            assert 0 <= token < 50, "Tokens should be valid indices"

    def test_model_determinism(self):
        """Same seed should produce same output."""
        np.random.seed(42)
        model1 = Transformer(vocab_size=50, d_model=32, n_heads=2, n_layers=1)
        tokens = np.array([1, 2, 3])
        out1 = model1(tokens)

        np.random.seed(42)
        model2 = Transformer(vocab_size=50, d_model=32, n_heads=2, n_layers=1)
        out2 = model2(tokens)

        assert np.allclose(out1, out2), "Same seed should give same output"

    def test_different_architectures(self):
        """Different architecture choices should all work."""
        configs = [
            {'use_swiglu': True, 'use_rmsnorm': True},
            {'use_swiglu': True, 'use_rmsnorm': False},
            {'use_swiglu': False, 'use_rmsnorm': True},
            {'use_swiglu': False, 'use_rmsnorm': False},
        ]

        for config in configs:
            model = Transformer(
                vocab_size=50, d_model=32, n_heads=2, n_layers=1, **config
            )
            tokens = np.array([1, 2, 3])
            logits = model(tokens)
            assert logits.shape == (3, 50), f"Config {config} should work"
            assert not np.any(np.isnan(logits)), f"Config {config} should not produce NaN"


# =============================================================================
# Test Runner
# =============================================================================

def run_tests():
    """Run all tests."""
    test_classes = [
        TestUtilities,
        TestNormalization,
        TestPositionalEncoding,
        TestMultiHeadAttention,
        TestFFN,
        TestTransformerBlock,
        TestTransformer,
        TestCharTokenizer,
        TestBPETokenizer,
        TestTrainingUtils,
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

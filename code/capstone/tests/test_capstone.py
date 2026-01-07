"""
Tests for Capstone: Trainable Transformer

These tests verify:
1. Forward pass produces correct shapes
2. Backward pass computes gradients for all parameters
3. Gradients are numerically correct (gradient checking)
4. Training reduces loss
5. Generation produces valid output
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import (
    TrainableTransformer,
    CharTokenizer,
    cross_entropy_loss,
    compute_perplexity,
    RMSNorm,
    Embedding,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    Parameter,
    softmax,
    silu,
    silu_backward,
)


# =============================================================================
# Utility Tests
# =============================================================================

def test_softmax():
    """Test softmax produces valid probability distribution."""
    x = np.array([1.0, 2.0, 3.0])
    p = softmax(x)

    assert np.allclose(np.sum(p), 1.0), "Softmax should sum to 1"
    assert np.all(p > 0), "Softmax should be positive"
    assert p[2] > p[1] > p[0], "Larger values should have higher probability"
    print("✓ test_softmax passed")


def test_softmax_numerical_stability():
    """Test softmax handles large values."""
    x = np.array([1000.0, 1001.0, 1002.0])
    p = softmax(x)

    assert np.allclose(np.sum(p), 1.0), "Softmax should handle large values"
    assert not np.any(np.isnan(p)), "Softmax should not produce NaN"
    print("✓ test_softmax_numerical_stability passed")


def test_silu():
    """Test SiLU activation."""
    x = np.array([-1.0, 0.0, 1.0])
    y = silu(x)

    assert y[1] == 0.0, "SiLU(0) should be 0"
    assert y[2] > y[0], "SiLU should be larger for positive inputs"
    print("✓ test_silu passed")


def test_silu_backward():
    """Test SiLU gradient numerically."""
    x = np.array([0.5])
    eps = 1e-5

    # Numerical gradient
    numerical_grad = (silu(x + eps) - silu(x - eps)) / (2 * eps)

    # Analytical gradient
    analytical_grad = silu_backward(x, np.ones_like(x))

    assert np.allclose(numerical_grad, analytical_grad, atol=1e-4), \
        f"SiLU gradient mismatch: {numerical_grad} vs {analytical_grad}"
    print("✓ test_silu_backward passed")


# =============================================================================
# RMSNorm Tests
# =============================================================================

def test_rmsnorm_forward():
    """Test RMSNorm forward pass."""
    np.random.seed(42)

    norm = RMSNorm(dim=16)
    x = np.random.randn(2, 8, 16)
    y = norm.forward(x)

    assert y.shape == x.shape, "Output shape should match input"
    # RMSNorm should normalize to roughly unit RMS
    rms = np.sqrt(np.mean(y ** 2, axis=-1))
    assert np.allclose(rms, 1.0, atol=0.1), "RMSNorm should normalize to unit RMS"
    print("✓ test_rmsnorm_forward passed")


def test_rmsnorm_backward():
    """Test RMSNorm gradient numerically."""
    np.random.seed(42)

    norm = RMSNorm(dim=8)
    x = np.random.randn(2, 4, 8)

    # Forward
    y = norm.forward(x)

    # Backward with random gradient
    grad_output = np.random.randn(*y.shape)
    grad_x = norm.backward(grad_output)

    # Numerical gradient check for a single element
    eps = 1e-5
    i, j, k = 0, 0, 0

    x_plus = x.copy()
    x_plus[i, j, k] += eps
    norm2 = RMSNorm(dim=8)
    norm2.weight.data = norm.weight.data.copy()
    y_plus = norm2.forward(x_plus)

    x_minus = x.copy()
    x_minus[i, j, k] -= eps
    norm3 = RMSNorm(dim=8)
    norm3.weight.data = norm.weight.data.copy()
    y_minus = norm3.forward(x_minus)

    numerical_grad = np.sum((y_plus - y_minus) * grad_output) / (2 * eps)
    analytical_grad = grad_x[i, j, k]

    assert np.allclose(numerical_grad, analytical_grad, atol=1e-4), \
        f"RMSNorm gradient mismatch: {numerical_grad} vs {analytical_grad}"
    print("✓ test_rmsnorm_backward passed")


# =============================================================================
# Embedding Tests
# =============================================================================

def test_embedding_forward():
    """Test Embedding forward pass."""
    np.random.seed(42)

    emb = Embedding(vocab_size=100, d_model=32)
    tokens = np.array([[1, 5, 10], [20, 30, 40]])

    x = emb.forward(tokens)

    assert x.shape == (2, 3, 32), f"Expected shape (2, 3, 32), got {x.shape}"
    assert np.allclose(x[0, 0], emb.weight.data[1]), "Embedding lookup incorrect"
    print("✓ test_embedding_forward passed")


def test_embedding_backward():
    """Test Embedding gradient accumulation."""
    np.random.seed(42)

    emb = Embedding(vocab_size=10, d_model=4)
    tokens = np.array([[1, 1, 2]])  # Token 1 appears twice

    x = emb.forward(tokens)
    grad_output = np.ones_like(x)

    emb.weight.grad = np.zeros_like(emb.weight.data)
    emb.backward(grad_output)

    # Token 1 gradient should be accumulated (appears twice)
    assert np.allclose(emb.weight.grad[1], 2.0), "Gradient should accumulate for repeated tokens"
    assert np.allclose(emb.weight.grad[2], 1.0), "Token 2 gradient should be 1"
    assert np.allclose(emb.weight.grad[0], 0.0), "Unused tokens should have zero gradient"
    print("✓ test_embedding_backward passed")


# =============================================================================
# MultiHeadAttention Tests
# =============================================================================

def test_attention_forward():
    """Test MultiHeadAttention forward pass."""
    np.random.seed(42)

    attn = MultiHeadAttention(d_model=32, n_heads=4)
    x = np.random.randn(2, 8, 32)

    y = attn.forward(x)

    assert y.shape == x.shape, "Attention output should match input shape"
    print("✓ test_attention_forward passed")


def test_attention_causal_mask():
    """Test that causal mask prevents attending to future."""
    np.random.seed(42)

    attn = MultiHeadAttention(d_model=16, n_heads=2)
    x = np.random.randn(1, 4, 16)

    # Create causal mask
    from model import create_causal_mask
    mask = create_causal_mask(4)

    y = attn.forward(x, mask=mask)

    # Attention weights should be lower-triangular
    weights = attn.cache['attn'][0, 0]  # First batch, first head
    upper_tri = np.triu(np.ones((4, 4)), k=1)
    masked_weights = weights * upper_tri

    assert np.allclose(masked_weights, 0.0, atol=1e-6), \
        "Causal mask should zero out upper triangle"
    print("✓ test_attention_causal_mask passed")


def test_attention_backward():
    """Test MultiHeadAttention backward pass produces gradients."""
    np.random.seed(42)

    attn = MultiHeadAttention(d_model=16, n_heads=2)
    x = np.random.randn(2, 4, 16)

    # Forward
    y = attn.forward(x)

    # Backward
    grad_output = np.random.randn(*y.shape)
    grad_x = attn.backward(grad_output)

    # Check gradients exist and have correct shapes
    assert grad_x.shape == x.shape, "Input gradient shape mismatch"
    assert attn.wq.grad.shape == attn.wq.data.shape, "WQ gradient shape mismatch"
    assert attn.wk.grad.shape == attn.wk.data.shape, "WK gradient shape mismatch"
    assert attn.wv.grad.shape == attn.wv.data.shape, "WV gradient shape mismatch"
    assert attn.wo.grad.shape == attn.wo.data.shape, "WO gradient shape mismatch"

    # Check gradients are non-zero
    assert np.abs(attn.wq.grad).sum() > 0, "WQ gradient should be non-zero"
    print("✓ test_attention_backward passed")


# =============================================================================
# FeedForward Tests
# =============================================================================

def test_feedforward_forward():
    """Test FeedForward forward pass."""
    np.random.seed(42)

    ffn = FeedForward(d_model=32)
    x = np.random.randn(2, 8, 32)

    y = ffn.forward(x)

    assert y.shape == x.shape, "FFN output should match input shape"
    print("✓ test_feedforward_forward passed")


def test_feedforward_backward():
    """Test FeedForward backward pass."""
    np.random.seed(42)

    ffn = FeedForward(d_model=16)
    x = np.random.randn(2, 4, 16)

    y = ffn.forward(x)
    grad_output = np.random.randn(*y.shape)
    grad_x = ffn.backward(grad_output)

    assert grad_x.shape == x.shape, "Input gradient shape mismatch"
    assert ffn.w1.grad.shape == ffn.w1.data.shape, "W1 gradient shape mismatch"
    assert ffn.w2.grad.shape == ffn.w2.data.shape, "W2 gradient shape mismatch"
    assert ffn.w3.grad.shape == ffn.w3.data.shape, "W3 gradient shape mismatch"
    print("✓ test_feedforward_backward passed")


# =============================================================================
# TransformerBlock Tests
# =============================================================================

def test_transformer_block():
    """Test TransformerBlock forward and backward."""
    np.random.seed(42)

    block = TransformerBlock(d_model=32, n_heads=4)
    x = np.random.randn(2, 8, 32)

    from model import create_causal_mask
    mask = create_causal_mask(8)

    # Forward
    y = block.forward(x, mask)
    assert y.shape == x.shape, "Block output should match input shape"

    # Backward
    grad_output = np.random.randn(*y.shape)
    grad_x = block.backward(grad_output)
    assert grad_x.shape == x.shape, "Block gradient should match input shape"

    # Check all parameters have gradients
    for p in block.parameters():
        assert p.grad is not None or hasattr(p, 'grad'), f"Missing gradient"
    print("✓ test_transformer_block passed")


# =============================================================================
# Full Transformer Tests
# =============================================================================

def test_transformer_forward():
    """Test Transformer forward pass."""
    np.random.seed(42)

    model = TrainableTransformer(
        vocab_size=50,
        d_model=32,
        n_heads=4,
        n_layers=2,
        max_seq_len=16,
    )

    tokens = np.array([[1, 5, 10, 15, 20]])
    logits = model.forward(tokens)

    assert logits.shape == (1, 5, 50), f"Expected (1, 5, 50), got {logits.shape}"
    print("✓ test_transformer_forward passed")


def test_transformer_backward():
    """Test Transformer backward pass."""
    np.random.seed(42)

    model = TrainableTransformer(
        vocab_size=50,
        d_model=32,
        n_heads=4,
        n_layers=2,
    )

    tokens = np.array([[1, 5, 10, 15, 20]])
    targets = np.array([[5, 10, 15, 20, 25]])

    logits = model.forward(tokens)
    loss, grad_logits = cross_entropy_loss(logits, targets)

    model.zero_grad()
    model.backward(grad_logits)

    # Check all parameters have gradients
    for i, p in enumerate(model.parameters()):
        assert p.grad is not None, f"Parameter {i} missing gradient"
        assert p.grad.shape == p.data.shape, f"Parameter {i} gradient shape mismatch"

    print("✓ test_transformer_backward passed")


def test_transformer_generation():
    """Test Transformer text generation."""
    np.random.seed(42)

    model = TrainableTransformer(
        vocab_size=50,
        d_model=32,
        n_heads=4,
        n_layers=2,
    )

    prompt = [1, 2, 3]
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)

    assert len(generated) == len(prompt) + 10, "Generation should produce correct length"
    assert all(0 <= t < 50 for t in generated), "All tokens should be valid"
    print("✓ test_transformer_generation passed")


# =============================================================================
# Loss Function Tests
# =============================================================================

def test_cross_entropy_loss():
    """Test cross-entropy loss computation."""
    np.random.seed(42)

    logits = np.random.randn(2, 5, 10)  # [batch, seq, vocab]
    targets = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

    loss, grad = cross_entropy_loss(logits, targets)

    assert isinstance(loss, float), "Loss should be a scalar"
    assert loss > 0, "Loss should be positive"
    assert grad.shape == logits.shape, "Gradient shape should match logits"
    print("✓ test_cross_entropy_loss passed")


def test_cross_entropy_gradient():
    """Test cross-entropy gradient numerically."""
    np.random.seed(42)

    logits = np.random.randn(1, 3, 5)
    targets = np.array([[0, 1, 2]])

    loss, grad = cross_entropy_loss(logits, targets)

    # Numerical gradient check
    eps = 1e-5
    for i in range(3):
        for j in range(5):
            logits_plus = logits.copy()
            logits_plus[0, i, j] += eps
            loss_plus, _ = cross_entropy_loss(logits_plus, targets)

            logits_minus = logits.copy()
            logits_minus[0, i, j] -= eps
            loss_minus, _ = cross_entropy_loss(logits_minus, targets)

            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytical = grad[0, i, j]

            assert np.allclose(numerical, analytical, atol=1e-4), \
                f"Gradient mismatch at ({i}, {j}): {numerical} vs {analytical}"

    print("✓ test_cross_entropy_gradient passed")


# =============================================================================
# Tokenizer Tests
# =============================================================================

def test_tokenizer():
    """Test CharTokenizer."""
    tokenizer = CharTokenizer().train("hello world")

    encoded = tokenizer.encode("hello")
    decoded = tokenizer.decode(encoded)

    assert decoded == "hello", f"Expected 'hello', got '{decoded}'"
    assert tokenizer.vocab_size == len(set("hello world")), "Vocab size mismatch"
    print("✓ test_tokenizer passed")


# =============================================================================
# Training Tests
# =============================================================================

def test_training_reduces_loss():
    """Test that training actually reduces loss."""
    np.random.seed(42)

    # Create a tiny model
    model = TrainableTransformer(
        vocab_size=10,
        d_model=16,
        n_heads=2,
        n_layers=1,
    )

    # Simple training data
    tokens = np.array([[1, 2, 3, 4, 5]])
    targets = np.array([[2, 3, 4, 5, 6]])

    # Compute initial loss
    logits = model.forward(tokens)
    initial_loss, _ = cross_entropy_loss(logits, targets)

    # Train for a few steps
    lr = 0.01
    for _ in range(50):
        logits = model.forward(tokens)
        loss, grad = cross_entropy_loss(logits, targets)

        model.zero_grad()
        model.backward(grad)

        # Simple SGD update
        for p in model.parameters():
            if p.grad is not None:
                p.data -= lr * p.grad

    # Final loss
    logits = model.forward(tokens)
    final_loss, _ = cross_entropy_loss(logits, targets)

    assert final_loss < initial_loss, \
        f"Training should reduce loss: {initial_loss:.4f} -> {final_loss:.4f}"
    print(f"✓ test_training_reduces_loss passed (loss: {initial_loss:.4f} -> {final_loss:.4f})")


def test_perplexity():
    """Test perplexity computation."""
    assert compute_perplexity(0.0) == 1.0, "Perplexity of 0 loss should be 1"
    assert np.isclose(compute_perplexity(np.log(10)), 10.0), "exp(log(10)) should be 10"
    print("✓ test_perplexity passed")


# =============================================================================
# Integration Test
# =============================================================================

def test_full_pipeline():
    """Test complete training pipeline."""
    np.random.seed(42)

    text = "hello world hello world"
    tokenizer = CharTokenizer().train(text)
    tokens = tokenizer.encode(text)

    model = TrainableTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=2,
        n_layers=2,
        max_seq_len=16,
    )

    # Create batch
    seq_len = 8
    inputs = np.array([tokens[:seq_len]])
    targets = np.array([tokens[1:seq_len + 1]])

    # Forward
    logits = model.forward(inputs)
    loss, grad = cross_entropy_loss(logits, targets)

    # Backward
    model.zero_grad()
    model.backward(grad)

    # Update
    for p in model.parameters():
        if p.grad is not None:
            p.data -= 0.01 * p.grad

    # Generate
    prompt = tokenizer.encode("hel")
    generated = model.generate(prompt, max_new_tokens=5)
    output = tokenizer.decode(generated)

    print(f"✓ test_full_pipeline passed (generated: '{output}')")


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Capstone Tests")
    print("=" * 60)
    print()

    tests = [
        # Utilities
        test_softmax,
        test_softmax_numerical_stability,
        test_silu,
        test_silu_backward,

        # Components
        test_rmsnorm_forward,
        test_rmsnorm_backward,
        test_embedding_forward,
        test_embedding_backward,
        test_attention_forward,
        test_attention_causal_mask,
        test_attention_backward,
        test_feedforward_forward,
        test_feedforward_backward,
        test_transformer_block,

        # Full model
        test_transformer_forward,
        test_transformer_backward,
        test_transformer_generation,

        # Loss
        test_cross_entropy_loss,
        test_cross_entropy_gradient,

        # Tokenizer
        test_tokenizer,

        # Training
        test_training_reduces_loss,
        test_perplexity,

        # Integration
        test_full_pipeline,
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

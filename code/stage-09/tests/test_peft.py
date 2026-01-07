"""
Tests for Stage 9: Fine-tuning & Parameter-Efficient Methods
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from peft import (
    LoRALayer,
    LoRALinear,
    Adapter,
    PrefixTuning,
    PromptTuning,
    count_trainable_parameters,
    compute_parameter_efficiency,
    apply_lora_to_attention,
    compare_peft_methods,
)


# =============================================================================
# LoRA Tests
# =============================================================================

def test_lora_layer_initialization():
    """Test LoRA layer initializes correctly."""
    lora = LoRALayer(in_features=256, out_features=256, rank=8)

    assert lora.A.shape == (8, 256), f"A shape wrong: {lora.A.shape}"
    assert lora.B.shape == (256, 8), f"B shape wrong: {lora.B.shape}"
    assert np.allclose(lora.B, 0), "B should initialize to zeros"
    print("✓ test_lora_layer_initialization passed")


def test_lora_forward():
    """Test LoRA forward pass."""
    np.random.seed(42)

    lora = LoRALayer(128, 128, rank=4, alpha=8)
    W = np.random.randn(128, 128) * 0.02
    x = np.random.randn(2, 16, 128)

    output = lora.forward(x, W)

    assert output.shape == (2, 16, 128), f"Output shape wrong: {output.shape}"
    # At initialization (B=0), output should equal base output
    expected = x @ W.T
    assert np.allclose(output, expected, atol=1e-6), "Initial output should match base"
    print("✓ test_lora_forward passed")


def test_lora_forward_with_adaptation():
    """Test LoRA forward with non-zero adaptation."""
    np.random.seed(42)

    lora = LoRALayer(64, 64, rank=4, alpha=8)
    W = np.random.randn(64, 64) * 0.02
    x = np.random.randn(2, 8, 64)

    # Modify B to get non-trivial adaptation
    lora.B = np.random.randn(64, 4) * 0.01

    output = lora.forward(x, W)
    base_output = x @ W.T

    # Output should differ from base
    assert not np.allclose(output, base_output), "Adaptation should change output"
    print("✓ test_lora_forward_with_adaptation passed")


def test_lora_backward():
    """Test LoRA backward pass produces gradients."""
    np.random.seed(42)

    lora = LoRALayer(64, 64, rank=4)
    W = np.random.randn(64, 64) * 0.02
    x = np.random.randn(2, 8, 64)

    # Forward
    output = lora.forward(x, W)

    # Backward
    grad_output = np.random.randn(*output.shape)
    grad_x = lora.backward(grad_output)

    assert grad_x.shape == x.shape, f"Input gradient shape wrong: {grad_x.shape}"
    assert lora.A_grad is not None, "A gradient should exist"
    assert lora.B_grad is not None, "B gradient should exist"
    assert lora.A_grad.shape == lora.A.shape, "A gradient shape wrong"
    assert lora.B_grad.shape == lora.B.shape, "B gradient shape wrong"
    print("✓ test_lora_backward passed")


def test_lora_gradient_numerical():
    """Test LoRA gradients numerically."""
    np.random.seed(42)

    lora = LoRALayer(32, 32, rank=2, alpha=4)
    W = np.random.randn(32, 32) * 0.02
    x = np.random.randn(1, 4, 32)

    # Set non-zero B for non-trivial gradients
    lora.B = np.random.randn(32, 2) * 0.1

    # Forward and backward
    output = lora.forward(x, W)
    grad_output = np.ones_like(output)
    lora.backward(grad_output)

    # Numerical gradient check for A[0,0]
    eps = 1e-5
    lora_plus = LoRALayer(32, 32, rank=2, alpha=4)
    lora_plus.A = lora.A.copy()
    lora_plus.B = lora.B.copy()
    lora_plus.A[0, 0] += eps
    out_plus = lora_plus.forward(x, W)

    lora_minus = LoRALayer(32, 32, rank=2, alpha=4)
    lora_minus.A = lora.A.copy()
    lora_minus.B = lora.B.copy()
    lora_minus.A[0, 0] -= eps
    out_minus = lora_minus.forward(x, W)

    numerical_grad = np.sum((out_plus - out_minus) * grad_output) / (2 * eps)
    analytical_grad = lora.A_grad[0, 0]

    assert np.isclose(numerical_grad, analytical_grad, rtol=1e-3), \
        f"Gradient mismatch: numerical={numerical_grad}, analytical={analytical_grad}"
    print("✓ test_lora_gradient_numerical passed")


def test_lora_merge_weights():
    """Test merging LoRA into base weights."""
    np.random.seed(42)

    lora = LoRALayer(64, 64, rank=4, alpha=8)
    W = np.random.randn(64, 64) * 0.02
    lora.B = np.random.randn(64, 4) * 0.01

    x = np.random.randn(2, 8, 64)

    # Output with LoRA
    output_lora = lora.forward(x, W)

    # Merge weights
    W_merged = lora.merge_weights(W)

    # Output with merged weights (no LoRA)
    output_merged = x @ W_merged.T

    assert np.allclose(output_lora, output_merged, atol=1e-6), \
        "Merged output should match LoRA output"
    print("✓ test_lora_merge_weights passed")


def test_lora_compression_ratio():
    """Test LoRA compression ratio calculation."""
    lora = LoRALayer(4096, 4096, rank=8)

    full_params = 4096 * 4096  # 16.7M
    lora_params = lora.num_parameters()  # 8 * (4096 + 4096) = 65K

    ratio = lora.compression_ratio()

    assert ratio < 0.01, f"LoRA should be < 1% of full params, got {ratio:.2%}"
    assert lora_params == 8 * (4096 + 4096), f"Wrong param count: {lora_params}"
    print("✓ test_lora_compression_ratio passed")


# =============================================================================
# LoRALinear Tests
# =============================================================================

def test_lora_linear():
    """Test LoRA-enhanced linear layer."""
    np.random.seed(42)

    weight = np.random.randn(128, 64) * 0.02
    bias = np.random.randn(128) * 0.01

    layer = LoRALinear(weight, bias, lora_rank=4)

    x = np.random.randn(2, 16, 64)
    output = layer.forward(x)

    assert output.shape == (2, 16, 128), f"Output shape wrong: {output.shape}"
    print("✓ test_lora_linear passed")


def test_lora_linear_merge():
    """Test merging LoRA linear layer."""
    np.random.seed(42)

    weight = np.random.randn(64, 64) * 0.02
    layer = LoRALinear(weight, lora_rank=4)

    # Modify LoRA weights
    layer.lora.B = np.random.randn(64, 4) * 0.01

    merged_w, merged_b = layer.merge()

    assert merged_w.shape == weight.shape
    assert not np.allclose(merged_w, weight), "Merged should differ from original"
    print("✓ test_lora_linear_merge passed")


# =============================================================================
# Adapter Tests
# =============================================================================

def test_adapter_initialization():
    """Test adapter initialization."""
    adapter = Adapter(d_model=256, bottleneck_dim=64)

    assert adapter.W_down.shape == (256, 64)
    assert adapter.W_up.shape == (64, 256)
    print("✓ test_adapter_initialization passed")


def test_adapter_forward():
    """Test adapter forward pass."""
    np.random.seed(42)

    adapter = Adapter(d_model=128, bottleneck_dim=32)
    x = np.random.randn(2, 16, 128)

    output = adapter.forward(x)

    assert output.shape == x.shape, f"Output shape wrong: {output.shape}"
    # With small initialization, output should be close to input (residual)
    assert np.allclose(output, x, atol=0.1), "Initial adapter should be near-identity"
    print("✓ test_adapter_forward passed")


def test_adapter_backward():
    """Test adapter backward pass."""
    np.random.seed(42)

    adapter = Adapter(d_model=64, bottleneck_dim=16)
    x = np.random.randn(2, 8, 64)

    output = adapter.forward(x)
    grad_output = np.random.randn(*output.shape)
    grad_x = adapter.backward(grad_output)

    assert grad_x.shape == x.shape
    assert hasattr(adapter, 'W_down_grad')
    assert hasattr(adapter, 'W_up_grad')
    print("✓ test_adapter_backward passed")


def test_adapter_parameter_count():
    """Test adapter parameter counting."""
    adapter = Adapter(d_model=4096, bottleneck_dim=64)

    params = adapter.num_parameters()
    expected = 2 * 4096 * 64  # down + up

    assert params == expected, f"Expected {expected}, got {params}"
    print("✓ test_adapter_parameter_count passed")


# =============================================================================
# Prefix Tuning Tests
# =============================================================================

def test_prefix_tuning_initialization():
    """Test prefix tuning initialization."""
    prefix = PrefixTuning(num_layers=12, num_heads=8, d_head=64, prefix_length=10)

    expected_shape = (12, 2, 10, 8, 64)
    assert prefix.prefix.shape == expected_shape, f"Wrong shape: {prefix.prefix.shape}"
    print("✓ test_prefix_tuning_initialization passed")


def test_prefix_tuning_get_prefix():
    """Test getting prefixes for a layer."""
    prefix = PrefixTuning(num_layers=4, num_heads=4, d_head=32, prefix_length=5)

    k, v = prefix.get_prefix(layer_idx=2)

    assert k.shape == (5, 4, 32), f"Key prefix shape wrong: {k.shape}"
    assert v.shape == (5, 4, 32), f"Value prefix shape wrong: {v.shape}"
    print("✓ test_prefix_tuning_get_prefix passed")


def test_prefix_tuning_params():
    """Test prefix tuning parameter count."""
    prefix = PrefixTuning(num_layers=32, num_heads=32, d_head=128, prefix_length=10)

    params = prefix.num_parameters()
    expected = 32 * 2 * 10 * 32 * 128  # layers * (k,v) * length * heads * d_head

    assert params == expected
    print("✓ test_prefix_tuning_params passed")


# =============================================================================
# Prompt Tuning Tests
# =============================================================================

def test_prompt_tuning_initialization():
    """Test prompt tuning initialization."""
    prompt = PromptTuning(d_model=256, prompt_length=20)

    assert prompt.prompt.shape == (20, 256)
    print("✓ test_prompt_tuning_initialization passed")


def test_prompt_tuning_forward():
    """Test prompt tuning forward pass."""
    np.random.seed(42)

    prompt = PromptTuning(d_model=128, prompt_length=10)
    x = np.random.randn(2, 32, 128)

    output = prompt.forward(x)

    expected_length = 10 + 32  # prompt + input
    assert output.shape == (2, expected_length, 128), f"Wrong shape: {output.shape}"
    print("✓ test_prompt_tuning_forward passed")


def test_prompt_tuning_backward():
    """Test prompt tuning backward pass."""
    np.random.seed(42)

    prompt = PromptTuning(d_model=64, prompt_length=5)
    x = np.random.randn(2, 16, 64)

    output = prompt.forward(x)
    grad_output = np.random.randn(*output.shape)
    grad_x = prompt.backward(grad_output)

    assert grad_x.shape == x.shape, f"Gradient shape wrong: {grad_x.shape}"
    assert prompt.prompt_grad is not None
    assert prompt.prompt_grad.shape == prompt.prompt.shape
    print("✓ test_prompt_tuning_backward passed")


def test_prompt_tuning_init_from_vocab():
    """Test prompt tuning initialization from vocabulary."""
    np.random.seed(42)

    vocab_embeddings = np.random.randn(1000, 128)
    prompt = PromptTuning(d_model=128, prompt_length=10, init_from_vocab=vocab_embeddings)

    # Each prompt should be copied from vocab
    assert prompt.prompt.shape == (10, 128)
    print("✓ test_prompt_tuning_init_from_vocab passed")


# =============================================================================
# Utility Tests
# =============================================================================

def test_count_trainable_parameters():
    """Test parameter counting utility."""
    params = [np.zeros((10, 20)), np.zeros((30,))]
    count = count_trainable_parameters(params)

    assert count == 10 * 20 + 30
    print("✓ test_count_trainable_parameters passed")


def test_compute_parameter_efficiency():
    """Test efficiency computation."""
    result = compute_parameter_efficiency(trainable=1000, total=100000)

    assert result['trainable_ratio'] == 0.01
    assert result['efficiency_gain'] == 100
    print("✓ test_compute_parameter_efficiency passed")


def test_apply_lora_to_attention():
    """Test applying LoRA to attention matrices."""
    np.random.seed(42)

    d = 64
    wq = np.random.randn(d, d)
    wk = np.random.randn(d, d)
    wv = np.random.randn(d, d)
    wo = np.random.randn(d, d)

    layers = apply_lora_to_attention(wq, wk, wv, wo, rank=4, target_modules=['q', 'v'])

    assert isinstance(layers['q'], LoRALinear), "Q should have LoRA"
    assert isinstance(layers['v'], LoRALinear), "V should have LoRA"
    assert isinstance(layers['k'], np.ndarray), "K should be frozen"
    assert isinstance(layers['o'], np.ndarray), "O should be frozen"
    print("✓ test_apply_lora_to_attention passed")


def test_compare_peft_methods():
    """Test PEFT comparison."""
    comparison = compare_peft_methods(
        d_model=512,
        num_layers=6,
        num_heads=8,
        lora_rank=4,
    )

    assert 'lora' in comparison
    assert 'adapters' in comparison
    assert 'prefix_tuning' in comparison
    assert 'prompt_tuning' in comparison

    # LoRA should be much smaller than full fine-tuning
    assert comparison['lora']['ratio'] < 0.1
    print("✓ test_compare_peft_methods passed")


# =============================================================================
# Integration Tests
# =============================================================================

def test_lora_training_step():
    """Test a complete LoRA training step."""
    np.random.seed(42)

    # Create layer
    d = 64
    W = np.random.randn(d, d) * 0.02
    lora = LoRALayer(d, d, rank=4, alpha=8)
    lora.B = np.random.randn(d, 4) * 0.01  # Non-zero for training

    # Forward
    x = np.random.randn(4, 8, d)
    output = lora.forward(x, W)

    # Fake loss gradient
    target = np.random.randn(*output.shape)
    grad = 2 * (output - target) / output.size

    # Backward
    lora.backward(grad)

    # Update (simple SGD)
    lr = 0.01
    lora.A -= lr * lora.A_grad
    lora.B -= lr * lora.B_grad

    # Verify parameters changed
    assert not np.allclose(lora.A, np.zeros_like(lora.A))
    print("✓ test_lora_training_step passed")


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Stage 9 PEFT Tests")
    print("=" * 60)
    print()

    tests = [
        # LoRA
        test_lora_layer_initialization,
        test_lora_forward,
        test_lora_forward_with_adaptation,
        test_lora_backward,
        test_lora_gradient_numerical,
        test_lora_merge_weights,
        test_lora_compression_ratio,

        # LoRALinear
        test_lora_linear,
        test_lora_linear_merge,

        # Adapter
        test_adapter_initialization,
        test_adapter_forward,
        test_adapter_backward,
        test_adapter_parameter_count,

        # Prefix Tuning
        test_prefix_tuning_initialization,
        test_prefix_tuning_get_prefix,
        test_prefix_tuning_params,

        # Prompt Tuning
        test_prompt_tuning_initialization,
        test_prompt_tuning_forward,
        test_prompt_tuning_backward,
        test_prompt_tuning_init_from_vocab,

        # Utilities
        test_count_trainable_parameters,
        test_compute_parameter_efficiency,
        test_apply_lora_to_attention,
        test_compare_peft_methods,

        # Integration
        test_lora_training_step,
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

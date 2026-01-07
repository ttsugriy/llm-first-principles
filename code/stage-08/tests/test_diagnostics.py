"""
Tests for Stage 8: Training Dynamics & Debugging
"""

import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from diagnostics import (
    TrainingHistory,
    GradientStats,
    LearningRateFinder,
    ActivationStats,
    ActivationMonitor,
    TrainingDebugger,
    compute_activation_stats,
    compute_layer_gradient_stats,
    clip_gradients,
    detect_dead_neurons,
    check_initialization,
)


# =============================================================================
# TrainingHistory Tests
# =============================================================================

def test_training_history_record():
    """Test recording training metrics."""
    history = TrainingHistory()

    for i in range(10):
        history.record(step=i, loss=1.0 - i * 0.1, grad_norm=0.5, lr=1e-3)

    assert len(history.loss) == 10
    assert len(history.step) == 10
    assert history.loss[0] == 1.0
    assert np.isclose(history.loss[-1], 0.1, atol=1e-10)
    print("✓ test_training_history_record passed")


def test_training_history_diagnose_healthy():
    """Test diagnosis of healthy training."""
    history = TrainingHistory()

    # Simulate healthy decreasing loss
    for i in range(100):
        loss = 3.0 * np.exp(-i / 30) + 0.5
        history.record(step=i, loss=loss, grad_norm=1.0, lr=1e-3)

    diagnosis = history.diagnose()
    assert diagnosis['status'] == 'healthy', f"Expected healthy, got {diagnosis['status']}"
    assert len(diagnosis['issues']) == 0
    print("✓ test_training_history_diagnose_healthy passed")


def test_training_history_detect_explosion():
    """Test detection of loss explosion."""
    history = TrainingHistory()

    # Normal loss then explosion
    for i in range(50):
        history.record(step=i, loss=1.0, grad_norm=1.0, lr=1e-3)

    for i in range(50, 60):
        history.record(step=i, loss=1000.0, grad_norm=1.0, lr=1e-3)

    diagnosis = history.diagnose()
    assert 'LOSS_EXPLOSION' in diagnosis['issues']
    print("✓ test_training_history_detect_explosion passed")


def test_training_history_detect_plateau():
    """Test detection of loss plateau."""
    history = TrainingHistory()

    # Flat loss
    for i in range(200):
        history.record(step=i, loss=1.0 + np.random.normal(0, 0.0001), grad_norm=1.0, lr=1e-3)

    diagnosis = history.diagnose()
    assert 'LOSS_PLATEAU' in diagnosis['issues']
    print("✓ test_training_history_detect_plateau passed")


def test_training_history_detect_vanishing_gradients():
    """Test detection of vanishing gradients."""
    history = TrainingHistory()

    for i in range(50):
        history.record(step=i, loss=1.0, grad_norm=1e-10, lr=1e-3)

    diagnosis = history.diagnose()
    assert 'VANISHING_GRADIENTS' in diagnosis['issues']
    print("✓ test_training_history_detect_vanishing_gradients passed")


def test_training_history_save_load():
    """Test saving and loading history."""
    history = TrainingHistory()
    for i in range(10):
        history.record(step=i, loss=float(i), grad_norm=0.5, lr=1e-3)

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        path = f.name

    try:
        history.save(path)
        loaded = TrainingHistory.load(path)

        assert loaded.loss == history.loss
        assert loaded.step == history.step
        print("✓ test_training_history_save_load passed")
    finally:
        os.unlink(path)


# =============================================================================
# GradientStats Tests
# =============================================================================

def test_gradient_stats_basic():
    """Test basic gradient statistics."""
    grads = [np.random.randn(100) for _ in range(5)]
    stats = GradientStats.from_gradients(grads)

    assert stats.total_elements == 500
    assert stats.norm > 0
    assert stats.num_nans == 0
    assert stats.num_infs == 0
    print("✓ test_gradient_stats_basic passed")


def test_gradient_stats_detect_nan():
    """Test NaN detection in gradients."""
    grads = [np.array([1.0, np.nan, 2.0])]
    stats = GradientStats.from_gradients(grads)

    assert stats.num_nans == 1
    healthy, issues = stats.is_healthy()
    assert not healthy
    assert any('NaN' in i for i in issues)
    print("✓ test_gradient_stats_detect_nan passed")


def test_gradient_stats_detect_inf():
    """Test Inf detection in gradients."""
    grads = [np.array([1.0, np.inf, 2.0])]
    stats = GradientStats.from_gradients(grads)

    assert stats.num_infs == 1
    healthy, issues = stats.is_healthy()
    assert not healthy
    print("✓ test_gradient_stats_detect_inf passed")


def test_gradient_stats_detect_large_norm():
    """Test detection of large gradient norm."""
    grads = [np.ones(1000) * 100]  # Very large gradients
    stats = GradientStats.from_gradients(grads)

    healthy, issues = stats.is_healthy()
    assert not healthy
    assert any('large' in i.lower() for i in issues)
    print("✓ test_gradient_stats_detect_large_norm passed")


def test_compute_layer_gradient_stats():
    """Test per-layer gradient statistics."""
    grads = [np.random.randn(50), np.random.randn(100), np.random.randn(25)]
    names = ['fc1', 'fc2', 'fc3']

    layer_stats = compute_layer_gradient_stats(grads, names)

    assert len(layer_stats) == 3
    assert 'fc1' in layer_stats
    assert layer_stats['fc1'].total_elements == 50
    print("✓ test_compute_layer_gradient_stats passed")


# =============================================================================
# LearningRateFinder Tests
# =============================================================================

def test_lr_finder_basic():
    """Test basic LR finder functionality."""
    finder = LearningRateFinder(min_lr=1e-5, max_lr=1.0, num_steps=20)

    def train_fn(lr):
        return 2.0 - lr * 10 if lr < 0.1 else 2.0 + (lr - 0.1) * 100

    result = finder.range_test(None, train_fn)

    assert 'suggested_lr' in result
    assert result['suggested_lr'] > 0
    assert len(result['lrs']) > 0
    print("✓ test_lr_finder_basic passed")


def test_lr_finder_stops_on_explosion():
    """Test LR finder stops when loss explodes."""
    finder = LearningRateFinder(min_lr=1e-5, max_lr=10.0, num_steps=50)

    def train_fn(lr):
        if lr > 0.01:
            return float('inf')
        return 1.0

    result = finder.range_test(None, train_fn)

    # Should stop before all steps complete
    assert len(result['lrs']) < 50
    print("✓ test_lr_finder_stops_on_explosion passed")


# =============================================================================
# Activation Tests
# =============================================================================

def test_activation_stats():
    """Test activation statistics computation."""
    activations = np.random.randn(32, 64)
    stats = compute_activation_stats(activations)

    assert stats.total_elements == 32 * 64
    assert stats.std > 0
    print("✓ test_activation_stats passed")


def test_activation_stats_dead_neurons():
    """Test detection of dead neurons in activations."""
    # Mostly zeros (like dead ReLU)
    activations = np.maximum(0, np.random.randn(32, 64) - 3)  # Mostly negative -> zeros
    stats = compute_activation_stats(activations)

    assert stats.zero_ratio > 0.9  # Should be mostly zeros
    print("✓ test_activation_stats_dead_neurons passed")


def test_activation_monitor():
    """Test activation monitoring over time."""
    monitor = ActivationMonitor(window_size=10)

    for _ in range(15):
        monitor.record('layer1', np.random.randn(32, 64))
        # Dead layer
        monitor.record('layer2', np.zeros((32, 64)))

    issues = monitor.diagnose()
    assert 'layer2' in issues
    assert any('Dead' in i or 'zero' in i.lower() for i in issues.get('layer2', []))
    print("✓ test_activation_monitor passed")


# =============================================================================
# Utility Function Tests
# =============================================================================

def test_clip_gradients():
    """Test gradient clipping."""
    grads = [np.ones(100) * 10]  # Large gradients
    clipped, original_norm = clip_gradients(grads, max_norm=1.0)

    clipped_norm = np.sqrt(sum(np.sum(g ** 2) for g in clipped))

    assert original_norm > 1.0, "Original should exceed max_norm"
    assert clipped_norm <= 1.0 + 1e-6, "Clipped should be at max_norm"
    print("✓ test_clip_gradients passed")


def test_clip_gradients_no_change():
    """Test that small gradients aren't clipped."""
    grads = [np.ones(10) * 0.1]
    clipped, original_norm = clip_gradients(grads, max_norm=1.0)

    assert np.allclose(clipped[0], grads[0])
    print("✓ test_clip_gradients_no_change passed")


def test_detect_dead_neurons():
    """Test dead neuron detection."""
    # Half neurons are dead (always zero)
    activations = np.zeros((100, 20))
    activations[:, :10] = np.random.randn(100, 10)

    num_dead, ratio = detect_dead_neurons(activations)

    assert ratio >= 0.4  # At least 40% dead
    print("✓ test_detect_dead_neurons passed")


def test_check_initialization():
    """Test initialization checking."""
    # Good initialization (Xavier-like)
    good_weights = [np.random.randn(100, 50) * np.sqrt(2.0 / 100)]
    result = check_initialization(good_weights)

    assert result['all_ok'] or result['layers'][0]['status'] == 'ok'
    print("✓ test_check_initialization passed")


def test_check_initialization_bad():
    """Test detection of bad initialization."""
    # Very small weights
    bad_weights = [np.random.randn(100, 50) * 0.0001]
    result = check_initialization(bad_weights)

    assert result['layers'][0]['status'] == 'too_small'
    print("✓ test_check_initialization_bad passed")


# =============================================================================
# TrainingDebugger Tests
# =============================================================================

def test_training_debugger():
    """Test full training debugger."""
    debugger = TrainingDebugger()

    for i in range(20):
        loss = 2.0 - i * 0.05
        grads = [np.random.randn(50) * 0.1]
        debugger.step(step=i, loss=loss, gradients=grads, learning_rate=1e-3)

    assert debugger.quick_check()
    report = debugger.report()
    assert 'status' in report
    print("✓ test_training_debugger passed")


def test_training_debugger_detects_nan():
    """Test debugger detects NaN."""
    debugger = TrainingDebugger()

    # Normal steps
    for i in range(10):
        debugger.step(step=i, loss=1.0, gradients=[np.ones(10)], learning_rate=1e-3)

    # NaN step
    debugger.step(step=10, loss=float('nan'), gradients=[np.ones(10)], learning_rate=1e-3)

    assert not debugger.quick_check()
    print("✓ test_training_debugger_detects_nan passed")


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Stage 8 Training Diagnostics Tests")
    print("=" * 60)
    print()

    tests = [
        # TrainingHistory
        test_training_history_record,
        test_training_history_diagnose_healthy,
        test_training_history_detect_explosion,
        test_training_history_detect_plateau,
        test_training_history_detect_vanishing_gradients,
        test_training_history_save_load,

        # GradientStats
        test_gradient_stats_basic,
        test_gradient_stats_detect_nan,
        test_gradient_stats_detect_inf,
        test_gradient_stats_detect_large_norm,
        test_compute_layer_gradient_stats,

        # LearningRateFinder
        test_lr_finder_basic,
        test_lr_finder_stops_on_explosion,

        # Activation
        test_activation_stats,
        test_activation_stats_dead_neurons,
        test_activation_monitor,

        # Utilities
        test_clip_gradients,
        test_clip_gradients_no_change,
        test_detect_dead_neurons,
        test_check_initialization,
        test_check_initialization_bad,

        # TrainingDebugger
        test_training_debugger,
        test_training_debugger_detects_nan,
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

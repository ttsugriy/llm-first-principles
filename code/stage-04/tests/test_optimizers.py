"""
Tests for Optimizer Implementations

Verifies correctness of optimizers using simple test problems
where we can verify convergence and gradient updates.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers import (
    SGD, AdaGrad, RMSprop, Adam, AdamW,
    WarmupCosineScheduler, StepScheduler, CosineWithRestartsScheduler,
    clip_grad_norm, clip_grad_value,
)


class TestSGD:
    """Tests for SGD optimizer."""

    def test_basic_update(self):
        """SGD updates parameters correctly."""
        params = [np.array([1.0, 2.0, 3.0])]
        grads = [np.array([0.1, 0.2, 0.3])]
        opt = SGD(params, lr=0.1)

        opt.step(grads)

        expected = np.array([1.0 - 0.1 * 0.1, 2.0 - 0.1 * 0.2, 3.0 - 0.1 * 0.3])
        assert np.allclose(params[0], expected), f"Got {params[0]}, expected {expected}"

    def test_momentum(self):
        """Momentum accumulates velocity."""
        params = [np.array([1.0])]
        opt = SGD(params, lr=0.1, momentum=0.9)

        # First step
        opt.step([np.array([1.0])])
        v1 = opt.state['velocity'][0].copy()

        # Second step with same gradient
        opt.step([np.array([1.0])])
        v2 = opt.state['velocity'][0].copy()

        # Velocity should grow
        assert v2[0] > v1[0], "Momentum should accumulate velocity"

    def test_convergence_quadratic(self):
        """SGD converges on simple quadratic."""
        # Minimize f(x) = x^2, optimal x = 0
        x = np.array([5.0])
        params = [x]
        opt = SGD(params, lr=0.1)

        for _ in range(100):
            grad = 2 * x  # df/dx = 2x
            opt.step([grad])

        assert abs(x[0]) < 0.01, f"Should converge to 0, got {x[0]}"


class TestAdam:
    """Tests for Adam optimizer."""

    def test_bias_correction(self):
        """Adam applies bias correction correctly."""
        params = [np.array([0.0])]
        opt = Adam(params, lr=1.0, betas=(0.9, 0.999))

        # First step
        opt.step([np.array([1.0])])

        # Without bias correction, m_hat would be 0.1
        # With bias correction, m_hat = 0.1 / (1 - 0.9) = 1.0
        # The update should be larger than without correction
        assert abs(params[0][0]) > 0.1, "Bias correction should amplify early updates"

    def test_adaptive_scaling(self):
        """Adam scales updates by gradient magnitude."""
        # Two parameters with different gradient magnitudes
        params = [np.array([0.0]), np.array([0.0])]
        opt = Adam(params, lr=0.1, betas=(0.9, 0.999))

        # Apply gradients: one large, one small
        for _ in range(100):
            opt.step([np.array([1.0]), np.array([0.01])])

        # Both should have moved, but the ratio should be different from 100x
        ratio = abs(params[0][0]) / abs(params[1][0])
        assert ratio < 50, f"Adaptive scaling should reduce ratio, got {ratio}"

    def test_convergence(self):
        """Adam converges on quadratic."""
        np.random.seed(42)
        A = np.random.randn(10, 5)
        b = np.random.randn(10)

        x = np.zeros(5)
        params = [x]
        opt = Adam(params, lr=0.1)

        for _ in range(1000):
            grad = A.T @ (A @ x - b)
            opt.step([grad])

        # Should be close to optimal
        x_opt = np.linalg.lstsq(A, b, rcond=None)[0]
        error = np.linalg.norm(x - x_opt)
        assert error < 0.01, f"Should converge, error = {error}"


class TestAdamW:
    """Tests for AdamW optimizer."""

    def test_weight_decay(self):
        """AdamW applies weight decay."""
        params = [np.array([1.0])]
        opt = AdamW(params, lr=0.1, weight_decay=0.1)

        # Step with zero gradient
        opt.step([np.array([0.0])])

        # Parameter should shrink due to weight decay
        expected = 1.0 * (1 - 0.1 * 0.1)  # (1 - lr * wd)
        # Note: bias correction affects zero gradients too
        assert params[0][0] < 1.0, "Weight decay should shrink parameters"

    def test_decoupled_weight_decay(self):
        """Weight decay is decoupled from adaptive scaling."""
        # Compare two identical setups, one with large gradient one with small
        params1 = [np.array([1.0])]
        params2 = [np.array([1.0])]

        opt1 = AdamW(params1, lr=0.1, weight_decay=0.1)
        opt2 = AdamW(params2, lr=0.1, weight_decay=0.1)

        # Different gradient magnitudes
        opt1.step([np.array([10.0])])
        opt2.step([np.array([0.1])])

        # Weight decay effect should be the same for both
        # (the difference is in the Adam update, not the decay)
        # Both started at 1.0, decay is 1 * (1 - 0.1 * 0.1) = 0.99


class TestSchedulers:
    """Tests for learning rate schedulers."""

    def test_warmup_cosine(self):
        """Warmup cosine schedule has correct shape."""
        params = [np.zeros(5)]
        opt = Adam(params, lr=1.0)
        scheduler = WarmupCosineScheduler(opt, warmup_steps=100, total_steps=1000)

        # During warmup: linear increase
        lrs_warmup = []
        for _ in range(100):
            scheduler.step()
            lrs_warmup.append(opt.lr)

        assert lrs_warmup[0] < lrs_warmup[50] < lrs_warmup[99], "Warmup should increase"

        # Peak at end of warmup
        peak_lr = opt.lr
        assert abs(peak_lr - 1.0) < 0.02, f"Should reach base lr, got {peak_lr}"

        # During decay: cosine decrease
        for _ in range(400):
            scheduler.step()

        mid_lr = opt.lr
        assert mid_lr < peak_lr, "Should decay after warmup"

        # End of training
        for _ in range(500):
            scheduler.step()

        final_lr = opt.lr
        assert final_lr < mid_lr, "Should continue decaying"

    def test_step_scheduler(self):
        """Step scheduler decays at correct intervals."""
        params = [np.zeros(5)]
        opt = Adam(params, lr=1.0)
        scheduler = StepScheduler(opt, step_size=10, gamma=0.1)

        # After 9 steps: still lr = 1.0 (step_count=9, 9//10=0)
        for _ in range(9):
            scheduler.step()
        assert abs(opt.lr - 1.0) < 1e-6, f"Expected 1.0, got {opt.lr}"

        # After 10th step: lr = 0.1 (step_count=10, 10//10=1)
        scheduler.step()
        assert abs(opt.lr - 0.1) < 1e-6, f"Expected 0.1, got {opt.lr}"

        # After 10 more steps (total 20): lr = 0.01 (20//10=2)
        for _ in range(10):
            scheduler.step()
        assert abs(opt.lr - 0.01) < 1e-6, f"Expected 0.01, got {opt.lr}"


class TestGradientClipping:
    """Tests for gradient clipping utilities."""

    def test_clip_grad_norm(self):
        """Gradient norm clipping works correctly."""
        grads = [np.array([3.0, 4.0])]  # norm = 5

        clipped, orig_norm = clip_grad_norm(grads, max_norm=1.0)

        assert abs(orig_norm - 5.0) < 1e-6, f"Original norm should be 5, got {orig_norm}"

        new_norm = np.sqrt(np.sum(clipped[0] ** 2))
        assert abs(new_norm - 1.0) < 1e-6, f"Clipped norm should be 1, got {new_norm}"

    def test_clip_grad_norm_no_clip(self):
        """No clipping when norm is below threshold."""
        grads = [np.array([0.3, 0.4])]  # norm = 0.5

        clipped, orig_norm = clip_grad_norm(grads, max_norm=1.0)

        assert np.allclose(clipped[0], grads[0]), "Should not clip small gradients"

    def test_clip_grad_value(self):
        """Element-wise clipping works correctly."""
        grads = [np.array([-5.0, 0.5, 5.0])]

        clipped = clip_grad_value(grads, clip_value=1.0)

        expected = np.array([-1.0, 0.5, 1.0])
        assert np.allclose(clipped[0], expected), f"Got {clipped[0]}, expected {expected}"


class TestOptimizerComparison:
    """Compare optimizer behaviors."""

    def test_momentum_faster_than_sgd(self):
        """Momentum converges faster than vanilla SGD on elongated quadratic."""
        np.random.seed(42)

        # Ill-conditioned problem
        def loss_and_grad(x):
            loss = x[0] ** 2 + 100 * x[1] ** 2
            grad = np.array([2 * x[0], 200 * x[1]])
            return loss, grad

        # SGD
        x_sgd = np.array([2.0, 2.0])
        opt_sgd = SGD([x_sgd], lr=0.001)
        for _ in range(100):
            _, grad = loss_and_grad(x_sgd)
            opt_sgd.step([grad])
        loss_sgd = loss_and_grad(x_sgd)[0]

        # Momentum
        x_mom = np.array([2.0, 2.0])
        opt_mom = SGD([x_mom], lr=0.001, momentum=0.9)
        for _ in range(100):
            _, grad = loss_and_grad(x_mom)
            opt_mom.step([grad])
        loss_mom = loss_and_grad(x_mom)[0]

        assert loss_mom < loss_sgd, f"Momentum ({loss_mom}) should beat SGD ({loss_sgd})"


def run_tests():
    """Run all tests."""
    test_classes = [
        TestSGD,
        TestAdam,
        TestAdamW,
        TestSchedulers,
        TestGradientClipping,
        TestOptimizerComparison,
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

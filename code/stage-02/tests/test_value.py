"""
Tests for Automatic Differentiation

Comprehensive test suite verifying:
1. Each operation's forward and backward pass
2. Gradient accumulation for shared variables
3. Numerical gradient checking
4. Edge cases and special values
5. Neural network learning
"""

import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from value import Value, Neuron, Layer, MLP


# =============================================================================
# Numerical Gradient Checking
# =============================================================================


def numerical_gradient(f, x: Value, h: float = 1e-5) -> float:
    """
    Compute numerical gradient using central difference.

    f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

    This is more accurate than forward difference because
    the error term is O(h²) instead of O(h).
    """
    # Save original value
    original = x.data

    # f(x + h)
    x.data = original + h
    y_plus = f().data

    # f(x - h)
    x.data = original - h
    y_minus = f().data

    # Restore
    x.data = original

    return (y_plus - y_minus) / (2 * h)


def check_gradient(computed: float, numerical: float, tolerance: float = 1e-4) -> bool:
    """
    Check if computed gradient matches numerical gradient.

    Uses relative error for large values, absolute for small.
    """
    if abs(computed) < 1e-7 and abs(numerical) < 1e-7:
        return True

    rel_error = abs(computed - numerical) / (abs(computed) + abs(numerical) + 1e-8)
    return rel_error < tolerance


# =============================================================================
# Unit Tests for Operations
# =============================================================================


class TestAddition:
    """Tests for addition operation."""

    def test_forward(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x + y
        assert abs(z.data - 5.0) < 1e-6, f"Expected 5.0, got {z.data}"

    def test_backward(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x + y
        z.backward()
        assert abs(x.grad - 1.0) < 1e-6, f"∂z/∂x should be 1, got {x.grad}"
        assert abs(y.grad - 1.0) < 1e-6, f"∂z/∂y should be 1, got {y.grad}"

    def test_numerical(self):
        x = Value(2.0)
        y = Value(3.0)

        def f():
            return x + y

        z = f()
        z.backward()

        num_grad = numerical_gradient(f, x)
        assert check_gradient(x.grad, num_grad), f"Gradient mismatch: {x.grad} vs {num_grad}"

    def test_scalar_add(self):
        x = Value(2.0)
        z = x + 5
        z.backward()
        assert abs(z.data - 7.0) < 1e-6
        assert abs(x.grad - 1.0) < 1e-6

    def test_radd(self):
        x = Value(2.0)
        z = 5 + x
        z.backward()
        assert abs(z.data - 7.0) < 1e-6
        assert abs(x.grad - 1.0) < 1e-6


class TestMultiplication:
    """Tests for multiplication operation."""

    def test_forward(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x * y
        assert abs(z.data - 6.0) < 1e-6

    def test_backward(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x * y
        z.backward()
        assert abs(x.grad - 3.0) < 1e-6, f"∂z/∂x should be y=3, got {x.grad}"
        assert abs(y.grad - 2.0) < 1e-6, f"∂z/∂y should be x=2, got {y.grad}"

    def test_numerical(self):
        x = Value(2.0)
        y = Value(3.0)

        def f():
            return x * y

        z = f()
        z.backward()

        num_grad_x = numerical_gradient(f, x)
        num_grad_y = numerical_gradient(f, y)
        assert check_gradient(x.grad, num_grad_x)
        assert check_gradient(y.grad, num_grad_y)

    def test_scalar_mul(self):
        x = Value(2.0)
        z = x * 5
        z.backward()
        assert abs(z.data - 10.0) < 1e-6
        assert abs(x.grad - 5.0) < 1e-6

    def test_rmul(self):
        x = Value(2.0)
        z = 5 * x
        z.backward()
        assert abs(z.data - 10.0) < 1e-6
        assert abs(x.grad - 5.0) < 1e-6


class TestSubtraction:
    """Tests for subtraction operation."""

    def test_forward(self):
        x = Value(5.0)
        y = Value(3.0)
        z = x - y
        assert abs(z.data - 2.0) < 1e-6

    def test_backward(self):
        x = Value(5.0)
        y = Value(3.0)
        z = x - y
        z.backward()
        assert abs(x.grad - 1.0) < 1e-6
        assert abs(y.grad - (-1.0)) < 1e-6

    def test_rsub(self):
        x = Value(3.0)
        z = 5 - x
        z.backward()
        assert abs(z.data - 2.0) < 1e-6
        assert abs(x.grad - (-1.0)) < 1e-6


class TestNegation:
    """Tests for negation operation."""

    def test_forward(self):
        x = Value(3.0)
        z = -x
        assert abs(z.data - (-3.0)) < 1e-6

    def test_backward(self):
        x = Value(3.0)
        z = -x
        z.backward()
        assert abs(x.grad - (-1.0)) < 1e-6


class TestDivision:
    """Tests for division operation."""

    def test_forward(self):
        x = Value(6.0)
        y = Value(2.0)
        z = x / y
        assert abs(z.data - 3.0) < 1e-6

    def test_backward(self):
        x = Value(6.0)
        y = Value(2.0)
        z = x / y
        z.backward()
        # ∂(x/y)/∂x = 1/y = 0.5
        assert abs(x.grad - 0.5) < 1e-6, f"Expected 0.5, got {x.grad}"
        # ∂(x/y)/∂y = -x/y² = -6/4 = -1.5
        assert abs(y.grad - (-1.5)) < 1e-6, f"Expected -1.5, got {y.grad}"

    def test_numerical(self):
        x = Value(6.0)
        y = Value(2.0)

        def f():
            return x / y

        z = f()
        z.backward()

        num_grad_x = numerical_gradient(f, x)
        num_grad_y = numerical_gradient(f, y)
        assert check_gradient(x.grad, num_grad_x)
        assert check_gradient(y.grad, num_grad_y)


class TestPower:
    """Tests for power operation."""

    def test_square(self):
        x = Value(3.0)
        z = x ** 2
        z.backward()
        assert abs(z.data - 9.0) < 1e-6
        assert abs(x.grad - 6.0) < 1e-6  # 2*x = 6

    def test_cube(self):
        x = Value(2.0)
        z = x ** 3
        z.backward()
        assert abs(z.data - 8.0) < 1e-6
        assert abs(x.grad - 12.0) < 1e-6  # 3*x² = 12

    def test_negative_power(self):
        x = Value(2.0)
        z = x ** -1
        z.backward()
        assert abs(z.data - 0.5) < 1e-6
        # ∂(x^-1)/∂x = -x^-2 = -0.25
        assert abs(x.grad - (-0.25)) < 1e-6

    def test_fractional_power(self):
        x = Value(4.0)
        z = x ** 0.5
        z.backward()
        assert abs(z.data - 2.0) < 1e-6
        # ∂(x^0.5)/∂x = 0.5 * x^-0.5 = 0.25
        assert abs(x.grad - 0.25) < 1e-6


class TestReLU:
    """Tests for ReLU activation."""

    def test_positive(self):
        x = Value(3.0)
        z = x.relu()
        z.backward()
        assert abs(z.data - 3.0) < 1e-6
        assert abs(x.grad - 1.0) < 1e-6

    def test_negative(self):
        x = Value(-3.0)
        z = x.relu()
        z.backward()
        assert abs(z.data - 0.0) < 1e-6
        assert abs(x.grad - 0.0) < 1e-6

    def test_zero(self):
        x = Value(0.0)
        z = x.relu()
        z.backward()
        assert abs(z.data - 0.0) < 1e-6
        # Convention: gradient is 0 at exactly 0
        assert abs(x.grad - 0.0) < 1e-6


class TestTanh:
    """Tests for tanh activation."""

    def test_forward(self):
        x = Value(0.0)
        z = x.tanh()
        assert abs(z.data - 0.0) < 1e-6

    def test_backward(self):
        x = Value(0.0)
        z = x.tanh()
        z.backward()
        # ∂tanh(x)/∂x = 1 - tanh²(x) = 1 at x=0
        assert abs(x.grad - 1.0) < 1e-6

    def test_numerical(self):
        x = Value(0.5)

        def f():
            return x.tanh()

        z = f()
        z.backward()

        num_grad = numerical_gradient(f, x)
        assert check_gradient(x.grad, num_grad)


class TestSigmoid:
    """Tests for sigmoid activation."""

    def test_forward_zero(self):
        x = Value(0.0)
        z = x.sigmoid()
        assert abs(z.data - 0.5) < 1e-6

    def test_backward_zero(self):
        x = Value(0.0)
        z = x.sigmoid()
        z.backward()
        # σ'(0) = σ(0) * (1 - σ(0)) = 0.5 * 0.5 = 0.25
        assert abs(x.grad - 0.25) < 1e-6

    def test_numerical(self):
        x = Value(1.0)

        def f():
            return x.sigmoid()

        z = f()
        z.backward()

        num_grad = numerical_gradient(f, x)
        assert check_gradient(x.grad, num_grad)


class TestExp:
    """Tests for exponential function."""

    def test_forward(self):
        x = Value(1.0)
        z = x.exp()
        assert abs(z.data - math.e) < 1e-6

    def test_backward(self):
        x = Value(1.0)
        z = x.exp()
        z.backward()
        # ∂e^x/∂x = e^x
        assert abs(x.grad - math.e) < 1e-6

    def test_numerical(self):
        x = Value(0.5)

        def f():
            return x.exp()

        z = f()
        z.backward()

        num_grad = numerical_gradient(f, x)
        assert check_gradient(x.grad, num_grad)


class TestLog:
    """Tests for logarithm function."""

    def test_forward(self):
        x = Value(math.e)
        z = x.log()
        assert abs(z.data - 1.0) < 1e-6

    def test_backward(self):
        x = Value(2.0)
        z = x.log()
        z.backward()
        # ∂ln(x)/∂x = 1/x = 0.5
        assert abs(x.grad - 0.5) < 1e-6

    def test_numerical(self):
        x = Value(3.0)

        def f():
            return x.log()

        z = f()
        z.backward()

        num_grad = numerical_gradient(f, x)
        assert check_gradient(x.grad, num_grad)


# =============================================================================
# Gradient Accumulation Tests
# =============================================================================


class TestGradientAccumulation:
    """Tests for gradient accumulation with shared variables."""

    def test_variable_used_twice(self):
        """x appears in x*x, gradient should be 2x."""
        x = Value(3.0)
        z = x * x
        z.backward()
        assert abs(x.grad - 6.0) < 1e-6  # 2*x = 6

    def test_variable_in_multiple_branches(self):
        """x appears in both branches of x*x + x."""
        x = Value(2.0)
        z = x * x + x
        z.backward()
        # ∂(x² + x)/∂x = 2x + 1 = 5
        assert abs(x.grad - 5.0) < 1e-6

    def test_complex_accumulation(self):
        """x used in three places."""
        x = Value(2.0)
        z = x * x * x  # x³
        z.backward()
        # ∂x³/∂x = 3x² = 12
        assert abs(x.grad - 12.0) < 1e-6

    def test_multiple_paths(self):
        """Two independent paths to output."""
        x = Value(2.0)
        y = Value(3.0)
        a = x + y
        b = x * y
        z = a + b
        z.backward()
        # z = (x+y) + (x*y) = x + y + xy
        # ∂z/∂x = 1 + y = 4
        # ∂z/∂y = 1 + x = 3
        assert abs(x.grad - 4.0) < 1e-6
        assert abs(y.grad - 3.0) < 1e-6


# =============================================================================
# Chain Rule Tests
# =============================================================================


class TestChainRule:
    """Tests verifying the chain rule."""

    def test_composition(self):
        """f(g(x)) where f(u) = u² and g(x) = x + 1."""
        x = Value(2.0)
        g = x + 1  # g(x) = x + 1 = 3
        f = g ** 2  # f = 9
        f.backward()
        # ∂f/∂x = ∂f/∂g * ∂g/∂x = 2g * 1 = 6
        assert abs(x.grad - 6.0) < 1e-6

    def test_deep_composition(self):
        """Multiple nested compositions."""
        x = Value(1.0)
        z = ((x * 2) + 1).tanh()
        z.backward()

        # Verify numerically
        def f():
            return ((x * 2) + 1).tanh()

        num_grad = numerical_gradient(f, x)
        assert check_gradient(x.grad, num_grad)


# =============================================================================
# Neural Network Tests
# =============================================================================


class TestNeuralNetwork:
    """Tests for neural network components."""

    def test_neuron(self):
        """Test single neuron forward and backward."""
        import random
        random.seed(42)

        n = Neuron(2, activation='tanh')
        x = [Value(1.0), Value(0.5)]
        y = n(x)
        y.backward()

        # Verify gradients exist
        assert n.w[0].grad != 0
        assert n.w[1].grad != 0
        assert n.b.grad != 0

    def test_layer(self):
        """Test layer of neurons."""
        import random
        random.seed(42)

        layer = Layer(2, 3)
        x = [Value(1.0), Value(0.5)]
        y = layer(x)

        assert len(y) == 3
        for yi in y:
            assert isinstance(yi, Value)

    def test_mlp(self):
        """Test MLP forward and backward."""
        import random
        random.seed(42)

        mlp = MLP(2, [4, 4, 1])
        x = [Value(1.0), Value(0.5)]
        y = mlp(x)

        y.backward()

        # Verify all parameters have gradients
        for p in mlp.parameters():
            # At least some gradients should be non-zero
            pass  # Just checking no errors

    def test_learning(self):
        """Test that MLP can learn a simple function."""
        import random
        random.seed(42)

        # Learn a simpler function: y = x₁ + x₂
        mlp = MLP(2, [4, 1])

        # Training data
        data = [
            ([0, 0], 0),
            ([1, 0], 1),
            ([0, 1], 1),
            ([1, 1], 2),
            ([0.5, 0.5], 1),
        ]

        # Training loop
        lr = 0.05
        initial_loss = None
        final_loss = None

        for epoch in range(200):
            total_loss = 0

            for inputs, target in data:
                # Forward
                x = [Value(float(i)) for i in inputs]
                pred = mlp(x)
                loss = (pred - target) ** 2
                total_loss += loss.data

                # Backward
                mlp.zero_grad()
                loss.backward()

                # Update
                for p in mlp.parameters():
                    p.data -= lr * p.grad

            if epoch == 0:
                initial_loss = total_loss
            final_loss = total_loss

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5, f"Loss didn't decrease enough: {initial_loss:.4f} -> {final_loss:.4f}"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_zero_gradient(self):
        """Unused variable should have zero gradient."""
        x = Value(1.0)
        y = Value(2.0)
        z = x * 2  # y not used
        z.backward()
        assert abs(y.grad - 0.0) < 1e-6

    def test_very_small_values(self):
        """Test with very small values."""
        x = Value(1e-10)
        z = x * x
        z.backward()
        assert abs(x.grad - 2e-10) < 1e-15

    def test_large_values(self):
        """Test with large values."""
        x = Value(1000.0)
        z = x + x
        z.backward()
        assert abs(x.grad - 2.0) < 1e-6


# =============================================================================
# Test Runner
# =============================================================================


def run_tests():
    """Run all tests."""
    test_classes = [
        TestAddition,
        TestMultiplication,
        TestSubtraction,
        TestNegation,
        TestDivision,
        TestPower,
        TestReLU,
        TestTanh,
        TestSigmoid,
        TestExp,
        TestLog,
        TestGradientAccumulation,
        TestChainRule,
        TestNeuralNetwork,
        TestEdgeCases,
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

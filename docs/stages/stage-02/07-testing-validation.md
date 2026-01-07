# Section 2.7: Testing and Validating Your Autograd

Building an autograd system is one thing. *Trusting* it is another.

Gradient bugs are among the most insidious in machine learning. Your code runs, produces numbers, and even seems to learn—but if gradients are slightly wrong, training fails in mysterious ways or converges to suboptimal solutions.

**This section covers rigorous testing strategies to ensure your autograd is correct.**

## The Fundamental Test: Numerical Gradient Checking

The gold standard for testing gradients is comparison with numerical approximations.

### The Central Difference Formula

For any differentiable function f, the derivative can be approximated as:

$$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$


This is the **central difference** formula. It's more accurate than the forward difference (f(x+h) - f(x))/h because errors cancel.

### Why It Works

Taylor series expansion:

$$f(x + h) = f(x) + f'(x)h + \frac{f''(x)}{2}h^2 + O(h^3)$$

$$f(x - h) = f(x) - f'(x)h + \frac{f''(x)}{2}h^2 + O(h^3)$$


Subtracting:

$$f(x + h) - f(x - h) = 2f'(x)h + O(h^3)$$


So:

$$\frac{f(x + h) - f(x - h)}{2h} = f'(x) + O(h^2)$$


The error is O(h²), meaning it shrinks quadratically as h decreases.

### Implementation

```python
def numerical_gradient(f, x, h=1e-5):
    """Compute numerical gradient of f at x using central differences."""
    return (f(x + h) - f(x - h)) / (2 * h)
```

### Testing a Single Operation

```python
def test_multiply_gradient():
    """Test that multiplication gradient is correct."""
    x = Value(2.0)
    y = Value(3.0)

    # Compute analytical gradient
    z = x * y
    z.backward()

    # Compute numerical gradient for x
    def f_x(val):
        return val * y.data
    numerical_x = numerical_gradient(f_x, x.data)

    # Compute numerical gradient for y
    def f_y(val):
        return x.data * val
    numerical_y = numerical_gradient(f_y, y.data)

    # Compare
    assert abs(x.grad - numerical_x) < 1e-5, f"x.grad={x.grad}, numerical={numerical_x}"
    assert abs(y.grad - numerical_y) < 1e-5, f"y.grad={y.grad}, numerical={numerical_y}"

    print("Multiply gradient test passed!")
```

## Choosing the Right h

The step size h involves a tradeoff:

- **Too large**: Approximation error dominates
- **Too small**: Floating-point rounding error dominates

For float64 (standard Python floats), h ≈ 1e-5 to 1e-7 is usually good.

### Relative Error

For comparing gradients, use relative error to handle different magnitudes:

```python
def relative_error(computed, numerical):
    """Compute relative error between computed and numerical gradients."""
    numerator = abs(computed - numerical)
    denominator = max(abs(computed), abs(numerical), 1e-8)
    return numerator / denominator

def check_gradient(computed, numerical, tolerance=1e-5):
    """Check if computed gradient matches numerical approximation."""
    rel_err = relative_error(computed, numerical)
    return rel_err < tolerance
```

## Testing Complex Expressions

For expressions involving multiple operations, we test the entire composition:

```python
def test_complex_expression():
    """Test gradient through a complex expression."""
    def make_computation(x_val, y_val):
        x = Value(x_val)
        y = Value(y_val)
        z = (x * y + x) ** 2 - y.tanh()
        return z, x, y

    # Get analytical gradients
    z, x, y = make_computation(2.0, 3.0)
    z.backward()

    # Numerical gradient for x
    def f_x(val):
        z, _, _ = make_computation(val, 3.0)
        return z.data
    numerical_x = numerical_gradient(f_x, 2.0)

    # Numerical gradient for y
    def f_y(val):
        z, _, _ = make_computation(2.0, val)
        return z.data
    numerical_y = numerical_gradient(f_y, 3.0)

    # Compare
    assert check_gradient(x.grad, numerical_x), \
        f"x gradient mismatch: {x.grad} vs {numerical_x}"
    assert check_gradient(y.grad, numerical_y), \
        f"y gradient mismatch: {y.grad} vs {numerical_y}"

    print("Complex expression test passed!")
```

## Comprehensive Unit Tests

Every operation needs its own test:

```python
import unittest

class TestAutograd(unittest.TestCase):

    def test_add(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x + y
        z.backward()

        self.assertAlmostEqual(z.data, 5.0)
        self.assertAlmostEqual(x.grad, 1.0)
        self.assertAlmostEqual(y.grad, 1.0)

    def test_multiply(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x * y
        z.backward()

        self.assertAlmostEqual(z.data, 6.0)
        self.assertAlmostEqual(x.grad, 3.0)  # dz/dx = y
        self.assertAlmostEqual(y.grad, 2.0)  # dz/dy = x

    def test_power(self):
        x = Value(3.0)
        z = x ** 2
        z.backward()

        self.assertAlmostEqual(z.data, 9.0)
        self.assertAlmostEqual(x.grad, 6.0)  # dz/dx = 2x

    def test_division(self):
        x = Value(6.0)
        y = Value(2.0)
        z = x / y
        z.backward()

        self.assertAlmostEqual(z.data, 3.0)
        self.assertAlmostEqual(x.grad, 0.5)    # dz/dx = 1/y
        self.assertAlmostEqual(y.grad, -1.5)   # dz/dy = -x/y²

    def test_relu_positive(self):
        x = Value(3.0)
        z = x.relu()
        z.backward()

        self.assertAlmostEqual(z.data, 3.0)
        self.assertAlmostEqual(x.grad, 1.0)

    def test_relu_negative(self):
        x = Value(-3.0)
        z = x.relu()
        z.backward()

        self.assertAlmostEqual(z.data, 0.0)
        self.assertAlmostEqual(x.grad, 0.0)

    def test_tanh(self):
        import math
        x = Value(1.0)
        z = x.tanh()
        z.backward()

        expected_data = math.tanh(1.0)
        expected_grad = 1 - expected_data ** 2

        self.assertAlmostEqual(z.data, expected_data)
        self.assertAlmostEqual(x.grad, expected_grad)

    def test_exp(self):
        import math
        x = Value(2.0)
        z = x.exp()
        z.backward()

        expected = math.exp(2.0)
        self.assertAlmostEqual(z.data, expected)
        self.assertAlmostEqual(x.grad, expected)

    def test_log(self):
        import math
        x = Value(2.0)
        z = x.log()
        z.backward()

        self.assertAlmostEqual(z.data, math.log(2.0))
        self.assertAlmostEqual(x.grad, 0.5)  # d/dx(ln x) = 1/x

if __name__ == '__main__':
    unittest.main()
```

## Testing Gradient Accumulation

A critical test: when a variable is used multiple times, its gradient should accumulate:

```python
def test_gradient_accumulation():
    """Verify gradients accumulate when variable used multiple times."""
    x = Value(3.0)
    y = x + x  # x used twice

    y.backward()

    assert x.grad == 2.0, f"Expected 2.0, got {x.grad}"
    print("Gradient accumulation test passed!")

def test_multiple_paths():
    """Verify gradients sum over multiple paths."""
    x = Value(2.0)
    y = Value(3.0)

    a = x * y  # path 1: x contributes via multiplication
    b = x + y  # path 2: x contributes via addition
    z = a + b

    z.backward()

    # dz/dx = da/dx + db/dx = y + 1 = 3 + 1 = 4
    assert x.grad == 4.0, f"Expected 4.0, got {x.grad}"
    # dz/dy = da/dy + db/dy = x + 1 = 2 + 1 = 3
    assert y.grad == 3.0, f"Expected 3.0, got {y.grad}"

    print("Multiple paths test passed!")
```

## Property-Based Testing

Instead of testing specific values, test properties that should always hold:

### Property 1: Chain Rule Composition

For any f and g, (f∘g)'(x) = f'(g(x)) · g'(x)

```python
import random

def test_chain_rule_property():
    """Test that chain rule holds for composed operations."""
    for _ in range(100):  # Random testing
        x_val = random.uniform(-5, 5)
        if abs(x_val) < 0.1:  # Avoid near-zero for log/division
            continue

        x = Value(x_val)

        # Compose: tanh(x²)
        z = (x ** 2).tanh()
        z.backward()

        # Numerical check
        def f(v):
            return math.tanh(v ** 2)
        numerical = numerical_gradient(f, x_val)

        assert check_gradient(x.grad, numerical), \
            f"Chain rule failed at x={x_val}: {x.grad} vs {numerical}"

    print("Chain rule property test passed!")
```

### Property 2: Linearity of Gradient

∂(af + bg)/∂x = a·∂f/∂x + b·∂g/∂x

```python
def test_linearity_property():
    """Test that gradient is linear in the output."""
    for _ in range(100):
        x_val = random.uniform(-5, 5)
        a = random.uniform(-3, 3)
        b = random.uniform(-3, 3)

        # Compute gradients separately
        x1 = Value(x_val)
        f = x1 ** 2
        f.backward()
        grad_f = x1.grad

        x2 = Value(x_val)
        g = x2 ** 3
        g.backward()
        grad_g = x2.grad

        # Compute gradient of linear combination
        x3 = Value(x_val)
        h = a * (x3 ** 2) + b * (x3 ** 3)
        h.backward()
        grad_h = x3.grad

        # Should match
        expected = a * grad_f + b * grad_g
        assert check_gradient(grad_h, expected), \
            f"Linearity failed: {grad_h} vs {expected}"

    print("Linearity property test passed!")
```

### Property 3: Zero Gradient at Stationary Points

Where f'(x) = 0, our autograd should give 0:

```python
def test_stationary_points():
    """Test gradient is zero at known stationary points."""
    # x² has stationary point at x=0
    x = Value(0.0)
    z = x ** 2
    z.backward()
    assert abs(x.grad) < 1e-10, f"Expected 0 gradient at x=0, got {x.grad}"

    # sin(x) has stationary points at x = π/2, 3π/2, ...
    x = Value(math.pi / 2)
    z = Value(math.sin(x.data))  # Note: need to implement sin properly
    # For now, we can test tanh which has gradient → 0 for large |x|
    x = Value(10.0)
    z = x.tanh()
    z.backward()
    assert abs(x.grad) < 0.01, f"tanh gradient should be ~0 for large x, got {x.grad}"

    print("Stationary points test passed!")
```

## Comparison with PyTorch

The ultimate validation: compare with a production framework.

```python
def test_against_pytorch():
    """Compare our gradients with PyTorch's."""
    import torch

    # Test case: complex expression
    x_val, y_val = 2.0, 3.0

    # Our implementation
    x = Value(x_val)
    y = Value(y_val)
    z = (x * y + x ** 2).tanh() + y.exp()
    z.backward()
    our_x_grad = x.grad
    our_y_grad = y.grad

    # PyTorch
    x_torch = torch.tensor(x_val, requires_grad=True)
    y_torch = torch.tensor(y_val, requires_grad=True)
    z_torch = torch.tanh(x_torch * y_torch + x_torch ** 2) + torch.exp(y_torch)
    z_torch.backward()
    torch_x_grad = x_torch.grad.item()
    torch_y_grad = y_torch.grad.item()

    # Compare
    assert check_gradient(our_x_grad, torch_x_grad), \
        f"x gradient mismatch: ours={our_x_grad}, torch={torch_x_grad}"
    assert check_gradient(our_y_grad, torch_y_grad), \
        f"y gradient mismatch: ours={our_y_grad}, torch={torch_y_grad}"

    print("PyTorch comparison test passed!")
```

## Testing Neural Networks

For neural networks, we need to test that training actually works:

```python
def test_learning():
    """Test that a network can learn a simple function."""
    import random
    random.seed(42)

    # Create simple dataset: y = x²
    X = [[x] for x in [0.0, 0.5, 1.0, 1.5, 2.0]]
    y = [x[0] ** 2 for x in X]

    # Simple network: 1 -> 8 -> 1
    model = MLP(1, [8, 1])

    # Initial loss
    initial_predictions = [model(x) for x in X]
    initial_loss = sum((pred - target) ** 2
                       for pred, target in zip(initial_predictions, y))

    # Train
    for _ in range(200):
        predictions = [model(x) for x in X]
        loss = sum((pred - target) ** 2 for pred, target in zip(predictions, y))

        for p in model.parameters():
            p.grad = 0.0
        loss.backward()

        for p in model.parameters():
            p.data -= 0.01 * p.grad

    # Final loss should be much lower
    final_predictions = [model(x) for x in X]
    final_loss = sum((pred - target) ** 2
                     for pred, target in zip(final_predictions, y))

    assert final_loss.data < initial_loss.data * 0.1, \
        f"Training didn't reduce loss enough: {initial_loss.data} -> {final_loss.data}"

    print("Learning test passed!")
```

## Edge Cases and Corner Cases

Don't forget boundary conditions:

```python
def test_edge_cases():
    """Test edge cases that might break."""

    # Division by small number
    x = Value(1.0)
    y = Value(1e-8)
    z = x / y
    z.backward()
    assert not math.isnan(x.grad), "Gradient became NaN"
    assert not math.isinf(x.grad), "Gradient became infinite"

    # ReLU at zero (subgradient)
    x = Value(0.0)
    z = x.relu()
    z.backward()
    assert x.grad == 0.0, "ReLU gradient at 0 should be 0"

    # Very deep composition (test for stack overflow)
    x = Value(1.0)
    z = x
    for _ in range(100):
        z = z * Value(0.99) + Value(0.001)
    z.backward()
    assert not math.isnan(x.grad), "Deep composition produced NaN"

    # Zero gradient when output doesn't depend on input
    x = Value(2.0)
    y = Value(3.0)
    z = y ** 2  # z doesn't depend on x
    z.backward()
    assert x.grad == 0.0, "Gradient should be 0 for unused variable"

    print("Edge cases test passed!")
```

## Debugging Gradient Issues

When tests fail, here's how to debug:

### 1. Print the Graph

```python
def debug_print(v, depth=0):
    """Print the computation graph for debugging."""
    indent = "  " * depth
    print(f"{indent}{v._op or 'input'}: data={v.data:.4f}, grad={v.grad:.4f}")
    for p in v._parents:
        debug_print(p, depth + 1)
```

### 2. Check Intermediate Gradients

```python
def test_with_intermediate_checks():
    x = Value(2.0)
    y = Value(3.0)

    a = x * y
    print(f"a = x*y = {a.data}")

    b = a + x
    print(f"b = a+x = {b.data}")

    b.backward()

    print(f"b.grad = {b.grad} (should be 1)")
    print(f"a.grad = {a.grad} (should be 1)")
    print(f"x.grad = {x.grad} (should be y+1 = 4)")
    print(f"y.grad = {y.grad} (should be x = 2)")
```

### 3. Isolate the Failing Operation

If a complex test fails, binary search to find which operation is wrong:

```python
# Test each operation in isolation
test_add()
test_multiply()
test_power()
# etc.

# Then test pairs
test_add_then_multiply()
test_multiply_then_add()
# etc.
```

## Summary

| Testing Strategy | What It Catches |
|-----------------|-----------------|
| Numerical gradient check | Wrong gradient formulas |
| Unit tests | Individual operation bugs |
| Accumulation tests | += vs = bugs |
| Property tests | Systematic errors |
| PyTorch comparison | Complex interaction bugs |
| Edge case tests | Numerical instability |
| Learning tests | End-to-end failures |

**Key insights**:

1. **Trust but verify**: Always compare with numerical gradients
2. **Test the edges**: Zero, very large, very small values
3. **Test compositions**: Individual ops can be correct but compose wrongly
4. **Test learning**: The ultimate test is whether training works

## A Complete Test Suite

```python
def run_all_tests():
    """Run complete autograd test suite."""
    print("Running autograd test suite...\n")

    # Unit tests
    test_multiply_gradient()
    test_gradient_accumulation()
    test_multiple_paths()
    test_complex_expression()

    # Property tests
    test_chain_rule_property()
    test_linearity_property()
    test_stationary_points()

    # Edge cases
    test_edge_cases()

    # Learning test
    test_learning()

    # PyTorch comparison (if available)
    try:
        test_against_pytorch()
    except ImportError:
        print("Skipping PyTorch comparison (not installed)")

    print("\n✓ All tests passed!")

if __name__ == '__main__':
    run_all_tests()
```

## Exercises

1. **Find the bug**: This backward function for multiply has a subtle bug. Find it:
   ```python
   def _backward():
       self.grad = out.grad * other.data  # Bug!
       other.grad = out.grad * self.data
   ```

2. **Test coverage**: Write a test that would catch the bug in exercise 1.

3. **Numerical stability**: For what range of x values does `x.exp()` give reasonable gradients? Write a test to find the limits.

4. **Random testing**: Implement a function that generates random computational graphs and tests them automatically.

5. **Gradient checking function**: Write a general-purpose `check_gradients(f, inputs)` function that numerically verifies all gradients.

## What's Next

We now have:

- ✓ Mathematical foundations (Sections 2.1-2.3)
- ✓ Computational graphs (Section 2.4)
- ✓ Forward vs reverse mode (Section 2.5)
- ✓ Working implementation (Section 2.6)
- ✓ Testing framework (Section 2.7)

In Stage 3, we'll build on this foundation to implement our first complete language model—using the autograd system we just built to train it from scratch.

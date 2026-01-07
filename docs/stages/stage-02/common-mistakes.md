# Stage 2: Common Mistakes

## Mistake 1: Forgetting to Zero Gradients

**Symptom**: Gradients keep growing, training becomes unstable

**Wrong code**:
```python
for batch in data:
    loss = model.forward(batch)
    loss.backward()
    optimizer.step()
    # Gradients accumulate across batches!
```

**The fix**:
```python
for batch in data:
    model.zero_grad()  # Clear previous gradients
    loss = model.forward(batch)
    loss.backward()
    optimizer.step()
```

---

## Mistake 2: In-Place Operations

**Symptom**: Autograd fails with cryptic errors

**Wrong code**:
```python
x = x + 1  # OK
x += 1     # In-place, breaks gradient tracking!
```

**The fix**: Avoid in-place operations on tensors that need gradients
```python
x = x + 1  # Creates new tensor, gradient tracking preserved
```

---

## Mistake 3: Gradient Through Integer Operations

**Symptom**: Gradients are None or zero unexpectedly

**Example**:
```python
index = int(x)  # Rounding destroys gradient
y = array[index]  # No gradient flows back!
```

**The fix**: Use soft indexing or other differentiable alternatives

---

## Mistake 4: Numerical Instability in Softmax

**Wrong code**:
```python
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
# When x has large values, exp(x) overflows!
```

**The fix**: Subtract max for numerical stability
```python
def softmax(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)  # Now exp() input is ≤ 0
    return exp_x / np.sum(exp_x)
```

---

## Mistake 5: Wrong Chain Rule Order

**Wrong thinking**:
```
d/dx f(g(x)) = f'(x) * g'(x)  # WRONG!
```

**Correct**:
```
d/dx f(g(x)) = f'(g(x)) * g'(x)  # Outer derivative evaluated at inner function
```

**Example**:
```python
# f(x) = sin(x²)
# f'(x) = cos(x²) * 2x  ← Note cos(x²), not cos(x)!
```

---

## Mistake 6: Scalar vs Element-wise Gradients

**Symptom**: Shape mismatch in gradient computation

**Example**: For y = W @ x where W is (M, N) and x is (N,):
```python
# grad_W should be (M, N), not (M,) or (N,)
grad_W = np.outer(grad_y, x)  # Correct: (M,) × (N,) → (M, N)
```

---

## Mistake 7: Forgetting to Cache Forward Values

**Symptom**: Cannot compute backward pass

**Wrong code**:
```python
class MyLayer:
    def forward(self, x):
        return self.w * x  # Didn't save x!

    def backward(self, grad):
        # Need x to compute grad_w, but it's gone!
        self.grad_w = grad * ???
```

**The fix**: Cache everything needed for backward
```python
class MyLayer:
    def forward(self, x):
        self.x = x  # Cache input
        return self.w * x

    def backward(self, grad):
        self.grad_w = grad * self.x  # Now we have x
        return grad * self.w
```

---

## Mistake 8: Gradient Checking Tolerance

**Symptom**: Gradient check fails even when correct

**Problem**: Using too tight tolerance
```python
assert abs(analytical - numerical) < 1e-10  # Too strict!
```

**The fix**: Use relative error with reasonable tolerance
```python
relative_error = abs(analytical - numerical) / (abs(analytical) + 1e-8)
assert relative_error < 1e-5  # More reasonable
```

---

## Mistake 9: Broadcasting Errors in Backward

**Symptom**: Gradient shape doesn't match parameter shape

**Example**: Bias addition with broadcasting
```python
# Forward: y = x + b where x is (batch, dim) and b is (dim,)
# Wrong backward:
grad_b = grad_y  # Shape (batch, dim), but b is (dim,)!

# Correct:
grad_b = np.sum(grad_y, axis=0)  # Sum over batch dimension
```

---

## Mistake 10: Not Understanding Computational Graph

**Symptom**: Confused about which gradients to compute

**Key insight**: The computational graph determines gradient flow.

```
x → [multiply by 2] → y → [add 3] → z

If you want ∂z/∂x, you must:
1. Compute ∂z/∂y = 1 (from add)
2. Compute ∂y/∂x = 2 (from multiply)
3. Chain: ∂z/∂x = ∂z/∂y * ∂y/∂x = 1 * 2 = 2
```

**The fix**: Draw the graph, then backpropagate step by step.

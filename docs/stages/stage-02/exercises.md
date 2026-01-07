# Stage 2 Exercises

## Conceptual Questions

### Exercise 2.1: Gradient Intuition
For f(x) = x², we know f'(x) = 2x.

**a)** At x = 3, which direction decreases f(x)?
**b)** If we take a step of size 0.1 in the negative gradient direction, what is the new x?
**c)** What is the new f(x)? Is it smaller?

### Exercise 2.2: Chain Rule by Hand
Compute the derivative of f(x) = sin(x²) step by step.

**a)** Identify the outer function and inner function
**b)** Apply the chain rule
**c)** Verify using the computational graph approach

### Exercise 2.3: Graph Topology
Draw the computational graph for:
```
z = (a + b) * (a - b)
```

**a)** How many nodes? How many edges?
**b)** What are the intermediate values for a=3, b=2?
**c)** Compute ∂z/∂a and ∂z/∂b using backpropagation

### Exercise 2.4: Forward vs Reverse Mode
For f: R^n → R^m:

**a)** When is forward mode more efficient?
**b)** When is reverse mode more efficient?
**c)** Why does deep learning always use reverse mode?

---

## Implementation Exercises

### Exercise 2.5: Implement Division
Add division to the autograd system:

```python
class Div(Op):
    """z = a / b"""

    def forward(self, a: float, b: float) -> float:
        # TODO: Implement
        pass

    def backward(self, grad_output: float) -> Tuple[float, float]:
        # d(a/b)/da = 1/b
        # d(a/b)/db = -a/b²
        # TODO: Implement
        pass
```

**Test**: Verify gradients numerically for a=6, b=2.

### Exercise 2.6: Implement Power
Add the power operation x^n:

```python
class Pow(Op):
    """z = a^n where n is a constant"""

    def __init__(self, n: float):
        self.n = n

    def forward(self, a: float) -> float:
        # TODO
        pass

    def backward(self, grad_output: float) -> float:
        # d(a^n)/da = n * a^(n-1)
        # TODO
        pass
```

### Exercise 2.7: Sigmoid Gradient
Implement sigmoid and its gradient:

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(x, grad_output):
    # Hint: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    # TODO
    pass
```

### Exercise 2.8: Numerical Gradient Check
Implement a function to verify gradients numerically:

```python
def check_gradient(f, x, epsilon=1e-5):
    """
    Compare analytical gradient to numerical approximation.

    Numerical gradient: (f(x+ε) - f(x-ε)) / (2ε)
    """
    # TODO: Implement
    # Return True if analytical and numerical match within tolerance
    pass
```

---

## Challenge Exercises

### Exercise 2.9: Matrix Gradient
For matrix multiplication Y = XW where X is (N, D) and W is (D, M):

**a)** What is the shape of ∂L/∂W given ∂L/∂Y?
**b)** Derive the formula for ∂L/∂X
**c)** Implement both gradients

```python
def matmul_backward(X, W, grad_Y):
    """
    Returns: (grad_X, grad_W)
    """
    # TODO
    pass
```

### Exercise 2.10: Build a Simple Neural Network
Using your autograd system, build a 2-layer neural network:

```python
def forward(x, W1, b1, W2, b2):
    """
    z1 = x @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    return z2
    """
    pass

def backward(x, W1, b1, W2, b2, grad_output):
    """
    Return gradients for all parameters.
    """
    pass
```

### Exercise 2.11: Gradient Flow Analysis
Consider a 10-layer network where each layer multiplies by 2.

**a)** If the final gradient is 1, what is the gradient at layer 1?
**b)** What if each layer multiplies by 0.5?
**c)** This demonstrates vanishing/exploding gradients. Propose a solution.

---

## Checking Your Work

- **Test suite**: See `code/stage-02/tests/test_value.py` for expected behavior
- **Reference implementation**: Compare with `code/stage-02/value.py`
- **Self-check**: Use numerical gradient checking to verify your derivatives
---

## Mini-Project: Autograd Engine

Build a complete automatic differentiation engine that can train a small neural network.

### Requirements

1. **Value class**: Implement forward and backward for +, *, -, /, **
2. **Activations**: Add tanh, relu, and sigmoid with proper gradients
3. **Training**: Train a 2-layer MLP to learn XOR

### Deliverables

- [ ] Value class with all basic operations
- [ ] Gradient checking (numerical vs. autograd)
- [ ] XOR network that converges to <0.01 loss
- [ ] Visualization of the computational graph (optional)

### Extension

Add support for matrix operations (matmul, sum, mean). Can you train a simple image classifier?

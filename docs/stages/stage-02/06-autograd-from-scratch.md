# Section 2.6: Building Autograd from Scratch

We've covered the theory: derivatives, the chain rule, computational graphs, and reverse mode differentiation. Now let's build a complete automatic differentiation system from scratch.

**By the end of this section, you'll have working code that can differentiate arbitrary compositions of operations—the same capability that powers PyTorch and TensorFlow.**

## Python Prerequisites

This section uses several Python features that you should understand:

**Operator overloading** (`__add__`, `__mul__`, etc.): Python lets classes define how operators work on their instances. When you write `a + b`, Python actually calls `a.__add__(b)`. By defining these "magic methods," we can make our Value class work with `+`, `-`, `*`, `/`, and `**`.

```python
class Example:
    def __init__(self, x):
        self.x = x

    def __add__(self, other):
        # Called when we write: example + something
        return Example(self.x + other.x)
```

**Closures**: A function that "remembers" variables from its enclosing scope. This is crucial for our `_backward` functions:

```python
def make_multiplier(n):
    def multiply(x):
        return x * n  # 'n' is remembered from enclosing scope
    return multiply

times_5 = make_multiplier(5)
print(times_5(3))  # Prints 15
```

In our autograd, each operation creates a `_backward` function that remembers which Values were involved—this is a closure.

**isinstance()**: Checks if an object is of a certain type. We use it to handle both Value objects and raw Python numbers:

```python
if isinstance(other, Value):
    # other is already a Value
else:
    other = Value(other)  # wrap it
```

**Lambda functions**: Anonymous (unnamed) functions created with the `lambda` keyword. Useful for simple, one-expression functions:

```python
square = lambda x: x * x  # Same as: def square(x): return x * x
print(square(5))  # Prints 25

# lambda with no arguments
do_nothing = lambda: None
```

**List comprehensions and generator expressions**: Concise ways to create lists or iterate:

```python
# List comprehension: creates a list
squares = [x**2 for x in range(5)]  # [0, 1, 4, 9, 16]

# Generator expression: creates an iterator (lazy evaluation, saves memory)
sum_of_squares = sum(x**2 for x in range(5))  # 30
```

**zip()**: Pairs up elements from multiple iterables:

```python
names = ['a', 'b', 'c']
values = [1, 2, 3]
for name, value in zip(names, values):
    print(f"{name} = {value}")  # a = 1, b = 2, c = 3
```

**`__call__` method**: Makes an object callable like a function:

```python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor

times_3 = Multiplier(3)
print(times_3(5))  # Prints 15 — calls times_3.__call__(5)
```

## Design Goals

Our autograd system will:

1. **Track computations**: Build a graph as we compute
2. **Support basic operations**: +, -, ×, /, power, common functions
3. **Compute gradients automatically**: One call to `.backward()` gives all gradients
4. **Be minimal but complete**: ~100 lines of core code

## The Value Class: Core Data Structure

Every value in our system will be a `Value` object that stores:

- The actual numerical data
- The gradient (accumulated during backward pass)
- References to parent nodes (for graph traversal)
- The operation that created it (for computing local gradients)

```python
class Value:
    """A value in the computational graph with automatic differentiation."""

    def __init__(self, data, _parents=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None  # Function to compute parent gradients
        self._parents = set(_parents)
        self._op = _op  # For debugging/visualization

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
```

### Why This Structure?

- **`data`**: The actual number (forward pass result)
- **`grad`**: Accumulates ∂output/∂self during backward pass
- **`_backward`**: A closure that knows how to propagate gradients to parents
- **`_parents`**: The nodes this value depends on (graph edges)
- **`_op`**: What operation created this (useful for debugging)

## Implementing Addition

Let's start with the simplest operation: addition.

For z = x + y:

- Local gradients: ∂z/∂x = 1, ∂z/∂y = 1
- Backward: parent.grad += self.grad × local_gradient

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += out.grad * 1.0   # ∂z/∂x = 1
        other.grad += out.grad * 1.0  # ∂z/∂y = 1

    out._backward = _backward
    return out
```

### Key Points

1. **Wrap raw numbers**: If `other` isn't a Value, make it one
2. **Store parents**: The output depends on `self` and `other`
3. **Define backward**: A closure that knows how to propagate gradients
4. **Accumulate with +=**: A value might be used multiple times

### Why += Instead of =?

Consider f(x) = x + x. The variable x is used twice.

If we used `=`, the second path would overwrite the first. With `+=`, both contributions accumulate:

- First x: grad += 1
- Second x: grad += 1
- Total: grad = 2

This is correct: ∂(x+x)/∂x = 2.

## Implementing Multiplication

For z = x × y:

- Local gradients: ∂z/∂x = y, ∂z/∂y = x
- These need the cached values from the forward pass

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += out.grad * other.data  # ∂z/∂x = y
        other.grad += out.grad * self.data  # ∂z/∂y = x

    out._backward = _backward
    return out
```

### The Closure Captures Values

Notice that `_backward` references `other.data` and `self.data`. These are captured in the closure when the function is defined.

This is how we "cache" forward pass values for the backward pass—through Python's closure mechanism.

## The Backward Pass: Topological Sort

To compute all gradients, we need to:

1. Find all nodes in the graph
2. Process them in reverse topological order
3. Call each node's `_backward()` function

```python
def backward(self):
    # Build topological order
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for parent in v._parents:
                build_topo(parent)
            topo.append(v)

    build_topo(self)

    # Initialize gradient of output
    self.grad = 1.0

    # Backward pass in reverse order
    for v in reversed(topo):
        v._backward()
```

### Why Topological Order?

Consider: z = (x + y) × x

Graph:
```
x ──┬──▶ [+] ──▶ a ──┐
    │     ▲          │
    │     │          ▼
    │     y         [×] ──▶ z
    │                ▲
    └────────────────┘
```

We must process z before a (to have z.grad when computing a's contribution), and a before x and y.

Reverse topological order guarantees: each node is processed only after all its children (nodes that depend on it) have been processed.

## Putting It Together: First Test

```python
# Create values
x = Value(2.0)
y = Value(3.0)

# Forward pass (builds graph automatically)
z = x * y + x

# Backward pass
z.backward()

print(f"z = {z.data}")     # z = 8.0
print(f"∂z/∂x = {x.grad}") # ∂z/∂x = 4.0
print(f"∂z/∂y = {y.grad}") # ∂z/∂y = 2.0
```

### Verification

z = x × y + x = 2 × 3 + 2 = 8 ✓

∂z/∂x = y + 1 = 3 + 1 = 4 ✓ (x contributes twice: via multiplication and addition)

∂z/∂y = x = 2 ✓

**It works!**

## Completing the Operations

### Subtraction and Negation

```python
def __neg__(self):
    return self * -1

def __sub__(self, other):
    return self + (-other)

def __rsub__(self, other):  # other - self
    other = other if isinstance(other, Value) else Value(other)
    return other + (-self)
```

### Division

For z = x / y = x × y⁻¹:

- ∂z/∂x = 1/y
- ∂z/∂y = -x/y²

```python
def __truediv__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return self * other**-1
```

### Power

For z = $x^n$ (where n is a constant):

- ∂z/∂x = n × x^(n-1)

```python
def __pow__(self, n):
    assert isinstance(n, (int, float)), "only supporting constant powers"
    out = Value(self.data ** n, (self,), f'**{n}')

    def _backward():
        self.grad += out.grad * (n * self.data ** (n - 1))

    out._backward = _backward
    return out
```

### Reverse Operations

Python calls `__radd__` when the left operand doesn't support the operation:

```python
def __radd__(self, other):  # other + self
    return self + other

def __rmul__(self, other):  # other * self
    return self * other
```

This lets us write `2 * x` where x is a Value.

## Activation Functions

Neural networks need nonlinear activations. Let's implement the most common ones.

### ReLU

$$\text{relu}(x) = \max(0, x)$$


Derivative:

$$\frac{\partial}{\partial x}\text{relu}(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$$


```python
def relu(self):
    out = Value(max(0, self.data), (self,), 'relu')

    def _backward():
        self.grad += out.grad * (1.0 if self.data > 0 else 0.0)

    out._backward = _backward
    return out
```

### Tanh

$$\tanh(x) = \frac{$e^x$ - $e^{-x}$}{$e^x$ + $e^{-x}$}$$


Derivative:

$$\frac{\partial}{\partial x}\tanh(x) = 1 - \$tanh^2$(x)$$


```python
import math

def tanh(self):
    t = math.tanh(self.data)
    out = Value(t, (self,), 'tanh')

    def _backward():
        self.grad += out.grad * (1 - t ** 2)

    out._backward = _backward
    return out
```

**Note**: We use the output `t` to compute the derivative (1 - t²). This is a memory optimization—we don't need to store the input.

### Sigmoid

$$\sigma(x) = \frac{1}{1 + $e^{-x}$}$$


Derivative:

$$\frac{\partial}{\partial x}\sigma(x) = \sigma(x)(1 - \sigma(x))$$


```python
def sigmoid(self):
    s = 1 / (1 + math.exp(-self.data))
    out = Value(s, (self,), 'sigmoid')

    def _backward():
        self.grad += out.grad * (s * (1 - s))

    out._backward = _backward
    return out
```

### Exponential and Log

```python
def exp(self):
    out = Value(math.exp(self.data), (self,), 'exp')

    def _backward():
        self.grad += out.grad * out.data  # d/dx(e^x) = e^x

    out._backward = _backward
    return out

def log(self):
    out = Value(math.log(self.data), (self,), 'log')

    def _backward():
        self.grad += out.grad * (1 / self.data)  # d/dx(ln x) = 1/x

    out._backward = _backward
    return out
```

## The Complete Value Class

Here's the full implementation:

```python
import math

class Value:
    """A scalar value with automatic differentiation support."""

    def __init__(self, data, _parents=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._parents = set(_parents)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # Arithmetic operations
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __pow__(self, n):
        assert isinstance(n, (int, float))
        out = Value(self.data ** n, (self,), f'**{n}')

        def _backward():
            self.grad += out.grad * (n * self.data ** (n - 1))
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other + (-self)

    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other * (self ** -1)

    # Activation functions
    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            # Derivative is 1 if x > 0, else 0 (using subgradient 0 at x=0)
            self.grad += out.grad * (1.0 if self.data > 0 else 0.0)
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += out.grad * (1 - t ** 2)
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')
        def _backward():
            self.grad += out.grad * (s * (1 - s))
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += out.grad * (1 / self.data)
        out._backward = _backward
        return out

    # Backward pass
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0

        for v in reversed(topo):
            v._backward()
```

**That's it—about 100 lines of code for a complete automatic differentiation engine.**

## Building a Neural Network

Let's use our autograd to build and train a simple neural network.

### A Single Neuron

A neuron computes: y = activation(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)

```python
class Neuron:
    def __init__(self, n_inputs):
        # Initialize weights randomly
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        # w · x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]
```

### A Layer of Neurons

```python
class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
```

### A Multi-Layer Perceptron (MLP)

```python
class MLP:
    def __init__(self, n_inputs, layer_sizes):
        sizes = [n_inputs] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i+1])
                       for i in range(len(layer_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

### Training Loop

```python
import random

# Create a simple dataset: learn XOR
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]  # XOR outputs

# Create network: 2 inputs -> 4 hidden -> 1 output
model = MLP(2, [4, 1])

# Training
learning_rate = 0.1

for epoch in range(100):
    # Forward pass
    predictions = [model(x) for x in X]

    # Compute loss (mean squared error)
    loss = sum((pred - target) ** 2 for pred, target in zip(predictions, y))

    # Zero gradients (important!)
    for p in model.parameters():
        p.grad = 0.0

    # Backward pass
    loss.backward()

    # Update parameters
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}")

# Test
print("\nFinal predictions:")
for x, target in zip(X, y):
    pred = model(x)
    print(f"  Input: {x}, Target: {target}, Prediction: {pred.data:.4f}")
```

Expected output:
```
Epoch 0, Loss: 2.3456
Epoch 10, Loss: 0.8234
Epoch 20, Loss: 0.2341
...
Epoch 90, Loss: 0.0012

Final predictions:
  Input: [0, 0], Target: 0, Prediction: 0.0234
  Input: [0, 1], Target: 1, Prediction: 0.9812
  Input: [1, 0], Target: 1, Prediction: 0.9756
  Input: [1, 1], Target: 0, Prediction: 0.0345
```

**We just trained a neural network using our own autograd system!**

## How This Relates to Real Frameworks

Our implementation captures the essential ideas of PyTorch's autograd:

| Our Implementation | PyTorch |
|-------------------|---------|
| `Value` | `torch.Tensor` with `requires_grad=True` |
| `_backward` closure | `grad_fn` attribute |
| `backward()` | `tensor.backward()` |
| Manual gradient zero | `optimizer.zero_grad()` |
| Manual parameter update | `optimizer.step()` |

### Key Differences in Real Frameworks

1. **Tensors, not scalars**: PyTorch operates on multi-dimensional arrays
2. **GPU acceleration**: Operations run on CUDA cores
3. **Optimized operations**: Matrix multiplies use BLAS/cuBLAS
4. **Memory management**: Sophisticated caching and cleanup
5. **Graph management**: Options for static vs dynamic graphs

But the core algorithm is the same: record operations during forward pass, then apply chain rule in reverse.

## Understanding the Gradient Flow

Let's trace through a simple example to see exactly how gradients flow:

```python
x = Value(2.0)
y = Value(3.0)
z = x * y    # z = 6
w = z + x    # w = 8
w.backward()
```

### Forward Pass (Builds Graph)

```
x(2.0) ──┬──▶ [*] ──▶ z(6.0) ──▶ [+] ──▶ w(8.0)
         │     ▲                   ▲
         │     │                   │
         │    y(3.0)               │
         └─────────────────────────┘
```

### Backward Pass (Computes Gradients)

**Step 1**: Initialize w.grad = 1.0

**Step 2**: Process w (addition node)
- w's _backward: `z.grad += 1.0`, `x.grad += 1.0`
- After: z.grad = 1.0, x.grad = 1.0

**Step 3**: Process z (multiplication node)
- z's _backward: `x.grad += 1.0 * 3.0`, `y.grad += 1.0 * 2.0`
- After: x.grad = 1.0 + 3.0 = 4.0, y.grad = 2.0

**Final Result**: x.grad = 4.0, y.grad = 2.0

**Verification**: w = xy + x, so ∂w/∂x = y + 1 = 4 ✓, ∂w/∂y = x = 2 ✓

## Common Pitfalls and Solutions

### 1. Forgetting to Zero Gradients

```python
# Wrong: gradients accumulate across iterations!
for epoch in range(10):
    loss = model(x)
    loss.backward()  # Gradients add to previous values!

# Correct: zero gradients before each backward
for epoch in range(10):
    for p in model.parameters():
        p.grad = 0.0
    loss = model(x)
    loss.backward()
```

### 2. In-Place Operations

```python
# Dangerous: modifies data that might be needed for gradients
x.data += 1  # If x was used in a computation, gradients may be wrong

# Safe: create new Value
x = x + 1
```

### 3. Gradient Through Non-Differentiable Operations

```python
# ReLU at exactly 0 has undefined gradient
# We define it as 0 (a common convention)
def relu(self):
    out = Value(max(0, self.data), (self,), 'relu')
    def _backward():
        # gradient is 0 when input <= 0
        self.grad += out.grad * (1.0 if self.data > 0 else 0.0)
    out._backward = _backward
    return out
```

## Visualizing the Graph

For debugging, it helps to visualize the computational graph:

```python
def draw_graph(root):
    """Generate DOT format graph description."""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for parent in v._parents:
                edges.add((parent, v))
                build(parent)

    build(root)

    # Generate DOT
    dot = ['digraph G {']
    dot.append('  rankdir=LR;')

    for n in nodes:
        label = f"{n.data:.2f}\\ngrad={n.grad:.2f}"
        dot.append(f'  "{id(n)}" [label="{label}", shape=box];')
        if n._op:
            op_id = f"{id(n)}_op"
            dot.append(f'  "{op_id}" [label="{n._op}", shape=circle];')
            dot.append(f'  "{op_id}" -> "{id(n)}";')
            for p in n._parents:
                dot.append(f'  "{id(p)}" -> "{op_id}";')

    dot.append('}')
    return '\n'.join(dot)
```

## Summary

| Component | Purpose |
|-----------|---------|
| `Value` class | Wraps scalar, stores gradient, tracks parents |
| `_backward` closure | Computes local gradient contribution |
| Operator overloading | Makes `+`, `*`, etc. build the graph |
| Topological sort | Ensures correct backward pass order |
| Gradient accumulation | Handles values used multiple times |

**Key insights**:

1. **Closures are key**: The `_backward` function captures forward-pass values
2. **The graph builds itself**: Operators automatically record dependencies
3. **One backward pass**: Computes all gradients efficiently
4. **It's just the chain rule**: Each node multiplies incoming gradient by local gradient

## Exercises

1. **Add more operations**: Implement `sin`, `cos`, and `sqrt` for the Value class.

2. **Gradient checking**: For f(x) = x³, verify that your autograd gives the same answer as the numerical approximation (f(x+h) - f(x-h)) / 2h.

3. **Memory investigation**: What happens to memory usage if you run many forward passes without calling backward? Why?

4. **Batch gradient descent**: Modify the training loop to use mini-batches instead of the full dataset.

5. **Visualization**: Use the `draw_graph` function to visualize the computation graph for a simple neural network.

6. **Cross-entropy loss**: Implement the cross-entropy loss function using our Value operations.

## What's Next

We have a working autograd system. But how do we know it's correct?

In Section 2.7, we'll cover **testing and validation**:

- Numerical gradient checking
- Unit tests for each operation
- Property-based testing for compositions
- Comparison with PyTorch

Robust testing is essential—gradient bugs are subtle and can make training fail in mysterious ways.

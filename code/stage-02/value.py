"""
Automatic Differentiation from Scratch

This module implements reverse-mode automatic differentiation (autograd),
the core mechanism that powers neural network training.

The implementation follows the computational graph approach:
1. Build a graph as we compute (forward pass)
2. Traverse the graph in reverse to compute gradients (backward pass)

Usage:
    from value import Value

    # Create values
    x = Value(2.0)
    y = Value(3.0)

    # Build computation
    z = x * y + x ** 2

    # Compute gradients
    z.backward()

    print(f"∂z/∂x = {x.grad}")  # 7.0 = y + 2*x = 3 + 4
    print(f"∂z/∂y = {y.grad}")  # 2.0 = x
"""

import math
from typing import Tuple, Set, Callable, Union


class Value:
    """
    A scalar value that tracks its computation history for automatic differentiation.

    Each Value stores:
    - data: The actual numerical value
    - grad: The gradient ∂L/∂self (accumulated during backward pass)
    - _backward: A function computing local gradients
    - _parents: Set of Values this was computed from
    - _op: String describing the operation (for debugging)

    The gradient is computed via reverse-mode autodiff:
    1. Forward pass builds the computation graph
    2. backward() traverses in reverse topological order
    3. Each node's _backward() propagates gradients to parents

    Attributes:
        data: The scalar value
        grad: Accumulated gradient (starts at 0, filled by backward())
    """

    def __init__(
        self,
        data: float,
        _parents: Tuple['Value', ...] = (),
        _op: str = ''
    ):
        """
        Create a Value.

        Args:
            data: The numerical value
            _parents: Tuple of parent Values (internal use)
            _op: Operation string for debugging (internal use)
        """
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._parents: Set['Value'] = set(_parents)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # =========================================================================
    # Arithmetic Operations
    # =========================================================================

    def __add__(self, other: Union['Value', float]) -> 'Value':
        """
        Addition: z = self + other

        Local gradients:
            ∂z/∂self = 1
            ∂z/∂other = 1

        The chain rule gives:
            ∂L/∂self += ∂L/∂z * 1 = ∂L/∂z
            ∂L/∂other += ∂L/∂z * 1 = ∂L/∂z
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0

        out._backward = _backward
        return out

    def __radd__(self, other: float) -> 'Value':
        """Handle float + Value by converting to Value + Value."""
        return self + other

    def __sub__(self, other: Union['Value', float]) -> 'Value':
        """
        Subtraction: z = self - other

        Implemented as self + (-other) to reuse existing operations.
        """
        return self + (-other)

    def __rsub__(self, other: float) -> 'Value':
        """Handle float - Value."""
        return Value(other) - self

    def __neg__(self) -> 'Value':
        """
        Negation: z = -self

        Local gradient: ∂z/∂self = -1
        """
        return self * -1

    def __mul__(self, other: Union['Value', float]) -> 'Value':
        """
        Multiplication: z = self * other

        Local gradients:
            ∂z/∂self = other
            ∂z/∂other = self

        The chain rule gives:
            ∂L/∂self += ∂L/∂z * other
            ∂L/∂other += ∂L/∂z * self
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __rmul__(self, other: float) -> 'Value':
        """Handle float * Value."""
        return self * other

    def __truediv__(self, other: Union['Value', float]) -> 'Value':
        """
        Division: z = self / other

        Implemented as self * other^(-1) to reuse power operation.
        """
        return self * (other ** -1)

    def __rtruediv__(self, other: float) -> 'Value':
        """Handle float / Value."""
        return Value(other) / self

    def __pow__(self, n: Union[int, float]) -> 'Value':
        """
        Power: z = self^n (where n is a constant)

        Local gradient: ∂z/∂self = n * self^(n-1)

        Note: n must be a Python number, not a Value.
        For Value^Value, use exp(other * log(self)).
        """
        assert isinstance(n, (int, float)), "Power exponent must be a number"
        out = Value(self.data ** n, (self,), f'**{n}')

        def _backward():
            # ∂L/∂self += ∂L/∂z * n * self^(n-1)
            self.grad += out.grad * n * (self.data ** (n - 1))

        out._backward = _backward
        return out

    # =========================================================================
    # Activation Functions
    # =========================================================================

    def relu(self) -> 'Value':
        """
        ReLU activation: z = max(0, self)

        Local gradient:
            ∂z/∂self = 1 if self > 0 else 0

        At exactly 0, we use the subgradient 0 (a common convention).
        """
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += out.grad * (1.0 if self.data > 0 else 0.0)

        out._backward = _backward
        return out

    def tanh(self) -> 'Value':
        """
        Hyperbolic tangent: z = tanh(self)

        Local gradient: ∂z/∂self = 1 - tanh²(self) = 1 - z²

        We use the output z to compute the derivative efficiently.
        """
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # ∂L/∂self += ∂L/∂z * (1 - z²)
            self.grad += out.grad * (1 - t ** 2)

        out._backward = _backward
        return out

    def sigmoid(self) -> 'Value':
        """
        Sigmoid activation: z = 1 / (1 + e^(-self)) = σ(self)

        Local gradient: ∂z/∂self = σ(self) * (1 - σ(self)) = z * (1 - z)

        We cache σ to compute the derivative efficiently.
        """
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            # ∂L/∂self += ∂L/∂z * z * (1 - z)
            self.grad += out.grad * s * (1 - s)

        out._backward = _backward
        return out

    def exp(self) -> 'Value':
        """
        Exponential: z = e^self

        Local gradient: ∂z/∂self = e^self = z

        The exponential is its own derivative!
        """
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            # ∂L/∂self += ∂L/∂z * z
            self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def log(self) -> 'Value':
        """
        Natural logarithm: z = ln(self)

        Local gradient: ∂z/∂self = 1/self

        Requires self.data > 0 (undefined for non-positive values).
        """
        assert self.data > 0, f"log undefined for {self.data}"
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            # ∂L/∂self += ∂L/∂z * (1/self)
            self.grad += out.grad * (1 / self.data)

        out._backward = _backward
        return out

    # =========================================================================
    # Backward Pass
    # =========================================================================

    def backward(self) -> None:
        """
        Compute gradients via reverse-mode automatic differentiation.

        This implements the backward pass of backpropagation:
        1. Build a topological ordering of the computation graph
        2. Set ∂L/∂self = 1 (self is the loss)
        3. Traverse in reverse order, calling each node's _backward()

        After calling backward(), every Value in the graph will have its
        .grad attribute set to ∂self/∂value.

        Note:
            - Call this on a scalar (typically the loss)
            - Gradients accumulate, so zero them before a new backward pass
            - The graph is traversed only once (O(nodes + edges))
        """
        # Build topological order via DFS
        topo: list[Value] = []
        visited: Set[Value] = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # Set gradient of output to 1
        self.grad = 1.0

        # Backpropagate in reverse topological order
        for v in reversed(topo):
            v._backward()

    def zero_grad(self) -> None:
        """
        Zero out gradients in the computation graph rooted at this node.

        Call this before a new forward/backward pass to avoid gradient accumulation.
        """
        visited: Set[Value] = set()

        def _zero(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                v.grad = 0.0
                for parent in v._parents:
                    _zero(parent)

        _zero(self)


# =============================================================================
# Neural Network Building Blocks
# =============================================================================


class Neuron:
    """
    A single neuron: y = activation(w·x + b)

    This demonstrates how Value can be used to build neural networks.
    Each parameter is a Value, so gradients propagate automatically.
    """

    def __init__(self, n_inputs: int, activation: str = 'tanh'):
        """
        Initialize a neuron with random weights.

        Args:
            n_inputs: Number of input features
            activation: Activation function ('tanh', 'relu', 'sigmoid', 'linear')
        """
        import random
        # Xavier-like initialization
        scale = 1.0 / (n_inputs ** 0.5)
        self.w = [Value(random.uniform(-scale, scale)) for _ in range(n_inputs)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x: list) -> Value:
        """
        Forward pass: compute w·x + b with activation.
        """
        # Weighted sum
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # Apply activation
        if self.activation == 'tanh':
            return act.tanh()
        elif self.activation == 'relu':
            return act.relu()
        elif self.activation == 'sigmoid':
            return act.sigmoid()
        else:  # linear
            return act

    def parameters(self) -> list:
        """Return all trainable parameters."""
        return self.w + [self.b]


class Layer:
    """A layer of neurons."""

    def __init__(self, n_inputs: int, n_outputs: int, activation: str = 'tanh'):
        """
        Initialize a layer.

        Args:
            n_inputs: Number of input features
            n_outputs: Number of neurons in this layer
            activation: Activation function for all neurons
        """
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_outputs)]

    def __call__(self, x: list) -> list:
        """Forward pass through all neurons."""
        return [n(x) for n in self.neurons]

    def parameters(self) -> list:
        """Return all trainable parameters."""
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """
    Multi-Layer Perceptron (Feed-Forward Neural Network).

    A sequence of fully-connected layers with non-linear activations.
    """

    def __init__(self, n_inputs: int, layer_sizes: list, activation: str = 'tanh'):
        """
        Initialize an MLP.

        Args:
            n_inputs: Number of input features
            layer_sizes: List of layer output sizes (e.g., [16, 16, 1])
            activation: Activation function for hidden layers
        """
        sizes = [n_inputs] + layer_sizes
        self.layers = []
        for i in range(len(layer_sizes)):
            # Use linear activation for output layer
            act = 'linear' if i == len(layer_sizes) - 1 else activation
            self.layers.append(Layer(sizes[i], sizes[i + 1], act))

    def __call__(self, x: list) -> Union[Value, list]:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        # Return scalar if single output
        return x[0] if len(x) == 1 else x

    def parameters(self) -> list:
        """Return all trainable parameters."""
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        """Zero all parameter gradients."""
        for p in self.parameters():
            p.grad = 0.0


# =============================================================================
# Demo
# =============================================================================


def demo():
    """Demonstrate automatic differentiation."""
    print("=" * 60)
    print("Automatic Differentiation Demo")
    print("=" * 60)

    # Example 1: Simple expression
    print("\n1. Simple expression: z = x*y + x^2")
    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x ** 2

    z.backward()
    print(f"   x = {x.data}, y = {y.data}")
    print(f"   z = x*y + x^2 = {z.data}")
    print(f"   ∂z/∂x = y + 2x = {y.data} + {2*x.data} = {x.grad}")
    print(f"   ∂z/∂y = x = {y.grad}")

    # Example 2: Chain rule
    print("\n2. Chain rule: z = tanh(x^2 + y)")
    x = Value(1.0)
    y = Value(2.0)
    z = (x ** 2 + y).tanh()

    z.backward()
    print(f"   x = {x.data}, y = {y.data}")
    print(f"   z = tanh({x.data**2 + y.data}) = {z.data:.4f}")
    print(f"   ∂z/∂x = {x.grad:.4f} (via chain rule)")
    print(f"   ∂z/∂y = {y.grad:.4f}")

    # Example 3: Multiple paths
    print("\n3. Multiple paths: z = x*x + x*x (same as 2*x^2)")
    x = Value(3.0)
    z = x * x + x * x

    z.backward()
    print(f"   x = {x.data}")
    print(f"   z = 2*x^2 = {z.data}")
    print(f"   ∂z/∂x = 4x = {x.grad} (gradient accumulated from two paths)")

    # Example 4: Neural network
    print("\n4. Simple neural network: 2-input, 2-hidden, 1-output")
    import random
    random.seed(42)

    mlp = MLP(2, [3, 1])
    inputs = [Value(1.0), Value(0.5)]
    output = mlp(inputs)
    output.backward()

    print(f"   Output: {output.data:.4f}")
    print(f"   Number of parameters: {len(mlp.parameters())}")
    print(f"   First weight gradient: {mlp.layers[0].neurons[0].w[0].grad:.4f}")


if __name__ == '__main__':
    demo()

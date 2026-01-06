# Section 3.3: Feed-Forward Neural Networks

We can now represent tokens as continuous vectors. The next step: build a function that transforms these vectors into predictions.

**Feed-forward neural networks** (also called multi-layer perceptrons or MLPs) are the simplest such functions. They're the building blocks of all modern deep learning.

This section derives neural networks from first principles and explains what makes them powerful.

## The Basic Unit: A Linear Layer

### Mathematical Definition

A linear layer transforms an input vector x ∈ ℝⁿ into an output vector y ∈ ℝᵐ:

$$y = Wx + b$$


Where:
- W ∈ ℝᵐˣⁿ is the **weight matrix**
- b ∈ ℝᵐ is the **bias vector**
- x ∈ ℝⁿ is the input
- y ∈ ℝᵐ is the output

### What It Computes

Each output component yᵢ is a weighted sum of inputs plus a bias:

$$y_i = \sum_{j=1}^{n} W_{ij} x_j + b_i$$


This is a **linear combination** of the inputs.

### Geometric Interpretation

A linear layer performs:
1. **Rotation/scaling**: W rotates and scales the input space
2. **Translation**: b shifts the result

It maps the input space to a new space with potentially different dimensionality.

### Example

Input: x = [2, 3] (n = 2)
Output dimension: m = 3

$$W = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad b = \begin{bmatrix} 0 \\ 1 \\ -1 \end{bmatrix}$$


$$y = \begin{bmatrix} 1·2 + 0·3 + 0 \\ 0·2 + 1·3 + 1 \\ 1·2 + 1·3 - 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 4 \\ 4 \end{bmatrix}$$


## The Problem: Linear Functions Are Limited

### Composition of Linear Functions is Linear

If f(x) = W₁x + b₁ and g(x) = W₂x + b₂, then:

$$g(f(x)) = W_2(W_1 x + b_1) + b_2 = W_2 W_1 x + (W_2 b_1 + b_2)$$


This is still a linear function! Let W' = W₂W₁ and b' = W₂b₁ + b₂:

$$g(f(x)) = W' x + b'$$


**No matter how many linear layers we stack, we get a linear function.**

### Why This Is a Problem

Linear functions can only:
- Draw straight decision boundaries
- Compute linear combinations

They cannot:
- Compute XOR (or any non-linearly-separable function)
- Model complex patterns
- Learn hierarchical features

## The Solution: Nonlinear Activation Functions

To break linearity, we apply a **nonlinear function** after each linear layer.

### The Pattern

$$h = \sigma(Wx + b)$$


Where σ is a nonlinear **activation function** applied element-wise.

### Common Activation Functions

**ReLU (Rectified Linear Unit)**:

$$\text{ReLU}(x) = \max(0, x)$$


- Simple and fast
- Derivative: 1 if x > 0, else 0
- Most popular for hidden layers

**Sigmoid**:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$


- Outputs between 0 and 1
- Historically popular, less used now due to **vanishing gradients**: sigmoid's derivative is at most 0.25, so when gradients pass through many sigmoid layers, they shrink exponentially (0.25 × 0.25 × ... → 0), making early layers learn very slowly

**Tanh**:

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$


- Outputs between -1 and 1
- Zero-centered (better than sigmoid)

**GELU (Gaussian Error Linear Unit)**:

$$\text{GELU}(x) = x \cdot \Phi(x)$$


Where Φ is the standard normal CDF. Approximation:

$$\text{GELU}(x) \approx 0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)])$$


- Used in transformers
- Smooth version of ReLU

### ReLU: A Closer Look

ReLU is piecewise linear:
- For x ≤ 0: output = 0
- For x > 0: output = x

This simple nonlinearity is enough to enable universal approximation.

**Derivative**:

$$\frac{d}{dx}\text{ReLU}(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \\ \text{undefined} & \text{if } x = 0 \end{cases}$$


At x = 0, we use the subgradient 0.

## Building Deep Networks

### Stacking Layers

A deep network is a composition of layers:

$$h_1 = \sigma(W_1 x + b_1)$$

$$h_2 = \sigma(W_2 h_1 + b_2)$$

$$h_3 = \sigma(W_3 h_2 + b_3)$$

$$\vdots$$

$$y = W_L h_{L-1} + b_L$$


Note: The last layer often has no activation (for regression) or a special activation (softmax for classification).

### Why Depth Matters

Each layer can learn increasingly abstract features:

- **Layer 1**: Low-level patterns (character combinations)
- **Layer 2**: Mid-level patterns (syllables, common sequences)
- **Layer 3**: High-level patterns (word-like structures)

Deep networks can express functions that would require exponentially wide shallow networks.

### Width vs Depth Trade-off

| Architecture | Advantages | Disadvantages |
|--------------|------------|---------------|
| Wide + Shallow | Easy to train, stable | Many parameters, limited abstraction |
| Narrow + Deep | Parameter efficient, hierarchical | Harder to train, vanishing gradients |

In practice, moderate depth (2-4 layers for character LM) works well.

## The Universal Approximation Theorem

### Statement

A feed-forward network with:
- One hidden layer
- Sufficient width (number of neurons)
- Nonlinear activation

Can approximate any continuous function on a compact domain to arbitrary precision.

### What This Means

Neural networks are **universal function approximators**. Given enough neurons, they can learn any reasonable input-output mapping.

### What This Doesn't Mean

- Doesn't say how many neurons needed (could be enormous)
- Doesn't say the function is easy to find (optimization might fail)
- Doesn't guarantee generalization (might overfit)

### Intuition via ReLU

A ReLU network creates a **piecewise linear function**:

- Each neuron contributes a "hinge" point
- With enough hinges, any continuous curve can be approximated
- More neurons = finer approximation

```
                      /\
                     /  \
        /\          /    \
       /  \        /      \___
    __/    \______/
```

Each change in slope corresponds to a neuron "turning on" or "off".

## The Output Layer for Language Modeling

For language modeling, we need to output a probability distribution over the vocabulary.

### The Softmax Function

Given logits z ∈ ℝ^|V|, softmax converts to probabilities:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{|V|} e^{z_j}}$$


### Properties of Softmax

1. **Valid probability distribution**:

$$\sum_i \text{softmax}(z)_i = 1$$


2. **All positive**:

$$\text{softmax}(z)_i > 0 \quad \forall i$$


3. **Monotonic**:
   Higher logit → higher probability

4. **Differentiable**:
   Smooth gradients everywhere

### The Complete Output Stage

$$\text{logits} = W_{\text{out}} h_L + b_{\text{out}}$$

$$P(\text{token} | \text{context}) = \text{softmax}(\text{logits})$$


Where:
- h_L ∈ ℝ^h is the final hidden state
- W_out ∈ ℝ^{|V|×h} projects to vocabulary size
- logits ∈ ℝ^|V| are unnormalized scores
- softmax normalizes to probabilities

## Putting It Together: The Full Architecture

For a character-level language model:

### Forward Pass

```
Input: context = [c_{t-k}, ..., c_{t-1}]  (k previous characters)

1. EMBED: For each position i:
   e_i = E[c_{t-k+i}]  ∈ ℝ^d

2. CONCATENATE:
   x = [e_0; e_1; ...; e_{k-1}]  ∈ ℝ^{k·d}

3. HIDDEN LAYER 1:
   h_1 = ReLU(W_1 x + b_1)  ∈ ℝ^{h_1}

4. HIDDEN LAYER 2:
   h_2 = ReLU(W_2 h_1 + b_2)  ∈ ℝ^{h_2}

5. OUTPUT:
   logits = W_out h_2 + b_out  ∈ ℝ^{|V|}

6. SOFTMAX:
   P(c_t | context) = softmax(logits)  ∈ ℝ^{|V|}
```

### Parameter Count

For context length k, embedding dim d, hidden sizes h₁ and h₂, vocabulary |V|:

| Component | Parameters |
|-----------|------------|
| Embedding | |V| × d |
| Layer 1 | (k·d) × h₁ + h₁ |
| Layer 2 | h₁ × h₂ + h₂ |
| Output | h₂ × |V| + |V| |
| **Total** | |V|d + kdh₁ + h₁ + h₁h₂ + h₂ + h₂|V| + |V| |

**Example**: |V| = 80, d = 32, k = 8, h₁ = h₂ = 128

- Embedding: 80 × 32 = 2,560
- Layer 1: 256 × 128 + 128 = 32,896
- Layer 2: 128 × 128 + 128 = 16,512
- Output: 128 × 80 + 80 = 10,320

**Total: ~62,000 parameters**

Compare to 5-gram: 80^6 ≈ 262 billion parameters. Neural wins by a factor of 4 million!

## Implementing a Layer

Using our Stage 2 autograd:

```python
class Linear:
    """A linear layer: y = Wx + b"""

    def __init__(self, in_features, out_features):
        # Xavier initialization
        scale = (2.0 / (in_features + out_features)) ** 0.5
        self.W = [[Value(random.gauss(0, scale))
                   for _ in range(in_features)]
                  for _ in range(out_features)]
        self.b = [Value(0.0) for _ in range(out_features)]

    def __call__(self, x):
        """x is a list of Values, returns list of Values."""
        out = []
        for i in range(len(self.b)):
            # Compute W[i] · x + b[i]
            activation = self.b[i]
            for j in range(len(x)):
                activation = activation + self.W[i][j] * x[j]
            out.append(activation)
        return out

    def parameters(self):
        return [w for row in self.W for w in row] + self.b
```

### ReLU Layer

```python
def relu(x):
    """Apply ReLU to a list of Values."""
    return [v.relu() for v in x]
```

### Softmax Layer

```python
def softmax(logits):
    """Convert logits to probabilities."""
    # For numerical stability, subtract max
    max_logit = max(v.data for v in logits)
    exp_logits = [(v - max_logit).exp() for v in logits]
    sum_exp = sum(exp_logits, Value(0.0))
    return [e / sum_exp for e in exp_logits]
```

## Numerical Stability: The Log-Sum-Exp Trick

### The Problem

Softmax involves exponentials. For large logits:

$$e^{100} \approx 2.7 \times 10^{43}$$

(overflow!)

$$e^{-100} \approx 3.7 \times 10^{-44}$$

(underflow!)

### The Solution

Softmax is invariant to constant shifts:

$$\text{softmax}(z - c) = \text{softmax}(z)$$


**Proof**:

$$\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i} e^{-c}}{\sum_j e^{z_j} e^{-c}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$


So we compute:

$$\text{softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$


After subtracting max, all exponents are ≤ 0, preventing overflow.

### For Log Probabilities

We often need log softmax:

$$\log \text{softmax}(z)_i = z_i - \log\sum_j e^{z_j}$$


The log-sum-exp (LSE) function:

$$\text{LSE}(z) = \log\sum_j e^{z_j} = \max(z) + \log\sum_j e^{z_j - \max(z)}$$


This is numerically stable.

## Information Flow in the Network

### Forward Pass

Information flows from input to output:

```
Context → Embeddings → Concatenate → Hidden₁ → Hidden₂ → Logits → Softmax
```

Each layer transforms representations, adding capacity to model complex patterns.

### Backward Pass

Gradients flow from output to input (via backpropagation):

```
Loss → ∂/∂Softmax → ∂/∂Logits → ∂/∂Hidden₂ → ∂/∂Hidden₁ → ∂/∂Embeddings
```

Every parameter receives a gradient signal indicating how to change to reduce loss.

### The Chain Rule at Work

For a parameter W₁[i,j] in the first layer:

$$\frac{\partial L}{\partial W_1[i,j]} = \frac{\partial L}{\partial h_1[i]} \cdot \frac{\partial h_1[i]}{\partial W_1[i,j]}$$


The gradient "chains" through all intermediate layers—exactly what we built in Stage 2.

## Summary

| Concept | Description |
|---------|-------------|
| Linear layer | y = Wx + b, affine transformation |
| Activation | Nonlinear function (ReLU, tanh, etc.) |
| Deep network | Composition of linear + activation |
| Universal approximation | Any function can be approximated |
| Softmax | Converts logits to probability distribution |
| Log-sum-exp trick | Numerically stable softmax computation |

**Key insight**: A neural network is a composition of simple functions (linear + nonlinear). Each layer transforms representations, and the composition can approximate any function. Training adjusts the parameters so this composition predicts well.

## Exercises

1. **Linearity proof**: Show that composing two linear functions y = Ax + a and z = By + b gives z = Cx + c where C = BA and c = Ba + b.

2. **ReLU universality**: Sketch how a ReLU network with 4 neurons could approximate f(x) = |x| on [-1, 1].

3. **Parameter counting**: For a network with input 256, hidden layers [512, 256, 128], and output 100, calculate the total number of parameters.

4. **Softmax verification**: Verify that softmax([2, 1, 0]) sums to 1 by computing it explicitly.

5. **Stability test**: Compute softmax([1000, 1001, 1002]) directly and with the max-subtraction trick. What happens in each case?

## What's Next

We have the architecture. But how do we train it?

In Section 3.4, we'll derive the **cross-entropy loss function**—the objective that tells the network how wrong its predictions are and how to improve.

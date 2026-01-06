# Section 2.5: Forward Mode vs Reverse Mode

There are two fundamentally different ways to apply the chain rule through a computational graph. The choice between them has massive implications for efficiency.

**Bottom line**: For neural networks (many inputs, one output), reverse mode is exponentially faster. This is why backpropagation works.

## The Two Modes

Both modes compute exact derivatives. They differ in *direction*:

| Mode | Computes | Direction | Efficient when |
|------|----------|-----------|----------------|
| Forward | ∂(all outputs)/∂(one input) | input → output | few inputs |
| Reverse | ∂(one output)/∂(all inputs) | output → input | few outputs |

## Forward Mode: Intuition

In forward mode, we pick one input xᵢ and compute how it affects everything downstream.

We propagate **derivatives** forward alongside values:
- Each node stores: (value, ∂value/∂xᵢ)
- This pair flows through the graph
- At the end, we have ∂output/∂xᵢ

To get derivatives with respect to *all* inputs, we need *n* forward passes (one per input).

## Forward Mode: Algorithm

**Input**: Computational graph, input values, index i of input to differentiate

**Process**:
1. Set derivative of xᵢ to 1, all other inputs to 0
2. For each node v in topological order:
   - Compute v's value from parents (standard forward pass)
   - Compute ∂v/∂xᵢ using chain rule:

$$\frac{\partial v}{\partial x_i} = \sum_{p \in \text{parents}(v)} \frac{\partial v}{\partial p} \cdot \frac{\partial p}{\partial x_i}$$


**Output**: ∂output/∂xᵢ

## Forward Mode: Example

Consider f(x₁, x₂) = x₁·x₂ + sin(x₁).

Let x₁ = 2, x₂ = 3. Let's compute ∂f/∂x₁.

**Graph**:
```
x₁ ──┬──────▶ [×] ──▶ a ──┐
     │         ▲          │
     │         │          ▼
     │        x₂         [+] ──▶ f
     │                    ▲
     └──▶ [sin] ──▶ b ────┘
```

**Forward pass with derivatives** (dx means ∂/∂x₁):

| Node | Value | Derivative ∂·/∂x₁ |
|------|-------|------------------|
| x₁ | 2 | 1 (seed) |
| x₂ | 3 | 0 |
| a = x₁·x₂ | 6 | x₂·1 + x₁·0 = 3 |
| b = sin(x₁) | sin(2) ≈ 0.91 | cos(x₁)·1 = cos(2) ≈ -0.42 |
| f = a + b | 6.91 | 3 + (-0.42) = 2.58 |

So ∂f/∂x₁ ≈ 2.58.

**To get ∂f/∂x₂**, we need another forward pass with seed (0, 1).

## Dual Numbers: The Math of Forward Mode

Forward mode has an elegant mathematical formulation using **dual numbers**.

A dual number has the form: a + bε where ε² = 0 but ε ≠ 0.

This is like complex numbers but with ε² = 0 instead of i² = -1.

**Key property**: f(a + bε) = f(a) + f'(a)·b·ε

The ε coefficient automatically carries the derivative!

**Example**: (2 + 1·ε)² = 4 + 4ε + ε² = 4 + 4ε (since ε² = 0)

This matches: d/dx(x²) at x=2 is 2·2 = 4.

## Reverse Mode: Intuition

In reverse mode, we pick one output and compute how *all* inputs affect it.

We propagate **gradients** backward from output to inputs:
- Start with ∂output/∂output = 1
- Work backward through the graph
- Accumulate ∂output/∂v for each node v

One backward pass gives derivatives with respect to *all* inputs!

## Reverse Mode: Algorithm

**Input**: Computational graph (already evaluated), output node

**Process**:
1. Forward pass: compute all values
2. Set gradient of output to 1
3. For each node v in reverse topological order:
   - For each parent p of v:

$$\frac{\partial \text{output}}{\partial p} \mathrel{+}= \frac{\partial \text{output}}{\partial v} \cdot \frac{\partial v}{\partial p}$$


**Output**: ∂output/∂(all inputs)

## Reverse Mode: Example

Same function: f(x₁, x₂) = x₁·x₂ + sin(x₁), with x₁ = 2, x₂ = 3.

**Forward pass** (compute values):
- a = x₁·x₂ = 6
- b = sin(x₁) = sin(2) ≈ 0.91
- f = a + b ≈ 6.91

**Backward pass**:

| Step | Node | Gradient ∂f/∂· | How |
|------|------|----------------|-----|
| 1 | f | 1 | seed |
| 2 | a | 1 | ∂f/∂a = 1 (from addition) |
| 3 | b | 1 | ∂f/∂b = 1 (from addition) |
| 4 | x₁ (via a) | x₂ = 3 | ∂f/∂a · ∂a/∂x₁ = 1 · 3 |
| 5 | x₂ (via a) | x₁ = 2 | ∂f/∂a · ∂a/∂x₂ = 1 · 2 |
| 6 | x₁ (via b) | cos(2) ≈ -0.42 | ∂f/∂b · ∂b/∂x₁ = 1 · cos(2) |
| 7 | x₁ (total) | 3 + (-0.42) ≈ 2.58 | sum both paths |

**Results from ONE backward pass**:
- ∂f/∂x₁ ≈ 2.58
- ∂f/∂x₂ = 2

## Complexity Comparison

Consider a function f: ℝⁿ → ℝᵐ (n inputs, m outputs).

| Mode | Passes needed | Total work |
|------|---------------|------------|
| Forward | n passes | O(n · graph size) |
| Reverse | m passes | O(m · graph size) |

**For neural networks**:
- n = millions of parameters
- m = 1 (scalar loss)
- Forward mode: millions of passes
- Reverse mode: ONE pass

**Reverse mode wins by a factor of n!**

This is why backpropagation (reverse mode autodiff) is the standard for training neural networks.

## The Jacobian Perspective

For f: ℝⁿ → ℝᵐ, the Jacobian is an m × n matrix:

$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$


- **Forward mode**: Computes one *column* of J per pass (how one input affects all outputs)
- **Reverse mode**: Computes one *row* of J per pass (how all inputs affect one output)

For a loss function L: ℝⁿ → ℝ, the Jacobian is 1 × n (a row vector = the gradient).
Reverse mode computes the entire gradient in one pass.

## When to Use Each Mode

| Scenario | n (inputs) | m (outputs) | Best mode |
|----------|------------|-------------|-----------|
| Neural network training | millions | 1 | Reverse |
| Sensitivity analysis | few | many | Forward |
| Jacobian-vector product | any | any | Depends on direction |
| Scientific computing | varies | varies | Often forward |

**Machine learning almost always uses reverse mode** because we have one scalar loss.

## Vector-Jacobian Products (VJP) vs Jacobian-Vector Products (JVP)

The two modes can be understood as different matrix-vector products:

**Forward mode (JVP)**: Given vector v (tangent), compute J·v
- "If inputs change by v, how do outputs change?"

**Reverse mode (VJP)**: Given vector u (cotangent), compute uᵀ·J
- "To change output by u, how much does each input contribute?"

For scalar loss with u = 1:
- VJP gives the full gradient
- This is exactly what we need for gradient descent

## Memory Trade-offs

| Mode | Memory requirement |
|------|-------------------|
| Forward | O(1) extra (just carry derivatives) |
| Reverse | O(graph size) (must cache forward pass values) |

Reverse mode needs to store intermediate values for the backward pass. For deep networks, this can be significant.

**Techniques to reduce memory**:
- Gradient checkpointing: Recompute some values instead of storing
- Trading compute for memory

## A Larger Example

Consider a 3-layer neural network:
```
x → [W₁] → relu → [W₂] → relu → [W₃] → y → [loss] → L
```

Parameters: W₁, W₂, W₃ (possibly millions of numbers)
Output: L (scalar)

**Forward mode**: To get ∂L/∂Wᵢⱼ for each weight, we need one pass per weight. With 1 million weights = 1 million forward passes.

**Reverse mode**: One forward pass + one backward pass = 2 total passes. Done.

## The Adjoint Method

Reverse mode is also called the **adjoint method** in scientific computing.

The key insight: instead of propagating "how does this input affect output" forward, propagate "how does output depend on this intermediate value" backward.

This is a duality—both compute the same derivatives, just in different orders.

## Summary

| Aspect | Forward Mode | Reverse Mode |
|--------|--------------|--------------|
| Direction | Input → Output | Output → Input |
| Computes | ∂(all)/∂(one input) | ∂(one output)/∂(all) |
| Passes for ℝⁿ→ℝ | n | 1 |
| Memory | Low | Must cache forward pass |
| Math | Dual numbers | Adjoint method |
| Used in | Sensitivity analysis | Deep learning |

**The key insight**: Neural network training involves computing ∂(scalar loss)/∂(all parameters). Reverse mode does this in O(1) passes; forward mode takes O(parameters) passes. This makes reverse mode essential for deep learning.

## Exercises

1. **Forward vs reverse count**: For f: ℝ¹⁰⁰ → ℝ¹⁰, how many passes does each mode need to compute the full Jacobian?

2. **Dual numbers**: Using dual numbers, compute d/dx(x³) at x = 2 by evaluating (2 + ε)³.

3. **Memory analysis**: In a 100-layer network, roughly how many intermediate values need to be cached for reverse mode?

4. **When forward wins**: Give an example where forward mode is more efficient than reverse mode.

5. **Thinking question**: If we have f: ℝⁿ → ℝⁿ and need the full Jacobian, which mode is better? Can we do better than n passes?

## What's Next

We understand the theory. Now let's build it.

In Section 2.6, we'll implement a complete automatic differentiation system from scratch—supporting forward pass recording and reverse mode gradient computation.

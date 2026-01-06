# Section 2.3: The Chain Rule — The Heart of Backpropagation

The chain rule is the single most important derivative rule for machine learning. Neural networks are compositions of functions, and the chain rule tells us how to differentiate compositions.

**This section is the mathematical foundation of backpropagation.**

## The Problem: Nested Functions

Consider h(x) = (x² + 1)³.

This isn't a simple polynomial or product. It's a **composition**: the cube function applied to (x² + 1).

If we write:
- g(x) = x² + 1 (inner function)
- f(u) = u³ (outer function)

Then h(x) = f(g(x)) = (g(x))³.

How do we find h'(x)?

## Intuition: Rates of Change Multiply

Suppose:
- x changes by a small amount Δx
- This causes g(x) to change by Δg
- Which causes f(g(x)) to change by Δf

The rate of change of f with respect to x is:

$$\frac{\Delta f}{\Delta x} = \frac{\Delta f}{\Delta g} \cdot \frac{\Delta g}{\Delta x}$$


The rates multiply!

If g is 3 times as sensitive to x as f is to g, then f is 3 times as sensitive to x overall.

## The Chain Rule: Formal Statement

For h(x) = f(g(x)), if g is differentiable at x and f is differentiable at g(x):

$$h'(x) = f'(g(x)) \cdot g'(x)$$


Or in Leibniz notation, if y = f(u) and u = g(x):

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$


The derivatives "chain" together—hence the name.

## Proof of the Chain Rule

**Setup**: Let h(x) = f(g(x)). We want to show h'(x) = f'(g(x)) · g'(x).

**Step 1**: Write the difference quotient for h.

$$h'(x) = \lim_{k \to 0} \frac{f(g(x+k)) - f(g(x))}{k}$$


**Step 2**: Let Δg = g(x+k) - g(x).

As k → 0, we have Δg → 0 (since g is continuous).

**Step 3**: Multiply and divide by Δg (when Δg ≠ 0).

$$\frac{f(g(x+k)) - f(g(x))}{k} = \frac{f(g(x) + \Delta g) - f(g(x))}{\Delta g} \cdot \frac{\Delta g}{k}$$


$$= \frac{f(g(x) + \Delta g) - f(g(x))}{\Delta g} \cdot \frac{g(x+k) - g(x)}{k}$$


**Step 4**: Take limits.

As k → 0:
- The first factor → f'(g(x)) (definition of derivative of f at g(x))
- The second factor → g'(x) (definition of derivative of g at x)

Therefore:

$$h'(x) = f'(g(x)) \cdot g'(x) \quad \blacksquare$$


*Technical note*: The proof needs care when Δg = 0 for some k ≠ 0. A rigorous proof handles this with a modified definition. The intuition above captures the essence.

## Example: h(x) = (x² + 1)³

**Identify the parts**:
- Inner: g(x) = x² + 1, so g'(x) = 2x
- Outer: f(u) = u³, so f'(u) = 3u²

**Apply chain rule**:

$$h'(x) = f'(g(x)) \cdot g'(x) = 3(x^2 + 1)^2 \cdot 2x = 6x(x^2 + 1)^2$$


**Verification**: Let's expand and differentiate directly (painful but correct).

$(x^2 + 1)^3 = x^6 + 3x^4 + 3x^2 + 1$

$\frac{d}{dx}(x^6 + 3x^4 + 3x^2 + 1) = 6x^5 + 12x^3 + 6x = 6x(x^4 + 2x^2 + 1) = 6x(x^2 + 1)^2$ ✓

## More Examples

### Example 1: e^{-x²}

This is exp(u) where u = -x².

- u = -x², so du/dx = -2x
- y = e^u, so dy/du = e^u

$$\frac{dy}{dx} = e^{-x^2} \cdot (-2x) = -2x e^{-x^2}$$


### Example 2: sin(3x + 2)

This is sin(u) where u = 3x + 2.

- du/dx = 3
- d(sin u)/du = cos(u)

$$\frac{d}{dx}\sin(3x+2) = \cos(3x+2) \cdot 3 = 3\cos(3x+2)$$


### Example 3: ln(x² + 1)

This is ln(u) where u = x² + 1.

- du/dx = 2x
- d(ln u)/du = 1/u

$$\frac{d}{dx}\ln(x^2+1) = \frac{1}{x^2+1} \cdot 2x = \frac{2x}{x^2+1}$$


### Example 4: Triple Composition

Let h(x) = sin(e^{x²}).

Break it down:
- Innermost: a = x², so da/dx = 2x
- Middle: b = e^a, so db/da = e^a
- Outer: y = sin(b), so dy/db = cos(b)

Chain them all:

$$\frac{dy}{dx} = \frac{dy}{db} \cdot \frac{db}{da} \cdot \frac{da}{dx} = \cos(e^{x^2}) \cdot e^{x^2} \cdot 2x$$


## The Chain Rule for Multiple Variables

In machine learning, we typically have functions of many variables. The chain rule generalizes.

### Scalar Case

If z = f(y) and y = g(x₁, x₂, ..., xₙ):

$$\frac{\partial z}{\partial x_i} = \frac{dz}{dy} \cdot \frac{\partial y}{\partial x_i}$$


### General Case: Multiple Paths

If z depends on y₁ and y₂, which both depend on x:

$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y_1} \cdot \frac{\partial y_1}{\partial x} + \frac{\partial z}{\partial y_2} \cdot \frac{\partial y_2}{\partial x}$$


We **sum over all paths** from z to x.

### Example: Multivariate Chain Rule

Let z = x·y where x = s² and y = s³.

By the chain rule:

$$\frac{dz}{ds} = \frac{\partial z}{\partial x}\frac{dx}{ds} + \frac{\partial z}{\partial y}\frac{dy}{ds}$$


$$= y \cdot 2s + x \cdot 3s^2 = s^3 \cdot 2s + s^2 \cdot 3s^2 = 2s^4 + 3s^4 = 5s^4$$


**Verification**: z = x·y = s²·s³ = s⁵, so dz/ds = 5s⁴ ✓

## Why This Matters for Neural Networks

A neural network is a composition of layers:

$$\text{output} = f_L(f_{L-1}(\cdots f_2(f_1(x))\cdots))$$


Each layer f_i transforms its input, and the chain rule tells us:

$$\frac{\partial \text{loss}}{\partial \text{parameters of layer } i} = \frac{\partial \text{loss}}{\partial \text{output of layer } i} \cdot \frac{\partial \text{output of layer } i}{\partial \text{parameters of layer } i}$$


The "gradient of loss with respect to output" flows backward through the network, getting multiplied by local gradients at each layer.

This is **backpropagation**—it's just the chain rule applied systematically from output to input.

## The Chain Rule as a Graph

We can visualize the chain rule as flow through a computation graph:

```
x → [g] → g(x) → [f] → f(g(x)) = h(x)
```

- Forward pass: compute values left to right
- Backward pass: compute gradients right to left
  - Start with ∂h/∂h = 1
  - At f: multiply by f'(g(x))
  - At g: multiply by g'(x)
  - Result: h'(x) = f'(g(x)) · g'(x)

This graph perspective leads directly to automatic differentiation.

## A Longer Chain

Consider:

$$y = \sin(\ln(x^2 + 1))$$


Let's trace through step by step:

| Variable | Expression | Derivative w.r.t. previous |
|----------|------------|---------------------------|
| a | x² + 1 | da/dx = 2x |
| b | ln(a) | db/da = 1/a |
| y | sin(b) | dy/db = cos(b) |

By chain rule:

$$\frac{dy}{dx} = \frac{dy}{db} \cdot \frac{db}{da} \cdot \frac{da}{dx} = \cos(\ln(x^2+1)) \cdot \frac{1}{x^2+1} \cdot 2x$$


$$= \frac{2x \cos(\ln(x^2+1))}{x^2+1}$$


## The Key Insight for Autodiff

Notice the pattern:
1. Forward pass: compute intermediate values (a, b, y)
2. Backward pass: multiply derivatives in reverse order

We don't need to derive a formula for the overall derivative. We just need:
- The derivative of each primitive operation (sin, ln, +, ×, etc.)
- A systematic way to apply the chain rule

This is automatic differentiation. We'll implement it in Section 2.6.

## Exercises

1. **Basic chain rule**: Find d/dx of:
   - (3x + 1)⁵
   - e^{2x}
   - ln(x³)
   - √(1 + x²)

2. **Multiple applications**: Find the derivative of sin(cos(x²)).

3. **Verify by expansion**: For (x+1)², compute the derivative using:
   - The chain rule
   - Expanding to x² + 2x + 1 and differentiating
   - Check they match.

4. **Multivariate**: If f(x,y) = x²y³, x = t², y = t³, find df/dt two ways:
   - Substitute and differentiate directly
   - Use the multivariate chain rule

5. **Neural network layer**: A layer computes y = σ(Wx + b) where σ is an activation function. If L is a loss depending on y, express ∂L/∂W using the chain rule.

## Summary

| Concept | Formula | Intuition |
|---------|---------|-----------|
| Chain rule (single) | (f∘g)' = f'(g) · g' | Rates multiply |
| Chain rule (multi) | ∂z/∂x = Σ (∂z/∂yᵢ)(∂yᵢ/∂x) | Sum over paths |
| Leibniz notation | dy/dx = (dy/du)(du/dx) | "Cancel" the du |
| Backprop | Gradients flow backward | Local gradients multiply |

**Key insight**: The chain rule turns differentiation of complex expressions into local operations connected by multiplication. This locality is what makes automatic differentiation possible and efficient.

Next: We'll represent computations as graphs and see how the chain rule applies to them systematically.

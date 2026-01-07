# Section 2.1: What is a Derivative?

In Stage 1, we found the optimal parameters for a Markov model by counting. The math worked out to a closed-form solution—no iteration required.

Neural networks are different. There's no closed-form solution. We need to *search* for good parameters by iteratively improving them. This requires knowing: **if I change a parameter slightly, how does the output change?**

This is exactly what derivatives measure.

## The Geometric Picture

Consider a function f(x). At any point x, we can ask: what's the **slope** of f at that point?

For a straight line f(x) = mx + b, the slope is constant: it's m everywhere.

For a curve like f(x) = x², the slope varies. At x=0, the curve is flat (slope 0). At x=2, it's steeper (slope 4). At x=-1, it slopes downward (slope -2).

**The derivative f'(x) gives us the slope at each point x.**

### Tangent Lines

Geometrically, f'(x) is the slope of the **tangent line** to f at x.

The tangent line is the straight line that:

1. Touches the curve at point (x, f(x))
2. Has the same "direction" as the curve at that point

If you zoom in far enough on any smooth curve, it looks like a straight line. That straight line is the tangent.

## The Algebraic Definition

How do we compute the slope at a single point? We can't directly—a single point has no "rise over run."

**The key insight**: Approximate the slope using two nearby points, then take a limit.

### The Difference Quotient

For two points x and x+h on the curve:

- Rise: f(x+h) - f(x)
- Run: h

The **difference quotient** is:

$$\frac{f(x+h) - f(x)}{h}$$


This gives the slope of the **secant line** connecting the two points.

### Taking the Limit

A **limit** describes the value a function approaches as its input approaches some value. The notation $\lim_{h \to 0} f(h)$ means "the value f(h) gets arbitrarily close to as h gets arbitrarily close to 0." We don't evaluate at h=0 directly (which might be undefined), but rather ask: what value is f(h) approaching?

As h → 0, the two points get closer together, and the secant line approaches the tangent line.

**Definition (Derivative)**: The derivative of f at x is:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$


if this limit exists.

Alternative notation:

- f'(x) — Lagrange notation
- df/dx — Leibniz notation (suggests "infinitesimal change in f per infinitesimal change in x")
- Df(x) — Operator notation

## Computing a Derivative from Definition

Let's derive the derivative of f(x) = x² from first principles.

**Step 1**: Write the difference quotient.

$$\frac{f(x+h) - f(x)}{h} = \frac{(x+h)^2 - x^2}{h}$$


**Step 2**: Expand the numerator.

$$(x+h)^2 = x^2 + 2xh + h^2$$

$$\frac{x^2 + 2xh + h^2 - x^2}{h} = \frac{2xh + h^2}{h}$$


**Step 3**: Simplify.

$$\frac{2xh + h^2}{h} = \frac{h(2x + h)}{h} = 2x + h$$


**Step 4**: Take the limit as h → 0.

$$\lim_{h \to 0} (2x + h) = 2x$$


**Result**: f'(x) = 2x

This means:

- At x=0: slope = 0 (the parabola is flat at the bottom)
- At x=1: slope = 2 (rising)
- At x=3: slope = 6 (rising steeply)
- At x=-2: slope = -4 (falling)

## Another Example: f(x) = 1/x

Let's derive the derivative of f(x) = 1/x for x ≠ 0.

**Step 1**: Difference quotient.

$$\frac{f(x+h) - f(x)}{h} = \frac{\frac{1}{x+h} - \frac{1}{x}}{h}$$


**Step 2**: Find common denominator for the numerator.

$$\frac{1}{x+h} - \frac{1}{x} = \frac{x - (x+h)}{x(x+h)} = \frac{-h}{x(x+h)}$$


**Step 3**: Divide by h.

$$\frac{-h}{h \cdot x(x+h)} = \frac{-1}{x(x+h)}$$


**Step 4**: Take limit.

$$\lim_{h \to 0} \frac{-1}{x(x+h)} = \frac{-1}{x \cdot x} = -\frac{1}{x^2}$$


**Result**: If f(x) = 1/x, then f'(x) = -1/x²

## Why Derivatives Matter for Optimization

Suppose we want to find the minimum of a function f(x). The derivative tells us which way is "downhill":

- If f'(x) > 0: f is increasing at x. Move left to decrease f.
- If f'(x) < 0: f is decreasing at x. Move right to decrease f.
- If f'(x) = 0: x might be a minimum (or maximum, or saddle point).

This is the foundation of **gradient descent**:

1. Start at some x
2. Compute f'(x)
3. Update: x ← x - α·f'(x) (where α is a small step size)
4. Repeat until f'(x) ≈ 0

The negative sign makes us move opposite to the gradient—toward lower values of f.

### Connection to Machine Learning

In machine learning:

- f is the **loss function** (measures how bad our predictions are)
- x is all the **parameters** (weights and biases)
- We want to minimize f (make predictions better)

The derivative tells us how to adjust each parameter to reduce the loss.

## Functions of Multiple Variables

Real neural networks have millions of parameters. We need derivatives with respect to each one.

**Partial Derivative**: For f(x, y), the partial derivative with respect to x is:

$$\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h}$$


We treat y as a constant and differentiate only with respect to x.

**Example**: f(x, y) = x²y + y³

$$\frac{\partial f}{\partial x} = 2xy$$


$$\frac{\partial f}{\partial y} = x^2 + 3y^2$$


**The Gradient**: The gradient collects all partial derivatives into a single object. We denote it with the symbol ∇ (called "nabla" or "del"):

$$\nabla f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right)$$


The result is a **vector**—an ordered list of numbers. A vector in n dimensions has n components, written as (v₁, v₂, ..., vₙ). For f with n variables, ∇f is an n-dimensional vector.

The gradient points in the direction of steepest ascent. To minimize, move opposite to the gradient.

## What Comes Next

We now know what a derivative is. But computing derivatives by hand is tedious and error-prone.

In Section 2.2, we'll derive the fundamental rules (power, product, chain) that let us compute derivatives of complex expressions systematically.

In Section 2.3, we'll focus on the **chain rule**—the key to computing derivatives of composed functions, which is exactly what neural networks are.

## Exercises

1. **Derive from definition**: Find f'(x) for f(x) = x³ using the limit definition.

2. **Verify geometrically**: Plot f(x) = x² and draw tangent lines at x = -1, 0, and 2. Verify the slopes match 2x.

3. **Partial derivatives**: For f(x,y) = sin(x)·cos(y), find ∂f/∂x and ∂f/∂y.

4. **Gradient descent by hand**: Starting at x = 5, apply 5 steps of gradient descent to minimize f(x) = x² with step size α = 0.1. What value do you approach?

5. **Thinking question**: The derivative is undefined when the limit doesn't exist. Give an example of a function that has no derivative at a particular point, and explain why geometrically.

## Summary

| Concept | Definition | Intuition |
|---------|------------|-----------|
| Derivative | lim_{h→0} [f(x+h)-f(x)]/h | Instantaneous rate of change |
| Geometric meaning | Slope of tangent line | How steep is the curve? |
| Partial derivative | Derivative with one variable | Hold others constant |
| Gradient | Vector of all partials | Direction of steepest ascent |
| Gradient descent | x ← x - α·∇f | Walk downhill to minimize |

**Key takeaway**: Derivatives tell us how outputs change when inputs change. For optimization, this tells us which direction to move to improve.

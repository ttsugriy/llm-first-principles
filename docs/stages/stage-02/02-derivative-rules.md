# Section 2.2: Derivative Rules from First Principles

Computing derivatives from the limit definition every time would be exhausting. Fortunately, a few key rules let us differentiate almost any function quickly.

But we won't just *state* these rules—we'll *derive* them from first principles. This ensures we understand not just *what* the rules are, but *why* they work.

## Constants and Linear Functions

### Constant Rule

If f(x) = c (a constant), what is f'(x)?

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} = \lim_{h \to 0} \frac{c - c}{h} = \lim_{h \to 0} \frac{0}{h} = 0$$


**Result**: The derivative of a constant is zero.

This makes sense: a horizontal line has slope 0 everywhere.

### Constant Multiple Rule

If g(x) = c·f(x), what is g'(x)?

$$g'(x) = \lim_{h \to 0} \frac{c \cdot f(x+h) - c \cdot f(x)}{h} = \lim_{h \to 0} c \cdot \frac{f(x+h) - f(x)}{h} = c \cdot f'(x)$$


**Result**: (c·f)' = c·f'

Constants factor out of derivatives.

## The Sum Rule

If h(x) = f(x) + g(x), what is h'(x)?

$$h'(x) = \lim_{k \to 0} \frac{[f(x+k) + g(x+k)] - [f(x) + g(x)]}{k}$$


Rearranging:

$$= \lim_{k \to 0} \frac{[f(x+k) - f(x)] + [g(x+k) - g(x)]}{k}$$


$$= \lim_{k \to 0} \frac{f(x+k) - f(x)}{k} + \lim_{k \to 0} \frac{g(x+k) - g(x)}{k}$$


$$= f'(x) + g'(x)$$


**Result**: (f + g)' = f' + g'

The derivative of a sum is the sum of derivatives. This is called **linearity**—differentiation is a linear operation.

## The Power Rule

The power rule states: if f(x) = $x^n$, then f'(x) = n·$x^{n-1}$.

Let's prove this for positive integers using the binomial theorem.

### Proof Using Binomial Expansion

For positive integer n:

$$(x+h)^n = \sum_{k=0}^{n} \binom{n}{k} x^{n-k} h^k = x^n + nx^{n-1}h + \frac{n(n-1)}{2}x^{n-2}h^2 + \cdots + h^n$$


The difference quotient:

$$\frac{(x+h)^n - x^n}{h} = \frac{nx^{n-1}h + \frac{n(n-1)}{2}x^{n-2}h^2 + \cdots + h^n}{h}$$


$$= nx^{n-1} + \frac{n(n-1)}{2}x^{n-2}h + \cdots + h^{n-1}$$


Taking the limit as h → 0:

$$\lim_{h \to 0} \left[ nx^{n-1} + \frac{n(n-1)}{2}x^{n-2}h + \cdots + h^{n-1} \right] = nx^{n-1}$$


All terms with h vanish, leaving only the first term.

**Result**: d/dx($x^n$) = n·$x^{n-1}$

### Examples

| Function | Derivative |
|----------|------------|
| x¹ | 1·x⁰ = 1 |
| x² | 2x |
| x³ | 3x² |
| $x^{10}$ | 10x⁹ |

### Extension to Negative and Fractional Powers

The power rule also works for negative and fractional exponents. Let's verify for n = -1:

We already proved: d/dx(1/x) = -1/x²

Using the power rule: d/dx($x^{-1}$) = -1·$x^{-2}$ = -1/x² ✓

For n = 1/2 (square root):

$$\frac{d}{dx}\sqrt{x} = \frac{d}{dx}x^{1/2} = \frac{1}{2}x^{-1/2} = \frac{1}{2\sqrt{x}}$$


This can be verified from the definition (more involved).

## The Product Rule

Now things get interesting. If h(x) = f(x)·g(x), is h'(x) = f'(x)·g'(x)?

**No!** Let's see what actually happens.

### Derivation

$$h'(x) = \lim_{k \to 0} \frac{f(x+k)g(x+k) - f(x)g(x)}{k}$$


The trick is to add and subtract a "bridge" term:

$$= \lim_{k \to 0} \frac{f(x+k)g(x+k) - f(x+k)g(x) + f(x+k)g(x) - f(x)g(x)}{k}$$


Grouping:

$$= \lim_{k \to 0} \frac{f(x+k)[g(x+k) - g(x)] + g(x)[f(x+k) - f(x)]}{k}$$


$$= \lim_{k \to 0} f(x+k) \cdot \frac{g(x+k) - g(x)}{k} + g(x) \cdot \lim_{k \to 0} \frac{f(x+k) - f(x)}{k}$$


Since f is continuous (differentiable functions are continuous):

$$\lim_{k \to 0} f(x+k) = f(x)$$


Therefore:

$$h'(x) = f(x) \cdot g'(x) + g(x) \cdot f'(x)$$


**Result**: (f·g)' = f·g' + f'·g

### Intuition

Why isn't the derivative of a product the product of derivatives?

Think of a rectangle with sides f and g. Its area is f·g.

If both sides grow:

- The area grows by f·Δg (original f, additional g)
- The area grows by g·Δf (original g, additional f)
- There's also a tiny Δf·Δg corner (negligible as Δ → 0)

Total growth ≈ f·Δg + g·Δf, which gives the product rule.

### Example

Let h(x) = x²·sin(x). (We'll derive sin'(x) = cos(x) later.)

Using the product rule:

$$h'(x) = x^2 \cdot \cos(x) + \sin(x) \cdot 2x = x^2\cos(x) + 2x\sin(x)$$


## The Quotient Rule

If h(x) = f(x)/g(x), what is h'(x)?

### Derivation

We can derive this from the product rule by writing f/g = f · (1/g).

First, we need d/dx(1/g). Let's use the limit definition:

$$\frac{d}{dx}\left(\frac{1}{g(x)}\right) = \lim_{k \to 0} \frac{\frac{1}{g(x+k)} - \frac{1}{g(x)}}{k}$$


$$= \lim_{k \to 0} \frac{g(x) - g(x+k)}{k \cdot g(x) \cdot g(x+k)}$$


$$= \lim_{k \to 0} \frac{-[g(x+k) - g(x)]}{k} \cdot \frac{1}{g(x) \cdot g(x+k)}$$


$$= -g'(x) \cdot \frac{1}{g(x)^2} = -\frac{g'(x)}{g(x)^2}$$


Now apply the product rule to h = f · (1/g):

$$h' = f \cdot \left(-\frac{g'}{g^2}\right) + \frac{1}{g} \cdot f' = \frac{f'}{g} - \frac{f \cdot g'}{g^2}$$


$$= \frac{f' \cdot g - f \cdot g'}{g^2}$$


**Result**: (f/g)' = (f'g - fg')/g²

### Memory Aid

Some remember this as "low d-high minus high d-low, over low squared":

$$\frac{d}{dx}\frac{f}{g} = \frac{g \cdot f' - f \cdot g'}{g^2}$$


## The Exponential Function

The exponential function $e^x$ is special: it's its own derivative.

### What is e?

The number e ≈ 2.71828... is defined as:

$$e = \lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n$$


Or equivalently, e is the unique number such that:

$$\lim_{h \to 0} \frac{e^h - 1}{h} = 1$$


### Derivative of $e^x$

$$\frac{d}{dx}e^x = \lim_{h \to 0} \frac{e^{x+h} - e^x}{h} = \lim_{h \to 0} \frac{e^x \cdot e^h - e^x}{h}$$


$$= e^x \cdot \lim_{h \to 0} \frac{e^h - 1}{h} = e^x \cdot 1 = e^x$$


**Result**: d/dx($e^x$) = $e^x$

This is why $e^x$ is so important in mathematics—it's the unique function (up to scaling) that equals its own derivative.

### General Exponential

For $a^x$ where a > 0:

Using $a^x$ = $e^{x·ln(a)}$ and the chain rule (next section):

$$\frac{d}{dx}a^x = a^x \cdot \ln(a)$$


## The Natural Logarithm

If y = ln(x), what is dy/dx?

### Derivation Using Inverse Functions

Since $e^{ln(x)}$ = x, differentiate both sides:

$$e^{\ln(x)} \cdot \frac{d}{dx}\ln(x) = 1$$


$$x \cdot \frac{d}{dx}\ln(x) = 1$$


$$\frac{d}{dx}\ln(x) = \frac{1}{x}$$


**Result**: d/dx(ln x) = 1/x

### Log of Other Bases

For log_a(x) = ln(x)/ln(a):

$$\frac{d}{dx}\log_a(x) = \frac{1}{x \cdot \ln(a)}$$


## Summary of Rules

| Rule | Formula | Derived From |
|------|---------|--------------|
| Constant | (c)' = 0 | Limit definition |
| Power | ($x^n$)' = $nx^{n-1}$ | Binomial theorem |
| Sum | (f+g)' = f'+g' | Linearity of limits |
| Product | (fg)' = fg' + f'g | Add-subtract trick |
| Quotient | (f/g)' = (f'g-fg')/g² | Product rule + reciprocal |
| Exponential | ($e^x$)' = $e^x$ | Definition of e |
| Logarithm | (ln x)' = 1/x | Inverse function |

## What's Missing: The Chain Rule

Notice we haven't handled **compositions** like sin(x²) or $e^{-x²}$ or ln(1+x).

These require the **chain rule**, which is so important it gets its own section. The chain rule is the heart of automatic differentiation.

## Exercises

1. **Derive the power rule for n=3** by directly expanding (x+h)³ - x³.

2. **Product rule practice**: Find d/dx(x³·$e^x$).

3. **Quotient rule practice**: Find d/dx(x²/(1+x)).

4. **Why the product rule?**: Give a geometric argument for why (fg)' ≠ f'g'.

5. **Verify log derivative**: Using the limit definition, show that d/dx(ln x) = 1/x by computing the limit directly. (Hint: use the substitution k = h/x and the definition of e.)

## Summary

We derived all fundamental derivative rules from the limit definition:

- Constants disappear, sums split, constants factor out
- The power rule handles polynomials
- The product rule handles products (it's not just f'g')
- The quotient rule is the product rule for reciprocals
- $e^x$ is its own derivative (remarkable!)
- ln(x) differentiates to 1/x

With these rules, we can differentiate any polynomial, rational function, or expression involving exponentials and logarithms—**as long as there's no function composition**.

For compositions, we need the chain rule.

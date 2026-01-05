# Stage 2: Automatic Differentiation

!!! note "Coming Soon"
    This stage is under development.

## Preview

In Stage 1, we built a language model by counting. The parameters were found by a closed-form solution—no iteration required.

But counting doesn't scale. To build models that *generalize*, we need to:

1. **Learn representations** — Map tokens to vectors that capture meaning
2. **Use neural networks** — Flexible function approximators
3. **Train with gradients** — Iteratively improve parameters

This requires **automatic differentiation**: computing gradients of complex functions efficiently.

## What You'll Learn

- What is a derivative? (Geometrically and algebraically)
- The chain rule for derivatives (not to be confused with probability chain rule)
- Forward mode vs. reverse mode differentiation
- Building an autograd system from scratch
- Computational graphs and backpropagation

## Why This Matters

Every neural network training loop does this:

```python
loss = model(inputs)
loss.backward()  # ← This is automatic differentiation
optimizer.step()
```

We'll build `backward()` from scratch, understanding exactly how gradients flow through a computation.

---

Check back soon for the full content!

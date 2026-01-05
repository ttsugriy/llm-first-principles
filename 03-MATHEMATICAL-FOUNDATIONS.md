# Mathematical Foundations

## Overview

This document catalogs all mathematical prerequisites and derivations required throughout the book. Rather than presenting all math upfront, we derive concepts just-in-time. This document serves as:

1. **Reference** — What math is needed where
2. **Derivation catalog** — All proofs and derivations in one place
3. **Dependency map** — Which math concepts depend on which

---

## 1. Prerequisites by Stage

### Stage 1: Markov Chains
- Basic probability (distributions, conditional probability)
- Summation notation
- Logarithms

### Stage 2: Automatic Differentiation
- Limits (conceptual)
- Derivatives (definition, common functions)
- Chain rule
- Partial derivatives

### Stage 3: Backpropagation Deep Dive
- Jacobian matrices
- Matrix calculus basics
- Vector calculus

### Stage 4: Neural Language Model
- Matrix multiplication
- Vectors and vector spaces
- Function composition

### Stage 5: Optimization
- Gradients and directional derivatives
- Convexity (basic)
- Taylor series (first-order)

### Stage 6: Training Stability
- Variance and expectation
- Random variable properties

### Stage 7-10: Transformer
- Linear algebra (eigenvalues intuition)
- Softmax properties
- Trigonometry (for positional encoding)

### Stage 11-13: Scaling
- Big-O notation
- Memory hierarchy concepts
- Communication complexity basics

### Stage 14-18: Modern Practice
- All of the above
- Information theory basics

---

## 2. Core Derivations Catalog

### 2.1 Probability Theory

#### 2.1.1 Chain Rule of Probability

**Statement:**
For events A₁, A₂, ..., Aₙ:

P(A₁, A₂, ..., Aₙ) = P(A₁) · P(A₂|A₁) · P(A₃|A₁,A₂) · ... · P(Aₙ|A₁,...,Aₙ₋₁)

**Proof:**
By definition of conditional probability: P(A|B) = P(A,B)/P(B)

Therefore: P(A,B) = P(B) · P(A|B)

For three events:
P(A,B,C) = P(A,B) · P(C|A,B) = P(A) · P(B|A) · P(C|A,B)

By induction, extends to n events. ∎

**Used in:** Stage 1 (autoregressive factorization)

---

#### 2.1.2 Bayes' Theorem

**Statement:**
P(A|B) = P(B|A) · P(A) / P(B)

**Proof:**
From conditional probability:
- P(A|B) = P(A,B)/P(B)
- P(B|A) = P(A,B)/P(A)

From the second: P(A,B) = P(B|A) · P(A)

Substituting into the first: P(A|B) = P(B|A) · P(A) / P(B) ∎

**Used in:** Stage 1 (smoothing as prior), Stage 16 (reward modeling)

---

#### 2.1.3 Maximum Likelihood Estimation

**Statement:**
Given observations x₁, ..., xₙ from distribution p(x|θ), the MLE is:

θ̂ = argmax_θ ∏ᵢ p(xᵢ|θ) = argmax_θ Σᵢ log p(xᵢ|θ)

**For categorical distribution:**
If we observe counts c₁, ..., cₖ for k categories:

θ̂ᵢ = cᵢ / Σⱼ cⱼ

**Proof:**
The log-likelihood is: L(θ) = Σᵢ cᵢ log θᵢ

Subject to constraint: Σᵢ θᵢ = 1

Using Lagrange multipliers:
∂/∂θᵢ [Σⱼ cⱼ log θⱼ - λ(Σⱼ θⱼ - 1)] = cᵢ/θᵢ - λ = 0

Therefore: θᵢ = cᵢ/λ

From constraint: Σᵢ θᵢ = Σᵢ cᵢ/λ = 1, so λ = Σᵢ cᵢ

Therefore: θ̂ᵢ = cᵢ / Σⱼ cⱼ ∎

**Used in:** Stage 1 (Markov chain training = MLE)

---

### 2.2 Information Theory

#### 2.2.1 Entropy

**Definition:**
For discrete random variable X with probability mass function p:

H(X) = -Σₓ p(x) log p(x)

**Interpretation:**
- Average number of bits needed to encode X
- Measure of uncertainty/surprise
- Maximum when uniform, minimum (0) when deterministic

**Properties:**
- H(X) ≥ 0
- H(X) ≤ log|X| (equality when uniform)
- H(X,Y) ≤ H(X) + H(Y) (equality when independent)

**Used in:** Stage 1 (perplexity), Stage 7 (attention entropy)

---

#### 2.2.2 Cross-Entropy

**Definition:**
For true distribution p and model distribution q:

H(p,q) = -Σₓ p(x) log q(x)

**Relation to entropy:**
H(p,q) = H(p) + D_KL(p||q) ≥ H(p)

**As loss function:**
When p is one-hot (true label) and q is predicted distribution:

H(p,q) = -log q(y_true)

This is the negative log probability of the correct class.

**Used in:** Stage 1, 4 (training loss)

---

#### 2.2.3 KL Divergence

**Definition:**
D_KL(p||q) = Σₓ p(x) log(p(x)/q(x)) = H(p,q) - H(p)

**Properties:**
- D_KL(p||q) ≥ 0 (Gibbs' inequality)
- D_KL(p||q) = 0 iff p = q
- Asymmetric: D_KL(p||q) ≠ D_KL(q||p)

**Proof of non-negativity (Gibbs' inequality):**
Using log x ≤ x - 1:

-D_KL(p||q) = Σₓ p(x) log(q(x)/p(x)) ≤ Σₓ p(x)(q(x)/p(x) - 1) = Σₓ q(x) - Σₓ p(x) = 0 ∎

**Used in:** Stage 16 (RLHF KL penalty)

---

#### 2.2.4 Perplexity

**Definition:**
PPL = exp(H(p,q)) = exp(-1/N Σᵢ log q(xᵢ))

**Interpretation:**
If perplexity is k, model is as uncertain as choosing uniformly among k options.

**Derivation of interpretation:**
For uniform distribution over k items: H = log k
Therefore: exp(H) = k

Perplexity = exp(cross-entropy) gives "effective vocabulary size"

**Used in:** Stage 1, 4 (evaluation metric)

---

### 2.3 Calculus

#### 2.3.1 Derivative Definition

**Definition:**
f'(x) = lim_{h→0} [f(x+h) - f(x)] / h

**Common derivatives:**

| Function | Derivative |
|----------|------------|
| xⁿ | nxⁿ⁻¹ |
| eˣ | eˣ |
| log x | 1/x |
| sin x | cos x |
| cos x | -sin x |

**Used in:** Stage 2 (autodiff foundation)

---

#### 2.3.2 Chain Rule

**Statement:**
If y = f(g(x)), then dy/dx = f'(g(x)) · g'(x)

Or in Leibniz notation: dy/dx = (dy/du) · (du/dx) where u = g(x)

**Proof (sketch):**
Δy/Δx = (Δy/Δu) · (Δu/Δx)

Taking limit as Δx → 0:
dy/dx = (dy/du) · (du/dx) ∎

**Multivariate version:**
If z = f(x,y) where x = x(t), y = y(t):

dz/dt = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt)

**Used in:** Stage 2, 3 (backpropagation)

---

#### 2.3.3 Gradient

**Definition:**
For f: ℝⁿ → ℝ:

∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

**Properties:**
- Points in direction of steepest ascent
- Magnitude is rate of change in that direction
- Perpendicular to level sets

**Used in:** Stage 2, 5 (optimization)

---

### 2.4 Linear Algebra

#### 2.4.1 Matrix Multiplication

**Definition:**
For A ∈ ℝᵐˣⁿ and B ∈ ℝⁿˣᵖ:

(AB)ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ

**Complexity:** O(mnp) for naive algorithm

**Three interpretations:**
1. Dot products: row of A · column of B
2. Linear combinations: columns of AB are combinations of columns of A
3. Outer products: AB = Σₖ A[:,k] ⊗ B[k,:]

**Used in:** Stage 4 (embeddings, layers), Stage 7 (attention)

---

#### 2.4.2 Matrix Gradient Rules

**For Y = XW where X ∈ ℝᵐˣⁿ, W ∈ ℝⁿˣᵖ, Y ∈ ℝᵐˣᵖ:**

Given upstream gradient ∂L/∂Y ∈ ℝᵐˣᵖ:

∂L/∂X = (∂L/∂Y) Wᵀ    [shape: m×n]
∂L/∂W = Xᵀ (∂L/∂Y)    [shape: n×p]

**Derivation:**
Consider scalar loss L. For single element:

∂L/∂Xᵢⱼ = Σₖ (∂L/∂Yᵢₖ)(∂Yᵢₖ/∂Xᵢⱼ) = Σₖ (∂L/∂Yᵢₖ) Wⱼₖ

In matrix form: ∂L/∂X = (∂L/∂Y) Wᵀ ∎

**Used in:** Stage 3 (manual backprop)

---

#### 2.4.3 Softmax Gradient

**Definition:**
softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)

**Jacobian:**
∂softmax(z)ᵢ/∂zⱼ = softmax(z)ᵢ (δᵢⱼ - softmax(z)ⱼ)

where δᵢⱼ is Kronecker delta.

**Derivation:**
Let sᵢ = softmax(z)ᵢ = exp(zᵢ)/Σₖexp(zₖ)

Case i = j:
∂sᵢ/∂zᵢ = [exp(zᵢ)·Σ - exp(zᵢ)·exp(zᵢ)] / Σ² = sᵢ(1 - sᵢ)

Case i ≠ j:
∂sᵢ/∂zⱼ = -exp(zᵢ)·exp(zⱼ) / Σ² = -sᵢsⱼ

Combined: ∂sᵢ/∂zⱼ = sᵢ(δᵢⱼ - sⱼ) ∎

**With cross-entropy:**
If L = -log(sᵧ) where y is true class:

∂L/∂zᵢ = sᵢ - δᵢᵧ = softmax(z)ᵢ - one_hot(y)ᵢ

Beautiful simplification!

**Used in:** Stage 3, 7 (attention gradient)

---

### 2.5 Variance Analysis

#### 2.5.1 Variance of Sum

**Statement:**
Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)

If independent: Var(X + Y) = Var(X) + Var(Y)

**Used in:** Stage 6 (initialization analysis)

---

#### 2.5.2 Variance of Product

**Statement:**
For independent X, Y:

Var(XY) = Var(X)Var(Y) + Var(X)E[Y]² + Var(Y)E[X]²

If E[X] = E[Y] = 0:
Var(XY) = Var(X)Var(Y)

**Used in:** Stage 6 (initialization), Stage 7 (√d scaling)

---

#### 2.5.3 Xavier Initialization

**Problem:** Keep variance stable through layers.

**Setup:** y = Wx where W ∈ ℝᵐˣⁿ, x ∈ ℝⁿ

Assume:
- x has zero mean, variance σₓ²
- W entries iid with zero mean, variance σw²
- x and W independent

**Analysis:**
yᵢ = Σⱼ Wᵢⱼ xⱼ

E[yᵢ] = Σⱼ E[Wᵢⱼ]E[xⱼ] = 0

Var(yᵢ) = Σⱼ Var(Wᵢⱼ xⱼ) = Σⱼ σw² σₓ² = n σw² σₓ²

**For variance preservation:** n σw² = 1, so σw² = 1/n

**Xavier initialization:** W ~ N(0, 1/fan_in) or U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))

**Used in:** Stage 6

---

#### 2.5.4 Kaiming Initialization

**For ReLU:** Half the outputs are zero on average.

Var(ReLU(y)) = Var(y)/2 (for symmetric y around 0)

**To preserve variance:** Need Var(y) = 2σₓ²

So: n σw² = 2, giving σw² = 2/n

**Kaiming initialization:** W ~ N(0, 2/fan_in)

**Used in:** Stage 6

---

#### 2.5.5 Attention Scaling

**Problem:** Dot product of two random vectors has high variance.

**Setup:** q, k ∈ ℝᵈ with entries having variance 1.

**Analysis:**
q·k = Σᵢ qᵢkᵢ

Var(q·k) = Σᵢ Var(qᵢkᵢ) = Σᵢ 1·1 = d

**Solution:** Divide by √d to get variance 1.

score = (q·k) / √d → Var(score) = 1

**Used in:** Stage 7

---

### 2.6 Optimization

#### 2.6.1 Gradient Descent Convergence

**For convex, L-smooth function:**

If ||∇f(x) - ∇f(y)|| ≤ L||x - y||, and we use learning rate η ≤ 1/L:

f(xₜ) - f(x*) ≤ ||x₀ - x*||² / (2ηt)

Convergence rate: O(1/t)

**Used in:** Stage 5

---

#### 2.6.2 Momentum

**Update:**
vₜ = β vₜ₋₁ + ∇f(xₜ)
xₜ₊₁ = xₜ - η vₜ

**Interpretation:**
vₜ = Σᵢ₌₀ᵗ βᵗ⁻ⁱ ∇f(xᵢ)

Exponential moving average of gradients.

**Used in:** Stage 5

---

#### 2.6.3 Adam Derivation

**First moment (mean):**
mₜ = β₁ mₜ₋₁ + (1-β₁) gₜ

**Second moment (uncentered variance):**
vₜ = β₂ vₜ₋₁ + (1-β₂) gₜ²

**Bias correction:**
At t=1: E[mₜ] = (1-β₁) E[g], but we want E[g]
Correction: m̂ₜ = mₜ / (1-β₁ᵗ)

Similarly: v̂ₜ = vₜ / (1-β₂ᵗ)

**Update:**
θₜ₊₁ = θₜ - η m̂ₜ / (√v̂ₜ + ε)

**Used in:** Stage 5

---

### 2.7 Advanced Topics

#### 2.7.1 RoPE Derivation

**Idea:** Encode position through rotation.

**2D case:**
For position m, rotate query/key by angle mθ:

R(m) = [cos(mθ), -sin(mθ)]
       [sin(mθ),  cos(mθ)]

**Key property:**
q_m · k_n = (R(m)q) · (R(n)k) depends only on (m-n)

**Proof:**
(R(m)q)ᵀ(R(n)k) = qᵀ R(m)ᵀ R(n) k = qᵀ R(n-m) k

Since R(m)ᵀ = R(-m) and R(-m)R(n) = R(n-m) ∎

**Extension to d dimensions:**
Apply different frequencies to pairs of dimensions.

**Used in:** Stage 9, 14

---

#### 2.7.2 DPO Derivation

**Starting point:** RLHF objective

max_π E_x[E_y~π[r(x,y)] - β D_KL(π||π_ref)]

**Optimal policy:**
π*(y|x) = (1/Z(x)) π_ref(y|x) exp(r(x,y)/β)

where Z(x) = Σᵧ π_ref(y|x) exp(r(x,y)/β)

**Rearranging for reward:**
r(x,y) = β log(π*(y|x)/π_ref(y|x)) + β log Z(x)

**Substituting into Bradley-Terry:**
P(y₁ > y₂|x) = σ(r(x,y₁) - r(x,y₂))
             = σ(β log(π*(y₁|x)/π_ref(y₁|x)) - β log(π*(y₂|x)/π_ref(y₂|x)))

**DPO loss:**
L_DPO = -E[(log σ(β log(π(y_w|x)/π_ref(y_w|x)) - β log(π(y_l|x)/π_ref(y_l|x))))]

No reward model needed!

**Used in:** Stage 16

---

## 3. Notation Reference

### Scalars
- x, y, z — variables
- α, β, γ — hyperparameters
- η — learning rate
- ε — small constant

### Vectors
- **x**, **v** — bold lowercase
- xᵢ — i-th element

### Matrices
- **W**, **A** — bold uppercase
- Wᵢⱼ — element at row i, column j
- Wᵀ — transpose
- W⁻¹ — inverse

### Functions
- f, g, h — generic functions
- σ — sigmoid or softmax
- ReLU, GELU — activation functions

### Operators
- ∇ — gradient
- ∂ — partial derivative
- Σ — summation
- ∏ — product
- E[·] — expectation
- Var(·) — variance

### Sets
- ℝ — real numbers
- ℝⁿ — n-dimensional real vectors
- ℝᵐˣⁿ — m×n real matrices

### Complexity
- O(·) — big-O notation
- Θ(·) — tight bound
- Ω(·) — lower bound

---

## 4. Quick Reference Cards

### Probability Essentials
```
P(A,B) = P(A|B)P(B) = P(B|A)P(A)
P(A|B) = P(B|A)P(A)/P(B)         [Bayes]
H(X) = -Σ p(x) log p(x)          [Entropy]
H(p,q) = -Σ p(x) log q(x)        [Cross-entropy]
D_KL(p||q) = Σ p(x) log(p(x)/q(x)) [KL divergence]
```

### Calculus Essentials
```
d/dx[f(g(x))] = f'(g(x))g'(x)    [Chain rule]
∇f = [∂f/∂x₁, ..., ∂f/∂xₙ]       [Gradient]
∂(XW)/∂X = (∂L/∂Y)Wᵀ            [Matrix gradient]
∂(XW)/∂W = Xᵀ(∂L/∂Y)            [Matrix gradient]
```

### Linear Algebra Essentials
```
(AB)ᵢⱼ = Σₖ AᵢₖBₖⱼ               [Matrix multiply]
(AB)ᵀ = BᵀAᵀ                     [Transpose of product]
Tr(AB) = Tr(BA)                  [Trace cyclic]
||x||₂ = √(Σᵢ xᵢ²)               [L2 norm]
```

### Softmax
```
softmax(z)ᵢ = exp(zᵢ)/Σⱼexp(zⱼ)
∂softmax/∂z = diag(s) - ssᵀ      [Jacobian]
∂(-log s_y)/∂z = s - one_hot(y)  [With cross-entropy]
```

---

## 5. Stage-by-Stage Math Map

| Stage | New Math Introduced | Builds On |
|-------|---------------------|-----------|
| 1 | Probability, MLE, Entropy, Cross-entropy | — |
| 2 | Derivatives, Chain rule | — |
| 3 | Matrix gradients, Jacobians | 2 |
| 4 | Softmax, Cross-entropy loss | 1, 2, 3 |
| 5 | Optimization theory, Adam | 2, 3 |
| 6 | Variance analysis, Initialization | 5 |
| 7 | Attention math, Scaling | 4, 6 |
| 8 | Multi-head decomposition | 7 |
| 9 | RoPE, Trigonometry | 7 |
| 10 | Information theory, Compression | 1 |
| 11 | Complexity analysis | — |
| 12 | I/O complexity, Numerical precision | 11 |
| 13 | Communication complexity | 11, 12 |
| 14 | Scaling laws | 11 |
| 15 | Low-rank approximation | 4 |
| 16 | RL, Policy gradients, DPO | 1, 5 |
| 17 | Roofline model | 11, 12 |
| 18 | Quantization theory | 12 |

---

This document will be updated as stages are written and new derivations are needed.

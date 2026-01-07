# Section 3.4: Cross-Entropy Loss and Maximum Likelihood

We have a neural network that outputs a probability distribution. But how do we train it?

**Cross-entropy loss** is the answer. In this section, we'll derive it from first principles and prove it's equivalent to maximum likelihood estimation—connecting back to Stage 1.

## The Training Problem

### What We Have

A neural language model that computes:

$$\hat{P}(c_t | c_{t-k:t-1}; \theta)$$


Where:

- $c_{t-k:t-1}$ is the context (previous k characters)
- $c_t$ is the next character
- θ are all the model parameters (embeddings, weights, biases)
- $\hat{P}$ is the model's predicted probability distribution

### What We Want

Parameters θ* that make the model's predictions match the true data distribution as closely as possible.

### The Fundamental Question

How do we measure "how wrong" the model's predictions are?

## Maximum Likelihood: The Principled Approach

### From Stage 1

In Stage 1, we derived maximum likelihood estimation for n-gram models:

$$\theta^* = \arg\max_\theta P(\text{data} | \theta)$$


The optimal parameters are those that maximize the probability of the observed data.

### For Neural Language Models

The principle is exactly the same!

Given training data $D = \{(x_1, y_1), ..., (x_N, y_N)\}$ where:

- $x_i$ is the i-th context
- $y_i$ is the true next character

The likelihood is:

$$P(D | \theta) = \prod_{i=1}^{N} P(y_i | x_i; \theta)$$


(Assuming independence between examples.)

### Log-Likelihood

Products are numerically unstable and hard to differentiate. Take logarithms:

$$\log P(D | \theta) = \sum_{i=1}^{N} \log P(y_i | x_i; \theta)$$


This is the **log-likelihood**.

### From Maximization to Minimization

ML practitioners prefer **minimizing** a **loss**. Negating:

$$L(\theta) = -\log P(D | \theta) = -\sum_{i=1}^{N} \log P(y_i | x_i; \theta)$$


This is the **negative log-likelihood**.

**Minimizing NLL = Maximizing likelihood.**

### Average Loss

For numerical stability and comparison across dataset sizes, use the average:

$$L(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \log P(y_i | x_i; \theta)$$


This is our training objective.

## Cross-Entropy: The Information Theory View

### Cross-Entropy Definition

Given true distribution p and model distribution q, the **cross-entropy** is:

$$H(p, q) = -\sum_{x} p(x) \log q(x)$$


It measures the expected number of bits needed to encode samples from p using a code optimized for q.

### For Language Modeling

At each position in our data:

- True distribution: p(c_t | context) = 1 if c_t is the actual next character, else 0 (one-hot)
- Model distribution: $q(c) = \hat{P}(c | \text{context}; \theta)$

The cross-entropy at this position:

$$H(p, q) = -\sum_{c} p(c) \log q(c)$$


Since p(c) = 1 for c = y (the true character) and 0 otherwise:

$$H(p, q) = -1 \cdot \log q(y) + \sum_{c \neq y} 0 \cdot \log q(c) = -\log q(y)$$


This is exactly the negative log-probability!

### The Equivalence

Cross-entropy loss per example = Negative log-likelihood per example.

For the dataset:

$$\text{Cross-Entropy Loss} = -\frac{1}{N}\sum_{i=1}^{N} \log \hat{P}(y_i | x_i; \theta)$$


**This is the same as average NLL!**

## Why Cross-Entropy Is the Right Loss

### Reason 1: Maximum Likelihood

As shown, minimizing cross-entropy = maximizing likelihood.

MLE has strong theoretical justification:

- Consistent (converges to true parameters with infinite data)
- Asymptotically efficient (lowest variance among consistent estimators)
- Invariant under reparametrization

### Reason 2: Proper Scoring Rule

A scoring rule S(q, y) is **proper** if:

$$\mathbb{E}_{y \sim p}[S(p, y)] \leq \mathbb{E}_{y \sim p}[S(q, y)]$$


For any distribution q. That is, the best expected score is achieved when q = p.

**Cross-entropy is proper.**

**Proof**:

Expected cross-entropy when true distribution is p:

$$\mathbb{E}_{y \sim p}[H(p, q)] = \sum_y p(y) H(p, q) = -\sum_y p(y) \log q(y)$$


This is minimized when q = p, giving the entropy H(p):

$$H(p, p) = -\sum_y p(y) \log p(y) = H(p)$$


For any q ≠ p, we have H(p, q) > H(p, p) by Gibbs' inequality.

### Reason 3: Information-Theoretic Interpretation

Cross-entropy H(p, q) measures the inefficiency of using code for q when the true distribution is p.

Minimizing cross-entropy = finding the most efficient encoding of the data.

### Reason 4: Gradient Properties

We'll see shortly that cross-entropy combined with softmax has remarkably clean gradients.

## The Loss Function Explicitly

### For a Single Example

Given context x and true next character y:

$$L = -\log \hat{P}(y | x; \theta)$$


Recall that the model computes:

$$\hat{P}(c | x; \theta) = \text{softmax}(\text{logits})_c = \frac{e^{z_c}}{\sum_{c'} e^{z_{c'}}}$$


Where z are the logits (outputs of the final linear layer).

Substituting:

$$L = -\log \frac{e^{z_y}}{\sum_c e^{z_c}}$$


### Simplifying

$$L = -\log e^{z_y} + \log \sum_c e^{z_c}$$


$$L = -z_y + \log \sum_c e^{z_c}$$


This is the **softmax log-likelihood formula**.

The second term is the **log-sum-exp** (LSE) function:

$$\text{LSE}(z) = \log \sum_c e^{z_c}$$


### For a Batch

For N examples:

$$L = \frac{1}{N}\sum_{i=1}^{N} \left( -z_{y_i}^{(i)} + \text{LSE}(z^{(i)}) \right)$$


Where $z^{(i)}$ are the logits for example i, and $y_i$ is the true character index.

## Computing the Gradient

### Why We Need the Gradient

To train via gradient descent, we need:

$$\frac{\partial L}{\partial \theta}$$


For every parameter θ. Our Stage 2 autograd will compute this automatically, but understanding the gradient structure is valuable.

### Gradient w.r.t. Logits

The most important gradient: ∂L/∂z.

For a single example with true class y:

$$L = -z_y + \log \sum_c e^{z_c}$$


For the logit of the true class:

$$\frac{\partial L}{\partial z_y} = -1 + \frac{e^{z_y}}{\sum_c e^{z_c}} = -1 + \hat{P}(y | x)$$


For any other logit $z_c$ where c ≠ y:

$$\frac{\partial L}{\partial z_c} = 0 + \frac{e^{z_c}}{\sum_c e^{z_c}} = \hat{P}(c | x)$$


### The Beautiful Result

For all classes c:

$$\frac{\partial L}{\partial z_c} = \hat{P}(c | x) - \delta_{cy}$$


Where $\delta_{cy}$ is 1 if c = y (the true class), else 0.

In vector form:

$$\frac{\partial L}{\partial z} = \hat{p} - y_{\text{one-hot}}$$


The gradient is simply: **predicted probability minus true probability!**

### Why This Is Beautiful

- Magnitude of gradient = how wrong the prediction is
- If $\hat{P}(y) = 1$: gradient is 0 (perfect prediction)
- If $\hat{P}(y) = 0$: gradient is -1 (maximally wrong)
- Automatically scaled by confidence

This natural weighting makes gradient descent effective.

## Deriving the Softmax-CrossEntropy Gradient

Let's prove the result rigorously.

### Setup

Given logits $z \in \mathbb{R}^{|V|}$ and true class y:

$$\hat{P}(c | x) = \frac{e^{z_c}}{\sum_{c'} e^{z_{c'}}} = \frac{e^{z_c}}{Z}$$


Where $Z = \sum_c e^{z_c}$ (partition function).

Loss:

$$L = -\log \hat{P}(y | x) = -\log e^{z_y} + \log Z = -z_y + \log Z$$


### Derivative of Partition Function

$$\frac{\partial}{\partial z_c} \log Z = \frac{1}{Z} \frac{\partial Z}{\partial z_c} = \frac{1}{Z} e^{z_c} = \hat{P}(c | x)$$


### Derivative of Loss w.r.t. z_y

$$\frac{\partial L}{\partial z_y} = -1 + \frac{\partial \log Z}{\partial z_y} = -1 + \hat{P}(y | x)$$


### Derivative w.r.t. Other Logits

For c ≠ y:

$$\frac{\partial L}{\partial z_c} = 0 + \frac{\partial \log Z}{\partial z_c} = \hat{P}(c | x)$$


### Combined Result

$$\frac{\partial L}{\partial z_c} = \begin{cases} \hat{P}(y | x) - 1 & \text{if } c = y \\ \hat{P}(c | x) & \text{if } c \neq y \end{cases}$$


Which is exactly:

$$\frac{\partial L}{\partial z_c} = \hat{P}(c | x) - \delta_{cy}$$


**QED.**

## Connection to Perplexity (from Stage 1)

### Recall Perplexity

In Stage 1, we defined perplexity as:

$$\text{PPL} = \exp\left( -\frac{1}{N}\sum_{i=1}^{N} \log P(w_i | w_{<i}) \right)$$


### The Relationship

The exponent is exactly the average cross-entropy loss!

$$\text{PPL} = \exp(L)$$


Where L is the cross-entropy loss.

### Why This Matters

Training: minimize L (cross-entropy)

Evaluation: report exp(L) (perplexity)

Same underlying metric, different presentations:

- L ∈ [0, ∞): additive, for optimization
- PPL ∈ [1, ∞): interpretable, for humans

### Interpreting the Loss

If L = 2.3:

- PPL = exp(2.3) ≈ 10
- Model is "as uncertain as picking uniformly among 10 options"
- Lower is better

## Implementing Cross-Entropy Loss

Using our Stage 2 autograd:

```python
def cross_entropy_loss(logits, target_idx):
    """
    logits: list of Value objects (unnormalized scores)
    target_idx: int (index of true class)

    Returns: Value (scalar loss)
    """
    # Log-sum-exp for numerical stability
    max_logit = max(v.data for v in logits)

    # Subtract max for stability (doesn't change result)
    shifted = [v - max_logit for v in logits]

    # exp of shifted logits
    exp_logits = [v.exp() for v in shifted]

    # Sum
    sum_exp = sum(exp_logits, Value(0.0))

    # Log-sum-exp
    lse = sum_exp.log() + max_logit

    # Loss = -logit[target] + lse
    loss = lse - logits[target_idx]

    return loss
```

### Why LogSumExp?

Direct computation $\log(\sum e^{z_i})$ can overflow/underflow.

Using LSE trick:

$$\log \sum_i e^{z_i} = \max(z) + \log \sum_i e^{z_i - \max(z)}$$


After subtracting max, all exponents are ≤ 0, preventing overflow.

## Multiple Examples: Batch Loss

For a batch of N examples:

```python
def batch_cross_entropy(batch_logits, batch_targets):
    """
    batch_logits: list of lists of Values
    batch_targets: list of target indices

    Returns: Value (average loss)
    """
    losses = [cross_entropy_loss(logits, target)
              for logits, target in zip(batch_logits, batch_targets)]

    # Average
    total = sum(losses, Value(0.0))
    return total / len(losses)
```

## Summary

| Concept | Formula | Interpretation |
|---------|---------|----------------|
| MLE objective | max log P(data\|θ) | Most likely parameters |
| NLL loss | -log P(data\|θ) | Minimize to maximize likelihood |
| Cross-entropy | -Σ p log q | Expected bits using wrong code |
| Per-example loss | -log P(y\|x) | Surprise at true outcome |
| Softmax gradient | $\hat{p} - p$ | Predicted - true |
| Perplexity | exp(loss) | Interpretable uncertainty |

**Key insights**:

1. **Cross-entropy = MLE**: Same theoretical foundation as Stage 1
2. **Proper scoring rule**: Best achievable when q = p
3. **Beautiful gradient**: $\hat{p} - p$ is simple and effective
4. **Connection to perplexity**: exp(loss) gives interpretable metric

## Exercises

1. **Verify equivalence**: Show that for one-hot true distribution, H(p,q) = -log q(y).

2. **Compute loss**: Given logits [2.0, 1.0, 3.0] and true class 2, compute the cross-entropy loss by hand.

3. **Gradient check**: For the same logits and true class, compute ∂L/∂z for each logit. Verify the formula.

4. **Perplexity**: If cross-entropy loss is 1.5, what is the perplexity?

5. **Softmax temperature**: What happens to the loss if we compute softmax(z/T) for T → 0? For T → ∞?

## What's Next

We have:

- Embeddings (Section 3.2)
- Feed-forward networks (Section 3.3)
- Cross-entropy loss (Section 3.4)

Time to put it all together!

In Section 3.5, we'll **implement a complete character-level neural language model** using our Stage 2 autograd system.

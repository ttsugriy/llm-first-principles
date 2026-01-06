# Section 3.6: Training Dynamics

Having a model and a loss function isn't enough. Making neural networks actually learn requires understanding training dynamics—the interplay of learning rates, initialization, and optimization.

**This section covers the practical art of training neural language models.**

## The Optimization Landscape

### What Are We Optimizing?

The loss function L(θ) defines a surface over parameter space. For our language model:

$$L(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \log P(y_i | x_i; \theta)$$


Where θ includes all embeddings, weights, and biases.

### Visualizing the Landscape

For 2 parameters, we can plot L as a 3D surface:

```
L(θ)
  │    ╱\
  │   /  \\     /\
  │  /    \___/  \
  │ /            \
  └───────────────── θ
```

Real networks have millions of parameters—the surface is in million-dimensional space!

### Key Properties

**Non-convex**: Multiple local minima. No guarantee of finding the global minimum.

**High-dimensional**: In high dimensions, most critical points are saddle points, not local minima. This is actually good—harder to get stuck.

**Flat regions**: Some directions have near-zero gradient. Training can plateau.

## Gradient Descent: The Basic Algorithm

### The Update Rule

Given current parameters θ and learning rate η:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$


We move in the direction of steepest descent (negative gradient).

### Why It Works

Taylor expansion around current point:

$$L(\theta + \Delta\theta) \approx L(\theta) + \nabla L \cdot \Delta\theta$$


To decrease L, we want $\nabla L \cdot \Delta\theta < 0$.

Choosing $\Delta\theta = -\eta \nabla L$:

$$\nabla L \cdot (-\eta \nabla L) = -\eta ||\nabla L||^2 < 0$$


The loss decreases (for small enough η).

### Stochastic Gradient Descent (SGD)

Computing the gradient over all N examples is expensive. Instead:

1. Sample a single example (or mini-batch)
2. Compute gradient on that sample
3. Update parameters
4. Repeat

The gradient estimate is noisy but unbiased:

$$\mathbb{E}[\nabla L_i] = \nabla L$$


This noise can actually help escape local minima!

## The Learning Rate

The learning rate η is the most important hyperparameter.

### Too Small

- Very slow progress
- May never reach good solution
- Training takes forever

```
Loss
  │\
  │ \__________
  │            \_____
  │                  \____...
  └──────────────────────── Epochs
```

### Too Large

- Overshoots optimal values
- Loss oscillates or diverges
- Training becomes unstable

```
Loss
  │
  │    /\    /\    /\
  │   /  \  /  \  /
  │  /    \/    \/
  │ /
  └──────────────────── Epochs
```

### Just Right

- Steady decrease
- Converges to good minimum
- Can explore and then settle

```
Loss
  │\
  │ \
  │  \.
  │   '·..
  │       ''''·····
  └──────────────────── Epochs
```

### Finding the Right Learning Rate

**Rule of thumb**: Start with 0.01, adjust by factors of 10.

**Learning rate finder**: Gradually increase LR, plot loss. Use value just before loss explodes.

**Common values**:
- 0.1: Often too high for deep nets
- 0.01: Good starting point
- 0.001: Common for fine-tuning
- 0.0001: Very conservative

### Our Character LM

For our model, try:
- Start: η = 0.1 (aggressive)
- If unstable: reduce to 0.01
- If too slow: increase to 0.5

## Learning Rate Schedules

A fixed learning rate isn't optimal. Better: change η during training.

### Step Decay

Reduce by factor every K epochs:

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/K \rfloor}$$


Example: Start at 0.1, multiply by 0.5 every 10 epochs.

```python
def step_decay(initial_lr, epoch, decay_rate=0.5, decay_every=10):
    return initial_lr * (decay_rate ** (epoch // decay_every))
```

### Linear Decay

Decrease linearly to zero:

$$\eta_t = \eta_0 \cdot \left(1 - \frac{t}{T}\right)$$


```python
def linear_decay(initial_lr, step, total_steps):
    return initial_lr * (1 - step / total_steps)
```

### Cosine Annealing

Smooth decrease following cosine:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$


Popular in modern training.

### Warmup

Start with tiny learning rate, gradually increase:

$$\eta_t = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}$$


Then decay. Helps stabilize early training when gradients are large.

## Initialization

How we initialize parameters affects training dramatically.

### The Problem with Zeros

If all weights are zero:
- All neurons compute the same thing
- All gradients are the same
- Symmetry is never broken

**Never initialize weights to zero!** (Biases to zero are okay.)

### Random Initialization

Simple approach: small random values.

$$W_{ij} \sim \mathcal{N}(0, \sigma^2)$$


But what should σ be?

### The Variance Problem

Consider a single layer: y = Wx where x ∈ ℝⁿ.

If $x_i$ has variance $\text{Var}(x)$ and $W_{ij}$ has variance σ²:

$$\text{Var}(y_j) = \sum_{i=1}^{n} \text{Var}(W_{ij}) \cdot \text{Var}(x_i) = n \cdot \sigma^2 \cdot \text{Var}(x)$$


The variance grows by factor n!

For deep networks, this compounds: variance explodes or vanishes.

### Xavier/Glorot Initialization

Solution: scale by both input and output dimensions.

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$


Or for uniform distribution:

$$W_{ij} \sim \text{Uniform}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$


This balances variance preservation in both forward and backward passes.

**Simplified view** (forward pass only): If we just want forward variance preserved:

$$\text{Var}(y_j) = n_{\text{in}} \cdot \frac{2}{n_{\text{in}} + n_{\text{out}}} \cdot \text{Var}(x) \approx \text{Var}(x)$$


(Approximately preserved when n_in ≈ n_out.)

### He Initialization

For ReLU activations, Xavier underestimates because ReLU zeros out approximately half the neurons (those with negative input), effectively halving the variance of activations.

He initialization compensates for this:

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

The factor of 2 (compared to 1/n_in) compensates for ReLU zeroing half the activations, maintaining proper variance flow through the network.

### Our Implementation

```python
def init_weights(shape, activation='relu'):
    """Initialize weight matrix with appropriate scaling."""
    n_in, n_out = shape
    if activation == 'relu':
        scale = (2.0 / n_in) ** 0.5  # He
    else:
        scale = (2.0 / (n_in + n_out)) ** 0.5  # Xavier
    return [[Value(random.gauss(0, scale))
             for _ in range(n_in)]
            for _ in range(n_out)]
```

## Batching

Processing one example at a time is inefficient and noisy.

### Mini-Batch Gradient Descent

Process B examples together:

$$\nabla L = \frac{1}{B}\sum_{i=1}^{B} \nabla L_i$$


**Advantages**:
- More stable gradients (averaging reduces variance)
- Computational efficiency (parallelism)
- Better generalization (noise helps)

**Common batch sizes**: 32, 64, 128, 256

### Trade-offs

| Batch Size | Gradient Variance | Computation | Generalization |
|------------|-------------------|-------------|----------------|
| 1 | Very high | Slow | Good |
| 32-128 | Medium | Fast | Good |
| 1000+ | Low | Very fast | May overfit |

### Implementation

```python
def create_batches(examples, batch_size):
    """Split examples into mini-batches."""
    random.shuffle(examples)
    batches = []
    for i in range(0, len(examples), batch_size):
        batches.append(examples[i:i + batch_size])
    return batches


def train_batch(model, batch, learning_rate):
    """Train on a single mini-batch."""
    params = model.parameters()

    # Forward pass and accumulate loss
    total_loss = Value(0.0)
    for context, target in batch:
        loss = model.loss(context, target)
        total_loss = total_loss + loss

    avg_loss = total_loss / len(batch)

    # Zero gradients
    for p in params:
        p.grad = 0.0

    # Backward pass
    avg_loss.backward()

    # Update
    for p in params:
        p.data -= learning_rate * p.grad

    return avg_loss.data
```

## Overfitting and Regularization

### The Overfitting Problem

With enough parameters, the model can memorize training data perfectly—but fail on new data.

**Signs of overfitting**:
- Training loss keeps decreasing
- Validation loss starts increasing
- Large gap between train and validation loss

```
Loss
  │\
  │ \  training
  │  \____________________
  │      ╱
  │     /  validation
  │    /''·····
  └──────────────────────── Epochs
```

### Train/Validation Split

Always evaluate on held-out data:

```python
def split_data(examples, val_fraction=0.1):
    """Split examples into train and validation."""
    n_val = int(len(examples) * val_fraction)
    random.shuffle(examples)
    return examples[n_val:], examples[:n_val]
```

### Early Stopping

Stop training when validation loss stops improving:

```python
def train_with_early_stopping(model, train_examples, val_examples,
                               patience=5, max_epochs=100):
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        # Train one epoch
        train_loss = train_epoch(model, train_examples)

        # Evaluate
        val_loss = evaluate(model, val_examples)

        print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save best model weights here
        else:
            epochs_without_improvement += 1

        # Early stop
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
```

### Weight Decay (L2 Regularization)

Add penalty for large weights:

$$L_{\text{total}} = L + \lambda \sum_i \theta_i^2$$


This encourages smaller weights, reducing overfitting.

```python
def apply_weight_decay(params, learning_rate, weight_decay):
    """Apply L2 regularization."""
    for p in params:
        p.data -= learning_rate * weight_decay * p.data
```

In practice, combine with gradient update:

$$\theta_{t+1} = \theta_t - \eta(\nabla L + \lambda \theta_t) = (1 - \eta\lambda)\theta_t - \eta \nabla L$$


### Dropout

During training, randomly zero some activations:

$$h_i^{\text{dropped}} = h_i \cdot m_i$$


Where $m_i \sim \text{Bernoulli}(1 - p)$ and p is the dropout probability.

At test time, scale by (1-p) or use all activations.

```python
def dropout(x, p=0.5, training=True):
    """Apply dropout to list of Values."""
    if not training:
        return x
    mask = [1 if random.random() > p else 0 for _ in x]
    scale = 1.0 / (1.0 - p)  # Scale to maintain expected value
    return [v * m * scale for v, m in zip(x, mask)]
```

## Monitoring Training

### What to Track

1. **Training loss**: Should decrease
2. **Validation loss**: Should decrease, watch for divergence from training
3. **Perplexity**: exp(loss), more interpretable
4. **Gradient norms**: Should be stable, not exploding/vanishing
5. **Parameter norms**: Shouldn't grow unboundedly

### Implementation

```python
def compute_gradient_norm(params):
    """Compute L2 norm of all gradients."""
    total = sum(p.grad ** 2 for p in params)
    return total ** 0.5


def compute_param_norm(params):
    """Compute L2 norm of all parameters."""
    total = sum(p.data ** 2 for p in params)
    return total ** 0.5
```

### What to Watch For

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Loss stays flat | LR too small, or stuck | Increase LR, reinitialize |
| Loss explodes | LR too large | Reduce LR, gradient clipping |
| Val > Train | Overfitting | Regularization, early stopping |
| Loss oscillates | LR too large | Reduce LR |
| Gradients → 0 | Vanishing gradients | Better init, skip connections |
| Gradients → ∞ | Exploding gradients | Gradient clipping, smaller LR |

## Gradient Clipping

Prevent exploding gradients by capping the gradient norm:

```python
def clip_gradients(params, max_norm):
    """Clip gradients to maximum norm."""
    total_norm = compute_gradient_norm(params)
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for p in params:
            p.grad *= scale
```

This is especially important for language models, where certain inputs can cause large gradients.

## A Complete Training Function

Putting it all together:

```python
def train_model(model, train_data, val_data, config):
    """
    Complete training loop with all best practices.

    config: dict with hyperparameters
        - epochs: max training epochs
        - batch_size: mini-batch size
        - learning_rate: initial learning rate
        - weight_decay: L2 regularization strength
        - max_grad_norm: gradient clipping threshold
        - patience: early stopping patience
    """
    params = model.parameters()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        # Learning rate schedule (linear decay)
        lr = config['learning_rate'] * (1 - epoch / config['epochs'])

        # Training
        model.train_mode = True
        batches = create_batches(train_data, config['batch_size'])

        train_loss = 0.0
        for batch in batches:
            # Forward and backward
            batch_loss = train_batch(model, batch, lr)

            # Gradient clipping
            clip_gradients(params, config['max_grad_norm'])

            # Weight decay
            apply_weight_decay(params, lr, config['weight_decay'])

            train_loss += batch_loss

        train_loss /= len(batches)

        # Validation
        model.train_mode = False
        val_loss = evaluate(model, val_data)

        # Logging
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)
        print(f"Epoch {epoch+1}: "
              f"train_loss={train_loss:.4f} (PPL={train_ppl:.2f}), "
              f"val_loss={val_loss:.4f} (PPL={val_ppl:.2f})")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model
```

## Summary

| Concept | Description | Practical Tip |
|---------|-------------|---------------|
| Learning rate | Step size for updates | Start at 0.01, adjust |
| LR schedule | Change LR over time | Decay helps convergence |
| Initialization | Starting parameter values | Use He for ReLU |
| Batch size | Examples per update | 32-128 typical |
| Weight decay | L2 regularization | 1e-4 to 1e-2 |
| Gradient clipping | Prevent explosion | Max norm 1-5 |
| Early stopping | Prevent overfitting | Patience 5-10 |

**Key insight**: Training neural networks is empirical. Start with defaults, monitor carefully, adjust based on what you observe. There's no substitute for running experiments.

## Exercises

1. **Learning rate experiment**: Train the model with learning rates 0.001, 0.01, 0.1, and 1.0. Plot the training curves. What do you observe?

2. **Initialization comparison**: Compare training with Xavier init vs. random N(0, 1). How long until each converges?

3. **Batch size trade-off**: Train with batch sizes 1, 16, 64, and 256. Compare wall-clock time to reach the same loss.

4. **Early stopping**: Implement early stopping and compare final validation loss with and without it.

5. **Gradient analysis**: Add logging for gradient norms. At what point in training are gradients largest?

## What's Next

We can train our model. But how good is it really?

In Section 3.7, we'll **evaluate our neural language model** and compare it directly to the Markov models from Stage 1. We'll see concrete evidence of the neural advantage.

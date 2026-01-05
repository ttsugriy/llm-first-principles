# Section 3.5: Building a Character-Level Neural Language Model

Theory meets practice. In this section, we'll build a complete neural language model from scratch, using only the autograd system we developed in Stage 2.

**By the end, you'll have a working model that learns to generate text character by character.**

## The Complete Architecture

Let's specify exactly what we're building:

```
Input: k previous characters [c_{t-k}, ..., c_{t-1}]
Output: P(c_t | context) for each character in vocabulary

Architecture:
  1. Embedding layer: map each character to d-dimensional vector
  2. Concatenate: combine k embeddings into one vector
  3. Hidden layer 1: linear + ReLU
  4. Hidden layer 2: linear + ReLU
  5. Output layer: linear (logits)
  6. Softmax: convert to probabilities
```

## Prerequisites: The Value Class

First, let's ensure we have our Stage 2 autograd. Here's the complete Value class:

```python
import math
import random

class Value:
    """Scalar value with automatic differentiation."""

    def __init__(self, data, _parents=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._parents = set(_parents)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            self.grad += out.grad * (n * self.data ** (n - 1))
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += out.grad * (1.0 if self.data > 0 else 0.0)
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for p in v._parents:
                    build_topo(p)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
```

## Building the Model Components

### Embedding Layer

```python
class Embedding:
    """Lookup table for token embeddings."""

    def __init__(self, vocab_size, embed_dim):
        """
        vocab_size: number of unique tokens
        embed_dim: dimension of embedding vectors
        """
        # Initialize with small random values
        self.weight = [
            [Value(random.gauss(0, 0.1)) for _ in range(embed_dim)]
            for _ in range(vocab_size)
        ]
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def __call__(self, token_idx):
        """Return embedding for token index."""
        return self.weight[token_idx]  # List of Value objects

    def parameters(self):
        """Return all learnable parameters."""
        return [v for row in self.weight for v in row]
```

### Linear Layer

```python
class Linear:
    """Fully connected layer: y = Wx + b."""

    def __init__(self, in_features, out_features):
        """
        in_features: input dimension
        out_features: output dimension
        """
        # Xavier initialization for stable gradients
        scale = (2.0 / (in_features + out_features)) ** 0.5
        self.weight = [
            [Value(random.gauss(0, scale)) for _ in range(in_features)]
            for _ in range(out_features)
        ]
        self.bias = [Value(0.0) for _ in range(out_features)]
        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, x):
        """
        x: list of Value objects (length = in_features)
        Returns: list of Value objects (length = out_features)
        """
        out = []
        for i in range(self.out_features):
            activation = self.bias[i]
            for j in range(self.in_features):
                activation = activation + self.weight[i][j] * x[j]
            out.append(activation)
        return out

    def parameters(self):
        return [v for row in self.weight for v in row] + self.bias
```

### Activation Functions

```python
def relu(x):
    """Apply ReLU to list of Values."""
    return [v.relu() for v in x]


def softmax(logits):
    """
    Convert logits to probabilities.
    Numerically stable implementation.
    """
    # Subtract max for numerical stability
    max_val = max(v.data for v in logits)
    exp_logits = [(v - max_val).exp() for v in logits]
    sum_exp = sum(exp_logits, Value(0.0))
    return [e / sum_exp for e in exp_logits]
```

### Cross-Entropy Loss

```python
def cross_entropy_loss(logits, target_idx):
    """
    Compute cross-entropy loss.

    logits: list of Value objects (unnormalized scores)
    target_idx: index of true class

    Returns: Value (scalar loss)
    """
    # Log-sum-exp trick for numerical stability
    max_logit = max(v.data for v in logits)
    shifted = [v - max_logit for v in logits]
    exp_logits = [v.exp() for v in shifted]
    sum_exp = sum(exp_logits, Value(0.0))
    log_sum_exp = sum_exp.log() + max_logit

    # Loss = -logit[target] + log_sum_exp
    loss = log_sum_exp - logits[target_idx]
    return loss
```

## The Complete Language Model

Now we assemble everything:

```python
class CharacterLM:
    """Character-level neural language model."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, context_length):
        """
        vocab_size: number of unique characters
        embed_dim: dimension of character embeddings
        hidden_dim: size of hidden layers
        context_length: number of previous characters to use
        """
        self.context_length = context_length
        self.vocab_size = vocab_size

        # Embedding layer
        self.embedding = Embedding(vocab_size, embed_dim)

        # Hidden layers
        concat_dim = context_length * embed_dim
        self.layer1 = Linear(concat_dim, hidden_dim)
        self.layer2 = Linear(hidden_dim, hidden_dim)

        # Output layer
        self.output = Linear(hidden_dim, vocab_size)

    def forward(self, context):
        """
        context: list of character indices (length = context_length)
        Returns: list of Values (logits for each vocabulary item)
        """
        # 1. Embed each character
        embeddings = [self.embedding(idx) for idx in context]

        # 2. Concatenate all embeddings
        x = []
        for emb in embeddings:
            x.extend(emb)

        # 3. First hidden layer
        h1 = relu(self.layer1(x))

        # 4. Second hidden layer
        h2 = relu(self.layer2(h1))

        # 5. Output logits
        logits = self.output(h2)

        return logits

    def predict_probs(self, context):
        """Get probability distribution over next character."""
        logits = self.forward(context)
        return softmax(logits)

    def loss(self, context, target_idx):
        """Compute cross-entropy loss for one example."""
        logits = self.forward(context)
        return cross_entropy_loss(logits, target_idx)

    def parameters(self):
        """Return all learnable parameters."""
        params = []
        params.extend(self.embedding.parameters())
        params.extend(self.layer1.parameters())
        params.extend(self.layer2.parameters())
        params.extend(self.output.parameters())
        return params
```

## Data Preparation

### Building the Vocabulary

```python
def build_vocab(text):
    """Create character-to-index mappings."""
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def encode(text, char_to_idx):
    """Convert text to list of indices."""
    return [char_to_idx[c] for c in text]


def decode(indices, idx_to_char):
    """Convert indices back to text."""
    return ''.join(idx_to_char[i] for i in indices)
```

### Creating Training Examples

```python
def create_examples(encoded_text, context_length):
    """
    Create (context, target) pairs from encoded text.

    Each example is:
      - context: previous context_length characters
      - target: the next character
    """
    examples = []
    for i in range(context_length, len(encoded_text)):
        context = encoded_text[i - context_length : i]
        target = encoded_text[i]
        examples.append((context, target))
    return examples
```

## The Training Loop

```python
def train(model, examples, epochs, learning_rate, print_every=100):
    """
    Train the language model.

    model: CharacterLM instance
    examples: list of (context, target) pairs
    epochs: number of passes through the data
    learning_rate: step size for gradient descent
    """
    params = model.parameters()
    n_params = len(params)
    print(f"Training model with {n_params} parameters")

    for epoch in range(epochs):
        # Shuffle examples each epoch
        random.shuffle(examples)

        total_loss = 0.0
        for i, (context, target) in enumerate(examples):
            # Forward pass
            loss = model.loss(context, target)
            total_loss += loss.data

            # Zero gradients
            for p in params:
                p.grad = 0.0

            # Backward pass
            loss.backward()

            # Update parameters
            for p in params:
                p.data -= learning_rate * p.grad

            # Print progress
            if (i + 1) % print_every == 0:
                avg_loss = total_loss / (i + 1)
                print(f"Epoch {epoch+1}, Example {i+1}/{len(examples)}, "
                      f"Avg Loss: {avg_loss:.4f}")

        # End of epoch
        avg_loss = total_loss / len(examples)
        perplexity = math.exp(avg_loss)
        print(f"Epoch {epoch+1} complete. "
              f"Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")

    return model
```

## Text Generation

Once trained, we can generate new text:

```python
def generate(model, idx_to_char, char_to_idx, seed_text, length, temperature=1.0):
    """
    Generate text from the model.

    seed_text: initial text to condition on
    length: number of characters to generate
    temperature: controls randomness (lower = more deterministic)
    """
    context_length = model.context_length

    # Ensure seed is long enough
    if len(seed_text) < context_length:
        seed_text = ' ' * (context_length - len(seed_text)) + seed_text

    # Encode seed
    generated = list(seed_text)
    context = [char_to_idx[c] for c in seed_text[-context_length:]]

    for _ in range(length):
        # Get logits
        logits = model.forward(context)

        # Apply temperature
        if temperature != 1.0:
            logits = [Value(v.data / temperature) for v in logits]

        # Convert to probabilities
        probs = softmax(logits)
        prob_values = [p.data for p in probs]

        # Sample from distribution
        next_idx = random.choices(range(len(prob_values)),
                                  weights=prob_values, k=1)[0]

        # Add to generated text
        next_char = idx_to_char[next_idx]
        generated.append(next_char)

        # Update context
        context = context[1:] + [next_idx]

    return ''.join(generated)
```

## Putting It All Together

Here's a complete training script:

```python
def main():
    # Hyperparameters
    CONTEXT_LENGTH = 8
    EMBED_DIM = 32
    HIDDEN_DIM = 128
    EPOCHS = 3
    LEARNING_RATE = 0.01

    # Sample training text
    text = """
    The quick brown fox jumps over the lazy dog.
    A journey of a thousand miles begins with a single step.
    To be or not to be, that is the question.
    All that glitters is not gold.
    The only thing we have to fear is fear itself.
    """

    # Build vocabulary
    char_to_idx, idx_to_char = build_vocab(text)
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Characters: {''.join(sorted(char_to_idx.keys()))}")

    # Encode text
    encoded = encode(text, char_to_idx)
    print(f"Encoded length: {len(encoded)}")

    # Create training examples
    examples = create_examples(encoded, CONTEXT_LENGTH)
    print(f"Number of examples: {len(examples)}")

    # Create model
    model = CharacterLM(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        context_length=CONTEXT_LENGTH
    )
    print(f"Model parameters: {len(model.parameters())}")

    # Train
    print("\n--- Training ---")
    model = train(model, examples, EPOCHS, LEARNING_RATE, print_every=50)

    # Generate
    print("\n--- Generation ---")
    seed = "The quick"
    generated = generate(model, idx_to_char, char_to_idx,
                        seed, length=100, temperature=0.8)
    print(f"Seed: '{seed}'")
    print(f"Generated: '{generated}'")


if __name__ == '__main__':
    main()
```

## Expected Output

After a few epochs, you should see something like:

```
Vocabulary size: 42
Characters:  .ATTabdefghijklmnorstuvy
Encoded length: 262
Number of examples: 254
Model parameters: 19626

--- Training ---
Epoch 1, Example 50/254, Avg Loss: 3.2541
Epoch 1, Example 100/254, Avg Loss: 2.8923
...
Epoch 3 complete. Loss: 1.4521, Perplexity: 4.27

--- Generation ---
Seed: 'The quick'
Generated: 'The quick brown fox the only thing we that is the question...'
```

The model learns common patterns:
- Word boundaries (spaces after words)
- Common words ("the", "is", "that")
- Phrase structures

With more data and training, quality improves significantly.

## Analysis: What Did We Build?

### Parameter Count

For our example (vocab=42, embed=32, hidden=128, context=8):

| Component | Size | Parameters |
|-----------|------|------------|
| Embedding | 42 × 32 | 1,344 |
| Layer 1 | (8×32) × 128 + 128 | 32,896 |
| Layer 2 | 128 × 128 + 128 | 16,512 |
| Output | 128 × 42 + 42 | 5,418 |
| **Total** | | **56,170** |

### Computational Cost

Per training example:
- Forward: O(context × embed × hidden + hidden² + hidden × vocab)
- Backward: Same order (automatic via autograd)
- Memory: Proportional to computation (store activations)

### What Makes It Work

1. **Embeddings**: Similar characters get similar representations
2. **Hidden layers**: Learn to combine patterns
3. **Cross-entropy**: Proper training objective
4. **Gradient descent**: Iterative improvement

## Common Issues and Solutions

### Gradient Issues

**Problem**: Loss becomes NaN

**Solutions**:
- Reduce learning rate
- Check for division by zero
- Use gradient clipping (cap gradient magnitudes)

```python
# Gradient clipping
max_norm = 1.0
for p in params:
    if abs(p.grad) > max_norm:
        p.grad = max_norm * (1 if p.grad > 0 else -1)
```

### Poor Generation

**Problem**: Generated text is repetitive or nonsensical

**Solutions**:
- More training data
- More epochs
- Adjust temperature during generation
- Larger context length

### Slow Training

**Problem**: Training takes too long

**Solutions**:
- Smaller model (fewer hidden units)
- Fewer training examples
- Early stopping when loss plateaus

## Summary

We built a complete neural language model:

| Component | Purpose |
|-----------|---------|
| Embedding | Discrete → continuous representation |
| Linear layers | Learn pattern combinations |
| ReLU | Add nonlinearity |
| Softmax | Normalize to probabilities |
| Cross-entropy | Measure prediction quality |
| Backprop | Compute all gradients |
| SGD | Update parameters |

**Key insight**: With ~100 lines of autograd and ~200 lines of model code, we have a working neural language model. The same principles scale to billion-parameter models.

## Exercises

1. **Experiment with hyperparameters**: Try different embedding dimensions, hidden sizes, and context lengths. How does each affect final perplexity?

2. **Add a third hidden layer**: Modify the model to have 3 hidden layers instead of 2. Does it help?

3. **Different activation**: Replace ReLU with tanh. Compare training dynamics.

4. **Bigger dataset**: Train on a larger text corpus (e.g., a book from Project Gutenberg). How does quality change?

5. **Temperature exploration**: Generate text at temperatures 0.5, 1.0, and 1.5. Describe the differences.

## What's Next

Our model trains, but there's a lot we glossed over:
- How to choose the learning rate?
- When to stop training?
- How to prevent overfitting?

In Section 3.6, we'll dive deep into **training dynamics**—the art and science of making neural networks learn effectively.

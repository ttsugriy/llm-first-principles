# Stage 3 Exercises

## Conceptual Questions

### Exercise 3.1: Embedding Geometry
Word embeddings place words in a continuous space.

**a)** If "king" - "man" + "woman" ≈ "queen", what does this tell us about the embedding space?
**b)** Why can't we do this with one-hot encodings?
**c)** What would "Paris" - "France" + "Germany" approximate?

### Exercise 3.2: Hidden Layer Purpose
A neural language model has architecture: Embedding → Hidden → Softmax

**a)** What does the hidden layer learn to do?
**b)** What happens if we remove it (Embedding → Softmax directly)?
**c)** What happens if we add 10 more hidden layers?

### Exercise 3.3: Cross-Entropy Intuition
For a vocabulary of 10,000 words:

**a)** What is the cross-entropy loss if the model assigns probability 0.5 to the correct word?
**b)** What if it assigns 0.0001?
**c)** What is the minimum possible loss (perfect prediction)?

### Exercise 3.4: Softmax Temperature
The softmax function is: softmax(z)_i = exp(z_i) / Σ exp(z_j)

**a)** If all logits are equal, what is the output distribution?
**b)** If z = [10, 0, 0], approximately what are the probabilities?
**c)** What happens to the distribution as the largest logit grows to infinity?

---

## Implementation Exercises

### Exercise 3.5: Embedding from Scratch
Implement an embedding layer:

```python
class Embedding:
    def __init__(self, vocab_size: int, dim: int):
        self.W = np.random.randn(vocab_size, dim) * 0.01

    def forward(self, indices: np.ndarray) -> np.ndarray:
        """
        Args:
            indices: token indices [batch_size, seq_len]
        Returns:
            embeddings [batch_size, seq_len, dim]
        """
        # TODO: Implement (hint: fancy indexing)
        pass

    def backward(self, grad: np.ndarray, indices: np.ndarray) -> None:
        """Accumulate gradients into self.W_grad"""
        # TODO: Implement (hint: np.add.at)
        pass
```

### Exercise 3.6: Two-Layer Network
Implement a complete 2-layer neural language model:

```python
class NeuralLM:
    def __init__(self, vocab_size, embed_dim, hidden_dim, context_size):
        # TODO: Initialize weights
        pass

    def forward(self, context_indices):
        """
        context_indices: [batch, context_size]
        Returns: logits [batch, vocab_size]
        """
        # TODO: embedding → flatten → hidden → output
        pass

    def loss(self, logits, targets):
        """Cross-entropy loss"""
        # TODO
        pass
```

### Exercise 3.7: Softmax Stability
Implement numerically stable softmax and log-softmax:

```python
def stable_softmax(z: np.ndarray) -> np.ndarray:
    """Softmax that doesn't overflow"""
    # TODO: subtract max before exp
    pass

def log_softmax(z: np.ndarray) -> np.ndarray:
    """Log of softmax, computed stably"""
    # TODO: avoid computing softmax then taking log
    pass
```

### Exercise 3.8: Perplexity Tracking
Add perplexity tracking to training:

```python
def train_epoch(model, data, optimizer):
    total_loss = 0
    n_tokens = 0

    for batch in data:
        loss = model.train_step(batch)
        total_loss += loss * batch.n_tokens
        n_tokens += batch.n_tokens

    avg_loss = total_loss / n_tokens
    perplexity = ???  # TODO: compute from avg_loss
    return perplexity
```

---

## Challenge Exercises

### Exercise 3.9: Embedding Visualization
Train a small language model on a corpus, then:

**a)** Extract the learned embeddings
**b)** Apply t-SNE or PCA to reduce to 2D
**c)** Plot and look for clusters (do similar words cluster together?)

### Exercise 3.10: Context Window Analysis
Train models with different context sizes (1, 2, 4, 8) on the same data.

**a)** Plot perplexity vs context size
**b)** At what point do diminishing returns set in?
**c)** Why doesn't context size 1000 help much more than 8?

### Exercise 3.11: Compare to N-gram
On the same dataset:

**a)** Train a trigram model with smoothing
**b)** Train a neural LM with context size 2
**c)** Compare perplexity on test set
**d)** Compare generation quality

---

## Checking Your Work

- **Test suite**: See `code/stage-03/tests/test_neural_lm.py` for expected behavior
- **Reference implementation**: Compare with `code/stage-03/neural_lm.py`
- **Self-check**: Verify loss decreases during training and perplexity is reasonable
---

## Mini-Project: Neural Bigram Model

Build a neural language model that outperforms your Markov chain from Stage 1.

### Requirements

1. **Architecture**: Embedding → Linear → Softmax
2. **Training**: Train on the same Shakespeare data from Stage 1
3. **Comparison**: Compare perplexity with your Stage 1 model

### Deliverables

- [ ] Working neural bigram model
- [ ] Training loop with loss plotting
- [ ] Perplexity comparison table:
  | Model | Perplexity |
  |-------|------------|
  | Bigram Markov | ? |
  | Neural Bigram | ? |
- [ ] Generated text samples

### Extension

Extend to trigrams (2 token context). How much does adding context help?

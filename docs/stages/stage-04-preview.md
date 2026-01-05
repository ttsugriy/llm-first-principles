# Stage 4: Recurrent Neural Networks (Coming Soon)

## Unbounded Context: Learning to Remember

Our Stage 3 neural language model uses a fixed context window. But language has structure that spans paragraphs, pages, even entire documents. How do we model dependencies of arbitrary length?

**Recurrent Neural Networks (RNNs)** solve this by maintaining a hidden state that accumulates information over time.

## Planned Topics

### The Limitations of Fixed Context
Why feed-forward networks can't handle variable-length sequences naturally.

### The Recurrent Architecture
Sharing weights across time steps. How RNNs process sequences.

### Backpropagation Through Time
Extending our autograd to handle temporal dependencies. The chain rule across time.

### Vanishing and Exploding Gradients
The fundamental challenge of learning long-range dependencies. Mathematical analysis.

### LSTM and GRU
Gated architectures that enable learning over longer sequences. Deriving the gate mechanisms.

### Building an RNN Language Model
Implementation from scratch, building on our Stage 2-3 foundations.

## What You'll Build

A recurrent language model that:

- Processes sequences of arbitrary length
- Maintains memory across the sequence
- Uses gating to control information flow
- Generates more coherent long-form text

## Prerequisites

- Stage 1: Probability and language modeling
- Stage 2: Automatic differentiation
- Stage 3: Neural language models (embeddings, training)

---

*This stage is under active development. Check back soon!*

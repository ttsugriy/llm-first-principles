# Section 3.2: Embeddings — From Discrete to Continuous

The fundamental problem: language is made of discrete symbols (characters, words), but neural networks operate on continuous vectors.

**Embeddings** bridge this gap. They convert discrete tokens into continuous representations that capture meaning.

This section derives embeddings from first principles and shows why they're the key to neural language modeling.

## The Problem with Discrete Representations

### One-Hot Encoding

The naive way to represent tokens: one-hot vectors.

For vocabulary {a, b, c, d} with |V| = 4:

$$\text{one\_hot}(a) = [1, 0, 0, 0]$$

$$\text{one\_hot}(b) = [0, 1, 0, 0]$$

$$\text{one\_hot}(c) = [0, 0, 1, 0]$$

$$\text{one\_hot}(d) = [0, 0, 0, 1]$$


Each token gets a unique position; all other entries are zero.

### Why One-Hot Fails

**Problem 1: No Similarity**

For any two different tokens i ≠ j:

$$\text{one\_hot}(i) \cdot \text{one\_hot}(j) = 0$$


All tokens are orthogonal. "a" and "b" are as dissimilar as "a" and "7".

**Problem 2: Dimensionality**

Vector size = vocabulary size.

For characters: manageable (~100 dimensions)
For words: massive (~50,000+ dimensions)

**Problem 3: No Generalization**

A weight connecting to one-hot position i affects ONLY token i.

Learning about "cat" provides zero information about "dog".

### The Geometric View

In one-hot space:

- All vectors have length 1
- All pairs are distance √2 apart (since ||eᵢ - eⱼ||² = 2 for i ≠ j)
- No structure, no clusters, no relationships

The notation **||v||** denotes the **Euclidean length (norm)** of vector v, calculated as √(v₁² + v₂² + ... + vₙ²). The squared norm ||v||² is simply the sum of squared components.

This is a failure of representation, not of the neural network itself.

## The Embedding Solution

### The Key Idea

Instead of one-hot vectors, represent each token as a **learned dense vector**.

For vocabulary size |V| and embedding dimension d (where d << |V|):

$$E \in \mathbb{R}^{|V| \times d}$$


Each row of E is the embedding for one token:

$$\text{embed}(i) = E[i, :] \in \mathbb{R}^d$$


### Example

For vocabulary {a, b, c, d} with d = 3:

$$E = \begin{bmatrix} 0.2 & -0.5 & 0.1 \\ 0.3 & -0.4 & 0.2 \\ -0.1 & 0.8 & 0.3 \\ 0.5 & 0.2 & -0.7 \end{bmatrix}$$


Then:

- embed(a) = [0.2, -0.5, 0.1]
- embed(b) = [0.3, -0.4, 0.2]
- embed(c) = [-0.1, 0.8, 0.3]
- embed(d) = [0.5, 0.2, -0.7]

### What Changes?

| Property | One-Hot | Embedding |
|----------|---------|-----------|
| Dimension | |V| | d << |V| |
| Similarity | All orthogonal | Learned from data |
| Parameters | 0 (fixed) | |V| × d (learned) |
| Structure | None | Emerges from training |

## Mathematical Formulation

### Embeddings as Matrix Lookup

The embedding operation is mathematically simple:

$$\text{embed}(i) = E[i, :]$$


This is equivalent to:

$$\text{embed}(i) = \text{one\_hot}(i) \cdot E$$


**Proof**:

Let e_i = one_hot(i) be the i-th standard basis vector.

$$e_i \cdot E = [0, ..., 1, ..., 0] \cdot E = \text{i-th row of } E = E[i, :]$$


The one-hot vector "selects" the corresponding row of E.

### Why This Matters for Backpropagation

During training, we need gradients of the loss with respect to E.

If the loss is L and we used embedding E[i, :] in the forward pass:

$$\frac{\partial L}{\partial E[i, :]} = \frac{\partial L}{\partial \text{embed}(i)}$$


Only the i-th row of E gets a gradient—the rows for tokens not used in this example get zero gradient.

This is **sparse gradient updates**: each training example only modifies the embeddings of tokens it contains.

### Efficient Implementation

Although mathematically equivalent to matrix multiplication, we implement embeddings as **table lookup**:

```python
class Embedding:
    def __init__(self, vocab_size, embed_dim):
        # Initialize randomly
        self.weight = [[random.gauss(0, 0.1) for _ in range(embed_dim)]
                       for _ in range(vocab_size)]

    def forward(self, token_idx):
        # O(1) lookup, not O(vocab_size) multiplication
        return self.weight[token_idx]
```

This is O(d) instead of O(|V| × d)—a massive speedup.

## Why Embeddings Enable Generalization

### The Distributional Hypothesis

Linguist J.R. Firth (1957):

> "You shall know a word by the company it keeps."

Tokens that appear in similar contexts should have similar meanings.

### How Training Enforces This

Consider training on:

- "the **cat** sat on the mat"
- "the **dog** sat on the rug"

Both "cat" and "dog":

1. Follow "the"
2. Precede "sat"
3. Appear in similar grammatical positions

During training:

- Both receive gradients pushing them to predict "sat"
- Both receive gradients from "the" predictions
- These similar gradient signals push them toward similar embeddings

**Result**: embed("cat") ≈ embed("dog") emerges automatically!

### Formal Statement

If two tokens t₁ and t₂ appear in similar contexts:

$$P(\text{context} | t_1) \approx P(\text{context} | t_2)$$


Then training will push their embeddings together:

$$\text{embed}(t_1) \approx \text{embed}(t_2)$$


This is not programmed—it's a consequence of the optimization objective.

## The Geometry of Embedding Space

### Distances Encode Relationships

In a trained embedding space:

**Cosine Similarity**:

$$\text{sim}(u, v) = \frac{u \cdot v}{||u|| \cdot ||v||}$$


Ranges from -1 (opposite) to +1 (identical direction).

**Euclidean Distance**:

$$d(u, v) = ||u - v||_2$$


Small distance = similar tokens.

### What Emerges

After training on language data, embedding spaces exhibit:

1. **Clusters**: Similar tokens group together
   - Vowels cluster, consonants cluster
   - Digits cluster together
   - Punctuation forms its own region

2. **Directions**: Some directions encode specific properties
   - Moving in one direction might shift from "noun-like" to "verb-like"
   - Another direction might encode frequency

3. **Linear Relationships** (for word embeddings):
   - king - man + woman ≈ queen
   - Paris - France + Italy ≈ Rome

For character-level models, the structure is simpler but still meaningful.

## Embedding Dimensions

### How Many Dimensions?

This is a hyperparameter choice. Common values:

| Token Type | Vocabulary Size | Typical Embedding Dim |
|------------|-----------------|----------------------|
| Characters | 50-100 | 16-64 |
| Subwords | 30,000-50,000 | 256-768 |
| Words | 50,000-100,000 | 100-300 |

### Trade-offs

**Too few dimensions** (d too small):
- Can't capture all distinctions
- Different tokens forced to share representation
- Underfitting

**Too many dimensions** (d too large):
- More parameters to learn
- Risk of overfitting
- Diminishing returns

### The Right Number

Rule of thumb: d should be large enough to express the meaningful distinctions in your vocabulary.

For characters (~100 tokens): d = 32-64 usually sufficient
For words (~50,000 tokens): d = 256-512 often needed

A common heuristic: d ≈ |V|^{1/4} (fourth root of vocabulary size).

## Context Representation

For language modeling, we need to represent not just one token, but a sequence of context tokens.

### Concatenation

The simplest approach: concatenate embeddings.

For context [t₁, t₂, t₃] with embedding dimension d:

$$x = [\text{embed}(t_1); \text{embed}(t_2); \text{embed}(t_3)] \in \mathbb{R}^{3d}$$


This preserves position information:

- Dimensions [0:d] represent first position
- Dimensions [d:2d] represent second position
- Dimensions [2d:3d] represent third position

### Why Concatenation?

Position matters in language:

- "dog bites man" ≠ "man bites dog"

Concatenation lets the network learn position-specific patterns.

### Alternative: Addition (Not for Us)

Some architectures add embeddings instead:

$$x = \text{embed}(t_1) + \text{embed}(t_2) + \text{embed}(t_3)$$


This loses position information but reduces dimensionality.

We'll revisit this trade-off when we study attention mechanisms in Stage 7.

## Implementing Embeddings with Autograd

Using our Stage 2 Value class, here's a simple embedding layer:

```python
class Embedding:
    def __init__(self, vocab_size, embed_dim):
        # Each row is a token's embedding
        self.weight = [
            [Value(random.gauss(0, 0.1)) for _ in range(embed_dim)]
            for _ in range(vocab_size)
        ]

    def __call__(self, token_idx):
        """Return the embedding for token_idx."""
        return self.weight[token_idx]  # List of Value objects

    def parameters(self):
        """Return all learnable parameters."""
        return [v for row in self.weight for v in row]
```

### Gradient Flow

When we use an embedding in a forward pass:

```python
# Get embedding for token 5
emb = embedding(5)  # List of Value objects

# Use in computation
hidden = sum(w * e for w, e in zip(weights, emb))
loss = ...
loss.backward()

# Now embedding.weight[5] has gradients!
for i, v in enumerate(embedding.weight[5]):
    print(f"Gradient for dim {i}: {v.grad}")
```

The gradient flows from the loss back through the computation to the specific embedding row that was used.

## The Embedding Matrix as a Learned Dictionary

Another perspective: the embedding matrix E is a **dictionary** mapping tokens to their "meanings" (as vectors).

| Token Index | Traditional Dictionary | Embedding "Dictionary" |
|-------------|----------------------|----------------------|
| 0 ('a') | "First letter of alphabet" | [0.23, -0.15, 0.82, ...] |
| 1 ('b') | "Second letter" | [0.31, -0.22, 0.71, ...] |
| ... | ... | ... |

The difference: the vector "definitions" are:

- Learned from data, not written by humans
- Optimized for the prediction task
- Capture statistical patterns, not explicit semantics

## Initialization Matters

### Why Initialization?

Before training, embeddings must start somewhere. The initialization affects:

- How quickly training converges
- Whether training gets stuck in poor local minima
- The scale of activations in early layers

### Common Initializations

**Random Normal**:

$$E[i,j] \sim \mathcal{N}(0, \sigma^2)$$


With σ typically 0.01 to 0.1.

**Uniform**:

$$E[i,j] \sim \text{Uniform}(-a, a)$$


With a typically 0.05 to 0.5.

**Xavier/Glorot** (common for neural networks):

$$E[i,j] \sim \mathcal{N}\left(0, \frac{1}{d}\right)$$


For embeddings, we use 1/d since the "input" is effectively one-hot (dimension 1) and output is d.

### Our Choice

For character embeddings with d = 32:

$$E[i,j] \sim \mathcal{N}\left(0, 0.1\right)$$


Small random values that will be refined by training.

## Summary

| Concept | Explanation |
|---------|-------------|
| One-hot | Sparse, high-dimensional, no similarity |
| Embedding | Dense, low-dimensional, learned similarity |
| Embedding matrix | E ∈ ℝ^{|V|×d}, lookup table of vectors |
| Lookup operation | embed(i) = E[i,:], O(d) not O(|V|d) |
| Why it works | Similar contexts → similar gradients → similar embeddings |
| Geometry | Clusters, directions, linear relationships |
| Context representation | Concatenate embeddings, preserves position |

**Key insight**: Embeddings convert the discrete symbol problem into a continuous optimization problem. Similar tokens end up with similar embeddings because they receive similar training signals. This is the foundation of neural language modeling.

## Exercises

1. **One-hot distances**: Prove that for any two different one-hot vectors, the Euclidean distance is √2.

2. **Embedding equivalence**: Show that one_hot(i) · E = E[i,:] by writing out the matrix multiplication explicitly for a 3×2 embedding matrix.

3. **Parameter count**: For a vocabulary of 10,000 tokens and embedding dimension 256, how many parameters are in the embedding layer? How does this compare to one-hot dimension?

4. **Gradient sparsity**: If a training batch contains 32 examples, each with 10 tokens, at most how many rows of the embedding matrix receive non-zero gradients?

5. **Similarity emergence**: Describe a minimal training scenario (contexts and next tokens) that would cause embeddings for 'x' and 'y' to become similar.

## What's Next

We can now represent tokens as continuous vectors. But how do we use these vectors to predict the next token?

In Section 3.3, we'll build **feed-forward neural networks** that transform embedded contexts into probability distributions over the vocabulary.

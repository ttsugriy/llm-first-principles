# Spiral Structure: Complete Stage Breakdown

## Overview

This document details all 18 stages organized into 5 spirals. Each stage includes:
- Learning objectives
- Key concepts
- Mathematical derivations required
- Code implementations
- Visualizations
- Trade-offs to analyze
- Connections to other stages
- Estimated length

---

# SPIRAL 1: FOUNDATIONS & FIRST MODELS

**Theme:** "It works! But why?"

**Narrative Arc:** We build progressively sophisticated language models, from pure counting to neural networks. Each step is motivated by limitations of the previous approach.

---

## Stage 1: The Simplest Language Model (Markov Chains)

### Learning Objectives
- Understand language modeling as probability distribution over sequences
- Derive the autoregressive factorization from chain rule
- Implement training as counting (and prove it's MLE)
- Implement generation as sampling
- Understand perplexity as evaluation metric
- See the context-length trade-off clearly

### Key Concepts

| Concept | Intuition | Mathematical Form |
|---------|-----------|-------------------|
| Language Model | Probability over sequences | P(x₁, x₂, ..., xₙ) |
| Chain Rule | Factorize joint into conditionals | ∏ᵢ P(xᵢ\|x₁,...,xᵢ₋₁) |
| Markov Assumption | Future depends only on recent past | P(xᵢ\|x<ᵢ) ≈ P(xᵢ\|xᵢ₋ₖ,...,xᵢ₋₁) |
| Maximum Likelihood | Find parameters that maximize probability of data | argmax_θ ∏ P(data\|θ) |
| Perplexity | Effective vocabulary size / uncertainty | exp(-1/N · Σ log P(xᵢ\|context)) |

### Mathematical Derivations

1. **Chain Rule of Probability**
   - State and prove the general chain rule
   - Apply to sequences

2. **Maximum Likelihood Estimation**
   - Define likelihood and log-likelihood
   - Show that for categorical distributions, MLE = counting
   - Derive: P*(next|prev) = count(prev,next) / count(prev)

3. **Perplexity**
   - Derive from cross-entropy
   - Prove interpretation as "branching factor"
   - Show connection to compression (bits per character)

4. **Smoothing**
   - The zero-probability problem
   - Laplace (add-one) smoothing derivation
   - Why this is Bayesian (connection to Dirichlet prior)

### Code Implementations

```
stage-01/
├── markov_chain.py          # Core Markov chain class
├── train_bigram.py          # Training on text corpus
├── train_ngram.py           # Generalized n-gram training
├── generate.py              # Sampling and generation
├── evaluate.py              # Perplexity computation
├── visualize_transitions.py # Transition matrix heatmap
└── experiments/
    ├── order_comparison.py  # Compare n-gram orders
    └── smoothing_effects.py # Effect of smoothing on perplexity
```

### Visualizations

1. **Transition Matrix Heatmap** (order-1)
   - 27×27 matrix for character-level
   - Show sparsity at higher orders

2. **Perplexity vs. N-gram Order**
   - Training perplexity (decreases)
   - Validation perplexity (decreases then increases)
   - Highlight overfitting

3. **Sample Quality Ladder**
   - Show samples from order 1, 2, 3, 5
   - Visual progression of quality

4. **State Space Explosion**
   - |V|^k growth visualization
   - Show when most states are unseen

### Trade-offs to Analyze

| Trade-off | Low End | High End | Sweet Spot |
|-----------|---------|----------|------------|
| N-gram Order | Less context, worse predictions | More context, sparse data | 3-5 for characters |
| Vocabulary Size | More tokens per sequence | Fewer tokens, larger state space | Depends on data size |
| Smoothing Strength | Overfit to data | Too uniform | Validate empirically |

### Connections
- **Forward:** Motivates neural networks (Stage 4) — can we generalize beyond exact matches?
- **Forward:** Perplexity remains our metric through transformers
- **Forward:** Autoregressive generation is exactly the same pattern in GPT

### Pólya Structure

**Understand the Problem:**
- What is the simplest model that can generate text?
- What does "good generation" even mean?

**Devise a Plan:**
- Use the chain rule to factorize the problem
- Make a simplifying assumption (Markov property)
- Count transitions to estimate probabilities

**Carry Out the Plan:**
- Implement counting
- Implement sampling
- Evaluate with perplexity

**Look Back:**
- The counting *is* optimization (MLE equivalence)
- Perplexity has a beautiful interpretation
- But we can't capture long-range dependencies

### Estimated Length
- Main text: 8,000-10,000 words
- Code: ~500 lines
- Time to read: 45-60 minutes

---

## Stage 2: Automatic Differentiation (Micrograd Extended)

### Learning Objectives
- Understand derivatives from first principles
- Derive the chain rule with proof
- Build computational graphs
- Implement forward and backward passes
- Understand forward-mode vs reverse-mode AD
- Analyze memory cost of storing activations

### Key Concepts

| Concept | Intuition | Mathematical Form |
|---------|-----------|-------------------|
| Derivative | Instantaneous rate of change | lim_{h→0} [f(x+h) - f(x)] / h |
| Chain Rule | Derivative of composition | d/dx f(g(x)) = f'(g(x)) · g'(x) |
| Computational Graph | Recipe for computing a function | DAG of operations |
| Forward Mode AD | Compute derivatives alongside values | (v, v') pairs, v' = Σ (∂v/∂u) · u' |
| Reverse Mode AD | Backpropagate sensitivities | v̄ = Σ (∂w/∂v) · w̄ |

### Mathematical Derivations

1. **Derivative from First Principles**
   - The limit definition
   - Derive for: x², xⁿ, eˣ, log(x), sin(x)
   - Product rule and quotient rule

2. **Chain Rule**
   - Statement and proof
   - Multivariable generalization (Jacobian)
   - Why composition decomposes this way

3. **Forward vs Reverse Mode**
   - Forward: O(n) for n inputs, O(1) for single output
   - Reverse: O(1) for n inputs, O(m) for m outputs
   - Why reverse mode for neural networks (one loss, many parameters)

4. **Memory Analysis**
   - Why we store activations
   - Memory = O(depth × width)
   - Foreshadow gradient checkpointing

### Code Implementations

```
stage-02/
├── value.py                 # Scalar Value class with grad
├── tensor.py                # Tensor operations (extended)
├── operations/
│   ├── basic.py             # add, mul, pow, neg
│   ├── transcendental.py    # exp, log, sin, cos
│   ├── tensor_ops.py        # matmul, sum, reshape
│   └── backward.py          # All backward implementations
├── engine.py                # Topological sort, backward pass
├── nn.py                    # Simple neural network layers
├── visualize_graph.py       # Computational graph visualization
└── tests/
    ├── test_gradients.py    # Numerical gradient checking
    └── test_chain_rule.py   # Verify chain rule
```

### Visualizations

1. **Computational Graph**
   - DAG showing forward computation
   - Annotate with values and gradients

2. **Gradient Flow**
   - Animate backward pass
   - Show how gradients accumulate

3. **Forward vs Reverse Mode**
   - Side-by-side comparison
   - Highlight different orders of computation

4. **Memory Timeline**
   - Show activation storage during forward
   - Show release during backward

### Trade-offs to Analyze

| Trade-off | Option A | Option B | Modern Choice |
|-----------|----------|----------|---------------|
| AD Mode | Forward (simple) | Reverse (efficient for NN) | Reverse |
| Graph Type | Static (fast) | Dynamic (flexible) | Dynamic (PyTorch) |
| Precision | FP64 (accurate) | FP32 (faster) | FP32/FP16 mixed |

### Connections
- **Backward:** Relies on Stage 1's understanding of optimization goal
- **Forward:** Foundation for all neural network training
- **Forward:** Memory analysis sets up Stage 11-12

### Pólya Structure

**Understand the Problem:**
- How do we compute gradients of complex functions?
- Why do we need gradients at all? (optimization)

**Devise a Plan:**
- Break complex functions into simple operations
- Derive gradient for each simple operation
- Chain them together

**Carry Out the Plan:**
- Build Value class with gradient storage
- Implement backward() methods
- Create topological sort for ordering

**Look Back:**
- Forward vs reverse mode: why reverse for NNs?
- Memory cost: we must store activations
- This is exactly what PyTorch does

### Estimated Length
- Main text: 10,000-12,000 words
- Code: ~600 lines
- Time to read: 60-75 minutes

---

## Stage 3: Backpropagation Deep Dive

### Learning Objectives
- Derive gradients for every common operation by hand
- Understand the "local × upstream" pattern deeply
- See gradient flow through networks
- Understand vanishing and exploding gradients
- Connect to numerical stability

### Key Concepts

| Concept | Intuition | Mathematical Form |
|---------|-----------|-------------------|
| Local Gradient | Derivative of one operation | ∂output/∂input |
| Upstream Gradient | Signal from later layers | ∂L/∂output |
| Backprop Rule | Multiply and accumulate | ∂L/∂input = ∂L/∂output · ∂output/∂input |
| Vanishing Gradients | Signal dies through depth | ∏ᵢ \|∂hᵢ₊₁/∂hᵢ\| → 0 |
| Exploding Gradients | Signal grows uncontrollably | ∏ᵢ \|∂hᵢ₊₁/∂hᵢ\| → ∞ |

### Mathematical Derivations

1. **Matrix Multiply Gradient**
   - Y = XW: derive ∂L/∂X and ∂L/∂W
   - The transpose pattern

2. **Softmax Gradient**
   - The Jacobian matrix
   - Simplification when combined with cross-entropy

3. **Layer Normalization Gradient**
   - Full derivation (non-trivial)
   - Why statistics create dependencies

4. **Attention Gradient**
   - Gradient through softmax
   - Gradient through QK^T/√d

5. **Residual Connection Gradient**
   - Why y = x + f(x) helps
   - The "gradient highway"

### Code Implementations

```
stage-03/
├── manual_grads/
│   ├── matmul.py            # Matrix multiply gradients
│   ├── softmax.py           # Softmax + cross-entropy
│   ├── layernorm.py         # Layer normalization
│   ├── attention.py         # Full attention gradient
│   └── residual.py          # Residual connections
├── verify.py                # Compare manual vs autograd
├── visualize_flow.py        # Gradient magnitude through layers
└── experiments/
    ├── vanishing.py         # Demonstrate vanishing gradients
    └── exploding.py         # Demonstrate exploding gradients
```

### Visualizations

1. **Gradient Magnitude Through Depth**
   - Plot |gradient| vs layer
   - Show vanishing/exploding patterns

2. **Softmax Jacobian**
   - Heatmap of the Jacobian matrix
   - Show off-diagonal structure

3. **Residual vs Non-Residual**
   - Compare gradient flow
   - Show how residuals maintain signal

### Trade-offs to Analyze

| Consideration | Problem | Solution | Trade-off |
|---------------|---------|----------|-----------|
| Vanishing | Deep nets can't learn | Residuals, careful init | More parameters |
| Exploding | Training unstable | Gradient clipping | May slow learning |
| Numerical | Underflow in softmax | Log-space computation | Code complexity |

### Connections
- **Backward:** Builds on Stage 2's autodiff foundation
- **Forward:** Essential for understanding Stage 6 (stability)
- **Forward:** Softmax gradient critical for Stage 7 (attention)

### Pólya Structure

**Understand the Problem:**
- We have autodiff, but do we really understand what it computes?
- What can go wrong with gradients?

**Devise a Plan:**
- Derive each gradient by hand
- Verify against autodiff
- Study gradient flow through deep networks

**Carry Out the Plan:**
- Manual derivations for all key operations
- Visualization of gradient magnitudes
- Experiments with vanishing/exploding

**Look Back:**
- The "local × upstream" pattern is universal
- Residual connections solve the gradient highway problem
- Understanding gradients is prerequisite to optimization

### Estimated Length
- Main text: 8,000-10,000 words
- Code: ~400 lines
- Time to read: 50-60 minutes

---

## Stage 4: Neural Language Model (MLP)

### Learning Objectives
- Understand embeddings as learned representations
- Build multi-layer perceptrons
- Implement forward and backward passes
- Train on real text data
- Compare to Markov baselines
- Understand generalization

### Key Concepts

| Concept | Intuition | Mathematical Form |
|---------|-----------|-------------------|
| Embedding | Dense representation of tokens | E ∈ ℝ^{vocab×dim}, lookup E[token] |
| MLP | Sequence of linear + nonlinear | h = σ(Wx + b) |
| Cross-Entropy Loss | Negative log probability | -Σ y_true · log(y_pred) |
| Softmax | Probabilities from scores | exp(zᵢ) / Σⱼ exp(zⱼ) |
| Generalization | Perform well on unseen data | Train/val gap |

### Mathematical Derivations

1. **Embeddings as Lookup**
   - One-hot × embedding matrix = lookup
   - Why learned embeddings work

2. **Universal Approximation Theorem**
   - Statement (what it says and doesn't say)
   - Intuition for why depth helps

3. **Softmax Derivation**
   - From maximum entropy principle
   - Numerical stability (log-sum-exp trick)

4. **Cross-Entropy = MLE**
   - Prove equivalence
   - Why minimizing cross-entropy maximizes likelihood

5. **Activation Functions**
   - ReLU: max(0, x) — derivative, dying ReLU
   - GELU: x · Φ(x) — smooth approximation
   - SwiGLU: preview for Stage 14

### Code Implementations

```
stage-04/
├── embeddings.py            # Embedding layer
├── mlp.py                   # Multi-layer perceptron
├── activations.py           # ReLU, GELU, Swish
├── softmax.py               # Numerically stable softmax
├── loss.py                  # Cross-entropy loss
├── language_model.py        # Full MLP language model
├── train.py                 # Training loop
├── generate.py              # Text generation
├── evaluate.py              # Perplexity evaluation
└── experiments/
    ├── depth_width.py       # Compare architectures
    ├── activation_compare.py # Compare activations
    └── context_length.py    # Effect of context window
```

### Visualizations

1. **Embedding Space**
   - t-SNE or PCA of learned embeddings
   - Show similar characters cluster

2. **Loss Curves**
   - Training vs validation loss
   - Show generalization gap

3. **Perplexity Comparison**
   - Markov (order 1, 2, 3) vs MLP
   - Show improvement

4. **Activation Patterns**
   - ReLU sparsity visualization
   - GELU smoothness

### Trade-offs to Analyze

| Trade-off | Low End | High End | Guidance |
|-----------|---------|----------|----------|
| Embedding Dim | Less capacity | More parameters | 64-256 for char-level |
| Hidden Size | Underfitting | Overfitting | 128-512 |
| Depth | Limited representation | Training difficulty | 2-4 layers |
| Context Length | Poor predictions | Memory cost | 8-64 characters |

### Connections
- **Backward:** Improves on Stage 1's Markov baseline
- **Backward:** Uses Stage 2-3's autodiff and backprop
- **Forward:** Architecture for Stage 7's transformer blocks

### Pólya Structure

**Understand the Problem:**
- Markov chains can't generalize beyond exact matches
- Can we learn patterns that transfer?

**Devise a Plan:**
- Replace counting with learned function
- Use embeddings for input representation
- Stack layers for capacity

**Carry Out the Plan:**
- Implement embeddings and MLP
- Train on text data
- Compare to Markov baseline

**Look Back:**
- Generalization is the key difference
- Cross-entropy = MLE (beautiful connection)
- But fixed context window is limiting

### Estimated Length
- Main text: 10,000-12,000 words
- Code: ~600 lines
- Time to read: 60-75 minutes

---

# SPIRAL 2: TRAINING DYNAMICS

**Theme:** "Understanding why training works (or doesn't)"

**Narrative Arc:** We dive deep into optimization and stability, understanding why some configurations train well and others fail.

---

## Stage 5: Optimization Theory

### Learning Objectives
- Understand loss landscapes and their geometry
- Derive SGD from first principles
- Derive momentum from physical intuition
- Derive Adam from RMSprop + momentum
- Understand learning rate's central role
- Implement and compare optimizers

### Key Concepts

| Concept | Intuition | Mathematical Form |
|---------|-----------|-------------------|
| Loss Landscape | Surface defined by parameters | L(θ): ℝⁿ → ℝ |
| Gradient | Direction of steepest ascent | ∇L = [∂L/∂θ₁, ..., ∂L/∂θₙ] |
| Learning Rate | Step size | θₜ₊₁ = θₜ - η·∇L |
| Momentum | Accumulate gradient direction | vₜ = βvₜ₋₁ + ∇L; θₜ₊₁ = θₜ - η·vₜ |
| Adam | Adaptive learning rates | First + second moment estimates |
| Condition Number | Landscape difficulty | κ = λmax/λmin |

### Mathematical Derivations

1. **Gradient Descent Convergence**
   - For convex functions: rate depends on condition number
   - For non-convex: convergence to stationary points

2. **Momentum Derivation**
   - Physical analogy: ball rolling down hill
   - Mathematical: exponential moving average of gradients

3. **RMSprop**
   - Adapt learning rate per parameter
   - Divide by running average of gradient magnitude

4. **Adam Full Derivation**
   - Combine momentum (first moment)
   - With RMSprop (second moment)
   - Bias correction: why and derivation

5. **Learning Rate Warmup**
   - Gradient variance at initialization
   - Why start small and increase

### Code Implementations

```
stage-05/
├── optimizers/
│   ├── sgd.py               # Vanilla SGD
│   ├── momentum.py          # SGD with momentum
│   ├── rmsprop.py           # RMSprop
│   ├── adam.py              # Adam with bias correction
│   └── adamw.py             # AdamW (decoupled weight decay)
├── learning_rate/
│   ├── constant.py          # Constant LR
│   ├── cosine.py            # Cosine annealing
│   └── warmup.py            # Linear warmup
├── visualization/
│   ├── loss_landscape.py    # 2D loss surface visualization
│   ├── trajectory.py        # Optimizer trajectory on surface
│   └── lr_sensitivity.py    # Learning rate sweep
└── experiments/
    ├── optimizer_compare.py  # Compare on language model
    └── lr_sweep.py          # Learning rate sensitivity
```

### Visualizations

1. **Loss Landscape (2D Slice)**
   - Contour plot with optimizer trajectories
   - Show SGD vs momentum vs Adam paths

2. **Learning Rate Sensitivity**
   - Loss vs learning rate curve
   - Show optimal range

3. **Adam Moment Evolution**
   - First and second moments over time
   - Show bias correction effect

4. **Warmup Effect**
   - Loss curves with/without warmup
   - Gradient variance at start

### Trade-offs to Analyze

| Optimizer | Memory | Convergence Speed | Hyperparameter Sensitivity |
|-----------|--------|-------------------|---------------------------|
| SGD | 1× | Slow | High (LR critical) |
| Momentum | 2× | Medium | Medium |
| Adam | 3× | Fast | Low |
| AdamW | 3× | Fast | Low (better generalization) |

### Connections
- **Backward:** Requires Stage 2-3's gradient computation
- **Forward:** Sets up Stage 6's stability discussion
- **Forward:** Adam used throughout remaining stages

### Estimated Length
- Main text: 10,000-12,000 words
- Code: ~500 lines
- Time to read: 60-75 minutes

---

## Stage 6: Training Stability

### Learning Objectives
- Understand initialization theory deeply
- Derive Xavier and Kaiming initialization
- Understand normalization techniques
- Derive BatchNorm and LayerNorm
- Understand Pre-LN vs Post-LN for transformers
- Debug training instabilities

### Key Concepts

| Concept | Intuition | Mathematical Form |
|---------|-----------|-------------------|
| Variance Preservation | Keep signal strength through layers | Var(output) ≈ Var(input) |
| Xavier Init | For tanh/sigmoid | W ~ N(0, 1/fan_in) |
| Kaiming Init | For ReLU | W ~ N(0, 2/fan_in) |
| Batch Normalization | Normalize across batch | (x - μ_batch) / σ_batch |
| Layer Normalization | Normalize across features | (x - μ_layer) / σ_layer |
| Pre-LN | Norm before sublayer | x + sublayer(norm(x)) |
| Post-LN | Norm after sublayer | norm(x + sublayer(x)) |

### Mathematical Derivations

1. **Variance Analysis Through Layers**
   - For y = Wx: Var(y) = fan_in · Var(W) · Var(x)
   - Derive condition for Var(y) = Var(x)

2. **Xavier Initialization**
   - For symmetric activations (tanh)
   - Derive Var(W) = 1/fan_in or 2/(fan_in + fan_out)

3. **Kaiming Initialization**
   - Account for ReLU zeroing half the outputs
   - Derive Var(W) = 2/fan_in

4. **BatchNorm Forward and Backward**
   - Forward: normalize, scale, shift
   - Backward: gradients through mean and variance

5. **LayerNorm vs BatchNorm**
   - Why LN for transformers (sequence length independence)
   - RMSNorm simplification

6. **Pre-LN Gradient Analysis**
   - Show more stable gradient flow
   - Why it allows deeper models

### Code Implementations

```
stage-06/
├── initialization/
│   ├── xavier.py            # Xavier/Glorot init
│   ├── kaiming.py           # He init
│   └── analyze_variance.py  # Track variance through depth
├── normalization/
│   ├── batchnorm.py         # Full BatchNorm with running stats
│   ├── layernorm.py         # LayerNorm
│   ├── rmsnorm.py           # RMSNorm
│   └── backward.py          # Manual backward passes
├── architectures/
│   ├── pre_ln_block.py      # Pre-normalization block
│   └── post_ln_block.py     # Post-normalization block
└── experiments/
    ├── init_comparison.py    # Compare initializations
    ├── norm_comparison.py    # Compare normalizations
    └── depth_scaling.py     # How deep can we go?
```

### Visualizations

1. **Activation Statistics Through Depth**
   - Mean and variance per layer
   - Compare with/without proper init

2. **Gradient Magnitude Through Depth**
   - Show vanishing with bad init
   - Show stable flow with good init

3. **Pre-LN vs Post-LN**
   - Gradient norms comparison
   - Training curves

4. **BatchNorm Running Statistics**
   - How they evolve during training
   - Train vs eval mode

### Trade-offs to Analyze

| Normalization | Batch Dependency | Sequence Agnostic | Cost | Use Case |
|---------------|------------------|-------------------|------|----------|
| BatchNorm | Yes | N/A | Medium | CNNs |
| LayerNorm | No | Yes | Medium | Transformers |
| RMSNorm | No | Yes | Low | Modern LLMs |

### Connections
- **Backward:** Uses Stage 5's optimization framework
- **Forward:** Normalization essential for Stage 7's transformers
- **Forward:** Sets up Stage 12's numerical precision discussion

### Estimated Length
- Main text: 10,000-12,000 words
- Code: ~500 lines
- Time to read: 60-75 minutes

---

# SPIRAL 3: THE TRANSFORMER

**Theme:** "The architecture that changed everything"

**Narrative Arc:** We build transformers from first principles, understanding why each component exists and how they work together.

---

## Stage 7: Attention from First Principles

### Learning Objectives
- Understand why attention was invented (long-range dependencies)
- Derive attention from content-based addressing
- Understand the Query-Key-Value formulation
- Derive the scaling factor √d
- Analyze O(n²) complexity and its implications

### Key Concepts

| Concept | Intuition | Mathematical Form |
|---------|-----------|-------------------|
| Attention | Dynamic weighted average | Σᵢ αᵢ · vᵢ |
| Query | "What am I looking for?" | q = Wq · x |
| Key | "What do I contain?" | k = Wk · x |
| Value | "What do I provide?" | v = Wv · x |
| Attention Weight | Relevance score | αᵢⱼ = softmax(qᵢ · kⱼ / √d) |
| Self-Attention | Attend within same sequence | Q, K, V from same input |

### Mathematical Derivations

1. **Attention as Soft Dictionary Lookup**
   - Hard attention: lookup exact key
   - Soft attention: weighted average by similarity
   - Connection to memory networks

2. **Query-Key-Value Decomposition**
   - Why separate projections?
   - Degrees of freedom analysis

3. **Scaling Factor Derivation**
   - Variance of dot product: Var(q·k) = d · Var(qᵢ) · Var(kᵢ)
   - If inputs have unit variance: Var(q·k) = d
   - Division by √d restores unit variance
   - Why this matters for softmax stability

4. **Complexity Analysis**
   - Time: O(n² · d) for computing all attention weights
   - Memory: O(n²) for storing attention matrix
   - Why this limits context length

5. **Causal Masking**
   - Autoregressive constraint: can't see future
   - Implementation as -inf before softmax

### Code Implementations

```
stage-07/
├── attention/
│   ├── scaled_dot_product.py  # Core attention function
│   ├── causal_mask.py         # Autoregressive masking
│   └── attention_patterns.py  # Visualize attention weights
├── analysis/
│   ├── variance_analysis.py   # Verify √d scaling
│   ├── complexity.py          # Time and memory profiling
│   └── softmax_stability.py   # Numerical issues
└── experiments/
    ├── scaling_ablation.py    # With/without √d
    └── sequence_length.py     # How n² affects memory
```

### Visualizations

1. **Attention Weights Heatmap**
   - Show learned attention patterns
   - Causal mask structure

2. **√d Scaling Effect**
   - Softmax output entropy with/without scaling
   - Gradient magnitude comparison

3. **O(n²) Memory Growth**
   - Memory vs sequence length plot
   - Annotate with actual GPU limits

4. **Attention as Information Flow**
   - Diagram showing query attending to keys

### Trade-offs to Analyze

| Aspect | Short Context | Long Context | Implication |
|--------|---------------|--------------|-------------|
| Memory | O(n²) manageable | O(n²) prohibitive | Need optimizations |
| Quality | May miss patterns | Better understanding | Longer is better |
| Speed | Fast | Slow | Quadratic growth |

### Connections
- **Backward:** Solves MLP's fixed context problem (Stage 4)
- **Forward:** Multi-head extension in Stage 8
- **Forward:** Memory optimization in Stage 12

### Estimated Length
- Main text: 10,000-12,000 words
- Code: ~400 lines
- Time to read: 60-75 minutes

---

## Stage 8: Multi-Head Attention and Architecture

### Learning Objectives
- Understand why multiple attention heads help
- Build complete transformer blocks
- Understand the feed-forward network's role
- Analyze residual connections and normalization
- Stack blocks into a full transformer

### Key Concepts

| Concept | Intuition | Mathematical Form |
|---------|-----------|-------------------|
| Multi-Head | Multiple attention subspaces | Concat(head₁, ..., headₕ) · Wₒ |
| Head Dimension | Dimension per head | d_head = d_model / n_heads |
| Feed-Forward | Per-position processing | FFN(x) = W₂ · act(W₁ · x) |
| Expansion Factor | FFN hidden size | Usually 4× d_model |
| Residual Connection | Skip connection | y = x + f(x) |
| Transformer Block | Self-attention + FFN | See full structure |

### Mathematical Derivations

1. **Multi-Head as Subspace Decomposition**
   - Each head attends in different subspace
   - Concatenation reassembles full space
   - Why this is better than one big head

2. **Head Dimension Trade-off**
   - More heads = more diverse attention
   - But smaller d_head = less capacity per head
   - Typical: 64-128 per head

3. **Feed-Forward Expansion**
   - Why 4×? Capacity analysis
   - Information bottleneck perspective

4. **Residual + Norm Analysis**
   - Gradient flow through residuals
   - Pre-LN: more stable gradients
   - Post-LN: original, but harder to train deep

### Code Implementations

```
stage-08/
├── attention/
│   ├── multi_head.py          # Multi-head attention
│   └── head_analysis.py       # Analyze individual heads
├── feed_forward/
│   ├── ffn.py                 # Standard FFN
│   └── gelu.py                # GELU activation
├── transformer/
│   ├── block.py               # Single transformer block
│   ├── pre_ln_block.py        # Pre-norm variant
│   └── transformer.py         # Full stacked transformer
├── analysis/
│   ├── head_diversity.py      # Are heads learning different things?
│   └── ablations.py           # Component ablations
└── experiments/
    ├── num_heads.py           # Effect of head count
    └── ffn_ratio.py           # Effect of expansion ratio
```

### Visualizations

1. **Multi-Head Attention Patterns**
   - Show different heads attending differently
   - Some syntactic, some semantic, some positional

2. **Information Flow Diagram**
   - Full transformer block with all components
   - Residual paths highlighted

3. **Ablation Results**
   - Performance with/without each component
   - Quantify contribution

### Trade-offs to Analyze

| Component | Fewer/Smaller | More/Larger | Modern Choice |
|-----------|---------------|-------------|---------------|
| Heads | Less diverse | More memory | 32-128 |
| FFN Ratio | Less capacity | More compute | 4× or 8/3× (SwiGLU) |
| Layers | Faster | Better quality | Scale with compute |

### Connections
- **Backward:** Builds on Stage 7's attention mechanism
- **Forward:** Position encodings needed (Stage 9)
- **Forward:** Modern variations (Stage 14)

### Estimated Length
- Main text: 10,000-12,000 words
- Code: ~600 lines
- Time to read: 60-75 minutes

---

## Stage 9: Positional Information

### Learning Objectives
- Understand why position information is needed
- Derive sinusoidal positional encodings
- Understand learned position embeddings
- Derive Rotary Position Embeddings (RoPE)
- Compare ALiBi approach
- Analyze extrapolation behavior

### Key Concepts

| Concept | Intuition | Mathematical Form |
|---------|-----------|-------------------|
| Position Agnostic | Attention doesn't know order | Permutation invariant |
| Sinusoidal PE | Encode position as frequencies | PE(pos,i) = sin/cos(pos/10000^(2i/d)) |
| Learned PE | Learn position embeddings | E_pos ∈ ℝ^{max_len × d} |
| RoPE | Rotate embeddings by position | (q,k) → (R(pos) · q, R(pos) · k) |
| ALiBi | Linear attention bias | attn + linear_bias(i-j) |

### Mathematical Derivations

1. **Permutation Invariance of Attention**
   - Prove: shuffling inputs shuffles outputs
   - Therefore: need explicit position info

2. **Sinusoidal Properties**
   - Relative position as linear transformation: PE(pos+k) = f(PE(pos))
   - Prove this property
   - Wavelength interpretation

3. **Rotary Position Embeddings**
   - Complex number formulation
   - 2D rotation matrices
   - How q·k captures relative position
   - Full derivation with rotation

4. **ALiBi**
   - Linear bias in attention
   - No learned parameters
   - Extrapolation properties

### Code Implementations

```
stage-09/
├── encodings/
│   ├── sinusoidal.py          # Sinusoidal PE
│   ├── learned.py             # Learned PE
│   ├── rope.py                # Rotary embeddings
│   ├── alibi.py               # ALiBi
│   └── visualization.py       # Visualize encodings
├── analysis/
│   ├── relative_position.py   # Verify relative position property
│   ├── extrapolation.py       # Test beyond training length
│   └── efficiency.py          # Memory/compute comparison
└── experiments/
    ├── encoding_comparison.py  # Compare on language model
    └── length_generalization.py # Extrapolation test
```

### Visualizations

1. **Sinusoidal Encoding Matrix**
   - Heatmap of PE across positions and dimensions
   - Show wavelength variation

2. **RoPE Rotation Visualization**
   - 2D rotation diagrams
   - How angle varies with position

3. **Extrapolation Performance**
   - Perplexity vs sequence length (train vs test)
   - Compare different encodings

### Trade-offs to Analyze

| Encoding | Memory | Extrapolation | Relative Position | Modern Use |
|----------|--------|---------------|-------------------|------------|
| Sinusoidal | O(max_len × d) | Poor | Implicit | Rare |
| Learned | O(max_len × d) | Poor | No | GPT-2 era |
| RoPE | O(1) cached | Good | Yes | Llama, modern |
| ALiBi | O(1) | Excellent | Yes | Some models |

### Connections
- **Backward:** Completes transformer architecture
- **Forward:** RoPE important for Stage 14 modern architectures
- **Forward:** Enables efficient KV caching (Stage 17)

### Estimated Length
- Main text: 8,000-10,000 words
- Code: ~400 lines
- Time to read: 50-60 minutes

---

## Stage 10: Tokenization Deep Dive

### Learning Objectives
- Understand tokenization as compression
- Connect to information theory
- Implement Byte Pair Encoding from scratch
- Understand vocabulary size trade-offs
- Analyze tokenization pathologies

### Key Concepts

| Concept | Intuition | Mathematical Form |
|---------|-----------|-------------------|
| Tokenization | Segment text into units | text → [token_ids] |
| Compression | Represent text in fewer units | Avg tokens < avg characters |
| BPE | Merge most frequent pairs | Iterative pair merging |
| Vocabulary Size | Number of unique tokens | \|V\| |
| Subword | Between characters and words | "unhappiness" → "un" "happiness" |

### Mathematical Derivations

1. **Tokenization as Compression**
   - Information theory view
   - Optimal code lengths (Shannon)
   - Connection to Huffman coding

2. **Vocabulary Size Trade-offs**
   - Embedding memory: O(|V| × d)
   - Softmax computation: O(|V|)
   - Sequence length: more tokens = longer sequences = O(n²) attention
   - Derive optimal vocabulary size range

3. **BPE Algorithm**
   - Greedy compression
   - Merge most frequent pair iteratively
   - Complexity analysis

4. **Unigram Model (Brief)**
   - Probabilistic tokenization
   - EM-style training

### Code Implementations

```
stage-10/
├── tokenizers/
│   ├── bpe.py                 # BPE from scratch
│   ├── bpe_fast.py            # Optimized BPE
│   ├── vocab_builder.py       # Build vocabulary
│   └── encode_decode.py       # Tokenization/detokenization
├── analysis/
│   ├── compression_ratio.py   # Tokens per character
│   ├── vocab_analysis.py      # Analyze vocabulary
│   ├── pathologies.py         # Find problematic cases
│   └── arithmetic.py          # Why tokenization breaks math
├── visualization/
│   └── tokenization_viz.py    # Show token boundaries
└── experiments/
    ├── vocab_size_sweep.py    # Effect of vocab size
    └── compare_tokenizers.py  # BPE vs others
```

### Visualizations

1. **Tokenization Examples**
   - Show token boundaries on real text
   - Color-coded tokens

2. **Vocabulary Size Effect**
   - Sequence length vs vocab size
   - Perplexity vs vocab size
   - Memory vs vocab size

3. **BPE Merge Evolution**
   - Show merges over iterations
   - Vocabulary growth

4. **Pathologies**
   - Arithmetic examples: how numbers tokenize
   - Rare words splitting oddly

### Trade-offs to Analyze

| Vocab Size | Compression | Memory | Rare Token Handling |
|------------|-------------|--------|---------------------|
| Small (1K) | Poor | Low | Good (seen often) |
| Medium (32K) | Good | Medium | Medium |
| Large (100K+) | Excellent | High | Poor (rarely seen) |

### Connections
- **Backward:** Now O(n²) attention stakes are clear
- **Forward:** Tokenization affects all downstream stages
- **Forward:** Important for inference optimization (Stage 17)

### Estimated Length
- Main text: 8,000-10,000 words
- Code: ~500 lines
- Time to read: 50-60 minutes

---

[Continued in Part 2: Spirals 4-5]

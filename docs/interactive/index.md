# Interactive Visualizations

Explore the concepts from each stage with these interactive tools. All visualizations run entirely in your browser—no server required.

<div class="grid cards" markdown>

-   :material-graph-outline:{ .lg .middle } __Autograd Visualizer__

    ---

    Watch automatic differentiation in action. Build computational graphs, run forward passes, and see gradients flow backward.

    [:octicons-arrow-right-24: Launch Autograd Visualizer](./autograd.html){ .md-button target="_blank" }

    _From Stage 2: Automatic Differentiation_

-   :material-thermometer:{ .lg .middle } __Temperature Sampling__

    ---

    See how temperature transforms probability distributions. Experiment with different temperatures and sample tokens.

    [:octicons-arrow-right-24: Launch Temperature Explorer](./temperature.html){ .md-button target="_blank" }

    _From Stage 1: Markov Chains_

-   :material-state-machine:{ .lg .middle } __N-gram State Machine__

    ---

    Visualize Markov chains as state machines. Train on custom text, watch state transitions, and generate text step-by-step.

    [:octicons-arrow-right-24: Launch N-gram Visualizer](./ngram.html){ .md-button target="_blank" }

    _From Stage 1: Markov Chains_

-   :material-chart-line:{ .lg .middle } __Gradient Descent Visualizer__

    ---

    Watch optimizers navigate loss landscapes. Compare SGD, momentum, RMSprop, and Adam on different surfaces.

    [:octicons-arrow-right-24: Launch Optimizer Visualizer](./optimizer.html){ .md-button target="_blank" }

    _From Stage 4: Optimization_

</div>

## What These Visualizations Teach

### Gradient Descent Visualizer (Stage 4) — NEW

The optimizer visualizer demonstrates the core concepts from Sections 4.2-4.5:

- **Loss landscapes**: See how different surfaces create different optimization challenges
- **Optimizer comparison**: Watch how SGD, momentum, and Adam behave differently
- **Hyperparameter effects**: Explore how learning rate and momentum coefficients affect convergence
- **Condition number**: Observe zigzagging on elongated valleys

Try these experiments:

| Surface | Optimizer | What it demonstrates |
|---------|-----------|---------------------|
| Elongated Valley | SGD | Zigzag problem, slow convergence |
| Elongated Valley | Momentum | Dampens oscillation, faster |
| Rosenbrock | Adam | Navigates curved valleys |
| Saddle Point | Any | Escape behavior (or getting stuck) |
| Rastrigin | Adam | Local minima challenges |

### N-gram State Machine (Stage 1)

The n-gram visualizer demonstrates the core concepts from Sections 1.1-1.3:

- **State machine view**: Markov chains are finite state automata
- **Training = counting**: Watch how observations become transition probabilities
- **Generation**: Sample from the model one token at a time
- **Context dependence**: See how history determines the next token distribution

Try these experiments:

| Training Text | What it demonstrates |
|--------------|---------------------|
| `abab` | Deterministic patterns |
| `the cat sat on the mat` | Natural language structure |
| `to be or not to be` | Repeated patterns create loops |
| `aaaaabbbbb` | Imbalanced distributions |

### Autograd Visualizer (Stage 2)

The autograd visualizer demonstrates the core concepts from Section 2.4-2.6:

- **Computational graphs**: See how mathematical expressions become directed acyclic graphs
- **Forward pass**: Watch values propagate from inputs to outputs
- **Backward pass**: Observe gradients flow in reverse via the chain rule
- **Local gradients**: Each operation contributes its local derivative

Try these expressions to explore different patterns:

| Expression | What it demonstrates |
|------------|---------------------|
| `(x + y) * z` | Basic operations, gradient accumulation |
| `x * x + y * y` | Sum of squares, independent gradients |
| `(x * y) + (y * z)` | Shared variable (y appears twice) |
| `x * x * x` | Power rule in action |

### Temperature Sampling (Stage 1)

The temperature explorer demonstrates concepts from Section 1.6:

- **Probability distributions**: How language models represent uncertainty
- **Temperature scaling**: The formula P_T(t) = P(t)^(1/T) / Z
- **Entropy**: How "spread out" the distribution is
- **Effective vocabulary**: Perplexity as the "equivalent uniform vocabulary size"

Key insights to discover:

| Temperature | Effect | Use case |
|-------------|--------|----------|
| T → 0 | Greedy (argmax) | Deterministic outputs |
| T = 0.7 | Slightly focused | Coherent generation |
| T = 1.0 | Original distribution | Balanced sampling |
| T = 1.5 | More random | Creative writing |
| T → ∞ | Uniform | Maximum diversity |

## Technical Notes

These visualizations are built with:

- **Vanilla JavaScript**: No build tools required, matching the "from scratch" philosophy
- **D3.js**: For reactive data visualization
- **Portable design**: Work offline, embed anywhere

The autograd visualizer is a direct port of the Python `Value` class from Stage 2, demonstrating that the same concepts work across languages.

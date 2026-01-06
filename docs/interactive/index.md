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

</div>

## What These Visualizations Teach

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

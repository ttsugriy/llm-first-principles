# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Stage 1: The Simplest Language Model

        ## Markov Chains — Where It All Begins

        Welcome to the first stage of **Building LLMs from First Principles**.

        We start with the simplest possible language model: one that predicts the next
        character based only on counting what came before. This humble model introduces
        concepts that echo through every modern LLM:

        - **Autoregressive generation**: predicting one token at a time
        - **Training = optimization**: learning from data
        - **Perplexity**: measuring how "surprised" a model is
        - **Temperature**: controlling randomness in generation

        By the end of this notebook, you'll understand *exactly* why these concepts work.
        """
    )
    return


@app.cell
def _():
    # Core imports
    from collections import defaultdict, Counter
    from typing import Dict, List, Tuple, Optional
    import math
    import random
    return Counter, Dict, List, Optional, Tuple, defaultdict, math, random


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Part 1: What is Language Modeling?

        Given the text "The cat sat on the", what word comes next?

        A **language model** assigns probabilities to text. Formally, it learns:

        $$P(x_1, x_2, \ldots, x_n)$$

        The probability of an entire sequence. But sequences can be arbitrarily long!
        With vocabulary size $|V|$ and sequence length $n$, there are $|V|^n$ possible
        sequences. That's impossibly large.

        ### The Chain Rule to the Rescue

        The chain rule of probability lets us factorize:

        $$P(x_1, x_2, x_3) = P(x_1) \cdot P(x_2 | x_1) \cdot P(x_3 | x_1, x_2)$$

        In general:

        $$P(x_1, \ldots, x_n) = \prod_{i=1}^{n} P(x_i | x_1, \ldots, x_{i-1})$$

        Now we've converted one impossible distribution into $n$ conditional distributions.
        But each conditional still depends on *all* previous tokens!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### The Markov Assumption

        **Simplifying assumption**: The future depends only on the *recent* past.

        - **Order-1 (bigram)**: $P(x_i | x_1, \ldots, x_{i-1}) \approx P(x_i | x_{i-1})$
        - **Order-k**: $P(x_i | x_1, \ldots, x_{i-1}) \approx P(x_i | x_{i-k}, \ldots, x_{i-1})$

        This is *wrong* for language — "The cat that sat on the mat next to the dog ... **was** sleeping"
        has long-range dependencies. But wrong assumptions can still be useful!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Part 2: The Implementation

        Let's build a complete Markov chain language model. The training algorithm is
        beautifully simple: **just count**.
        """
    )
    return


@app.cell
def _(Counter, Dict, List, Tuple, defaultdict, math, random):
    class MarkovChain:
        """N-gram language model using the Markov assumption."""

        def __init__(self, order: int = 1):
            """
            Initialize Markov chain.

            Args:
                order: Number of previous tokens to condition on (1 = bigram)
            """
            self.order = order
            self.counts: Dict[Tuple, Counter] = defaultdict(Counter)
            self.totals: Dict[Tuple, int] = defaultdict(int)
            self.vocab: set = set()

        def train(self, tokens: List[str]) -> None:
            """Train on a sequence of tokens by counting transitions."""
            # Add special start/end tokens
            padded = ["<START>"] * self.order + tokens + ["<END>"]

            # Count all n-grams
            for i in range(len(padded) - self.order):
                history = tuple(padded[i : i + self.order])
                next_token = padded[i + self.order]

                self.counts[history][next_token] += 1
                self.totals[history] += 1
                self.vocab.add(next_token)

        def probability(self, history: Tuple, next_token: str) -> float:
            """Get probability P(next_token | history)."""
            if history not in self.counts:
                return 0.0
            return self.counts[history][next_token] / self.totals[history]

        def get_distribution(self, history: Tuple) -> Dict[str, float]:
            """Get full probability distribution given history."""
            if history not in self.counts:
                return {}
            total = self.totals[history]
            return {token: count / total for token, count in self.counts[history].items()}

        def generate(
            self, max_length: int = 100, temperature: float = 1.0, seed: str = ""
        ) -> str:
            """Generate text using ancestral sampling."""
            if seed:
                history = tuple(list(seed)[-self.order :])
                if len(history) < self.order:
                    history = tuple(["<START>"] * (self.order - len(history))) + history
                generated = list(seed)
            else:
                history = tuple(["<START>"] * self.order)
                generated = []

            for _ in range(max_length):
                dist = self.get_distribution(history)
                if not dist:
                    break

                # Apply temperature
                if temperature != 1.0:
                    dist = self._apply_temperature(dist, temperature)

                # Sample
                tokens = list(dist.keys())
                probs = list(dist.values())
                next_token = random.choices(tokens, weights=probs, k=1)[0]

                if next_token == "<END>":
                    break

                generated.append(next_token)
                history = tuple(list(history)[1:] + [next_token])

            return "".join(generated)

        def _apply_temperature(
            self, dist: Dict[str, float], temp: float
        ) -> Dict[str, float]:
            """Apply temperature to distribution."""
            log_probs = {k: math.log(v + 1e-10) / temp for k, v in dist.items()}
            max_log = max(log_probs.values())
            exp_probs = {k: math.exp(v - max_log) for k, v in log_probs.items()}
            total = sum(exp_probs.values())
            return {k: v / total for k, v in exp_probs.items()}

        def perplexity(self, tokens: List[str]) -> float:
            """Compute perplexity on a sequence."""
            padded = ["<START>"] * self.order + tokens + ["<END>"]

            log_prob_sum = 0.0
            n_tokens = 0

            for i in range(len(padded) - self.order):
                history = tuple(padded[i : i + self.order])
                next_token = padded[i + self.order]

                prob = self.probability(history, next_token)
                if prob == 0:
                    return float("inf")

                log_prob_sum += math.log(prob)
                n_tokens += 1

            return math.exp(-log_prob_sum / n_tokens)

        def num_states(self) -> int:
            """Return number of unique history states."""
            return len(self.counts)

    return (MarkovChain,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Part 3: Interactive Exploration

        Now let's explore how Markov chains work! Use the controls below to:

        1. **Adjust the order** (context length) of the model
        2. **Change the temperature** for generation
        3. **See train vs test perplexity** to understand overfitting
        """
    )
    return


@app.cell
def _():
    # Sample text for training (Shakespeare excerpt)
    SAMPLE_TEXT = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die: to sleep;
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to: 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep, perchance to dream: ay, there's the rub:
    For in that sleep of death what dreams may come,
    When we have shuffled off this mortal coil,
    Must give us pause: there's the respect
    That makes calamity of so long life.
    For who would bear the whips and scorns of time,
    The oppressor's wrong, the proud man's contumely,
    The pangs of despised love, the law's delay,
    The insolence of office, and the spurns
    That patient merit of the unworthy takes,
    When he himself might his quietus make
    With a bare bodkin? Who would fardels bear,
    To grunt and groan under a weary life,
    But that the dread of something after death,
    The undiscovered country, from whose bourn
    No traveller returns, puzzles the will,
    And makes us rather bear those ills we have
    Than fly to others that we know not of?
    Thus conscience does make cowards of us all,
    And thus the native hue of resolution
    Is sicklied o'er with the pale cast of thought,
    And enterprises of great pith and moment,
    With this regard their currents turn awry
    And lose the name of action.
    """
    return (SAMPLE_TEXT,)


@app.cell
def _(mo):
    # Interactive controls
    order_slider = mo.ui.slider(
        1, 10, value=3, step=1, label="**Order** (context length)"
    )
    temp_slider = mo.ui.slider(
        0.1, 2.0, value=1.0, step=0.1, label="**Temperature** (sampling randomness)"
    )
    num_samples_slider = mo.ui.slider(
        1, 5, value=3, step=1, label="**Number of samples**"
    )

    mo.md(
        f"""
        ### Model Controls

        {order_slider}

        {temp_slider}

        {num_samples_slider}
        """
    )
    return num_samples_slider, order_slider, temp_slider


@app.cell
def _(MarkovChain, SAMPLE_TEXT, order_slider):
    # Train model with current order
    def train_model(order: int, text: str):
        tokens = list(text.lower())
        # Split into train/test (80/20)
        split_idx = int(len(tokens) * 0.8)
        train_tokens = tokens[:split_idx]
        test_tokens = tokens[split_idx:]

        model = MarkovChain(order=order)
        model.train(train_tokens)

        train_ppl = model.perplexity(train_tokens)
        test_ppl = model.perplexity(test_tokens)

        return model, train_ppl, test_ppl, train_tokens, test_tokens

    # This will reactively update when order_slider changes
    model, train_ppl, test_ppl, train_tokens, test_tokens = train_model(
        order_slider.value, SAMPLE_TEXT
    )
    return model, test_ppl, test_tokens, train_ppl, train_tokens, train_model


@app.cell
def _(mo, model, test_ppl, train_ppl):
    mo.md(
        f"""
        ### Model Statistics

        | Metric | Value |
        |--------|-------|
        | **Order (context length)** | {model.order} |
        | **Unique states (histories)** | {model.num_states():,} |
        | **Vocabulary size** | {len(model.vocab)} |
        | **Train perplexity** | {train_ppl:.2f} |
        | **Test perplexity** | {test_ppl:.2f if test_ppl != float('inf') else "∞ (unseen n-grams)"} |

        {"**Overfitting detected!** Test perplexity >> Train perplexity" if test_ppl > train_ppl * 1.5 and test_ppl != float('inf') else ""}
        {"**Infinite perplexity!** The test set contains n-grams never seen in training." if test_ppl == float('inf') else ""}
        """
    )
    return


@app.cell
def _(mo, model, num_samples_slider, temp_slider):
    # Generate samples
    samples = [
        model.generate(max_length=150, temperature=temp_slider.value)
        for _ in range(num_samples_slider.value)
    ]

    sample_text = "\n\n".join([f"**Sample {i+1}:**\n> {s}" for i, s in enumerate(samples)])

    mo.md(
        f"""
        ### Generated Samples

        Temperature = {temp_slider.value} ({"greedy" if temp_slider.value < 0.5 else "balanced" if temp_slider.value <= 1.0 else "creative"})

        {sample_text}
        """
    )
    return sample_text, samples


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Part 4: Understanding Temperature

        Temperature controls the "randomness" of sampling. Mathematically:

        $$P'(x) = \frac{\exp(\log P(x) / T)}{\sum_y \exp(\log P(y) / T)}$$

        - **T < 1**: Distribution becomes *sharper* (more deterministic)
        - **T = 1**: Original distribution
        - **T > 1**: Distribution becomes *flatter* (more random)

        As T → 0, we get argmax (greedy) decoding.
        As T → ∞, we get uniform random sampling.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Part 5: The Overfitting Problem

        Watch what happens as you increase the order:

        | Order | What Happens |
        |-------|--------------|
        | 1 | Very few states, can't capture patterns |
        | 2-3 | Good balance, captures some structure |
        | 5+ | Many states, starts memorizing training data |
        | 10+ | Almost every history is unique! |

        **The fundamental trade-off:**
        - More context → better predictions
        - More context → sparser observations

        This is why we need neural networks: they can **generalize** from similar patterns,
        rather than requiring exact matches.
        """
    )
    return


@app.cell
def _(MarkovChain, SAMPLE_TEXT, mo):
    import matplotlib.pyplot as plt
    import numpy as np

    # Compute perplexity across orders
    def compute_perplexity_curve(text: str, max_order: int = 8):
        tokens = list(text.lower())
        split_idx = int(len(tokens) * 0.8)
        train_tokens = tokens[:split_idx]
        test_tokens = tokens[split_idx:]

        orders = list(range(1, max_order + 1))
        train_ppls = []
        test_ppls = []
        num_states = []

        for order in orders:
            m = MarkovChain(order=order)
            m.train(train_tokens)

            train_ppls.append(m.perplexity(train_tokens))
            test_ppl = m.perplexity(test_tokens)
            # Cap infinite for plotting
            test_ppls.append(min(test_ppl, 100))
            num_states.append(m.num_states())

        return orders, train_ppls, test_ppls, num_states

    orders, train_ppls, test_ppls, num_states_list = compute_perplexity_curve(SAMPLE_TEXT)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Perplexity plot
    axes[0].plot(orders, train_ppls, "b-o", label="Train", linewidth=2, markersize=8)
    axes[0].plot(orders, test_ppls, "r-s", label="Test", linewidth=2, markersize=8)
    axes[0].set_xlabel("Order (context length)", fontsize=12)
    axes[0].set_ylabel("Perplexity (lower = better)", fontsize=12)
    axes[0].set_title("Train vs Test Perplexity", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    # State space plot
    axes[1].bar(orders, num_states_list, color="steelblue", edgecolor="black")
    axes[1].set_xlabel("Order (context length)", fontsize=12)
    axes[1].set_ylabel("Number of unique states", fontsize=12)
    axes[1].set_title("State Space Explosion", fontsize=14, fontweight="bold")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    mo.md(
        f"""
        ### Perplexity vs Order

        This plot shows the classic **overfitting pattern**:
        - Train perplexity keeps improving (blue line goes down)
        - Test perplexity first improves, then explodes (red line)

        The right plot shows *why*: the state space grows exponentially!
        """
    )
    return (
        axes,
        compute_perplexity_curve,
        fig,
        np,
        num_states_list,
        orders,
        plt,
        test_ppls,
        train_ppls,
    )


@app.cell
def _(fig, mo):
    mo.center(fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Part 6: Transition Matrix Visualization

        For a bigram (order-1) model, we can visualize the entire model as a matrix.

        Each cell shows $P(\text{column} | \text{row})$ — the probability of transitioning
        from the row character to the column character.
        """
    )
    return


@app.cell
def _(MarkovChain, SAMPLE_TEXT, mo, np, plt):
    # Build order-1 model for transition matrix
    tokens_for_matrix = list(SAMPLE_TEXT.lower())
    bigram_model = MarkovChain(order=1)
    bigram_model.train(tokens_for_matrix)

    # Get common characters for visualization
    char_counts = {}
    for c in tokens_for_matrix:
        char_counts[c] = char_counts.get(c, 0) + 1

    # Top 15 most common characters
    common_chars = sorted(char_counts.keys(), key=lambda x: -char_counts[x])[:15]

    # Build transition matrix
    n = len(common_chars)
    matrix = np.zeros((n, n))

    for i, from_char in enumerate(common_chars):
        dist = bigram_model.get_distribution((from_char,))
        for j, to_char in enumerate(common_chars):
            matrix[i, j] = dist.get(to_char, 0)

    # Plot
    fig2, ax = plt.subplots(figsize=(10, 8))

    # Create labels with escape sequences for special chars
    labels = [repr(c)[1:-1] if c in "\n\t " else c for c in common_chars]

    im = ax.imshow(matrix, cmap="YlOrRd")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Next character", fontsize=12)
    ax.set_ylabel("Current character", fontsize=12)
    ax.set_title("Bigram Transition Probabilities", fontsize=14, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Probability", fontsize=11)

    plt.tight_layout()

    mo.md(
        """
        ### Transition Matrix Heatmap

        Brighter colors = higher probability. Notice:
        - 'e' → space (words often end in 'e')
        - 't' → 'h' (common bigram 'th')
        - After space, common letters like 't', 'a', 'o' are likely
        """
    )
    return (
        ax,
        bigram_model,
        cbar,
        char_counts,
        common_chars,
        fig2,
        i,
        im,
        labels,
        matrix,
        n,
        tokens_for_matrix,
    )


@app.cell
def _(fig2, mo):
    mo.center(fig2)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Part 7: Key Insights

        ### Insight 1: Training = Counting = Maximum Likelihood

        Our counting procedure is actually **maximum likelihood estimation** (MLE).

        For a bigram model with observations, the log-likelihood is:

        $$\log L(\theta) = \sum_{a,b} \text{count}(a,b) \cdot \log \theta_{a \to b}$$

        Taking derivatives and using Lagrange multipliers for the constraint
        $\sum_b \theta_{a \to b} = 1$, we get:

        $$\theta^*_{a \to b} = \frac{\text{count}(a,b)}{\text{count}(a, \cdot)}$$

        **Counting *is* optimization!** This elegant equivalence is specific to categorical
        distributions, but the principle — training as optimization — carries through to
        neural networks.

        ### Insight 2: Perplexity as Branching Factor

        Perplexity has an intuitive interpretation:

        > If perplexity = 50, the model is as uncertain as choosing uniformly among 50 options.

        A random model over vocabulary of size $|V|$ has perplexity $|V|$.
        A perfect model that always predicts correctly has perplexity 1.

        ### Insight 3: The Fundamental Trade-off

        - **More context → better predictions** (we know more about what came before)
        - **More context → sparser data** (fewer examples of each exact history)

        This tension is fundamental. Neural networks resolve it by **generalizing** —
        similar histories get similar predictions, even if not identical.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## What's Next?

        We've built a working language model! But it has fundamental limitations:

        1. **Fixed context**: Can't look back arbitrarily far
        2. **No generalization**: Exact matches only
        3. **Exponential state space**: Higher orders quickly become impractical

        To move beyond these limitations, we need a way to:
        - Learn patterns that generalize
        - Capture long-range dependencies
        - Scale efficiently

        **That's what neural networks provide.**

        But to train neural networks, we need gradients. And to compute gradients efficiently,
        we need **automatic differentiation**.

        → **Stage 2: Automatic Differentiation**
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ---

        ## Exercises

        1. **Smoothing**: What happens when we see an n-gram we never saw in training?
           Implement Laplace smoothing: $P(b|a) = \\frac{\\text{count}(a,b) + 1}{\\text{count}(a,\\cdot) + |V|}$

        2. **Word-level model**: Modify the code to tokenize by words instead of characters.
           How does the optimal order change?

        3. **Temperature analysis**: Generate samples at T=0.5, 1.0, and 2.0.
           What's the mathematical relationship between temperature and entropy?

        4. **Different text**: Try training on different texts (poetry, code, Wikipedia).
           How do the transition matrices differ?
        """
    )
    return


if __name__ == "__main__":
    app.run()

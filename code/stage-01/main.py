#!/usr/bin/env python3
"""
Stage 1: The Simplest Language Model (Markov Chains)

This script demonstrates the complete Markov chain language model pipeline:
1. Load and preprocess text
2. Train models with different orders
3. Generate samples
4. Evaluate with perplexity
5. Visualize the trade-offs

Run: python main.py

Building LLMs from First Principles
"""

import random
from typing import List, Tuple

from data import get_sample_data, tokenize_chars, preprocess
from markov import MarkovChain, SmoothedMarkovChain
from generate import generate, generate_greedy, apply_temperature
from evaluate import compute_perplexity, evaluate_train_test_split


def print_separator(title: str = "") -> None:
    """Print a section separator."""
    print()
    print("=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)
    print()


def experiment_order_comparison(tokens: List[str], max_order: int = 8) -> None:
    """
    Compare models with different orders.

    This demonstrates the fundamental trade-off:
    - Higher order = better training fit
    - Higher order = worse generalization (overfitting)
    """
    print_separator("EXPERIMENT 1: Order Comparison")

    # Split data
    split = int(len(tokens) * 0.8)
    train = tokens[:split]
    test = tokens[split:]

    print(f"Training tokens: {len(train)}")
    print(f"Test tokens: {len(test)}")
    print()

    print(f"{'Order':<6} {'Train PPL':>10} {'Test PPL':>10} {'States':>10}")
    print("-" * 40)

    for order in range(1, max_order + 1):
        model = MarkovChain(order=order)
        model.train(train)

        train_ppl = compute_perplexity(model, train)
        test_ppl = compute_perplexity(model, test)

        # Format nicely, handling infinity
        train_str = f"{train_ppl:.2f}" if train_ppl < 1000 else "inf"
        test_str = f"{test_ppl:.2f}" if test_ppl < 1000 else "inf"

        print(f"{order:<6} {train_str:>10} {test_str:>10} {model.num_states():>10}")


def experiment_temperature(model: MarkovChain, temps: List[float] = [0.5, 1.0, 2.0]) -> None:
    """
    Demonstrate temperature sampling.

    Temperature controls the "sharpness" of the probability distribution:
    - T < 1: More deterministic (picks high-probability tokens)
    - T = 1: Original distribution
    - T > 1: More random (flattens probabilities)
    """
    print_separator("EXPERIMENT 2: Temperature Sampling")

    for temp in temps:
        desc = "greedy-ish" if temp < 0.8 else ("balanced" if temp <= 1.2 else "creative")
        print(f"Temperature = {temp} ({desc}):")
        for i in range(3):
            sample = generate(model, max_length=80, temperature=temp)
            print(f"  {i+1}. {sample[:60]}...")
        print()


def experiment_sample_quality(tokens: List[str]) -> None:
    """
    Show how sample quality improves with order.
    """
    print_separator("EXPERIMENT 3: Sample Quality vs Order")

    for order in [1, 2, 3, 5]:
        model = MarkovChain(order=order)
        model.train(tokens)

        sample = generate(model, max_length=100, temperature=1.0)

        print(f"Order {order} (states={model.num_states()}):")
        print(f"  {sample[:80]}...")
        print()


def experiment_smoothing(tokens: List[str]) -> None:
    """
    Demonstrate the effect of Laplace smoothing.

    Without smoothing, unseen n-grams get probability 0, leading to
    infinite perplexity on test data.

    With smoothing, we assign small probability to unseen events.
    """
    print_separator("EXPERIMENT 4: Effect of Smoothing")

    split = int(len(tokens) * 0.8)
    train = tokens[:split]
    test = tokens[split:]

    order = 3

    # Without smoothing
    model_no_smooth = MarkovChain(order=order)
    model_no_smooth.train(train)
    ppl_no_smooth = compute_perplexity(model_no_smooth, test)

    # With smoothing
    model_smooth = SmoothedMarkovChain(order=order, alpha=1.0)
    model_smooth.train(train)
    ppl_smooth = compute_perplexity(model_smooth, test)

    print(f"Order {order} Model Comparison:")
    print(f"  Without smoothing: Test PPL = {'inf' if ppl_no_smooth == float('inf') else f'{ppl_no_smooth:.2f}'}")
    print(f"  With smoothing:    Test PPL = {ppl_smooth:.2f}")
    print()
    print("Smoothing prevents infinite perplexity from unseen n-grams!")


def show_transition_matrix(model: MarkovChain, top_k: int = 10) -> None:
    """
    Display the transition probabilities for most common characters.
    """
    print_separator("TRANSITION MATRIX (Order 1)")

    if model.order != 1:
        print("(Skipped - transition matrix visualization is for order-1 models)")
        return

    # Find most common characters based on total counts
    char_totals = {}
    for (char,), counter in model.counts.items():
        if char not in ('<START>', '<END>'):
            char_totals[char] = sum(counter.values())

    top_chars = sorted(char_totals.keys(), key=lambda x: -char_totals.get(x, 0))[:top_k]

    # Print header
    print("P(column | row)")
    print()
    header = "     " + " ".join(f"{c:>5}" for c in top_chars)
    print(header)
    print("-" * len(header))

    # Print rows
    for from_char in top_chars:
        dist = model.get_distribution((from_char,))
        row_str = f"{from_char:>3} |"
        for to_char in top_chars:
            prob = dist.get(to_char, 0.0)
            if prob > 0.01:
                row_str += f" {prob:4.2f}"
            else:
                row_str += "    ."
        print(row_str)


def main():
    """Run all experiments."""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Stage 1: The Simplest Language Model (Markov Chains)    ║")
    print("║  Building LLMs from First Principles                     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Load and preprocess data
    text = get_sample_data("shakespeare")
    text = preprocess(text, lowercase=True)
    tokens = tokenize_chars(text)

    print(f"\nLoaded {len(tokens)} character tokens")
    print(f"Vocabulary size: {len(set(tokens))}")
    print(f"Sample: {text[:60]}...")

    # Run experiments
    experiment_order_comparison(tokens)

    # Train a model for temperature experiment
    model = MarkovChain(order=3)
    model.train(tokens)
    experiment_temperature(model)

    experiment_sample_quality(tokens)
    experiment_smoothing(tokens)

    # Show transition matrix for order-1 model
    bigram_model = MarkovChain(order=1)
    bigram_model.train(tokens)
    show_transition_matrix(bigram_model)

    print_separator("KEY INSIGHTS")
    print("""
    1. TRAINING = COUNTING
       Our simple counting procedure is actually maximum likelihood estimation.
       For categorical distributions, MLE = normalized counts.

    2. PERPLEXITY AS BRANCHING FACTOR
       If perplexity = 50, the model is as uncertain as choosing among
       50 equally likely options at each step.

    3. THE FUNDAMENTAL TRADE-OFF
       More context → better predictions
       More context → sparser observations
       This is why we need neural networks: they can generalize.

    4. TEMPERATURE CONTROLS RANDOMNESS
       T < 1: More deterministic, repeats common patterns
       T > 1: More random, more creative but also more errors
    """)

    print_separator("NEXT: Stage 2 - Automatic Differentiation")
    print("We've built a working language model with pure counting.")
    print("But it can't generalize. Neural networks can, but they need gradients.")
    print("That's what we'll build next.")
    print()


if __name__ == "__main__":
    main()

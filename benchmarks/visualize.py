#!/usr/bin/env python3
"""
Benchmark Visualization

Generates publication-quality plots of benchmark results for the book.
All plots follow Tufte's principles: maximize data-ink ratio, avoid chartjunk.
"""

import sys
from pathlib import Path

# Try to import matplotlib, provide helpful error if missing
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code" / "stage-01"))

from markov import MarkovChain, SmoothedMarkovChain
from evaluate import compute_perplexity
from data import get_sample_data, tokenize_chars, train_test_split


# ============================================================================
# Tufte-style plot configuration
# ============================================================================

def setup_tufte_style():
    """Configure matplotlib for Tufte-inspired minimalist plots."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.frameon': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,
    })


# ============================================================================
# Data Generation
# ============================================================================

def run_order_sweep(text: str, max_order: int = 7, use_smoothing: bool = False):
    """Run benchmark across model orders."""
    tokens = list(text.lower())
    train_tokens, test_tokens = train_test_split(tokens, test_ratio=0.2)

    results = {'orders': [], 'train_ppl': [], 'test_ppl': [], 'states': []}

    for order in range(1, max_order + 1):
        if use_smoothing:
            model = SmoothedMarkovChain(order=order, alpha=0.1)
        else:
            model = MarkovChain(order=order)

        model.train(train_tokens)

        train_ppl = compute_perplexity(model, train_tokens)
        test_ppl = compute_perplexity(model, test_tokens)

        results['orders'].append(order)
        results['train_ppl'].append(train_ppl if train_ppl < 1000 else float('nan'))
        results['test_ppl'].append(test_ppl if test_ppl < 1000 else float('nan'))
        results['states'].append(model.num_states())

    return results


def run_smoothing_sweep(text: str, order: int = 3):
    """Compare different smoothing parameters."""
    tokens = list(text.lower())
    train_tokens, test_tokens = train_test_split(tokens, test_ratio=0.2)

    alphas = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    results = {'alphas': alphas, 'train_ppl': [], 'test_ppl': []}

    for alpha in alphas:
        model = SmoothedMarkovChain(order=order, alpha=alpha)
        model.train(train_tokens)

        results['train_ppl'].append(compute_perplexity(model, train_tokens))
        results['test_ppl'].append(compute_perplexity(model, test_tokens))

    return results


# ============================================================================
# Plot Generation
# ============================================================================

def plot_order_sweep(results_smooth, results_no_smooth, output_path: str):
    """
    Plot perplexity vs. order showing the overfitting pattern.

    Key insight visualized: Train PPL decreases monotonically,
    test PPL has a U-shape indicating overfitting at high orders.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    orders = results_smooth['orders']

    # Plot smoothed results
    ax.plot(orders, results_smooth['train_ppl'], 'o-',
            color='#2ecc71', linewidth=1.5, markersize=6,
            label='Train (smoothed)')
    ax.plot(orders, results_smooth['test_ppl'], 's-',
            color='#e74c3c', linewidth=1.5, markersize=6,
            label='Test (smoothed)')

    # Find optimal order
    valid_test = [(o, p) for o, p in zip(orders, results_smooth['test_ppl'])
                  if p == p]  # filter NaN
    if valid_test:
        optimal_order, optimal_ppl = min(valid_test, key=lambda x: x[1])
        ax.axvline(x=optimal_order, color='#3498db', linestyle='--',
                   alpha=0.7, linewidth=1)
        ax.annotate(f'Optimal\n(order={optimal_order})',
                    xy=(optimal_order, optimal_ppl),
                    xytext=(optimal_order + 0.5, optimal_ppl * 1.3),
                    fontsize=9, color='#3498db')

    ax.set_xlabel('Model Order (context length)', fontsize=11)
    ax.set_ylabel('Perplexity', fontsize=11)
    ax.set_title('The Context-Sparsity Trade-off', fontsize=12, fontweight='bold')

    ax.legend(loc='upper right', fontsize=9)
    ax.set_xticks(orders)

    # Add annotation
    ax.text(0.02, 0.98,
            'Lower is better\nTest PPL shows U-shape: overfitting at high orders',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            style='italic', color='#666666')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_smoothing_comparison(results, output_path: str):
    """
    Plot perplexity vs. smoothing parameter.

    Shows the trade-off: too little smoothing → overfitting,
    too much smoothing → underfitting.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    alphas = results['alphas']

    ax.semilogx(alphas, results['train_ppl'], 'o-',
                color='#2ecc71', linewidth=1.5, markersize=6,
                label='Train')
    ax.semilogx(alphas, results['test_ppl'], 's-',
                color='#e74c3c', linewidth=1.5, markersize=6,
                label='Test')

    # Find optimal alpha
    optimal_idx = min(range(len(results['test_ppl'])),
                      key=lambda i: results['test_ppl'][i])
    optimal_alpha = alphas[optimal_idx]

    ax.axvline(x=optimal_alpha, color='#3498db', linestyle='--',
               alpha=0.7, linewidth=1)
    ax.annotate(f'Optimal α={optimal_alpha}',
                xy=(optimal_alpha, results['test_ppl'][optimal_idx]),
                xytext=(optimal_alpha * 3, results['test_ppl'][optimal_idx] * 1.1),
                fontsize=9, color='#3498db',
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=0.5))

    ax.set_xlabel('Smoothing Parameter (α)', fontsize=11)
    ax.set_ylabel('Perplexity', fontsize=11)
    ax.set_title('Effect of Laplace Smoothing', fontsize=12, fontweight='bold')

    ax.legend(loc='upper left', fontsize=9)

    # Add annotation
    ax.text(0.98, 0.98,
            'Too little α → zero probabilities\nToo much α → ignores data',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            horizontalalignment='right', style='italic', color='#666666')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_state_space_growth(results, output_path: str):
    """
    Plot the exponential growth of state space with order.

    Illustrates why high-order Markov models are impractical.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    orders = results['orders']
    states = results['states']

    ax.bar(orders, states, color='#3498db', alpha=0.8, edgecolor='#2980b9')

    # Add value labels on bars
    for i, (o, s) in enumerate(zip(orders, states)):
        ax.annotate(f'{s:,}',
                    xy=(o, s),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', fontsize=8)

    ax.set_xlabel('Model Order', fontsize=11)
    ax.set_ylabel('Number of States', fontsize=11)
    ax.set_title('State Space Explosion', fontsize=12, fontweight='bold')
    ax.set_xticks(orders)

    # Add theoretical line
    vocab_size = 27  # approximate for characters
    theoretical = [vocab_size ** o for o in orders]
    ax2 = ax.twinx()
    ax2.plot(orders, theoretical, 'r--', alpha=0.5, label='Theoretical max (27^k)')
    ax2.set_ylabel('Theoretical Maximum', color='red', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_temperature_distribution(output_path: str):
    """
    Visualize how temperature affects probability distributions.

    Shows the transformation from peaked (low T) to uniform (high T).
    """
    import math

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    # Original distribution (somewhat peaked)
    tokens = ['the', 'a', 'an', 'one', 'other']
    original_probs = [0.4, 0.25, 0.15, 0.12, 0.08]

    temperatures = [0.3, 0.7, 1.0, 2.0]
    temp_labels = ['T=0.3\n(focused)', 'T=0.7\n(slightly focused)',
                   'T=1.0\n(original)', 'T=2.0\n(diverse)']

    for ax, temp, label in zip(axes, temperatures, temp_labels):
        # Apply temperature
        log_probs = [math.log(p + 1e-10) / temp for p in original_probs]
        max_log = max(log_probs)
        exp_probs = [math.exp(lp - max_log) for lp in log_probs]
        total = sum(exp_probs)
        scaled_probs = [p / total for p in exp_probs]

        colors = ['#e74c3c' if i == 0 else '#3498db' for i in range(len(tokens))]
        ax.bar(tokens, scaled_probs, color=colors, alpha=0.8)
        ax.set_ylim(0, 1)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel('Probability' if ax == axes[0] else '')

        # Add entropy annotation
        entropy = -sum(p * math.log2(p + 1e-10) for p in scaled_probs)
        ax.text(0.95, 0.95, f'H={entropy:.2f} bits',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right')

    plt.suptitle('Temperature Scaling: Controlling Generation Diversity',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Generate all benchmark visualizations."""
    setup_tufte_style()

    # Create output directory
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)

    # Get sample data
    text = get_sample_data('shakespeare') * 10  # Repeat for more data

    print("Generating benchmark visualizations...")
    print("=" * 50)

    # 1. Order sweep comparison
    print("\n1. Running order sweep (with smoothing)...")
    results_smooth = run_order_sweep(text, max_order=7, use_smoothing=True)

    print("   Running order sweep (no smoothing)...")
    results_no_smooth = run_order_sweep(text, max_order=7, use_smoothing=False)

    plot_order_sweep(results_smooth, results_no_smooth,
                     str(output_dir / "order_sweep.png"))

    # 2. Smoothing comparison
    print("\n2. Running smoothing sweep...")
    smoothing_results = run_smoothing_sweep(text, order=3)
    plot_smoothing_comparison(smoothing_results,
                              str(output_dir / "smoothing_comparison.png"))

    # 3. State space growth
    print("\n3. Plotting state space growth...")
    plot_state_space_growth(results_smooth,
                            str(output_dir / "state_space.png"))

    # 4. Temperature visualization
    print("\n4. Plotting temperature distributions...")
    plot_temperature_distribution(str(output_dir / "temperature.png"))

    print("\n" + "=" * 50)
    print(f"All figures saved to: {output_dir}")
    print("\nTo use in the book, copy to docs/images/ and reference in markdown.")


if __name__ == "__main__":
    main()

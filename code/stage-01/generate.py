"""
Text Generation from Markov Chains

This module implements ancestral sampling from a trained Markov chain,
including temperature-controlled sampling.

Key concepts:
- Ancestral sampling: Sample each token from P(x_t | x_{<t})
- Temperature: Control randomness by rescaling log-probabilities
"""

from typing import Dict, Optional
import math
import random

from markov import MarkovChain


def sample_from_distribution(dist: Dict[str, float]) -> str:
    """
    Sample a token from a probability distribution.

    Uses weighted random sampling where each token is selected
    with probability proportional to its weight.

    Args:
        dist: Dictionary mapping tokens to probabilities

    Returns:
        A sampled token
    """
    if not dist:
        raise ValueError("Cannot sample from empty distribution")

    tokens = list(dist.keys())
    probs = list(dist.values())
    return random.choices(tokens, weights=probs, k=1)[0]


def apply_temperature(dist: Dict[str, float], temperature: float) -> Dict[str, float]:
    """
    Apply temperature scaling to a probability distribution.

    Temperature modifies the "sharpness" of the distribution:
    - T < 1.0: Sharper distribution (more deterministic)
    - T = 1.0: Original distribution
    - T > 1.0: Flatter distribution (more random)

    Mathematically:
    P'(x) = exp(log(P(x)) / T) / Z
    where Z is a normalizing constant.

    As T → 0, this approaches argmax (greedy decoding).
    As T → ∞, this approaches uniform distribution.

    Args:
        dist: Original probability distribution
        temperature: Temperature parameter (must be > 0)

    Returns:
        Temperature-scaled distribution
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}")

    if temperature == 1.0:
        return dist

    # Convert to log space, scale, convert back
    log_probs = {k: math.log(v + 1e-10) / temperature for k, v in dist.items()}

    # Subtract max for numerical stability (log-sum-exp trick)
    max_log = max(log_probs.values())
    exp_probs = {k: math.exp(v - max_log) for k, v in log_probs.items()}

    # Normalize
    total = sum(exp_probs.values())
    return {k: v / total for k, v in exp_probs.items()}


def generate(
    model: MarkovChain,
    max_length: int = 100,
    temperature: float = 1.0,
    seed: Optional[str] = None,
    stop_on_end: bool = True
) -> str:
    """
    Generate text from a trained Markov chain using ancestral sampling.

    Ancestral sampling generates each token by:
    1. Looking at the current history (last k tokens)
    2. Getting P(next | history) from the model
    3. Sampling next token from this distribution
    4. Appending to generated sequence and repeating

    Args:
        model: Trained MarkovChain
        max_length: Maximum tokens to generate
        temperature: Sampling temperature (1.0 = normal)
        seed: Optional starting text (will be used as initial history)
        stop_on_end: If True, stop when <END> token is generated

    Returns:
        Generated text as string

    Example:
        >>> model = MarkovChain(order=2)
        >>> model.train(list("hello world"))
        >>> generate(model, max_length=20)  # doctest: +SKIP
        'llo world'
    """
    # Initialize history from seed or start tokens
    if seed:
        seed_tokens = list(seed)
        if len(seed_tokens) >= model.order:
            history = tuple(seed_tokens[-model.order:])
        else:
            # Pad with START tokens if seed is shorter than order
            padding = ['<START>'] * (model.order - len(seed_tokens))
            history = tuple(padding + seed_tokens)
        generated = list(seed)
    else:
        history = tuple(['<START>'] * model.order)
        generated = []

    for _ in range(max_length):
        dist = model.get_distribution(history)

        if not dist:
            # No transitions from this history - stop generating
            break

        # Apply temperature
        if temperature != 1.0:
            dist = apply_temperature(dist, temperature)

        # Sample next token
        next_token = sample_from_distribution(dist)

        # Check for end token
        if stop_on_end and next_token == '<END>':
            break

        generated.append(next_token)

        # Update history: slide window by one position
        history = tuple(list(history)[1:] + [next_token])

    return ''.join(generated)


def generate_greedy(model: MarkovChain, max_length: int = 100, seed: Optional[str] = None) -> str:
    """
    Generate text using greedy (argmax) decoding.

    At each step, select the most probable next token.
    This is equivalent to temperature → 0.

    Note: Greedy decoding can get stuck in loops!

    Args:
        model: Trained MarkovChain
        max_length: Maximum tokens to generate
        seed: Optional starting text

    Returns:
        Generated text as string
    """
    if seed:
        seed_tokens = list(seed)
        if len(seed_tokens) >= model.order:
            history = tuple(seed_tokens[-model.order:])
        else:
            padding = ['<START>'] * (model.order - len(seed_tokens))
            history = tuple(padding + seed_tokens)
        generated = list(seed)
    else:
        history = tuple(['<START>'] * model.order)
        generated = []

    for _ in range(max_length):
        dist = model.get_distribution(history)

        if not dist:
            break

        # Greedy: select argmax
        next_token = max(dist.keys(), key=lambda k: dist[k])

        if next_token == '<END>':
            break

        generated.append(next_token)
        history = tuple(list(history)[1:] + [next_token])

    return ''.join(generated)


if __name__ == "__main__":
    # Quick test
    from markov import MarkovChain

    text = "to be or not to be that is the question"
    tokens = list(text)

    model = MarkovChain(order=2)
    model.train(tokens)

    print("Model:", model)
    print("\nSamples at different temperatures:")
    for temp in [0.5, 1.0, 2.0]:
        sample = generate(model, max_length=50, temperature=temp)
        print(f"  T={temp}: {sample}")

    print("\nGreedy sample:")
    print(f"  {generate_greedy(model, max_length=50)}")

"""
Neural Language Model from Scratch

This module implements a character-level neural language model using the
autograd system from Stage 2. It demonstrates how neural networks learn
language patterns through continuous representations.

Architecture:
    Input: k previous characters (context)
    → Embedding: lookup vectors for each character
    → Concatenate: combine all context embeddings
    → Hidden layers: learn patterns with nonlinear activations
    → Output: probability distribution over next character

Usage:
    from neural_lm import CharacterLM, build_vocab, train, generate

    # Prepare data
    text = "Hello world..."
    char_to_idx, idx_to_char, vocab_size = build_vocab(text)

    # Create and train model
    model = CharacterLM(vocab_size, embed_dim=32, hidden_dim=64, context_length=8)
    train(model, text, char_to_idx, epochs=10)

    # Generate text
    generated = generate(model, "The ", char_to_idx, idx_to_char, length=100)
"""

import sys
import os
import math
import random
from typing import List, Dict, Tuple, Optional

# Import Value from Stage 2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage-02'))
from value import Value


# =============================================================================
# Data Utilities
# =============================================================================


def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """
    Build character-to-index mappings from text.

    Args:
        text: Training text

    Returns:
        (char_to_idx, idx_to_char, vocab_size)
    """
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char, len(chars)


def encode(text: str, char_to_idx: Dict[str, int]) -> List[int]:
    """Convert text to list of character indices."""
    return [char_to_idx[ch] for ch in text]


def decode(indices: List[int], idx_to_char: Dict[int, str]) -> str:
    """Convert list of indices back to text."""
    return ''.join(idx_to_char[i] for i in indices)


def create_examples(
    encoded_text: List[int],
    context_length: int
) -> List[Tuple[List[int], int]]:
    """
    Create (context, target) pairs from encoded text.

    Args:
        encoded_text: List of token indices
        context_length: Number of context characters

    Returns:
        List of (context_indices, target_index) pairs
    """
    examples = []
    for i in range(len(encoded_text) - context_length):
        context = encoded_text[i:i + context_length]
        target = encoded_text[i + context_length]
        examples.append((context, target))
    return examples


# =============================================================================
# Activation Functions
# =============================================================================


def relu(x: List[Value]) -> List[Value]:
    """Apply ReLU activation element-wise."""
    return [xi.relu() for xi in x]


def softmax(logits: List[Value]) -> List[Value]:
    """
    Numerically stable softmax.

    softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    Subtracting max prevents overflow in exp().
    """
    # Find max for numerical stability
    max_val = max(l.data for l in logits)

    # Compute exp(x - max)
    exp_vals = [(l - max_val).exp() for l in logits]

    # Sum of exponentials
    sum_exp = sum(exp_vals, Value(0.0))

    # Normalize
    return [e / sum_exp for e in exp_vals]


# =============================================================================
# Loss Function
# =============================================================================


def cross_entropy_loss(logits: List[Value], target_idx: int) -> Value:
    """
    Cross-entropy loss with numerically stable log-softmax.

    loss = -log(softmax(logits)[target_idx])
        = -logits[target_idx] + log(sum(exp(logits)))

    Using log-sum-exp trick for stability:
        log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    # Log-sum-exp for stability
    max_val = max(l.data for l in logits)
    max_value = Value(max_val)

    # exp(x - max)
    shifted = [l - max_value for l in logits]
    exp_shifted = [s.exp() for s in shifted]

    # log(sum(exp(x - max))) + max
    sum_exp = sum(exp_shifted, Value(0.0))
    log_sum_exp = sum_exp.log() + max_value

    # -log(softmax) = log_sum_exp - logits[target]
    return log_sum_exp - logits[target_idx]


# =============================================================================
# Neural Network Layers
# =============================================================================


class Embedding:
    """
    Embedding layer: maps discrete tokens to continuous vectors.

    Each token id looks up a row in the embedding matrix.
    These vectors are learned during training.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Initialize embedding matrix.

        Args:
            vocab_size: Number of unique tokens
            embed_dim: Dimension of embedding vectors
        """
        # Initialize with small random values
        scale = 1.0 / math.sqrt(embed_dim)
        self.embedding = [
            [Value(random.uniform(-scale, scale)) for _ in range(embed_dim)]
            for _ in range(vocab_size)
        ]
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def __call__(self, token_idx: int) -> List[Value]:
        """Look up embedding vector for a token."""
        return self.embedding[token_idx]

    def parameters(self) -> List[Value]:
        """Return all learnable parameters."""
        return [v for row in self.embedding for v in row]


class Linear:
    """
    Fully connected layer: y = Wx + b

    Implements a linear transformation with learnable weights and biases.
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Initialize with Xavier/Glorot initialization.

        Args:
            in_features: Input dimension
            out_features: Output dimension
        """
        # Xavier initialization: scale = sqrt(2 / (in + out))
        scale = math.sqrt(2.0 / (in_features + out_features))

        self.weights = [
            [Value(random.uniform(-scale, scale)) for _ in range(in_features)]
            for _ in range(out_features)
        ]
        self.biases = [Value(0.0) for _ in range(out_features)]

        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, x: List[Value]) -> List[Value]:
        """
        Forward pass: compute Wx + b.

        Args:
            x: Input vector of Values

        Returns:
            Output vector of Values
        """
        output = []
        for i in range(self.out_features):
            # Dot product: sum(w_ij * x_j) + b_i
            activation = self.biases[i]
            for j in range(self.in_features):
                activation = activation + self.weights[i][j] * x[j]
            output.append(activation)
        return output

    def parameters(self) -> List[Value]:
        """Return all learnable parameters."""
        params = []
        for row in self.weights:
            params.extend(row)
        params.extend(self.biases)
        return params


# =============================================================================
# Character-Level Language Model
# =============================================================================


class CharacterLM:
    """
    Character-level neural language model.

    Architecture:
        1. Embed each context character
        2. Concatenate embeddings
        3. Pass through hidden layers with ReLU
        4. Output logits for next character
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        context_length: int = 8,
    ):
        """
        Initialize the language model.

        Args:
            vocab_size: Number of unique characters
            embed_dim: Embedding vector dimension
            hidden_dim: Hidden layer dimension
            context_length: Number of context characters
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length

        # Embedding layer
        self.embedding = Embedding(vocab_size, embed_dim)

        # Concatenated embedding size
        concat_size = context_length * embed_dim

        # Hidden layers
        self.fc1 = Linear(concat_size, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)

        # Output layer
        self.output = Linear(hidden_dim, vocab_size)

    def forward(self, context: List[int]) -> List[Value]:
        """
        Forward pass: compute logits for next character.

        Args:
            context: List of context character indices

        Returns:
            Logits for each character in vocabulary
        """
        # Embed each context character
        embeddings = []
        for idx in context:
            embeddings.extend(self.embedding(idx))

        # First hidden layer
        h1 = self.fc1(embeddings)
        h1 = relu(h1)

        # Second hidden layer
        h2 = self.fc2(h1)
        h2 = relu(h2)

        # Output logits
        logits = self.output(h2)

        return logits

    def predict_probs(self, context: List[int]) -> List[Value]:
        """Compute probability distribution over next character."""
        logits = self.forward(context)
        return softmax(logits)

    def loss(self, context: List[int], target_idx: int) -> Value:
        """Compute cross-entropy loss for a single example."""
        logits = self.forward(context)
        return cross_entropy_loss(logits, target_idx)

    def parameters(self) -> List[Value]:
        """Return all learnable parameters."""
        params = []
        params.extend(self.embedding.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.output.parameters())
        return params

    def zero_grad(self) -> None:
        """Zero all parameter gradients."""
        for p in self.parameters():
            p.grad = 0.0

    def num_parameters(self) -> int:
        """Count total parameters."""
        return len(self.parameters())


# =============================================================================
# Training
# =============================================================================


def train(
    model: CharacterLM,
    text: str,
    char_to_idx: Dict[str, int],
    epochs: int = 10,
    learning_rate: float = 0.1,
    print_every: int = 100,
) -> List[float]:
    """
    Train the language model using SGD.

    Args:
        model: CharacterLM instance
        text: Training text
        char_to_idx: Character to index mapping
        epochs: Number of training epochs
        learning_rate: SGD learning rate
        print_every: Print loss every N steps

    Returns:
        List of average losses per epoch
    """
    # Prepare training data
    encoded = encode(text, char_to_idx)
    examples = create_examples(encoded, model.context_length)

    epoch_losses = []

    for epoch in range(epochs):
        # Shuffle examples
        random.shuffle(examples)

        total_loss = 0.0
        num_examples = 0

        for i, (context, target) in enumerate(examples):
            # Forward pass
            loss = model.loss(context, target)
            total_loss += loss.data
            num_examples += 1

            # Backward pass
            model.zero_grad()
            loss.backward()

            # SGD update
            for p in model.parameters():
                p.data -= learning_rate * p.grad

            # Print progress
            if (i + 1) % print_every == 0:
                avg = total_loss / num_examples
                print(f"Epoch {epoch + 1}, Step {i + 1}/{len(examples)}: loss = {avg:.4f}")

        avg_loss = total_loss / num_examples
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} complete: avg_loss = {avg_loss:.4f}")

    return epoch_losses


# =============================================================================
# Evaluation
# =============================================================================


def compute_perplexity(
    model: CharacterLM,
    text: str,
    char_to_idx: Dict[str, int],
) -> float:
    """
    Compute perplexity on text.

    Perplexity = exp(average_cross_entropy_loss)

    Lower is better. Random guessing would give perplexity = vocab_size.
    """
    encoded = encode(text, char_to_idx)
    examples = create_examples(encoded, model.context_length)

    if not examples:
        return float('inf')

    total_loss = 0.0
    for context, target in examples:
        loss = model.loss(context, target)
        total_loss += loss.data

    avg_loss = total_loss / len(examples)
    return math.exp(avg_loss)


# =============================================================================
# Text Generation
# =============================================================================


def generate(
    model: CharacterLM,
    seed_text: str,
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
    length: int = 100,
    temperature: float = 1.0,
) -> str:
    """
    Generate text from the model.

    Args:
        model: Trained CharacterLM
        seed_text: Starting text (must be at least context_length chars)
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        length: Number of characters to generate
        temperature: Sampling temperature (< 1 = deterministic, > 1 = random)

    Returns:
        Generated text (seed + new characters)
    """
    # Encode seed
    context = encode(seed_text[-model.context_length:], char_to_idx)

    # Pad if needed
    while len(context) < model.context_length:
        context.insert(0, 0)

    generated = list(seed_text)

    for _ in range(length):
        # Get probabilities
        probs = model.predict_probs(context)
        probs_data = [p.data for p in probs]

        # Apply temperature
        if temperature != 1.0:
            # Scale logits by 1/temperature before softmax
            logits = [math.log(p + 1e-10) / temperature for p in probs_data]
            max_logit = max(logits)
            exp_logits = [math.exp(l - max_logit) for l in logits]
            sum_exp = sum(exp_logits)
            probs_data = [e / sum_exp for e in exp_logits]

        # Sample from distribution
        r = random.random()
        cumsum = 0.0
        next_idx = 0
        for i, p in enumerate(probs_data):
            cumsum += p
            if r < cumsum:
                next_idx = i
                break

        # Append to generated text
        generated.append(idx_to_char[next_idx])

        # Update context
        context = context[1:] + [next_idx]

    return ''.join(generated)


# =============================================================================
# Demo
# =============================================================================


def demo():
    """Demonstrate the neural language model."""
    print("=" * 60)
    print("Neural Language Model Demo")
    print("=" * 60)

    # Training text (Shakespeare excerpt)
    text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them."""

    print(f"\nTraining text ({len(text)} chars):")
    print(text[:100] + "...")

    # Build vocabulary
    char_to_idx, idx_to_char, vocab_size = build_vocab(text)
    print(f"\nVocabulary size: {vocab_size}")

    # Create model
    model = CharacterLM(
        vocab_size=vocab_size,
        embed_dim=16,
        hidden_dim=32,
        context_length=4,
    )
    print(f"Model parameters: {model.num_parameters():,}")

    # Initial perplexity (before training)
    initial_ppl = compute_perplexity(model, text, char_to_idx)
    print(f"Initial perplexity: {initial_ppl:.2f}")

    # Train
    print("\nTraining...")
    random.seed(42)
    train(model, text, char_to_idx, epochs=3, learning_rate=0.1, print_every=50)

    # Final perplexity
    final_ppl = compute_perplexity(model, text, char_to_idx)
    print(f"\nFinal perplexity: {final_ppl:.2f}")

    # Generate text
    print("\nGenerated text:")
    seed = "To b"
    generated = generate(model, seed, char_to_idx, idx_to_char, length=50, temperature=0.8)
    print(f'"{generated}"')


if __name__ == '__main__':
    demo()

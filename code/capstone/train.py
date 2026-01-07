#!/usr/bin/env python3
"""
Capstone: End-to-End Transformer Training

This script trains a small transformer language model from scratch,
demonstrating all concepts from Stages 1-6 working together:

- Data loading and tokenization (Stage 1 concepts)
- Backpropagation through the network (Stage 2)
- Neural network layers (Stage 3)
- Adam optimizer with learning rate scheduling (Stage 4)
- Attention mechanisms (Stage 5)
- Modern transformer architecture (Stage 6)

Usage:
    python train.py                           # Train with defaults
    python train.py --epochs 50 --lr 3e-4     # Custom settings
    python train.py --text-file mytext.txt    # Train on custom text
"""

import argparse
import time
import sys
import os
from typing import List, Tuple, Optional
import numpy as np

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage-04'))

from model import (
    TrainableTransformer,
    CharTokenizer,
    cross_entropy_loss,
    compute_perplexity,
    Parameter,
)


# =============================================================================
# Sample Data
# =============================================================================

SHAKESPEARE = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life;
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all;
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry,
And lose the name of action.
"""


# =============================================================================
# Adam Optimizer (from Stage 4 concepts)
# =============================================================================

class Adam:
    """
    Adam optimizer with decoupled weight decay (AdamW).

    This is the standard optimizer for training transformers.
    """

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        # Initialize moment estimates
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self) -> None:
        """Update parameters using stored gradients."""
        self.t += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            # Decoupled weight decay
            p.data *= (1 - self.lr * self.weight_decay)

            # Update biased first moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad

            # Update biased second moment
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay."""

    def __init__(
        self,
        optimizer: Adam,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-5,
    ):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.step_count = 0

    def step(self) -> float:
        """Update learning rate and return current lr."""
        self.step_count += 1

        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        self.optimizer.lr = lr
        return lr


# =============================================================================
# Data Loading and Batching
# =============================================================================

def create_batches(
    tokens: List[int],
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create training batches of (input, target) pairs.

    For language modeling, target[i] = input[i+1]
    """
    # Create sequences
    n_sequences = (len(tokens) - 1) // seq_len
    sequences = []

    for i in range(n_sequences):
        start = i * seq_len
        seq_input = tokens[start:start + seq_len]
        seq_target = tokens[start + 1:start + seq_len + 1]
        sequences.append((seq_input, seq_target))

    # Shuffle
    if shuffle:
        np.random.shuffle(sequences)

    # Create batches
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        if len(batch_seqs) < batch_size:
            continue  # Skip incomplete batches

        inputs = np.array([s[0] for s in batch_seqs])
        targets = np.array([s[1] for s in batch_seqs])
        batches.append((inputs, targets))

    return batches


# =============================================================================
# Gradient Clipping
# =============================================================================

def clip_grad_norm(params: List[Parameter], max_norm: float) -> float:
    """Clip gradients by global norm. Returns original norm."""
    total_norm_sq = sum(
        np.sum(p.grad ** 2) for p in params if p.grad is not None
    )
    total_norm = np.sqrt(total_norm_sq)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad *= scale

    return total_norm


# =============================================================================
# Training Loop
# =============================================================================

def train(
    text: str,
    epochs: int = 20,
    batch_size: int = 4,
    seq_len: int = 64,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    lr: float = 3e-4,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    log_interval: int = 10,
    sample_interval: int = 50,
    seed: int = 42,
) -> dict:
    """
    Train a transformer language model.

    Args:
        text: Training text
        epochs: Number of training epochs
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        lr: Learning rate
        warmup_ratio: Fraction of steps for warmup
        weight_decay: Weight decay coefficient
        max_grad_norm: Maximum gradient norm for clipping
        log_interval: Log every N steps
        sample_interval: Generate samples every N steps
        seed: Random seed

    Returns:
        Dictionary with training history
    """
    np.random.seed(seed)

    # Tokenize
    print("=" * 60)
    print("Capstone: Training Transformer Language Model")
    print("=" * 60)
    print(f"\nTokenizing {len(text)} characters...")

    tokenizer = CharTokenizer().train(text)
    tokens = tokenizer.encode(text)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Total tokens: {len(tokens)}")

    # Create model
    print(f"\nCreating model...")
    model = TrainableTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=seq_len,
    )
    print(f"Model parameters: {model.count_parameters():,}")

    # Create optimizer
    params = model.parameters()
    optimizer = Adam(params, lr=lr, weight_decay=weight_decay)

    # Calculate steps
    batches = create_batches(tokens, seq_len, batch_size, shuffle=False)
    steps_per_epoch = len(batches)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)

    print(f"Batches per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # Create scheduler
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    # Training history
    history = {
        'loss': [],
        'perplexity': [],
        'grad_norm': [],
        'lr': [],
    }

    # Training loop
    print(f"\nStarting training...")
    print("-" * 60)

    global_step = 0
    start_time = time.time()

    for epoch in range(epochs):
        # Create shuffled batches for this epoch
        batches = create_batches(tokens, seq_len, batch_size, shuffle=True)

        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, (inputs, targets) in enumerate(batches):
            # Forward pass
            logits = model.forward(inputs)
            loss, grad_logits = cross_entropy_loss(logits, targets)

            # Backward pass
            model.zero_grad()
            model.backward(grad_logits)

            # Gradient clipping
            grad_norm = clip_grad_norm(params, max_grad_norm)

            # Optimizer step
            optimizer.step()

            # Scheduler step
            current_lr = scheduler.step()

            # Record history
            epoch_loss += loss
            epoch_steps += 1
            global_step += 1

            history['loss'].append(loss)
            history['perplexity'].append(compute_perplexity(loss))
            history['grad_norm'].append(grad_norm)
            history['lr'].append(current_lr)

            # Log progress
            if global_step % log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                ppl = compute_perplexity(loss)
                print(f"Step {global_step:5d} | "
                      f"Loss {loss:.4f} | "
                      f"PPL {ppl:7.2f} | "
                      f"GradNorm {grad_norm:.3f} | "
                      f"LR {current_lr:.2e} | "
                      f"{steps_per_sec:.1f} steps/s")

            # Generate sample
            if global_step % sample_interval == 0:
                prompt = text[:seq_len // 2]
                prompt_tokens = tokenizer.encode(prompt)
                generated_tokens = model.generate(prompt_tokens, max_new_tokens=50, temperature=0.8)
                generated_text = tokenizer.decode(generated_tokens)
                print(f"\n--- Sample at step {global_step} ---")
                print(f"{generated_text[:150]}...")
                print("-" * 40 + "\n")

        # Epoch summary
        avg_loss = epoch_loss / epoch_steps
        avg_ppl = compute_perplexity(avg_loss)
        print(f"\nEpoch {epoch + 1}/{epochs} complete | Avg Loss: {avg_loss:.4f} | Avg PPL: {avg_ppl:.2f}\n")

    # Final summary
    total_time = time.time() - start_time
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Final perplexity: {history['perplexity'][-1]:.2f}")

    # Generate final sample
    print("\n--- Final Generation ---")
    prompt = text[:20]
    prompt_tokens = tokenizer.encode(prompt)
    generated_tokens = model.generate(prompt_tokens, max_new_tokens=100, temperature=0.7)
    generated_text = tokenizer.decode(generated_tokens)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

    return {
        'model': model,
        'tokenizer': tokenizer,
        'history': history,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train a transformer language model")

    # Data
    parser.add_argument('--text-file', type=str, default=None,
                        help='Path to training text file (uses Shakespeare if not provided)')

    # Model architecture
    parser.add_argument('--d-model', type=int, default=128,
                        help='Model dimension (default: 128)')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='Number of attention heads (default: 4)')
    parser.add_argument('--n-layers', type=int, default=4,
                        help='Number of transformer layers (default: 4)')

    # Training
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--seq-len', type=int, default=64,
                        help='Sequence length (default: 64)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')

    # Logging
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log every N steps (default: 10)')
    parser.add_argument('--sample-interval', type=int, default=50,
                        help='Generate sample every N steps (default: 50)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Load text
    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = SHAKESPEARE

    # Train
    result = train(
        text=text,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
        sample_interval=args.sample_interval,
        seed=args.seed,
    )

    return result


if __name__ == '__main__':
    main()

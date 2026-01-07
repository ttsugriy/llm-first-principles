# Glossary

A reference for key terms used throughout this book. Terms are organized alphabetically.

---

## A

### Adapter
A parameter-efficient fine-tuning method that inserts small bottleneck layers into a pretrained model. Only the adapter weights are trained while the base model remains frozen. Covered in Stage 9.

### Alignment
The process of making language models helpful, harmless, and honest. Bridges the gap between next-token prediction (what models learn) and human preferences (what we want). Covered in Stage 10.

### Attention
A mechanism that allows a model to focus on different parts of the input when producing each output. Computes weighted combinations of values based on query-key similarity. Foundation of transformer architecture. Covered in Stage 5.

### Autograd (Automatic Differentiation)
A technique for computing derivatives of functions expressed as computer programs. Builds a computational graph during forward pass and traverses it backward to compute gradients. Covered in Stage 2.

### Autoregressive
A model that predicts each element conditioned on all previous elements. Formally: P(x₁, ..., xₙ) = ∏ᵢ P(xᵢ | x₁, ..., xᵢ₋₁). All modern LLMs (GPT, Claude, LLaMA) are autoregressive.

---

## B

### Backpropagation
The algorithm for computing gradients in neural networks by applying the chain rule from outputs to inputs. A specific instance of reverse-mode automatic differentiation. Covered in Stage 2.

### Bigram
A sequence of two adjacent tokens. A bigram model predicts P(next | previous), using only one token of context.

### Bits per Character (BPC)
Cross-entropy measured in base-2 logarithms. Directly interpretable as the number of bits needed to encode each character. Shannon estimated English has ~1 bit/character of entropy.

### Bradley-Terry Model
A probabilistic model for pairwise comparisons: P(A > B) = σ(r_A - r_B), where σ is sigmoid and r is a score. Used in reward modeling to learn from human preferences. Covered in Stage 10.

### BPE (Byte Pair Encoding)
A subword tokenization algorithm that iteratively merges the most frequent pair of tokens. Used by GPT-2, GPT-3, and GPT-4. Covered in Stage 7.

---

## C

### Causal Masking
A technique that prevents attention from looking at future tokens. Implemented by setting future attention scores to -∞ before softmax. Essential for autoregressive generation. Covered in Stage 5.

### Chain Rule (Probability)
The identity P(A, B, C) = P(A) · P(B|A) · P(C|A,B). Generalizes to n events. Foundation of autoregressive modeling.

### Chain Rule (Calculus)
The identity d/dx[f(g(x))] = f'(g(x)) · g'(x). Foundation of backpropagation.

### Computational Graph
A directed acyclic graph representing a computation, where nodes are operations and edges are data dependencies. Enables automatic differentiation.

### Context Window
The maximum number of tokens a model can "see" when making predictions. Markov models have fixed, small context windows. Transformers have large but still finite windows (4K, 8K, 128K+ tokens).

### Cross-Entropy
A measure of the difference between two probability distributions:
H(p, q) = -∑ p(x) log q(x). In language modeling, p is the true distribution (empirical from data) and q is the model. Lower is better.

---

## D

### Dead Neurons
Neurons that always output zero (or the same value), typically caused by ReLU saturation or poor initialization. A symptom of training problems. Covered in Stage 8.

### Dirichlet Distribution
A probability distribution over probability distributions. Used as a prior in Bayesian smoothing. Parameter α controls how much we trust prior beliefs vs. observed data.

### DPO (Direct Preference Optimization)
An alignment technique that directly optimizes a policy from preference data without training a separate reward model. Simpler and more stable than RLHF. Covered in Stage 10.

---

## E

### Embedding
A learned vector representation of a discrete token. Maps tokens from a sparse one-hot space to a dense continuous space where similar tokens have similar vectors. Covered in Stage 3.

### Entropy
A measure of uncertainty in a probability distribution: H(p) = -∑ p(x) log p(x). Maximum when uniform, minimum (zero) when deterministic.

### Exploding Gradients
When gradients grow exponentially large during backpropagation, causing training instability or NaN values. Solved by gradient clipping. Covered in Stage 8.

---

## F

### Feed-Forward Network (FFN)
A fully connected network applied position-wise in transformers. Typically expands dimension by 4x, applies nonlinearity, then projects back. Covered in Stage 6.

### Fine-tuning
Adapting a pretrained model to a specific task by continuing training on task-specific data. Can be full (update all weights) or parameter-efficient (update a subset). Covered in Stage 9.

### FLOP (Floating Point Operation)
A single arithmetic operation on floating-point numbers. Used to measure computational cost. Modern LLMs require billions to trillions of FLOPs per token.

### Forward Pass
Computing the output of a model given inputs, following the direction of the computational graph from inputs to outputs.

---

## G

### GAE (Generalized Advantage Estimation)
A technique for computing advantages in reinforcement learning that balances bias and variance using a parameter λ. Used in PPO for RLHF. Covered in Stage 10.

### Gradient
The vector of partial derivatives of a function with respect to its inputs. Points in the direction of steepest ascent. Used to update parameters in training.

### Gradient Clipping
Scaling gradients when their norm exceeds a threshold, preventing exploding gradients. Standard practice for transformer training. Covered in Stage 8.

### Gradient Descent
An optimization algorithm that iteratively updates parameters by moving in the direction opposite to the gradient: θ ← θ - η∇L(θ), where η is the learning rate. Covered in Stage 4.

---

## K

### KL Divergence
A measure of how one probability distribution differs from another: KL(P || Q) = Σ P(x) log(P(x)/Q(x)). Used in RLHF to keep the policy close to a reference model. Covered in Stage 10.

### KV Cache
Storing computed key and value tensors during generation to avoid redundant computation. Essential for efficient autoregressive inference.

---

## L

### Laplace Smoothing
A technique to avoid zero probabilities by adding a constant (usually 1) to all counts:
P(b|a) = (count(a,b) + 1) / (count(a,·) + |V|). Also called add-one smoothing.

### LayerNorm (Layer Normalization)
Normalizing activations across the feature dimension (not batch). Standard in transformers. Pre-norm (before sublayer) is more stable than post-norm (after). Covered in Stage 6.

### Learning Rate
The step size in gradient descent. Too high causes divergence; too low causes slow convergence. Often varies during training (warmup, decay). Covered in Stage 4.

### Learning Rate Finder
A technique to find a good learning rate by training with exponentially increasing LR and plotting loss. Choose LR where loss decreases fastest. Covered in Stage 8.

### Likelihood
The probability of observed data given model parameters: L(θ) = P(data | θ). Used to evaluate how well parameters explain the data.

### Log-Likelihood
The natural logarithm of likelihood: ℓ(θ) = log P(data | θ). Converts products to sums, avoiding numerical underflow.

### LoRA (Low-Rank Adaptation)
A parameter-efficient fine-tuning method that represents weight updates as low-rank matrices: ΔW = BA where B is d×r and A is r×k with r << d,k. Covered in Stage 9.

### Loss Curve
A plot of training loss over time. Patterns reveal training health: healthy curves decrease smoothly; problematic curves plateau, oscillate, or explode. Covered in Stage 8.

### Loss Function
A function that measures how wrong a model's predictions are. Training minimizes the loss. Cross-entropy is the standard loss for language models.

---

## M

### Markov Assumption
The assumption that future states depend only on a fixed number of recent states, not the entire history: P(xₜ | x₁...xₜ₋₁) = P(xₜ | xₜ₋ₖ...xₜ₋₁). Covered in Stage 1.

### Maximum Likelihood Estimation (MLE)
Finding parameters that maximize the probability of observed data: θ* = argmax P(data | θ). For Markov models, MLE = counting and normalizing. Covered in Stage 1.

### Momentum
An optimization technique that accumulates gradients over time, smoothing updates and helping escape local minima. Covered in Stage 4.

### Multi-Head Attention
Running multiple attention operations in parallel with different learned projections, then concatenating results. Allows attending to different aspects simultaneously. Covered in Stage 5.

---

## N

### N-gram
A contiguous sequence of n tokens. Unigram (n=1), bigram (n=2), trigram (n=3), etc.

### Normalization
Ensuring probabilities sum to 1. The denominator Z in P(x) = exp(f(x)) / Z is the normalization constant.

---

## O

### Overfitting
When a model performs well on training data but poorly on new data. Detected when training loss decreases but validation loss increases. Covered in Stage 8.

---

## P

### PEFT (Parameter-Efficient Fine-Tuning)
Methods that fine-tune only a small fraction of model parameters while keeping most frozen. Includes LoRA, adapters, prefix tuning. Covered in Stage 9.

### Perplexity
The exponentiated cross-entropy: PPL = exp(H(p, q)). Interpretable as the "effective vocabulary size"—a perplexity of 100 means the model is as uncertain as if choosing uniformly among 100 equally likely tokens. Lower is better. Covered in Stage 1.

### Positional Encoding
Information added to embeddings to indicate token position. Can be sinusoidal (fixed), learned, or relative (RoPE). Covered in Stage 5.

### PPO (Proximal Policy Optimization)
A reinforcement learning algorithm that updates policies with clipped probability ratios to prevent too-large updates. Used in RLHF. Covered in Stage 10.

### Preference Data
Pairs of responses where humans indicate which is better. The fundamental training signal for alignment. Format: (prompt, chosen, rejected). Covered in Stage 10.

### Prefix Tuning
A PEFT method that prepends learned "prefix" vectors to the key and value sequences in attention layers. Covered in Stage 9.

### Prior (Bayesian)
A probability distribution representing beliefs before observing data. In smoothing, the prior expresses our belief that all tokens are somewhat likely.

### Posterior (Bayesian)
The updated probability distribution after combining prior beliefs with observed data. Posterior ∝ likelihood × prior.

### Prompt Tuning
A PEFT method that prepends learned "soft prompt" embeddings to the input. Even simpler than prefix tuning. Covered in Stage 9.

---

## R

### Reference Model
In alignment, the original model before RLHF/DPO training. Used to compute KL penalty, preventing the policy from diverging too far. Covered in Stage 10.

### Regularization
Techniques to prevent overfitting by adding constraints or penalties during training. L2 regularization, dropout, and smoothing are examples.

### Residual Connection
Adding the input to the output of a sublayer: output = input + sublayer(input). Enables gradient flow in deep networks. Essential for transformers. Covered in Stage 6.

### Reverse Mode Differentiation
Computing gradients by traversing the computational graph from outputs to inputs. Efficient when there are many inputs and few outputs (as in neural network training).

### Reward Hacking
When a model finds unexpected ways to maximize reward that don't align with human intent. A key challenge in RLHF. Covered in Stage 10.

### Reward Model
A model trained to predict human preferences, assigning scalar scores to (prompt, response) pairs. Used to provide reward signal in RLHF. Covered in Stage 10.

### RLHF (Reinforcement Learning from Human Feedback)
An alignment technique that trains a reward model on human preferences, then uses RL (typically PPO) to optimize the policy. Covered in Stage 10.

### RMSNorm (Root Mean Square Normalization)
A simpler alternative to LayerNorm that normalizes by the RMS of activations without centering: x / RMS(x). Used in LLaMA and modern architectures. Covered in Stage 6.

---

## S

### Scaling Laws
Empirical relationships between model size, data size, compute, and performance. Loss typically follows power laws: L ∝ N^(-α). Covered in Stage 6.

### Self-Attention
Attention where queries, keys, and values all come from the same sequence. Each position can attend to all other positions. Covered in Stage 5.

### Softmax
A function that converts a vector of real numbers to a probability distribution:
softmax(xᵢ) = exp(xᵢ) / ∑ⱼ exp(xⱼ). Ensures outputs are positive and sum to 1.

### Sparsity
The property of having mostly zero (or near-zero) values. High-order Markov models suffer from sparse observations—most n-grams never appear in training.

### Subword Tokenization
Breaking text into units between characters and words. Handles rare words by splitting, while keeping common words whole. Includes BPE, WordPiece, Unigram. Covered in Stage 7.

### SwiGLU
A gated activation function: SwiGLU(x) = (x · W₁ · σ(x · W₁)) · W₂, where σ is SiLU/Swish. Used in modern transformers like LLaMA. Covered in Stage 6.

---

## T

### Temperature
A parameter that controls the "sharpness" of a probability distribution during sampling:
Pₜ(x) ∝ P(x)^(1/T). T→0 approaches argmax (greedy), T→∞ approaches uniform, T=1 is unchanged. Covered in Stage 1.

### Token
The basic unit of text that a language model operates on. Can be characters, words, or subwords (BPE tokens). Covered in Stage 7.

### Tokenization
The process of converting raw text into discrete tokens for model input. A critical preprocessing step that affects model performance and efficiency. Covered in Stage 7.

### Transformer
The dominant architecture for modern LLMs. Uses self-attention to process entire sequences in parallel. Introduced in "Attention Is All You Need" (2017). Covered in Stage 6.

### Transformer Block
A single layer of a transformer: attention sublayer + FFN sublayer, each with residual connections and normalization. Covered in Stage 6.

---

## U

### Underflow
When a floating-point number becomes too small to represent (approaches zero). Avoided by working in log-space.

### Unigram (Tokenization)
A subword tokenization algorithm that starts with a large vocabulary and iteratively removes tokens to maximize likelihood. Used in SentencePiece. Covered in Stage 7.

---

## V

### Vanishing Gradients
When gradients become exponentially small during backpropagation, preventing learning in early layers. Solved by residual connections, careful initialization. Covered in Stage 8.

### Vocabulary
The set of all unique tokens a model can handle. Denoted |V|. Modern LLMs have vocabularies of 32K-128K tokens.

---

## W

### Warmup
Gradually increasing the learning rate from zero at the start of training. Helps stabilize early training when gradients are noisy. Covered in Stage 4.

### Weight Decay
A regularization technique that adds a penalty proportional to weight magnitude: L_total = L + λ||w||². Prevents weights from growing too large. Covered in Stage 4.

### Weight Tying
Sharing the embedding matrix between input embeddings and output projection. Reduces parameters by half for the vocabulary-related weights. Covered in Stage 6.

### WordPiece
A subword tokenization algorithm similar to BPE but uses likelihood-based scoring for merges. Used by BERT. Covered in Stage 7.

---

## Reference Table

| Symbol | Meaning |
|--------|---------|
| P(A) | Probability of event A |
| P(A\|B) | Probability of A given B |
| θ | Model parameters |
| &#124;V&#124; | Vocabulary size |
| n | Sequence length |
| k | Context size (Markov order) |
| H(p) | Entropy of distribution p |
| H(p,q) | Cross-entropy between p and q |
| PPL | Perplexity |
| ∇f | Gradient of function f |
| η | Learning rate |
| α | Smoothing parameter / LoRA scaling |
| β | Momentum coefficient / DPO temperature |
| r | LoRA rank |
| σ | Sigmoid function |
| KL(P\|\|Q) | KL divergence from Q to P |

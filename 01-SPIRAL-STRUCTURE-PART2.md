# Spiral Structure: Part 2 (Spirals 4-5)

---

# SPIRAL 4: MAKING IT FAST

**Theme:** "We hit walls. Memory, compute. Here's how to break through."

**Narrative Arc:** Training transformers exposes fundamental resource constraints. We analyze where resources go and develop principled optimizations.

---

## Stage 11: Memory Analysis

### Learning Objectives
- Understand exactly where GPU memory goes during training
- Break down: parameters, gradients, optimizer states, activations
- Understand why training uses much more memory than inference
- Profile real models to verify analysis
- Identify the dominant memory consumers

### Key Concepts

| Concept | Intuition | Mathematical Form |
|---------|-----------|-------------------|
| Parameter Memory | Storing model weights | 4 bytes × num_params (FP32) |
| Gradient Memory | Storing gradients | Same as parameters |
| Optimizer State | Adam's moments | 2× parameter memory (Adam) |
| Activation Memory | Intermediate values | O(batch × seq × hidden × layers) |
| Peak Memory | Maximum during training | Max over forward + backward |

### Mathematical Derivations

1. **Transformer Parameter Count**
   - Embeddings: vocab × d_model
   - Attention per layer: 4 × d_model² (Wq, Wk, Wv, Wo)
   - FFN per layer: 8 × d_model² (up and down, with 4× expansion)
   - Total: embedding + n_layers × (4d² + 8d²)

2. **Activation Memory Analysis**
   - Per attention layer: batch × seq × seq × heads (attention weights)
   - Per FFN layer: batch × seq × 4×d_model (after expansion)
   - Total: O(batch × seq² × layers) for attention-dominated

3. **Training vs Inference Memory**
   - Training: must store activations for backward pass
   - Inference: can discard after each layer
   - Ratio: O(layers) difference

4. **Memory Timeline**
   - Forward: memory grows (storing activations)
   - Backward: memory shrinks (releasing activations)
   - Peak: at end of forward pass

### Code Implementations

```
stage-11/
├── profiling/
│   ├── memory_profiler.py     # Track GPU memory over time
│   ├── parameter_count.py     # Count parameters precisely
│   ├── activation_size.py     # Compute activation memory
│   └── timeline.py            # Memory timeline visualization
├── analysis/
│   ├── breakdown.py           # Params/grads/optim/activations
│   ├── scaling.py             # Memory vs model size
│   └── batch_vs_seq.py        # Trade-off analysis
└── experiments/
    ├── profile_gpt2.py        # Profile GPT-2 variants
    └── find_max_batch.py      # Find maximum batch size
```

### Visualizations

1. **Memory Breakdown Pie Chart**
   - Parameters, gradients, optimizer, activations
   - Show activations dominate during training

2. **Memory Timeline**
   - Memory over forward → backward pass
   - Annotate peak memory

3. **Scaling Curves**
   - Memory vs model size
   - Memory vs sequence length (quadratic!)
   - Memory vs batch size (linear)

4. **Training vs Inference Comparison**
   - Side-by-side memory usage
   - Show the gap

### Trade-offs to Analyze

| Knob | Decrease Memory | Consequence |
|------|-----------------|-------------|
| Batch Size | Linear decrease | Slower training |
| Sequence Length | Quadratic decrease | Worse perplexity |
| Model Size | Linear decrease | Worse quality |
| Precision | 2× with FP16 | Potential instability |

### Connections
- **Backward:** Explains why Stage 4-8 models hit limits
- **Forward:** Motivates all optimizations in Stage 12-13
- **Forward:** Sets up mixed precision discussion

### Estimated Length
- Main text: 6,000-8,000 words
- Code: ~300 lines
- Time to read: 40-50 minutes

---

## Stage 12: Memory Optimization

### Learning Objectives
- Understand and implement gradient checkpointing
- Understand and derive Flash Attention
- Understand mixed precision training
- Analyze trade-offs for each optimization
- Combine optimizations effectively

### Key Concepts

| Concept | Intuition | Mathematical Form |
|---------|-----------|-------------------|
| Gradient Checkpointing | Recompute instead of store | Memory ↔ Compute trade-off |
| Flash Attention | I/O-aware attention | Tiling + online softmax |
| Mixed Precision | Lower precision math | FP16/BF16 for most operations |
| Loss Scaling | Prevent gradient underflow | Scale loss up, gradients down |

### Mathematical Derivations

1. **Gradient Checkpointing**
   - Without: O(n) memory for n layers
   - With k checkpoints: O(n/k) memory, O(n/k) extra compute
   - Optimal checkpoint placement: √n checkpoints for O(√n) memory

2. **Flash Attention Derivation**
   - Memory hierarchy: SRAM (fast, small) vs HBM (slow, large)
   - I/O complexity of standard attention: O(n² d) reads/writes
   - Tiling: process blocks that fit in SRAM
   - Online softmax: compute softmax incrementally
   - Result: O(n² d / M) I/O for SRAM size M

3. **Mixed Precision Math**
   - FP32: 1 sign + 8 exp + 23 mantissa (range ±3.4×10³⁸)
   - FP16: 1 sign + 5 exp + 10 mantissa (range ±65504)
   - BF16: 1 sign + 8 exp + 7 mantissa (same range as FP32)
   - Gradient magnitudes: when they underflow

4. **Loss Scaling Derivation**
   - If gradients are ~10⁻⁸, FP16 can't represent
   - Scale loss by 2¹⁶, gradients also scale
   - Now representable, divide back after

### Code Implementations

```
stage-12/
├── checkpointing/
│   ├── basic.py               # Simple checkpointing
│   ├── selective.py           # Selective layer checkpointing
│   └── optimal.py             # Optimal checkpoint placement
├── flash_attention/
│   ├── naive.py               # Standard attention (baseline)
│   ├── tiled.py               # Tiled attention
│   ├── online_softmax.py      # Online softmax algorithm
│   └── flash.py               # Full Flash Attention
├── mixed_precision/
│   ├── fp16_training.py       # FP16 with loss scaling
│   ├── bf16_training.py       # BF16 (no scaling needed)
│   └── autocast.py            # Automatic mixed precision
├── analysis/
│   ├── io_analysis.py         # Measure I/O operations
│   ├── precision_analysis.py  # Gradient magnitude distribution
│   └── combined.py            # All optimizations together
└── experiments/
    ├── checkpoint_sweep.py    # Memory vs recompute
    ├── flash_speedup.py       # Flash vs standard attention
    └── precision_comparison.py # FP32 vs FP16 vs BF16
```

### Visualizations

1. **Checkpointing Memory Trade-off**
   - Memory vs number of checkpoints
   - Compute overhead vs checkpoints

2. **Flash Attention I/O Pattern**
   - Diagram of tiled processing
   - SRAM vs HBM access pattern

3. **Online Softmax**
   - Algorithm visualization
   - Running max and sum

4. **Gradient Magnitude Distribution**
   - Histogram of gradient values
   - Show FP16 representable range
   - Show underflow region

### Trade-offs to Analyze

| Optimization | Memory Savings | Compute Cost | Complexity |
|--------------|----------------|--------------|------------|
| Checkpointing | O(√n) | +33% typically | Low |
| Flash Attention | O(n) → O(1) for attn | -20% (faster!) | High |
| FP16 | 2× | Neutral or faster | Medium |
| BF16 | 2× | Neutral | Low |

### Connections
- **Backward:** Directly addresses Stage 11's memory analysis
- **Forward:** Enables training larger models
- **Forward:** Flash Attention used in Stage 14 modern architectures

### Estimated Length
- Main text: 12,000-15,000 words
- Code: ~700 lines
- Time to read: 75-90 minutes

---

## Stage 13: Distributed Training

### Learning Objectives
- Understand why we need to distribute training
- Implement data parallelism
- Understand tensor parallelism
- Understand pipeline parallelism
- Analyze ZeRO optimization
- Understand how to combine strategies

### Key Concepts

| Concept | Intuition | Mathematical Form |
|---------|-----------|-------------------|
| Data Parallelism | Same model, different data | Replicate model, split batch |
| Tensor Parallelism | Split layers across GPUs | Column/row partitioning |
| Pipeline Parallelism | Different layers on different GPUs | Layer partitioning |
| ZeRO | Distribute optimizer state | Partition instead of replicate |
| All-Reduce | Average gradients | Ring all-reduce |

### Mathematical Derivations

1. **Data Parallelism Correctness**
   - Full batch gradient = average of minibatch gradients
   - Prove: ∇L(full) = (1/k) Σᵢ ∇L(minibatchᵢ)
   - Communication cost: O(params) per step

2. **Tensor Parallelism**
   - Matrix multiply splitting: Y = XW
   - Column parallel: W = [W₁, W₂], Y = [XW₁, XW₂]
   - Row parallel: W = [W₁; W₂], Y = X[W₁; W₂] (requires all-reduce)
   - Communication analysis

3. **Pipeline Parallelism**
   - Model split into stages
   - Microbatching to fill pipeline
   - Bubble analysis: idle time = (p-1)/(m+p-1) for p stages, m microbatches

4. **ZeRO Analysis**
   - Memory per GPU with replication: params + grads + optimizer
   - ZeRO Stage 1: partition optimizer states
   - ZeRO Stage 2: partition gradients too
   - ZeRO Stage 3: partition parameters too
   - Communication vs memory trade-off

### Code Implementations

```
stage-13/
├── data_parallel/
│   ├── naive.py               # Simple gradient averaging
│   ├── ring_allreduce.py      # Ring all-reduce
│   └── bucketed.py            # Bucketed gradient sync
├── tensor_parallel/
│   ├── column_parallel.py     # Column-parallel linear
│   ├── row_parallel.py        # Row-parallel linear
│   └── attention_parallel.py  # Parallel attention
├── pipeline_parallel/
│   ├── naive.py               # Sequential pipeline
│   ├── gpipe.py               # GPipe-style microbatching
│   └── bubble_analysis.py     # Analyze idle time
├── zero/
│   ├── stage1.py              # Optimizer state partitioning
│   ├── stage2.py              # Gradient partitioning
│   └── stage3.py              # Parameter partitioning
├── analysis/
│   ├── communication.py       # Communication volume analysis
│   ├── throughput.py          # Tokens per second
│   └── scaling_efficiency.py  # Efficiency vs GPU count
└── experiments/
    └── strategy_comparison.py # Compare all strategies
```

### Visualizations

1. **Data Parallelism**
   - Diagram showing model replication
   - Gradient averaging pattern

2. **Tensor Parallelism**
   - Matrix split diagrams
   - Communication patterns

3. **Pipeline Bubble**
   - Timeline showing idle time
   - Effect of microbatch count

4. **ZeRO Memory Reduction**
   - Memory per GPU vs ZeRO stage
   - Communication overhead

5. **Scaling Efficiency**
   - Throughput vs GPU count
   - Compare strategies

### Trade-offs to Analyze

| Strategy | Memory per GPU | Communication | Complexity | Best For |
|----------|----------------|---------------|------------|----------|
| Data Parallel | Full model | O(params) | Low | Small models |
| Tensor Parallel | Reduced | High frequency | Medium | Large layers |
| Pipeline Parallel | ~1/stages | Low frequency | High | Very deep |
| ZeRO | Configurable | Increases with stage | Medium | Large models |

### Connections
- **Backward:** Uses Stage 12's memory optimizations
- **Forward:** Enables training scale discussed in Stage 14
- **Forward:** Relevant for inference serving (Stage 17)

### Estimated Length
- Main text: 12,000-15,000 words
- Code: ~800 lines
- Time to read: 75-90 minutes

---

# SPIRAL 5: MODERN PRACTICE

**Theme:** "What state-of-the-art looks like and why"

**Narrative Arc:** We bring together everything into modern, production-ready LLM practice, from architectures to alignment to deployment.

---

## Stage 14: Architecture Evolution

### Learning Objectives
- Understand evolution from GPT-2 to modern architectures
- Understand and implement RMSNorm
- Understand and implement SwiGLU
- Understand Grouped Query Attention (GQA)
- Analyze scaling laws
- Design architectures for compute budgets

### Key Concepts

| Concept | Intuition | Mathematical Form |
|---------|-----------|-------------------|
| RMSNorm | LayerNorm without mean centering | x / RMS(x) |
| SwiGLU | Gated activation in FFN | Swish(xW_gate) ⊙ (xW_up) |
| GQA | Share KV heads | Fewer KV heads than Q heads |
| Scaling Laws | Performance vs resources | L = (C/C₀)^α |
| Chinchilla | Compute-optimal ratio | Tokens ≈ 20 × Parameters |

### Mathematical Derivations

1. **RMSNorm**
   - Standard LayerNorm: (x - μ) / σ
   - RMSNorm: x / √(mean(x²))
   - Why removing mean works
   - Computational savings

2. **SwiGLU Derivation**
   - GLU: sigmoid(xW_g) ⊙ xW_u
   - SwiGLU: swish(xW_g) ⊙ xW_u
   - Parameter count: 3 matrices instead of 2
   - Effective hidden dimension: 8/3 × d_model for same params

3. **GQA Analysis**
   - Standard MHA: h query heads, h KV heads
   - MQA: h query heads, 1 KV head
   - GQA: h query heads, g KV heads (g < h)
   - KV cache memory: O(seq × g × d_head) vs O(seq × h × d_head)

4. **Scaling Laws**
   - Empirical observations: L ∝ N^(-α) for N parameters
   - Chinchilla finding: compute-optimal requires balanced scaling
   - Derive: for compute budget C, optimal N ∝ C^0.5

### Code Implementations

```
stage-14/
├── normalization/
│   └── rmsnorm.py             # RMSNorm implementation
├── activations/
│   ├── swiglu.py              # SwiGLU activation
│   └── comparison.py          # Compare activations
├── attention/
│   ├── gqa.py                 # Grouped Query Attention
│   └── mqa.py                 # Multi-Query Attention
├── architectures/
│   ├── gpt2.py                # GPT-2 style
│   ├── llama.py               # Llama style
│   └── comparison.py          # Compare architectures
├── scaling/
│   ├── laws.py                # Scaling law analysis
│   ├── chinchilla.py          # Compute-optimal training
│   └── planner.py             # Architecture for budget
└── experiments/
    ├── architecture_ablation.py
    └── scaling_experiments.py
```

### Visualizations

1. **Architecture Comparison Diagram**
   - GPT-2 vs Llama side-by-side
   - Annotate differences

2. **SwiGLU vs ReLU/GELU**
   - Activation function plots
   - Information flow comparison

3. **GQA Memory Savings**
   - KV cache size vs number of KV heads
   - Quality vs memory trade-off

4. **Scaling Laws**
   - Loss vs compute curves
   - Chinchilla optimal frontier

### Trade-offs to Analyze

| Component | Original | Modern | Why |
|-----------|----------|--------|-----|
| Normalization | LayerNorm | RMSNorm | 10-15% faster, same quality |
| Activation | GELU | SwiGLU | Better quality |
| Attention | MHA | GQA | 2-8× less KV cache |
| Position | Learned | RoPE | Better extrapolation |

### Connections
- **Backward:** Applies all previous optimizations
- **Forward:** Architecture for fine-tuning (Stage 15)
- **Forward:** Architecture for inference (Stage 17)

### Estimated Length
- Main text: 10,000-12,000 words
- Code: ~600 lines
- Time to read: 60-75 minutes

---

## Stage 15: Fine-Tuning

### Learning Objectives
- Understand why fine-tuning works (feature reuse)
- Implement standard fine-tuning
- Understand and implement LoRA
- Understand QLoRA
- Analyze parameter-efficient methods

### Key Concepts

| Concept | Intuition | Mathematical Form |
|---------|-----------|-------------------|
| Fine-tuning | Adapt pretrained model | Continue training on new data |
| Feature Reuse | Early layers transfer | Freeze or small LR for early |
| LoRA | Low-rank weight updates | W' = W + BA where B,A are small |
| Rank | Dimension of update | r << d_model |
| QLoRA | LoRA on quantized model | 4-bit base + FP16 adapters |

### Mathematical Derivations

1. **Why Fine-tuning Works**
   - Pretrained weights as good initialization
   - Loss landscape has better structure
   - Features transfer across tasks

2. **LoRA Derivation**
   - Weight change is often low-rank
   - Full fine-tune: W' = W + ΔW
   - LoRA: ΔW ≈ BA where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}
   - Parameters: r(d+k) instead of dk
   - Merge at inference: W_merged = W + BA

3. **Rank Selection**
   - Higher rank = more capacity = more parameters
   - Typical: r = 8, 16, 32
   - Often just attention layers

4. **QLoRA Memory Analysis**
   - Base model: 4-bit quantized
   - Adapters: FP16
   - Gradients: only through adapters
   - Memory savings: ~4× vs full fine-tune

### Code Implementations

```
stage-15/
├── fine_tuning/
│   ├── standard.py            # Full fine-tuning
│   ├── freeze_layers.py       # Freeze early layers
│   └── lr_schedule.py         # Discriminative LR
├── lora/
│   ├── layer.py               # LoRA linear layer
│   ├── inject.py              # Inject LoRA into model
│   ├── merge.py               # Merge LoRA weights
│   └── rank_analysis.py       # Effect of rank
├── qlora/
│   ├── quantize.py            # 4-bit quantization
│   └── qlora.py               # QLoRA training
├── analysis/
│   ├── weight_change.py       # Analyze Δw rank
│   └── memory.py              # Memory comparison
└── experiments/
    ├── rank_sweep.py
    └── method_comparison.py
```

### Visualizations

1. **LoRA Architecture Diagram**
   - Original weight plus low-rank path
   - Show parameter counts

2. **Weight Change Rank Analysis**
   - Singular values of ΔW
   - Show rapid decay (low-rank)

3. **Memory Comparison**
   - Full fine-tune vs LoRA vs QLoRA
   - Training memory bar chart

4. **Quality vs Rank**
   - Performance for different r values
   - Diminishing returns

### Trade-offs to Analyze

| Method | Memory | Quality | Speed | Flexibility |
|--------|--------|---------|-------|-------------|
| Full Fine-tune | High | Best | Slowest | Full |
| LoRA (r=16) | Low | Good | Fast | Limited |
| QLoRA | Very Low | Good | Medium | Limited |

### Connections
- **Backward:** Requires understanding Stage 14 architectures
- **Forward:** Adapted models for alignment (Stage 16)

### Estimated Length
- Main text: 8,000-10,000 words
- Code: ~500 lines
- Time to read: 50-60 minutes

---

## Stage 16: Alignment (RLHF/DPO)

### Learning Objectives
- Understand the alignment problem
- Understand reward modeling
- Derive the RLHF objective
- Understand PPO for language models
- Derive DPO from RLHF
- Compare RLHF vs DPO trade-offs

### Key Concepts

| Concept | Intuition | Mathematical Form |
|---------|-----------|-------------------|
| Alignment | Match model behavior to preferences | Train to be helpful, harmless |
| Reward Model | Score response quality | r(x, y) : prompt, response → ℝ |
| Bradley-Terry | Model pairwise preferences | P(y₁ > y₂) = σ(r(y₁) - r(y₂)) |
| PPO | Policy gradient with clipping | Clip ratio, maximize advantage |
| KL Penalty | Stay close to base model | λ · KL(π || π_ref) |
| DPO | Direct preference optimization | No reward model needed |

### Mathematical Derivations

1. **Reward Model from Preferences**
   - Bradley-Terry model for pairwise comparisons
   - Loss: -log σ(r(y_w) - r(y_l)) for winner/loser
   - Why this is maximum likelihood

2. **RLHF Objective**
   - Maximize E[r(y)] - λ·KL(π || π_ref)
   - Why KL penalty (don't diverge too far from base)
   - Connection to KL-regularized RL

3. **PPO for Language Models**
   - Policy gradient: ∇ log π(a|s) · A(s,a)
   - PPO clipping: clip ratio to [1-ε, 1+ε]
   - Advantage estimation for language

4. **DPO Derivation**
   - Start from RLHF objective
   - Closed-form optimal policy: π*(y|x) ∝ π_ref(y|x) exp(r(y)/λ)
   - Substitute into preference model
   - Derive DPO loss: -log σ(β(log π(y_w) - log π(y_l) - log π_ref(y_w) + log π_ref(y_l)))
   - No reward model needed!

5. **RLHF vs DPO Trade-offs**
   - RLHF: online, can generate; harder to train
   - DPO: offline, simpler; needs paired data

### Code Implementations

```
stage-16/
├── reward_model/
│   ├── bradley_terry.py       # Bradley-Terry loss
│   ├── reward_head.py         # Add reward head to LM
│   └── train.py               # Train reward model
├── rlhf/
│   ├── ppo.py                 # PPO algorithm
│   ├── advantage.py           # Advantage estimation
│   ├── kl_penalty.py          # KL divergence penalty
│   └── train.py               # RLHF training loop
├── dpo/
│   ├── loss.py                # DPO loss function
│   ├── train.py               # DPO training
│   └── derivation.py          # Verify derivation numerically
├── analysis/
│   ├── reward_distribution.py # Analyze learned rewards
│   └── kl_analysis.py         # Monitor KL divergence
└── experiments/
    ├── rlhf_vs_dpo.py
    └── beta_sweep.py
```

### Visualizations

1. **Reward Model Training**
   - Preference accuracy over training
   - Reward distribution for good vs bad

2. **RLHF Training Dynamics**
   - Reward, KL divergence, loss over time
   - Balance between reward and KL

3. **DPO Derivation Steps**
   - Mathematical derivation visualized
   - Show each substitution step

4. **RLHF vs DPO Comparison**
   - Quality vs compute
   - Sample efficiency

### Trade-offs to Analyze

| Method | Data Needs | Compute | Stability | Quality |
|--------|------------|---------|-----------|---------|
| SFT only | Instructions | Low | High | Medium |
| RLHF | Preferences + RM | High | Low | High |
| DPO | Preferences only | Medium | High | High |
| IPO/ORPO | Preferences only | Medium | High | High |

### Connections
- **Backward:** Applies to fine-tuned models (Stage 15)
- **Forward:** Aligned models for deployment (Stage 17)

### Estimated Length
- Main text: 12,000-15,000 words
- Code: ~700 lines
- Time to read: 75-90 minutes

---

## Stage 17: Inference Optimization

### Learning Objectives
- Understand KV caching deeply
- Analyze memory-bound inference
- Understand speculative decoding
- Understand continuous batching
- Optimize for throughput vs latency

### Key Concepts

| Concept | Intuition | Mathematical Form |
|---------|-----------|-------------------|
| KV Cache | Cache key-value projections | Store K, V for past tokens |
| Memory Bandwidth | Speed of memory access | GB/s between GPU memory and compute |
| Arithmetic Intensity | Compute per byte | FLOPs / bytes accessed |
| Speculative Decoding | Draft then verify | Small model drafts, large verifies |
| Continuous Batching | Dynamic batch management | Add/remove sequences mid-batch |

### Mathematical Derivations

1. **KV Cache Size Analysis**
   - Per layer: 2 × seq × n_heads × d_head × precision
   - Total: 2 × layers × seq × d_model × precision
   - For 7B model, 4K context: ~500MB per sequence

2. **Memory-Bound Analysis**
   - Inference arithmetic intensity is low
   - For generation: read all parameters for each token
   - Bandwidth-limited, not compute-limited

3. **Speculative Decoding**
   - Draft k tokens with small model
   - Verify with large model in parallel
   - Acceptance probability: P(accept) = min(1, p_large/p_draft)
   - Speedup depends on acceptance rate

4. **Continuous Batching Throughput**
   - Static batching: wait for all to finish
   - Continuous: fill slots as they free
   - Throughput improvement analysis

### Code Implementations

```
stage-17/
├── kv_cache/
│   ├── basic.py               # Basic KV cache
│   ├── paged.py               # Paged attention (vLLM-style)
│   └── memory_analysis.py     # Cache memory profiling
├── speculative/
│   ├── draft_model.py         # Draft model setup
│   ├── verify.py              # Verification step
│   ├── speculative.py         # Full speculative decoding
│   └── analysis.py            # Acceptance rate analysis
├── batching/
│   ├── static.py              # Static batching
│   ├── continuous.py          # Continuous batching
│   └── scheduler.py           # Request scheduling
├── analysis/
│   ├── roofline.py            # Roofline analysis
│   ├── bandwidth.py           # Memory bandwidth profiling
│   └── throughput.py          # Tokens per second
└── experiments/
    ├── batch_size.py
    └── context_length.py
```

### Visualizations

1. **KV Cache Memory Scaling**
   - Memory vs sequence length
   - Per-sequence cost

2. **Roofline Model**
   - Compute intensity vs performance
   - Show inference is memory-bound

3. **Speculative Decoding Timeline**
   - Draft, verify, accept/reject phases
   - Speedup visualization

4. **Continuous Batching**
   - Timeline showing request handling
   - Compare to static batching

### Trade-offs to Analyze

| Optimization | Latency Impact | Throughput Impact | Complexity |
|--------------|----------------|-------------------|------------|
| KV Cache | Required | Required | Low |
| Speculative | Improves | Neutral | Medium |
| Continuous Batching | Neutral | Improves | High |
| Longer Batches | Hurts | Improves | Low |

### Connections
- **Backward:** Uses Stage 14's GQA for smaller KV cache
- **Forward:** Quantization (Stage 18) further reduces memory

### Estimated Length
- Main text: 10,000-12,000 words
- Code: ~600 lines
- Time to read: 60-75 minutes

---

## Stage 18: Quantization

### Learning Objectives
- Understand why quantization helps inference
- Derive quantization mathematics
- Implement INT8 quantization
- Understand INT4 and lower
- Analyze accuracy vs efficiency trade-offs
- Understand quantization-aware training

### Key Concepts

| Concept | Intuition | Mathematical Form |
|---------|-----------|-------------------|
| Quantization | Reduce precision for efficiency | FP16 → INT8 or INT4 |
| Scale | Map floating point to integer range | x_int = round(x_fp / scale) |
| Zero Point | Handle asymmetric distributions | x_int = round(x_fp / scale) + zp |
| Calibration | Find optimal scales | Minimize quantization error |
| Mixed Precision | Some layers in full precision | Sensitive layers stay FP16 |

### Mathematical Derivations

1. **Quantization Error Analysis**
   - Uniform quantization: max error = scale/2
   - For b bits: 2^b levels
   - Trade-off: fewer bits = more error

2. **Optimal Scale Selection**
   - Minimize E[(x - dequant(quant(x)))²]
   - For uniform distribution: scale = (max - min) / (2^b - 1)
   - For Gaussian: scale ≈ 6σ / (2^b - 1)

3. **Per-Channel vs Per-Tensor**
   - Per-tensor: one scale for entire tensor
   - Per-channel: one scale per output channel
   - Per-channel is more accurate, more complex

4. **GPTQ and AWQ**
   - Second-order information for better quantization
   - Activation-aware: use calibration data
   - Mathematical formulation

### Code Implementations

```
stage-18/
├── quantization/
│   ├── int8.py                # INT8 quantization
│   ├── int4.py                # INT4 quantization
│   ├── calibration.py         # Calibration for scales
│   └── mixed_precision.py     # Mixed precision selection
├── methods/
│   ├── naive.py               # Simple round-to-nearest
│   ├── gptq.py                # GPTQ algorithm
│   └── awq.py                 # Activation-aware quantization
├── analysis/
│   ├── error.py               # Quantization error analysis
│   ├── sensitivity.py         # Layer sensitivity analysis
│   └── perplexity.py          # Quality degradation
└── experiments/
    ├── bit_sweep.py           # Compare bit widths
    └── method_comparison.py   # Compare quantization methods
```

### Visualizations

1. **Quantization Error Histogram**
   - Distribution of errors
   - Compare bit widths

2. **Layer Sensitivity**
   - Perplexity impact per layer
   - Identify sensitive layers

3. **Bits vs Quality**
   - Perplexity vs bit width
   - Speedup vs bit width

4. **Weight Distributions**
   - Before and after quantization
   - Show clipping effects

### Trade-offs to Analyze

| Precision | Memory | Speed | Quality | Use Case |
|-----------|--------|-------|---------|----------|
| FP16 | 1× | 1× | Best | Training |
| INT8 | 0.5× | 1.5× | Good | Inference |
| INT4 | 0.25× | 2× | Medium | Consumer GPUs |
| INT2-3 | 0.125× | 2.5× | Degraded | Extreme edge |

### Connections
- **Backward:** Applies to optimized inference (Stage 17)
- **Backward:** QLoRA uses quantization (Stage 15)
- **Completion:** Final stage of the journey

### Estimated Length
- Main text: 10,000-12,000 words
- Code: ~500 lines
- Time to read: 60-75 minutes

---

# Summary Statistics

| Spiral | Stages | Est. Words | Est. Code Lines | Est. Read Time |
|--------|--------|------------|-----------------|----------------|
| 1: Foundations | 4 | 36,000-44,000 | 2,100 | 4-5 hours |
| 2: Training | 2 | 20,000-24,000 | 1,000 | 2-2.5 hours |
| 3: Transformer | 4 | 36,000-44,000 | 1,800 | 4-5 hours |
| 4: Making Fast | 3 | 30,000-38,000 | 1,800 | 3-4 hours |
| 5: Modern | 5 | 50,000-64,000 | 2,900 | 5-7 hours |
| **Total** | **18** | **172,000-214,000** | **9,600** | **18-24 hours** |

This is equivalent to a substantial technical book (200,000 words ≈ 500-600 pages).

---

# Cross-Stage Dependencies

```
Stage 1 (Markov) ─────────────────────────────────────────┐
    │                                                      │
    ▼                                                      │
Stage 2 (Autodiff) ───┐                                   │
    │                  │                                   │
    ▼                  ▼                                   │
Stage 3 (Backprop) ────────┐                              │
    │                       │                              │
    ▼                       ▼                              ▼
Stage 4 (MLP LM) ─────→ Stage 5 (Optimization) ─────→ Perplexity baseline
    │                       │
    ▼                       ▼
    │                  Stage 6 (Stability)
    │                       │
    ▼                       ▼
Stage 7 (Attention) ←──────┘
    │
    ▼
Stage 8 (Multi-head) ───→ Stage 9 (Position) ───→ Stage 10 (Tokenization)
    │                                                  │
    ▼                                                  ▼
Stage 11 (Memory Analysis) ←───────────────────────────┘
    │
    ▼
Stage 12 (Memory Opt) ───→ Stage 13 (Distributed)
    │                           │
    ▼                           ▼
Stage 14 (Modern Arch) ←────────┘
    │
    ▼
Stage 15 (Fine-tuning) ───→ Stage 16 (Alignment)
    │                           │
    ▼                           ▼
Stage 17 (Inference) ←──────────┘
    │
    ▼
Stage 18 (Quantization)
```

---

# Next Steps

1. Create `02-PEDAGOGICAL-FRAMEWORK.md` with detailed methodology guide
2. Create `03-MATHEMATICAL-FOUNDATIONS.md` with all prerequisites
3. Create first stage document: `stages/stage-01-markov-chains.md`
4. Begin code implementation for Stage 1

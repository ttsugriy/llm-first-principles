# Research Notes and References

## Overview

This document collects all research materials, references, and sources for the book. Organized by topic for easy lookup and citation.

---

## 1. Foundational Papers

### 1.1 Transformer Architecture

**"Attention Is All You Need" (Vaswani et al., 2017)**
- Paper: https://arxiv.org/abs/1706.03762
- Key contributions:
  - Introduced the Transformer architecture
  - Self-attention mechanism
  - Multi-head attention
  - Positional encoding (sinusoidal)
- Used in: Stage 7, 8, 9

**"BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)**
- Paper: https://arxiv.org/abs/1810.04805
- Key contributions:
  - Bidirectional pre-training
  - Masked language modeling
  - Next sentence prediction
- Used in: Stage 14 (historical context)

**"Language Models are Unsupervised Multitask Learners" (GPT-2, Radford et al., 2019)**
- Paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Key contributions:
  - Scaling up GPT
  - Zero-shot task performance
  - WebText dataset
- Used in: Stage 14

**"Language Models are Few-Shot Learners" (GPT-3, Brown et al., 2020)**
- Paper: https://arxiv.org/abs/2005.14165
- Key contributions:
  - 175B parameter model
  - In-context learning
  - Scaling laws emergence
- Used in: Stage 14

### 1.2 Modern Architectures

**"LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)**
- Paper: https://arxiv.org/abs/2302.13971
- Key contributions:
  - Open weights
  - RMSNorm, SwiGLU, RoPE combination
  - Efficient training recipe
- Used in: Stage 14

**"Llama 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al., 2023)**
- Paper: https://arxiv.org/abs/2307.09288
- Key contributions:
  - Grouped Query Attention (GQA)
  - Ghost Attention for multi-turn
  - Safety training
- Used in: Stage 14, 16

### 1.3 Attention Variants

**"Fast Transformer Decoding: One Write-Head is All You Need" (MQA, Shazeer, 2019)**
- Paper: https://arxiv.org/abs/1911.02150
- Key contribution: Multi-Query Attention for inference efficiency
- Used in: Stage 14, 17

**"GQA: Training Generalized Multi-Query Transformer Models" (Ainslie et al., 2023)**
- Paper: https://arxiv.org/abs/2305.13245
- Key contribution: Grouped Query Attention as MHA/MQA interpolation
- Used in: Stage 14, 17

**"FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)**
- Paper: https://arxiv.org/abs/2205.14135
- Key contributions:
  - I/O-aware attention algorithm
  - Tiling for SRAM
  - Online softmax
- Used in: Stage 12

**"FlashAttention-2: Faster Attention with Better Parallelism" (Dao, 2023)**
- Paper: https://arxiv.org/abs/2307.08691
- Key contributions:
  - Improved parallelism
  - Better GPU utilization
- Used in: Stage 12

### 1.4 Position Encodings

**"RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)**
- Paper: https://arxiv.org/abs/2104.09864
- Key contribution: Rotary Position Embeddings (RoPE)
- Used in: Stage 9, 14

**"Train Short, Test Long: Attention with Linear Biases" (ALiBi, Press et al., 2022)**
- Paper: https://arxiv.org/abs/2108.12409
- Key contribution: Attention with Linear Biases for length extrapolation
- Used in: Stage 9

### 1.5 Normalization

**"Layer Normalization" (Ba et al., 2016)**
- Paper: https://arxiv.org/abs/1607.06450
- Key contribution: Layer normalization for RNNs and Transformers
- Used in: Stage 6

**"Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)**
- Paper: https://arxiv.org/abs/1910.07467
- Key contribution: RMSNorm simplification
- Used in: Stage 6, 14

### 1.6 Activation Functions

**"GLU Variants Improve Transformer" (Shazeer, 2020)**
- Paper: https://arxiv.org/abs/2002.05202
- Key contribution: SwiGLU and other GLU variants
- Used in: Stage 14

**"Gaussian Error Linear Units (GELUs)" (Hendrycks & Gimpel, 2016)**
- Paper: https://arxiv.org/abs/1606.08415
- Key contribution: GELU activation
- Used in: Stage 4

---

## 2. Training and Optimization

### 2.1 Optimization Algorithms

**"Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)**
- Paper: https://arxiv.org/abs/1412.6980
- Key contribution: Adam optimizer
- Used in: Stage 5

**"Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)**
- Paper: https://arxiv.org/abs/1711.05101
- Key contribution: AdamW (decoupled weight decay)
- Used in: Stage 5

### 2.2 Initialization

**"Understanding the difficulty of training deep feedforward neural networks" (Glorot & Bengio, 2010)**
- Paper: http://proceedings.mlr.press/v9/glorot10a.html
- Key contribution: Xavier initialization
- Used in: Stage 6

**"Delving Deep into Rectifiers" (He et al., 2015)**
- Paper: https://arxiv.org/abs/1502.01852
- Key contribution: Kaiming initialization for ReLU
- Used in: Stage 6

### 2.3 Mixed Precision

**"Mixed Precision Training" (Micikevicius et al., 2017)**
- Paper: https://arxiv.org/abs/1710.03740
- Key contribution: FP16 training with loss scaling
- Used in: Stage 12

### 2.4 Distributed Training

**"Megatron-LM: Training Multi-Billion Parameter Language Models" (Shoeybi et al., 2019)**
- Paper: https://arxiv.org/abs/1909.08053
- Key contribution: Tensor parallelism
- Used in: Stage 13

**"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (Rajbhandari et al., 2020)**
- Paper: https://arxiv.org/abs/1910.02054
- Key contribution: ZeRO optimizer states partitioning
- Used in: Stage 13

**"GPipe: Efficient Training of Giant Neural Networks" (Huang et al., 2019)**
- Paper: https://arxiv.org/abs/1811.06965
- Key contribution: Pipeline parallelism with microbatching
- Used in: Stage 13

---

## 3. Scaling Laws

**"Scaling Laws for Neural Language Models" (Kaplan et al., 2020)**
- Paper: https://arxiv.org/abs/2001.08361
- Key contribution: Power law scaling relationships
- Used in: Stage 14

**"Training Compute-Optimal Large Language Models" (Chinchilla, Hoffmann et al., 2022)**
- Paper: https://arxiv.org/abs/2203.15556
- Key contribution: Compute-optimal training (more data, less params)
- Used in: Stage 14

---

## 4. Tokenization

**"Neural Machine Translation of Rare Words with Subword Units" (BPE, Sennrich et al., 2015)**
- Paper: https://arxiv.org/abs/1508.07909
- Key contribution: BPE for NMT
- Used in: Stage 10

**"SentencePiece: A simple and language independent subword tokenizer" (Kudo & Richardson, 2018)**
- Paper: https://arxiv.org/abs/1808.06226
- Key contribution: Language-independent tokenization
- Used in: Stage 10

**"Fast WordPiece Tokenization" (Song et al., 2020)**
- Paper: https://arxiv.org/abs/2012.15524
- Key contribution: Efficient WordPiece
- Used in: Stage 10

---

## 5. Fine-Tuning and Alignment

### 5.1 Parameter-Efficient Fine-Tuning

**"LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)**
- Paper: https://arxiv.org/abs/2106.09685
- Key contribution: Low-rank adaptation
- Used in: Stage 15

**"QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)**
- Paper: https://arxiv.org/abs/2305.14314
- Key contribution: 4-bit quantization + LoRA
- Used in: Stage 15

### 5.2 Alignment

**"Training language models to follow instructions with human feedback" (InstructGPT, Ouyang et al., 2022)**
- Paper: https://arxiv.org/abs/2203.02155
- Key contribution: RLHF for instruction following
- Used in: Stage 16

**"Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023)**
- Paper: https://arxiv.org/abs/2305.18290
- Key contribution: DPO as simpler alternative to RLHF
- Used in: Stage 16

**"Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022)**
- Paper: https://arxiv.org/abs/2212.08073
- Key contribution: RLAIF for alignment
- Used in: Stage 16 (mention)

---

## 6. Inference Optimization

**"Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2022)**
- Paper: https://arxiv.org/abs/2211.17192
- Key contribution: Speculative decoding
- Used in: Stage 17

**"Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM, Kwon et al., 2023)**
- Paper: https://arxiv.org/abs/2309.06180
- Key contribution: Paged attention, continuous batching
- Used in: Stage 17

---

## 7. Quantization

**"LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (Dettmers et al., 2022)**
- Paper: https://arxiv.org/abs/2208.07339
- Key contribution: Mixed INT8 inference
- Used in: Stage 18

**"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (Frantar et al., 2022)**
- Paper: https://arxiv.org/abs/2210.17323
- Key contribution: Second-order quantization
- Used in: Stage 18

**"AWQ: Activation-aware Weight Quantization" (Lin et al., 2023)**
- Paper: https://arxiv.org/abs/2306.00978
- Key contribution: Activation-aware quantization
- Used in: Stage 18

---

## 8. Educational Resources

### 8.1 Books

**"Deep Learning" (Goodfellow, Bengio, Courville, 2016)**
- URL: https://www.deeplearningbook.org/
- Use: Background on deep learning fundamentals
- Covers: Math foundations, MLPs, CNNs, RNNs, optimization

**"Build a Large Language Model (From Scratch)" (Raschka, 2024)**
- URL: https://www.manning.com/books/build-a-large-language-model-from-scratch
- Use: Implementation reference
- Covers: GPT implementation, training, fine-tuning

**"Mathematics for Machine Learning" (Deisenroth, Faisal, Ong, 2020)**
- URL: https://mml-book.github.io/
- Use: Mathematical foundations
- Covers: Linear algebra, calculus, probability

**"How to Solve It" (Pólya, 1945)**
- Use: Pedagogical framework
- Covers: Problem-solving methodology

**"The Visual Display of Quantitative Information" (Tufte, 1983)**
- Use: Visualization principles
- Covers: Information design

### 8.2 Courses and Videos

**Andrej Karpathy's "Neural Networks: Zero to Hero"**
- URL: https://karpathy.ai/zero-to-hero.html
- Use: Structure inspiration, implementation reference
- Covers: Micrograd, Makemore, GPT

**Stanford CS224N: Natural Language Processing with Deep Learning**
- URL: https://web.stanford.edu/class/cs224n/
- Use: Academic treatment of NLP
- Covers: Word vectors, RNNs, Transformers, pretraining

**Dive into Deep Learning (d2l.ai)**
- URL: https://d2l.ai/
- Use: Interactive reference
- Covers: Deep learning with code

### 8.3 Blog Posts and Articles

**"The Illustrated Transformer" (Jay Alammar)**
- URL: https://jalammar.github.io/illustrated-transformer/
- Use: Visual explanation of Transformer
- Excellent visualizations

**"The Annotated Transformer" (Harvard NLP)**
- URL: https://nlp.seas.harvard.edu/2018/04/03/attention.html
- Use: Line-by-line implementation
- PyTorch implementation with annotations

**Sebastian Raschka's Blog**
- URL: https://sebastianraschka.com/blog/
- Use: Implementation details, BPE from scratch
- Excellent technical writing

**Lilian Weng's Blog**
- URL: https://lilianweng.github.io/
- Use: Comprehensive surveys
- Covers: Attention, Transformers, RLHF

---

## 9. Codebases

### 9.1 Reference Implementations

**nanoGPT (Karpathy)**
- URL: https://github.com/karpathy/nanoGPT
- Use: Clean GPT implementation
- ~300 lines of core code

**llm.c (Karpathy)**
- URL: https://github.com/karpathy/llm.c
- Use: C/CUDA implementation
- Performance-focused

**minGPT (Karpathy)**
- URL: https://github.com/karpathy/minGPT
- Use: Minimal GPT implementation
- Educational focus

**lit-gpt (Lightning AI)**
- URL: https://github.com/Lightning-AI/lit-gpt
- Use: Production-quality implementations
- Multiple model architectures

### 9.2 Production Systems

**vLLM**
- URL: https://github.com/vllm-project/vllm
- Use: Inference optimization reference
- PagedAttention, continuous batching

**TensorRT-LLM (NVIDIA)**
- URL: https://github.com/NVIDIA/TensorRT-LLM
- Use: Optimized inference
- Quantization, kernel fusion

**Megatron-LM (NVIDIA)**
- URL: https://github.com/NVIDIA/Megatron-LM
- Use: Distributed training reference
- Tensor/pipeline parallelism

---

## 10. Benchmarks and Datasets

### 10.1 Pretraining Data

**The Pile**
- URL: https://pile.eleuther.ai/
- Use: Diverse pretraining corpus
- 800GB of text

**RedPajama**
- URL: https://github.com/togethercomputer/RedPajama-Data
- Use: Open reproduction of LLaMA training data

**FineWeb**
- URL: https://huggingface.co/datasets/HuggingFaceFW/fineweb
- Use: High-quality web data
- 15T tokens

### 10.2 Evaluation

**MMLU**
- URL: https://github.com/hendrycks/test
- Use: Multi-task language understanding

**HellaSwag**
- Use: Commonsense reasoning

**HumanEval**
- URL: https://github.com/openai/human-eval
- Use: Code generation

---

## 11. Tools

### 11.1 Training

**PyTorch**
- URL: https://pytorch.org/
- Primary framework for all code

**DeepSpeed**
- URL: https://github.com/microsoft/DeepSpeed
- ZeRO, mixed precision

**FSDP (PyTorch)**
- URL: https://pytorch.org/docs/stable/fsdp.html
- Fully Sharded Data Parallel

### 11.2 Profiling

**PyTorch Profiler**
- Built-in profiling for GPU/CPU

**NVIDIA Nsight Systems**
- Low-level GPU profiling

**Weights & Biases**
- Experiment tracking

### 11.3 Visualization

**Matplotlib**
- Standard plotting

**Plotly**
- Interactive plots

**TensorBoard**
- Training visualization

---

## 12. Research Directions to Watch

### 12.1 Efficiency

- Linear attention variants
- State space models (Mamba)
- Mixture of Experts at scale
- Speculative decoding improvements

### 12.2 Architecture

- Hybrid architectures (attention + SSM)
- Deeper scaling laws understanding
- Alternative position encodings

### 12.3 Training

- Synthetic data generation
- Curriculum learning
- Improved RLHF alternatives

### 12.4 Alignment

- Constitutional AI improvements
- Process supervision
- Interpretability for alignment

---

## 13. Citation Notes

When citing in the book, use format:
> As shown by Vaswani et al. (2017), the Transformer architecture...

For papers with many authors:
> The Llama 2 paper (Touvron et al., 2023) demonstrates...

For informal reference:
> Flash Attention (Dao et al.) achieves...

Include full bibliography at end of book.

---

## 14. Update Log

| Date | Update |
|------|--------|
| 2026-01-03 | Initial compilation |

---

*This document will be continuously updated as new relevant papers and resources are identified.*

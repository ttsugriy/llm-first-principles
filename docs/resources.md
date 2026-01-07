# Further Reading & Resources

*Curated resources for going deeper*

This page collects the most valuable external resources for each topic covered in this book.

---

## Papers

### Foundational Papers

| Paper | Year | Key Contribution | Stage |
|-------|------|------------------|-------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | The transformer architecture | 5, 6 |
| [Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html) | 2003 | Word embeddings for language modeling | 3 |
| [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) | 2014 | The Adam optimizer | 4 |
| [Layer Normalization](https://arxiv.org/abs/1607.06450) | 2016 | LayerNorm for transformers | 6 |
| [Deep Residual Learning](https://arxiv.org/abs/1512.03385) | 2015 | Residual connections | 6 |

### Tokenization Papers

| Paper | Year | Key Contribution | Stage |
|-------|------|------------------|-------|
| [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) | 2016 | BPE for NLP | 7 |
| [Google's Neural Machine Translation System](https://arxiv.org/abs/1609.08144) | 2016 | WordPiece | 7 |
| [SentencePiece](https://arxiv.org/abs/1808.06226) | 2018 | Unigram tokenization | 7 |

### PEFT Papers

| Paper | Year | Key Contribution | Stage |
|-------|------|------------------|-------|
| [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | 2021 | Low-rank fine-tuning | 9 |
| [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) | 2019 | Adapter layers | 9 |
| [Prefix-Tuning](https://arxiv.org/abs/2101.00190) | 2021 | Soft prefixes | 9 |
| [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) | 2021 | Prompt tuning | 9 |
| [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | 2023 | 4-bit fine-tuning | 9 |

### Alignment Papers

| Paper | Year | Key Contribution | Stage |
|-------|------|------------------|-------|
| [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) | 2022 | InstructGPT, RLHF | 10 |
| [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | 2023 | DPO | 10 |
| [Constitutional AI](https://arxiv.org/abs/2212.08073) | 2022 | Self-critique | 10 |
| [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | 2017 | PPO | 10 |

### Scaling & Architecture Papers

| Paper | Year | Key Contribution | Stage |
|-------|------|------------------|-------|
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | 2020 | Chinchilla scaling | 6 |
| [LLaMA: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2302.13971) | 2023 | Modern architecture | 6 |
| [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 2019 | GPT-2 | 6 |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | 2021 | RoPE | 5 |

---

## Libraries & Tools

### Essential Libraries

| Library | Purpose | Relevant Stages |
|---------|---------|-----------------|
| [NumPy](https://numpy.org/) | Array operations (used throughout this book) | All |
| [PyTorch](https://pytorch.org/) | Production deep learning | All |
| [JAX](https://github.com/google/jax) | Autodiff and accelerators | 2 |
| [Hugging Face Transformers](https://huggingface.co/transformers/) | Pre-trained models | 6, 9 |
| [PEFT](https://github.com/huggingface/peft) | LoRA and adapters | 9 |
| [TRL](https://github.com/huggingface/trl) | RLHF and DPO | 10 |

### Tokenization Libraries

| Library | Purpose |
|---------|---------|
| [tiktoken](https://github.com/openai/tiktoken) | OpenAI's BPE tokenizer |
| [SentencePiece](https://github.com/google/sentencepiece) | Unigram and BPE |
| [tokenizers](https://github.com/huggingface/tokenizers) | Fast tokenization |

### Training Tools

| Library | Purpose |
|---------|---------|
| [Weights & Biases](https://wandb.ai/) | Experiment tracking |
| [TensorBoard](https://www.tensorflow.org/tensorboard) | Training visualization |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | Distributed training |
| [Accelerate](https://github.com/huggingface/accelerate) | Multi-GPU training |

---

## Books

### Machine Learning Foundations

| Book | Author(s) | Focus |
|------|-----------|-------|
| *Deep Learning* | Goodfellow, Bengio, Courville | Comprehensive ML theory |
| *Pattern Recognition and Machine Learning* | Bishop | Probabilistic ML |
| *The Elements of Statistical Learning* | Hastie, Tibshirani, Friedman | Statistical methods |

### NLP & Language Models

| Book | Author(s) | Focus |
|------|-----------|-------|
| *Speech and Language Processing* | Jurafsky & Martin | NLP foundations |
| *Natural Language Processing with Transformers* | Tunstall, von Werra, Wolf | Practical transformers |
| *Dive into Deep Learning* | Zhang et al. | Interactive ML book |

---

## Courses

| Course | Institution | Focus |
|--------|-------------|-------|
| [CS231n](http://cs231n.stanford.edu/) | Stanford | CNNs, backprop basics |
| [CS224n](http://web.stanford.edu/class/cs224n/) | Stanford | NLP with deep learning |
| [CS324](https://stanford-cs324.github.io/winter2022/) | Stanford | Large language models |
| [fast.ai](https://course.fast.ai/) | fast.ai | Practical deep learning |

---

## Blog Posts & Tutorials

### Understanding Transformers

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual walkthrough
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - Code walkthrough
- [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) - Memory and compute

### Understanding Training

- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) - Karpathy's practical guide
- [Why Momentum Really Works](https://distill.pub/2017/momentum/) - Visual explanation

### Understanding Alignment

- [RLHF: Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) - Hugging Face overview
- [Illustrating RLHF](https://huggingface.co/blog/rlhf) - Visual guide

---

## Codebases to Study

### Educational Implementations

| Repo | Author | What to Learn |
|------|--------|---------------|
| [nanoGPT](https://github.com/karpathy/nanoGPT) | Karpathy | Minimal GPT training |
| [minGPT](https://github.com/karpathy/minGPT) | Karpathy | Simple GPT implementation |
| [micrograd](https://github.com/karpathy/micrograd) | Karpathy | Tiny autograd engine |
| [llm.c](https://github.com/karpathy/llm.c) | Karpathy | GPT in C |

### Production Implementations

| Repo | What to Learn |
|------|---------------|
| [llama](https://github.com/facebookresearch/llama) | Production transformer |
| [transformers](https://github.com/huggingface/transformers) | Library architecture |
| [vLLM](https://github.com/vllm-project/vllm) | Inference optimization |

---

## Datasets

### Language Modeling

| Dataset | Size | Use Case |
|---------|------|----------|
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | Small | Learning, debugging |
| [OpenWebText](https://huggingface.co/datasets/openwebtext) | Medium | GPT-2 reproduction |
| [The Pile](https://pile.eleuther.ai/) | Large | Serious pre-training |
| [RedPajama](https://github.com/togethercomputer/RedPajama-Data) | Large | LLaMA reproduction |

### Alignment

| Dataset | Purpose |
|---------|---------|
| [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) | Preference data |
| [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1) | Conversation data |
| [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) | Instruction data |

---

## Communities

- [Hugging Face Forums](https://discuss.huggingface.co/) - Library questions
- [r/MachineLearning](https://reddit.com/r/MachineLearning) - Research discussion
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA) - Running LLMs locally
- [EleutherAI Discord](https://discord.gg/eleutherai) - Open-source LLMs

---

## Staying Current

### Research Feeds

- [Papers With Code - Language Models](https://paperswithcode.com/task/language-modelling)
- [arXiv cs.CL](https://arxiv.org/list/cs.CL/recent) - NLP papers
- [arXiv cs.LG](https://arxiv.org/list/cs.LG/recent) - ML papers

### Newsletters

- [The Batch](https://www.deeplearning.ai/the-batch/) - DeepLearning.AI
- [Import AI](https://jack-clark.net/) - Jack Clark


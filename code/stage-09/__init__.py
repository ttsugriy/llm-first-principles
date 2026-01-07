"""
Stage 9: Fine-tuning & Parameter-Efficient Methods

This module implements PEFT methods from scratch:
- LoRA (Low-Rank Adaptation)
- Adapters
- Prefix Tuning
- Prompt Tuning
"""

from .peft import (
    LoRALayer,
    LoRALinear,
    Adapter,
    PrefixTuning,
    PromptTuning,
    count_trainable_parameters,
    count_frozen_parameters,
    compute_parameter_efficiency,
    apply_lora_to_attention,
    compare_peft_methods,
)

__all__ = [
    'LoRALayer',
    'LoRALinear',
    'Adapter',
    'PrefixTuning',
    'PromptTuning',
    'count_trainable_parameters',
    'count_frozen_parameters',
    'compute_parameter_efficiency',
    'apply_lora_to_attention',
    'compare_peft_methods',
]

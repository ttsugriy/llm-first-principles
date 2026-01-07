"""
Stage 10: Alignment - RLHF and DPO

This module implements alignment techniques from scratch.
"""

from .alignment import (
    PreferencePair,
    create_preference_dataset,
    RewardModel,
    reward_model_loss,
    DPOTrainer,
    compute_log_probs,
    PPOBuffer,
    ppo_loss,
    value_loss,
    kl_divergence,
    kl_penalty_reward,
    compare_alignment_methods,
)

__all__ = [
    'PreferencePair',
    'create_preference_dataset',
    'RewardModel',
    'reward_model_loss',
    'DPOTrainer',
    'compute_log_probs',
    'PPOBuffer',
    'ppo_loss',
    'value_loss',
    'kl_divergence',
    'kl_penalty_reward',
    'compare_alignment_methods',
]

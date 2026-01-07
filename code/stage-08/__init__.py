"""
Stage 8: Training Dynamics & Debugging

This module provides tools for understanding and debugging neural network training.
"""

from .diagnostics import (
    TrainingHistory,
    GradientStats,
    LearningRateFinder,
    ActivationStats,
    ActivationMonitor,
    TrainingDebugger,
    compute_activation_stats,
    compute_layer_gradient_stats,
    clip_gradients,
    detect_dead_neurons,
    check_initialization,
)

__all__ = [
    'TrainingHistory',
    'GradientStats',
    'LearningRateFinder',
    'ActivationStats',
    'ActivationMonitor',
    'TrainingDebugger',
    'compute_activation_stats',
    'compute_layer_gradient_stats',
    'clip_gradients',
    'detect_dead_neurons',
    'check_initialization',
]

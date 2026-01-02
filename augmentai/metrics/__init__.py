"""
Augmentation-aware robustness metrics module.

Provides tools for evaluating model sensitivity and invariance
to specific augmentation transforms.
"""

from augmentai.metrics.robustness import (
    RobustnessEvaluator,
    RobustnessScore,
    RobustnessReport,
)

__all__ = [
    "RobustnessEvaluator",
    "RobustnessScore",
    "RobustnessReport",
]

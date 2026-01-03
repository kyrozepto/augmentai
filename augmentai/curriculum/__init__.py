"""
Curriculum-Aware Dataset Preparation module.

Provides tools for curriculum learning:
- Score samples by difficulty
- Order data from easy → hard
- Adaptive augmentation strength over training epochs
"""

from augmentai.curriculum.difficulty_scorer import DifficultyScore, DifficultyScorer
from augmentai.curriculum.curriculum_scheduler import CurriculumSchedule, CurriculumScheduler
from augmentai.curriculum.adaptive_augmentation import AdaptiveAugmentation

__all__ = [
    "DifficultyScore",
    "DifficultyScorer",
    "CurriculumSchedule",
    "CurriculumScheduler",
    "AdaptiveAugmentation",
]

"""
AugmentAI - LLM-Powered Data Augmentation Policy Designer

Design domain-safe, task-aware augmentation policies through natural language.
"""

__version__ = "0.1.0"
__author__ = "AugmentAI Contributors"

from augmentai.core.policy import Policy, Transform
from augmentai.core.config import AugmentAIConfig

__all__ = ["Policy", "Transform", "AugmentAIConfig", "__version__"]

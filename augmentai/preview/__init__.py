"""
Augmentation preview module.

Provides visual preview and comparison of augmentation effects:
- Before/after image generation
- Per-transform diff visualization
- HTML/JSON dry-run reports
"""

from augmentai.preview.preview import (
    AugmentationPreview,
    PreviewResult,
    PreviewConfig,
)

__all__ = [
    "AugmentationPreview",
    "PreviewResult",
    "PreviewConfig",
]

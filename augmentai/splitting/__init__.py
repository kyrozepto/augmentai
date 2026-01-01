"""
Dataset splitting module.

Provides strategies for partitioning datasets into train/val/test splits.
"""

from augmentai.splitting.strategies import (
    SplitStrategy,
    SplitResult,
    DatasetSplitter,
)

__all__ = ["SplitStrategy", "SplitResult", "DatasetSplitter"]

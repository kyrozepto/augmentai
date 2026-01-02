"""
Dataset linting module.

Provides pre-prepare data quality checks including:
- Duplicate image detection
- Corrupt image detection  
- Mask-image mismatch detection
- Class imbalance warnings
- Label leakage detection
"""

from augmentai.linting.linter import (
    DatasetLinter,
    LintReport,
    LintIssue,
    LintSeverity,
    LintCategory,
)

__all__ = [
    "DatasetLinter",
    "LintReport", 
    "LintIssue",
    "LintSeverity",
    "LintCategory",
]

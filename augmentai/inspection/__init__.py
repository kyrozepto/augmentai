"""
Dataset inspection module.

Auto-detects dataset format, analyzes structure, and reports issues.
"""

from augmentai.inspection.detector import DatasetDetector
from augmentai.inspection.analyzer import DatasetAnalyzer, DatasetReport

__all__ = ["DatasetDetector", "DatasetAnalyzer", "DatasetReport"]

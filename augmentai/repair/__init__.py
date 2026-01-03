"""
Model-Guided Data Repair module.

Uses model uncertainty and feedback to suggest data repair actions:
- Relabel: Samples likely mislabeled
- Reweight: Samples that should have different training importance
- Remove: Corrupt or extremely noisy samples
"""

from augmentai.repair.sample_analysis import SampleAnalysis, SampleAnalyzer
from augmentai.repair.repair_suggestions import RepairSuggestion, DataRepair
from augmentai.repair.repair_report import RepairReport, RepairReportGenerator

__all__ = [
    "SampleAnalysis",
    "SampleAnalyzer",
    "RepairSuggestion",
    "DataRepair",
    "RepairReport",
    "RepairReportGenerator",
]

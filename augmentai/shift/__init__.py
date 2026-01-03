"""
Domain Shift Simulation module.

Generates controlled distribution shifts to stress-test model generalization:
- Covariate shift (brightness, contrast, noise, blur)
- Synthetic domain shifts (weather, compression artifacts)
"""

from augmentai.shift.shift_generator import ShiftConfig, ShiftGenerator
from augmentai.shift.shift_evaluator import ShiftResult, ShiftReport, ShiftEvaluator

__all__ = [
    "ShiftConfig",
    "ShiftGenerator",
    "ShiftResult",
    "ShiftReport",
    "ShiftEvaluator",
]

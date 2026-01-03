"""
AutoSearch module for automated augmentation policy optimization.

Provides:
- PolicySampler: LLM-based policy candidate generation
- PolicyEvaluator: Fast validation scoring  
- PolicyOptimizer: Evolutionary search algorithm
- SearchResult: Container for search results
"""

from augmentai.search.sampler import PolicySampler
from augmentai.search.evaluator import PolicyEvaluator
from augmentai.search.optimizer import PolicyOptimizer
from augmentai.search.result import SearchResult

__all__ = [
    "PolicySampler",
    "PolicyEvaluator", 
    "PolicyOptimizer",
    "SearchResult",
]

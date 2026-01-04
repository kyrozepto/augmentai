"""
AutoSearch module for automated augmentation policy optimization.

Provides:
- PolicySampler: LLM-based policy candidate generation
- PolicyEvaluator: Fast validation scoring  
- PolicyOptimizer: Evolutionary search algorithm
- OptimizerConfig: Configuration for the optimizer
- SearchResult: Container for search results
- quick_search: Convenience function for simple searches
"""

from augmentai.search.sampler import PolicySampler
from augmentai.search.evaluator import PolicyEvaluator
from augmentai.search.optimizer import PolicyOptimizer, OptimizerConfig, quick_search
from augmentai.search.result import SearchResult

__all__ = [
    "PolicySampler",
    "PolicyEvaluator", 
    "PolicyOptimizer",
    "OptimizerConfig",
    "SearchResult",
    "quick_search",
]


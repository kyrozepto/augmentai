"""
Search result container for AutoSearch.

Stores the best policy, search history, and all evaluated candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import json

from augmentai.core.policy import Policy


@dataclass
class SearchResult:
    """
    Result of an AutoSearch optimization run.
    
    Contains the best policy found, its score, and full search history
    for analysis and reproducibility.
    """
    
    # Best policy found during search
    best_policy: Policy
    
    # Score of the best policy (higher is better)
    best_score: float
    
    # Domain used for search
    domain: str
    
    # Total evaluation budget used
    budget_used: int
    
    # Time taken for search in seconds
    search_time: float
    
    # Generation-by-generation statistics
    history: list[dict[str, Any]] = field(default_factory=list)
    
    # All evaluated policies with their scores (sorted by score desc)
    all_candidates: list[tuple[Policy, float]] = field(default_factory=list)
    
    # Random seed used
    seed: int = 42
    
    # Timestamp of search completion
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def summary(self) -> str:
        """Get one-line summary of search results."""
        return (
            f"Best score: {self.best_score:.4f} | "
            f"Evaluations: {self.budget_used} | "
            f"Time: {self.search_time:.1f}s | "
            f"Transforms: {len(self.best_policy.transforms)}"
        )
    
    def top_policies(self, n: int = 5) -> list[tuple[Policy, float]]:
        """Get top N policies by score."""
        sorted_candidates = sorted(
            self.all_candidates, 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_candidates[:n]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_policy": self.best_policy.to_dict(),
            "best_score": self.best_score,
            "domain": self.domain,
            "budget_used": self.budget_used,
            "search_time": self.search_time,
            "history": self.history,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "n_candidates": len(self.all_candidates),
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, output_dir: Path) -> Path:
        """
        Save search results to directory.
        
        Creates:
        - search_result.json: Full search metadata
        - best_policy.yaml: The best policy
        
        Returns:
            Path to the result JSON file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save result metadata
        result_path = output_dir / "search_result.json"
        result_path.write_text(self.to_json())
        
        # Save best policy
        policy_path = output_dir / "best_policy.yaml"
        policy_path.write_text(self.best_policy.to_yaml())
        
        return result_path


@dataclass  
class GenerationStats:
    """Statistics for a single generation in evolutionary search."""
    
    generation: int
    best_score: float
    avg_score: float
    worst_score: float
    population_size: int
    mutations: int = 0
    crossovers: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "best_score": self.best_score,
            "avg_score": self.avg_score,
            "worst_score": self.worst_score,
            "population_size": self.population_size,
            "mutations": self.mutations,
            "crossovers": self.crossovers,
        }

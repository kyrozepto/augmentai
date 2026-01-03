"""
Difficulty scoring for curriculum learning.

Scores samples by difficulty using model feedback (loss, margin, etc.)
to enable easy→hard curriculum ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class DifficultyScore:
    """Difficulty score for a single sample.
    
    Attributes:
        sample_id: Unique identifier for the sample
        file_path: Path to the sample file
        score: Normalized difficulty score (0=easy, 1=hard)
        components: Breakdown of score components
        raw_loss: Raw loss value before normalization
        label: Ground truth label
    """
    sample_id: str
    file_path: Path
    score: float  # 0=easy, 1=hard
    components: dict[str, float] = field(default_factory=dict)
    raw_loss: float = 0.0
    label: str = ""
    
    def __post_init__(self) -> None:
        """Validate and clamp score."""
        self.score = max(0.0, min(1.0, self.score))
    
    @property
    def difficulty_level(self) -> str:
        """Get human-readable difficulty level."""
        if self.score < 0.25:
            return "easy"
        elif self.score < 0.5:
            return "medium"
        elif self.score < 0.75:
            return "hard"
        else:
            return "very hard"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "sample_id": self.sample_id,
            "file_path": str(self.file_path),
            "score": self.score,
            "difficulty_level": self.difficulty_level,
            "components": self.components,
            "raw_loss": self.raw_loss,
            "label": self.label,
        }


class DifficultyScorer:
    """Score samples by difficulty using model feedback.
    
    Difficulty is computed based on:
    - Loss: Higher loss = harder sample
    - Margin: Lower margin (confidence gap) = harder sample
    - Consistency: Samples misclassified often across augmentations = harder
    
    Example:
        def loss_fn(path: Path, label: str) -> float:
            return compute_loss(model, load_image(path), label)
        
        def margin_fn(path: Path) -> float:
            logits = model(load_image(path))
            top2 = sorted(logits, reverse=True)[:2]
            return top2[0] - top2[1]  # Margin between top-2
        
        scorer = DifficultyScorer(loss_fn, margin_fn)
        scores = scorer.score_dataset(samples)
        ranked = scorer.rank_by_difficulty(scores)  # Easy first
    """
    
    def __init__(
        self,
        loss_fn: Callable[[Path, str], float],
        margin_fn: Callable[[Path], float] | None = None,
        loss_weight: float = 0.7,
        margin_weight: float = 0.3,
    ):
        """Initialize difficulty scorer.
        
        Args:
            loss_fn: Function that computes loss for a sample and label
            margin_fn: Optional function that computes prediction margin
            loss_weight: Weight for loss component in final score
            margin_weight: Weight for margin component in final score
        """
        self.loss_fn = loss_fn
        self.margin_fn = margin_fn
        self.loss_weight = loss_weight
        self.margin_weight = margin_weight if margin_fn else 0.0
        
        # Normalize weights
        total_weight = self.loss_weight + self.margin_weight
        if total_weight > 0:
            self.loss_weight /= total_weight
            self.margin_weight /= total_weight
    
    def score_sample(
        self,
        sample_path: Path,
        label: str,
        sample_id: str | None = None,
    ) -> DifficultyScore:
        """Score a single sample's difficulty.
        
        Args:
            sample_path: Path to the sample file
            label: Ground truth label
            sample_id: Optional ID (defaults to filename)
            
        Returns:
            DifficultyScore (raw, not yet normalized to 0-1)
        """
        if sample_id is None:
            sample_id = sample_path.stem
        
        # Compute loss
        loss = self.loss_fn(sample_path, label)
        
        # Compute margin if available
        margin = None
        if self.margin_fn is not None:
            margin = self.margin_fn(sample_path)
        
        components = {"loss": loss}
        if margin is not None:
            components["margin"] = margin
        
        # Raw score (will be normalized later across dataset)
        raw_score = loss  # Use loss as base score
        
        return DifficultyScore(
            sample_id=sample_id,
            file_path=sample_path,
            score=0.0,  # Will be normalized later
            components=components,
            raw_loss=loss,
            label=label,
        )
    
    def score_dataset(
        self,
        samples: list[tuple[Path, str]],
    ) -> list[DifficultyScore]:
        """Score all samples in a dataset.
        
        Args:
            samples: List of (path, label) tuples
            
        Returns:
            List of DifficultyScore objects with normalized scores
        """
        # First pass: compute raw scores
        scores = []
        for i, (path, label) in enumerate(samples):
            sample_id = f"{i:05d}_{path.stem}"
            score = self.score_sample(path, label, sample_id)
            scores.append(score)
        
        # Second pass: normalize scores to 0-1
        self._normalize_scores(scores)
        
        return scores
    
    def _normalize_scores(self, scores: list[DifficultyScore]) -> None:
        """Normalize scores to [0, 1] range using min-max normalization."""
        if not scores:
            return
        
        # Get loss values
        losses = [s.raw_loss for s in scores]
        min_loss = min(losses)
        max_loss = max(losses)
        loss_range = max_loss - min_loss if max_loss > min_loss else 1.0
        
        # Get margin values if available
        margins = [s.components.get("margin", 0) for s in scores]
        min_margin = min(margins) if any(m != 0 for m in margins) else 0
        max_margin = max(margins) if any(m != 0 for m in margins) else 1
        margin_range = max_margin - min_margin if max_margin > min_margin else 1.0
        
        for score in scores:
            # Normalize loss (higher loss = higher difficulty)
            norm_loss = (score.raw_loss - min_loss) / loss_range
            
            # Normalize margin (lower margin = higher difficulty, so invert)
            if "margin" in score.components and margin_range > 0:
                norm_margin = 1 - (score.components["margin"] - min_margin) / margin_range
            else:
                norm_margin = 0
            
            # Weighted combination
            score.score = (
                self.loss_weight * norm_loss +
                self.margin_weight * norm_margin
            )
            score.score = max(0.0, min(1.0, score.score))
    
    def rank_by_difficulty(
        self,
        scores: list[DifficultyScore],
        ascending: bool = True,
    ) -> list[str]:
        """Rank samples by difficulty.
        
        Args:
            scores: List of difficulty scores
            ascending: If True, easy samples first (curriculum learning)
            
        Returns:
            Ordered list of sample IDs
        """
        sorted_scores = sorted(scores, key=lambda s: s.score, reverse=not ascending)
        return [s.sample_id for s in sorted_scores]
    
    def get_difficulty_distribution(
        self,
        scores: list[DifficultyScore],
        n_bins: int = 4,
    ) -> dict[str, list[str]]:
        """Get distribution of samples by difficulty level.
        
        Args:
            scores: List of difficulty scores
            n_bins: Number of difficulty bins
            
        Returns:
            Dictionary mapping difficulty level to sample IDs
        """
        distribution: dict[str, list[str]] = {}
        
        for score in scores:
            level = score.difficulty_level
            if level not in distribution:
                distribution[level] = []
            distribution[level].append(score.sample_id)
        
        return distribution
    
    def summary(self, scores: list[DifficultyScore]) -> dict[str, Any]:
        """Get summary statistics for scored dataset."""
        if not scores:
            return {"n_samples": 0}
        
        score_values = [s.score for s in scores]
        distribution = self.get_difficulty_distribution(scores)
        
        return {
            "n_samples": len(scores),
            "mean_difficulty": float(np.mean(score_values)),
            "std_difficulty": float(np.std(score_values)),
            "min_difficulty": float(min(score_values)),
            "max_difficulty": float(max(score_values)),
            "distribution": {k: len(v) for k, v in distribution.items()},
        }

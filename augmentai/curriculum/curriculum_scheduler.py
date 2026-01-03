"""
Curriculum scheduler for easy→hard data ordering.

Provides tools to schedule curriculum learning by ordering samples
based on difficulty scores and controlling the pacing of difficulty.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from augmentai.curriculum.difficulty_scorer import DifficultyScore


@dataclass
class CurriculumSchedule:
    """Curriculum schedule for training.
    
    Attributes:
        n_epochs: Total number of training epochs
        epoch_samples: Mapping from epoch number to ordered sample IDs
        pacing_function: Type of pacing function used
        warmup_epochs: Number of warmup epochs with easy samples only
        created_at: Schedule creation timestamp
        metadata: Additional metadata
    """
    n_epochs: int
    epoch_samples: dict[int, list[str]] = field(default_factory=dict)
    pacing_function: str = "linear"
    warmup_epochs: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def get_samples_for_epoch(self, epoch: int) -> list[str]:
        """Get ordered sample IDs for a specific epoch.
        
        Args:
            epoch: Epoch number (0-indexed)
            
        Returns:
            List of sample IDs in curriculum order
        """
        if epoch in self.epoch_samples:
            return self.epoch_samples[epoch]
        # Fall back to last available epoch
        if self.epoch_samples:
            return self.epoch_samples[max(self.epoch_samples.keys())]
        return []
    
    def get_difficulty_cutoff(self, epoch: int) -> float:
        """Get the maximum difficulty included at this epoch.
        
        Args:
            epoch: Epoch number (0-indexed)
            
        Returns:
            Maximum difficulty score (0-1) included at this epoch
        """
        if epoch < self.warmup_epochs:
            # During warmup, start with easier samples
            return 0.3 + (epoch / self.warmup_epochs) * 0.2
        
        # After warmup, ramp up to full difficulty
        remaining_epochs = self.n_epochs - self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / max(1, remaining_epochs)
        
        if self.pacing_function == "linear":
            return 0.5 + progress * 0.5
        elif self.pacing_function == "quadratic":
            return 0.5 + (progress ** 2) * 0.5
        elif self.pacing_function == "exponential":
            return 0.5 + (1 - np.exp(-3 * progress)) * 0.5
        else:
            return 1.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "n_epochs": self.n_epochs,
            "pacing_function": self.pacing_function,
            "warmup_epochs": self.warmup_epochs,
            "n_epoch_samples": len(self.epoch_samples),
            "metadata": self.metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class CurriculumScheduler:
    """Schedule curriculum learning from easy to hard.
    
    Creates a training schedule that introduces samples in order of difficulty,
    with configurable pacing functions to control how quickly hard samples
    are introduced.
    
    Example:
        scheduler = CurriculumScheduler(pacing="quadratic", warmup_epochs=5)
        schedule = scheduler.create_schedule(difficulty_scores, n_epochs=100)
        
        for epoch in range(100):
            sample_order = schedule.get_samples_for_epoch(epoch)
            train_on_samples(sample_order)
    """
    
    PACING_FUNCTIONS = ["linear", "quadratic", "exponential", "step"]
    
    def __init__(
        self,
        pacing: Literal["linear", "quadratic", "exponential", "step"] = "linear",
        warmup_epochs: int = 1,
        min_samples_per_epoch: float = 0.3,  # Minimum fraction of samples per epoch
    ):
        """Initialize curriculum scheduler.
        
        Args:
            pacing: Pacing function for introducing hard samples
                - linear: Constant rate of difficulty increase
                - quadratic: Slow start, faster increase later
                - exponential: Fast start, saturating
                - step: Discrete difficulty steps
            warmup_epochs: Number of epochs with only easy samples
            min_samples_per_epoch: Minimum fraction of dataset per epoch
        """
        if pacing not in self.PACING_FUNCTIONS:
            raise ValueError(f"Unknown pacing function: {pacing}. Use one of {self.PACING_FUNCTIONS}")
        
        self.pacing = pacing
        self.warmup_epochs = warmup_epochs
        self.min_samples_per_epoch = min_samples_per_epoch
    
    def create_schedule(
        self,
        scores: list[DifficultyScore],
        n_epochs: int,
    ) -> CurriculumSchedule:
        """Create a curriculum schedule from difficulty scores.
        
        Args:
            scores: List of difficulty scores for all samples
            n_epochs: Total number of training epochs
            
        Returns:
            CurriculumSchedule with per-epoch sample ordering
        """
        if not scores:
            return CurriculumSchedule(n_epochs=n_epochs, pacing_function=self.pacing)
        
        # Sort samples by difficulty (easy first)
        sorted_scores = sorted(scores, key=lambda s: s.score)
        all_sample_ids = [s.sample_id for s in sorted_scores]
        n_samples = len(all_sample_ids)
        
        # Create schedule
        schedule = CurriculumSchedule(
            n_epochs=n_epochs,
            pacing_function=self.pacing,
            warmup_epochs=self.warmup_epochs,
        )
        
        for epoch in range(n_epochs):
            cutoff = self._compute_cutoff(epoch, n_epochs)
            n_include = max(
                int(self.min_samples_per_epoch * n_samples),
                int(cutoff * n_samples),
            )
            n_include = min(n_include, n_samples)
            
            schedule.epoch_samples[epoch] = all_sample_ids[:n_include]
        
        schedule.metadata = {
            "n_total_samples": n_samples,
            "difficulty_range": (sorted_scores[0].score, sorted_scores[-1].score),
        }
        
        return schedule
    
    def _compute_cutoff(self, epoch: int, n_epochs: int) -> float:
        """Compute the fraction of samples to include at this epoch."""
        if epoch < self.warmup_epochs:
            # Start with easier samples during warmup
            return self.min_samples_per_epoch + (epoch / max(1, self.warmup_epochs)) * 0.2
        
        # Progress after warmup
        remaining = n_epochs - self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / max(1, remaining)
        
        if self.pacing == "linear":
            return self.min_samples_per_epoch + progress * (1 - self.min_samples_per_epoch)
        elif self.pacing == "quadratic":
            return self.min_samples_per_epoch + (progress ** 2) * (1 - self.min_samples_per_epoch)
        elif self.pacing == "exponential":
            return self.min_samples_per_epoch + (1 - np.exp(-3 * progress)) * (1 - self.min_samples_per_epoch)
        elif self.pacing == "step":
            n_steps = 4
            step = int(progress * n_steps)
            return self.min_samples_per_epoch + (step / n_steps) * (1 - self.min_samples_per_epoch)
        
        return 1.0
    
    def create_batched_schedule(
        self,
        scores: list[DifficultyScore],
        n_epochs: int,
        batch_size: int,
    ) -> dict[int, list[list[str]]]:
        """Create schedule with batches per epoch.
        
        Args:
            scores: Difficulty scores
            n_epochs: Number of epochs
            batch_size: Samples per batch
            
        Returns:
            Dict mapping epoch to list of batches (each batch is list of sample IDs)
        """
        schedule = self.create_schedule(scores, n_epochs)
        batched = {}
        
        for epoch, sample_ids in schedule.epoch_samples.items():
            batches = []
            for i in range(0, len(sample_ids), batch_size):
                batches.append(sample_ids[i:i+batch_size])
            batched[epoch] = batches
        
        return batched
    
    def save_schedule(
        self,
        schedule: CurriculumSchedule,
        output_path: Path,
    ) -> Path:
        """Save schedule to JSON file.
        
        Args:
            schedule: The curriculum schedule
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "n_epochs": schedule.n_epochs,
            "pacing_function": schedule.pacing_function,
            "warmup_epochs": schedule.warmup_epochs,
            "epoch_samples": schedule.epoch_samples,
            "metadata": schedule.metadata,
        }
        
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return output_path

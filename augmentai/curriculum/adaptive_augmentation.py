"""
Adaptive augmentation for curriculum learning.

Adjusts augmentation strength based on training progress and sample difficulty.
Easy samples get stronger augmentation, hard samples get weaker augmentation
early in training to facilitate learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from augmentai.core.policy import Policy, Transform


class AdaptiveAugmentation:
    """Adjust augmentation strength based on training progress and sample difficulty.
    
    Two main adaptation modes:
    1. **Epoch-based**: Increase augmentation strength over training epochs
    2. **Sample-based**: Strong augmentation for easy samples, weak for hard samples
    
    Example:
        adapter = AdaptiveAugmentation(
            base_policy,
            min_strength=0.2,
            max_strength=1.0,
            schedule="cosine",
        )
        
        for epoch in range(100):
            policy = adapter.get_policy_for_epoch(epoch, 100)
            # Train with adjusted augmentation
            
        # Or per-sample:
        for sample, difficulty in zip(samples, difficulty_scores):
            policy = adapter.get_policy_for_sample(difficulty)
            augmented = apply_policy(policy, sample)
    """
    
    SCHEDULES = ["linear", "cosine", "warmup", "constant"]
    
    def __init__(
        self,
        base_policy: Policy,
        min_strength: float = 0.2,
        max_strength: float = 1.0,
        schedule: Literal["linear", "cosine", "warmup", "constant"] = "linear",
        warmup_epochs: int = 5,
    ):
        """Initialize adaptive augmentation.
        
        Args:
            base_policy: The base augmentation policy to adapt
            min_strength: Minimum strength multiplier (0-1)
            max_strength: Maximum strength multiplier (0-1)
            schedule: How to vary strength over epochs
                - linear: Linear ramp from min to max
                - cosine: Cosine annealing from min to max
                - warmup: Linear warmup then constant
                - constant: Always use max_strength
            warmup_epochs: Epochs for warmup schedule
        """
        if schedule not in self.SCHEDULES:
            raise ValueError(f"Unknown schedule: {schedule}. Use one of {self.SCHEDULES}")
        
        self.base_policy = base_policy
        self.min_strength = min_strength
        self.max_strength = max_strength
        self.schedule = schedule
        self.warmup_epochs = warmup_epochs
    
    def get_strength_for_epoch(
        self,
        epoch: int,
        total_epochs: int,
    ) -> float:
        """Compute augmentation strength for a given epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
            
        Returns:
            Strength multiplier between min_strength and max_strength
        """
        if total_epochs <= 0:
            return self.max_strength
        
        progress = epoch / max(1, total_epochs - 1)
        
        if self.schedule == "constant":
            return self.max_strength
        
        elif self.schedule == "linear":
            return self.min_strength + progress * (self.max_strength - self.min_strength)
        
        elif self.schedule == "cosine":
            # Cosine annealing from min to max
            cosine_progress = (1 - np.cos(np.pi * progress)) / 2
            return self.min_strength + cosine_progress * (self.max_strength - self.min_strength)
        
        elif self.schedule == "warmup":
            if epoch < self.warmup_epochs:
                warmup_progress = epoch / self.warmup_epochs
                return self.min_strength + warmup_progress * (self.max_strength - self.min_strength)
            return self.max_strength
        
        return self.max_strength
    
    def get_policy_for_epoch(
        self,
        epoch: int,
        total_epochs: int,
    ) -> Policy:
        """Get policy with adjusted probabilities for an epoch.
        
        Args:
            epoch: Current epoch
            total_epochs: Total epochs
            
        Returns:
            Policy with adjusted transform probabilities
        """
        strength = self.get_strength_for_epoch(epoch, total_epochs)
        return self._scale_policy(strength)
    
    def get_strength_for_sample(
        self,
        sample_difficulty: float,
        invert: bool = True,
    ) -> float:
        """Compute augmentation strength for a sample based on difficulty.
        
        Args:
            sample_difficulty: Sample difficulty score (0=easy, 1=hard)
            invert: If True, easy samples get strong augmentation
            
        Returns:
            Strength multiplier
        """
        sample_difficulty = max(0.0, min(1.0, sample_difficulty))
        
        if invert:
            # Easy samples (low difficulty) → strong augmentation
            # Hard samples (high difficulty) → weak augmentation
            return self.max_strength - sample_difficulty * (self.max_strength - self.min_strength)
        else:
            # Hard samples → strong augmentation
            return self.min_strength + sample_difficulty * (self.max_strength - self.min_strength)
    
    def get_policy_for_sample(
        self,
        sample_difficulty: float,
        invert: bool = True,
    ) -> Policy:
        """Get policy adjusted for sample difficulty.
        
        Args:
            sample_difficulty: Sample difficulty (0=easy, 1=hard)
            invert: If True, easy samples get strong augmentation
            
        Returns:
            Adjusted policy
        """
        strength = self.get_strength_for_sample(sample_difficulty, invert)
        return self._scale_policy(strength)
    
    def _scale_policy(self, strength: float) -> Policy:
        """Create a scaled version of the base policy.
        
        Args:
            strength: Strength multiplier (0-1)
            
        Returns:
            New Policy with scaled probabilities and magnitudes
        """
        scaled_transforms = []
        
        for transform in self.base_policy.transforms:
            # Scale probability
            new_prob = transform.probability * strength
            new_prob = max(0.0, min(1.0, new_prob))
            
            # Scale magnitude if present
            new_magnitude = transform.magnitude
            if new_magnitude is not None:
                new_magnitude = int(new_magnitude * strength)
                new_magnitude = max(0, min(10, new_magnitude))
            
            # Scale applicable parameters
            new_params = self._scale_parameters(transform.parameters, strength)
            
            scaled_transforms.append(Transform(
                name=transform.name,
                probability=new_prob,
                parameters=new_params,
                category=transform.category,
                magnitude=new_magnitude,
            ))
        
        return Policy(
            name=f"{self.base_policy.name}_strength_{strength:.2f}",
            domain=self.base_policy.domain,
            transforms=scaled_transforms,
            description=f"Adapted from {self.base_policy.name} with strength={strength:.2f}",
            metadata={
                "base_policy": self.base_policy.name,
                "strength": strength,
                "schedule": self.schedule,
            }
        )
    
    def _scale_parameters(
        self,
        params: dict[str, Any],
        strength: float,
    ) -> dict[str, Any]:
        """Scale numeric parameters by strength.
        
        Only scales parameters that represent ranges or magnitudes.
        """
        scaled = {}
        
        # Parameters that should be scaled
        scalable = {
            "limit", "scale", "rotate_limit", "shift_limit",
            "brightness_limit", "contrast_limit", "hue_shift_limit",
            "sat_shift_limit", "val_shift_limit", "var_limit",
            "blur_limit", "sigma", "alpha", "p",
        }
        
        for key, value in params.items():
            if key in scalable:
                if isinstance(value, (int, float)):
                    scaled[key] = value * strength
                elif isinstance(value, tuple) and len(value) == 2:
                    # Range tuple like (-30, 30) or (0.5, 1.5)
                    scaled[key] = (value[0] * strength, value[1] * strength)
                else:
                    scaled[key] = value
            else:
                scaled[key] = value
        
        return scaled
    
    def get_schedule_table(
        self,
        total_epochs: int,
        step: int = 10,
    ) -> list[dict[str, Any]]:
        """Get a table showing strength at different epochs.
        
        Args:
            total_epochs: Total epochs
            step: Step between displayed epochs
            
        Returns:
            List of dicts with epoch and strength info
        """
        table = []
        for epoch in range(0, total_epochs, step):
            strength = self.get_strength_for_epoch(epoch, total_epochs)
            table.append({
                "epoch": epoch,
                "strength": round(strength, 3),
                "progress": round(epoch / max(1, total_epochs - 1), 3),
            })
        return table

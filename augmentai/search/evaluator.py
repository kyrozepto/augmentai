"""
Policy evaluation for AutoSearch.

Provides proxy metrics to score augmentation policies quickly
without requiring full model training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from augmentai.core.policy import Policy, Transform


@dataclass
class EvaluationResult:
    """Result of evaluating a single policy."""
    
    policy_name: str
    score: float
    metrics: dict[str, float]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "score": self.score,
            "metrics": self.metrics,
        }


class PolicyEvaluator:
    """
    Evaluate augmentation policies using proxy metrics.
    
    Proxy metrics allow fast scoring without full training:
    - Diversity: How varied are the augmentations?
    - Coverage: How many transform categories are used?
    - Strength: Overall augmentation intensity
    - Balance: Are probabilities well-distributed?
    """
    
    # Weights for combining metrics into final score
    DEFAULT_WEIGHTS = {
        "diversity": 0.3,
        "coverage": 0.25,
        "strength": 0.20,
        "balance": 0.15,
        "domain_fit": 0.10,
    }
    
    # Transform categories for diversity scoring
    TRANSFORM_CATEGORIES = {
        "geometric": {
            "HorizontalFlip", "VerticalFlip", "Rotate", "ShiftScaleRotate",
            "Perspective", "Affine", "RandomCrop", "RandomScale",
        },
        "color": {
            "RandomBrightnessContrast", "ColorJitter", "HueSaturationValue",
            "RandomGamma", "RandomToneCurve", "CLAHE", "Equalize",
        },
        "blur": {
            "GaussianBlur", "MotionBlur", "MedianBlur", "Blur",
        },
        "noise": {
            "GaussNoise", "ISONoise", "MultiplicativeNoise",
        },
        "dropout": {
            "CoarseDropout", "Cutout", "GridDropout",
        },
        "distortion": {
            "ElasticTransform", "GridDistortion", "OpticalDistortion",
        },
        "sharpness": {
            "Sharpen", "Emboss", "UnsharpMask",
        },
    }
    
    def __init__(
        self,
        weights: dict[str, float] | None = None,
        domain: str | None = None,
        custom_eval_fn: Callable[[Policy], float] | None = None,
    ) -> None:
        """
        Initialize the evaluator.
        
        Args:
            weights: Custom weights for metric combination
            domain: Domain for domain-fit scoring
            custom_eval_fn: Optional custom evaluation function
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.domain = domain
        self.custom_eval_fn = custom_eval_fn
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
    
    def evaluate(self, policy: Policy) -> EvaluationResult:
        """
        Evaluate a single policy.
        
        Args:
            policy: Policy to evaluate
            
        Returns:
            EvaluationResult with score and detailed metrics
        """
        metrics = {}
        
        # Calculate individual metrics
        metrics["diversity"] = self._score_diversity(policy)
        metrics["coverage"] = self._score_coverage(policy)
        metrics["strength"] = self._score_strength(policy)
        metrics["balance"] = self._score_balance(policy)
        metrics["domain_fit"] = self._score_domain_fit(policy)
        
        # Combine into final score
        score = sum(
            self.weights.get(metric, 0) * value
            for metric, value in metrics.items()
        )
        
        # If custom eval function provided, blend it in
        if self.custom_eval_fn:
            custom_score = self.custom_eval_fn(policy)
            # Blend 50-50 with proxy score
            score = 0.5 * score + 0.5 * custom_score
            metrics["custom"] = custom_score
        
        return EvaluationResult(
            policy_name=policy.name,
            score=score,
            metrics=metrics,
        )
    
    def evaluate_batch(
        self, 
        policies: list[Policy],
    ) -> list[EvaluationResult]:
        """
        Evaluate multiple policies.
        
        Args:
            policies: List of policies to evaluate
            
        Returns:
            List of evaluation results
        """
        return [self.evaluate(p) for p in policies]
    
    def _score_diversity(self, policy: Policy) -> float:
        """
        Score transform diversity (0.0 to 1.0).
        
        Higher when using transforms from different categories.
        """
        if not policy.transforms:
            return 0.0
        
        # Count unique categories
        categories_used = set()
        for t in policy.transforms:
            for cat, transforms in self.TRANSFORM_CATEGORIES.items():
                if t.name in transforms:
                    categories_used.add(cat)
                    break
        
        # Score based on category coverage
        max_categories = len(self.TRANSFORM_CATEGORIES)
        return len(categories_used) / max_categories
    
    def _score_coverage(self, policy: Policy) -> float:
        """
        Score transform count (0.0 to 1.0).
        
        Optimal around 5-7 transforms.
        """
        n = len(policy.transforms)
        
        # Bell curve around optimal count
        optimal = 6
        sigma = 3
        score = math.exp(-((n - optimal) ** 2) / (2 * sigma ** 2))
        
        return score
    
    def _score_strength(self, policy: Policy) -> float:
        """
        Score overall augmentation strength (0.0 to 1.0).
        
        Based on average probability and parameter aggressiveness.
        """
        if not policy.transforms:
            return 0.0
        
        # Average probability
        avg_prob = sum(t.probability for t in policy.transforms) / len(policy.transforms)
        
        # Clamp to reasonable range (0.3 to 0.7 is ideal)
        if avg_prob < 0.2:
            return avg_prob / 0.2 * 0.5  # Understrength
        elif avg_prob > 0.8:
            return 1.0 - (avg_prob - 0.8) / 0.2 * 0.5  # Overstrength
        else:
            return 0.7 + 0.3 * (1 - abs(avg_prob - 0.5) / 0.3)  # Sweet spot
    
    def _score_balance(self, policy: Policy) -> float:
        """
        Score probability distribution balance (0.0 to 1.0).
        
        Higher when probabilities are well-distributed, not all same.
        """
        if len(policy.transforms) < 2:
            return 0.5
        
        probs = [t.probability for t in policy.transforms]
        
        # Standard deviation of probabilities
        mean = sum(probs) / len(probs)
        variance = sum((p - mean) ** 2 for p in probs) / len(probs)
        std = math.sqrt(variance)
        
        # Some variation is good (0.1-0.2 std is ideal)
        if std < 0.05:
            return 0.5 + std / 0.05 * 0.3  # Too uniform
        elif std > 0.3:
            return max(0.3, 1.0 - (std - 0.3) / 0.2)  # Too varied
        else:
            return 0.8 + 0.2 * (1 - abs(std - 0.15) / 0.15)  # Sweet spot
    
    def _score_domain_fit(self, policy: Policy) -> float:
        """
        Score how well policy fits domain constraints (0.0 to 1.0).
        
        Higher when using recommended transforms, lower when near forbidden.
        """
        if not self.domain or not policy.transforms:
            return 0.5  # Neutral if no domain
        
        try:
            from augmentai.domains import get_domain
            domain_obj = get_domain(self.domain)
        except ValueError:
            return 0.5
        
        # Count recommended vs risky transforms
        recommended_count = 0
        risky_count = 0
        
        for t in policy.transforms:
            if t.name in domain_obj.recommended_transforms:
                recommended_count += 1
            # Penalize transforms close to forbidden (similar names)
            if t.name in domain_obj.forbidden_transforms:
                risky_count += 1  # Should not happen after enforcement
        
        n = len(policy.transforms)
        recommended_ratio = recommended_count / n
        
        return 0.5 + 0.5 * recommended_ratio

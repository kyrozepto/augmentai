"""
Augmentation safety tester.

Tests augmentation pipelines for domain-specific safety:
- Mask integrity: Verifies augmentations preserve mask-image correspondence
- OCR legibility: Measures text readability impact
- Boundary preservation: Checks anatomical boundaries for medical imaging
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from augmentai.core.policy import Policy, Transform
from augmentai.domains.base import Domain


@dataclass
class SafetyTestResult:
    """Result of testing augmentation safety."""
    
    passed: bool = True
    mask_integrity_score: float = 1.0  # 0-1, how much mask was preserved
    legibility_score: float = 1.0      # 0-1, for OCR domains
    transform_name: str = ""
    issues: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    
    def add_issue(self, message: str) -> None:
        """Add an issue and mark as failed."""
        self.issues.append(message)
        self.passed = False
    
    def summary(self) -> str:
        """Get a one-line summary."""
        if self.passed:
            return f"✓ {self.transform_name}: Safe (mask={self.mask_integrity_score:.2f})"
        return f"✗ {self.transform_name}: {len(self.issues)} issues"


@dataclass
class PolicyTestResult:
    """Result of testing a complete policy."""
    
    passed: bool = True
    transform_results: list[SafetyTestResult] = field(default_factory=list)
    overall_mask_integrity: float = 1.0
    overall_legibility: float = 1.0
    flagged_transforms: list[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Get a summary of the policy test."""
        if self.passed:
            return f"✓ Policy safe ({len(self.transform_results)} transforms tested)"
        return f"✗ Policy has issues: {', '.join(self.flagged_transforms)}"


class AugmentationSafetyTester:
    """
    Test augmentation pipelines for domain-specific safety.
    
    This class applies augmentations to sample images and measures
    their impact on label integrity, mask correspondence, and
    domain-specific constraints (e.g., OCR legibility).
    """
    
    # Threshold values for safety scoring
    MASK_INTEGRITY_THRESHOLD = 0.85    # Minimum IoU for mask preservation
    LEGIBILITY_THRESHOLD = 0.7         # Minimum legibility score for OCR
    EDGE_PRESERVATION_THRESHOLD = 0.8  # Minimum edge structure preservation
    
    def __init__(
        self,
        domain: Domain,
        strict: bool = False,
    ) -> None:
        """
        Initialize the safety tester.
        
        Args:
            domain: The domain to test against
            strict: If True, use stricter thresholds
        """
        self.domain = domain
        self.strict = strict
        
        if strict:
            self.MASK_INTEGRITY_THRESHOLD = 0.95
            self.LEGIBILITY_THRESHOLD = 0.85
    
    def test_transform(
        self,
        transform: Transform,
        sample_image: np.ndarray,
        sample_mask: np.ndarray | None = None,
        apply_fn: Callable[[np.ndarray, np.ndarray | None], tuple[np.ndarray, np.ndarray | None]] | None = None,
    ) -> SafetyTestResult:
        """
        Test a single transform for safety.
        
        Args:
            transform: The transform to test
            sample_image: Sample image as numpy array (H, W, C)
            sample_mask: Optional sample mask as numpy array (H, W)
            apply_fn: Optional function to apply the transform
                      Should return (augmented_image, augmented_mask)
        
        Returns:
            SafetyTestResult with scores and issues
        """
        result = SafetyTestResult(transform_name=transform.name)
        
        # Check if transform is forbidden in domain
        if transform.name in self.domain.forbidden_transforms:
            result.add_issue(f"Transform '{transform.name}' is forbidden in {self.domain.name} domain")
            return result
        
        # If we have an apply function and mask, test mask integrity
        if apply_fn is not None and sample_mask is not None:
            try:
                _, augmented_mask = apply_fn(sample_image, sample_mask)
                if augmented_mask is not None:
                    result.mask_integrity_score = self._compute_mask_integrity(
                        sample_mask, augmented_mask
                    )
                    
                    if result.mask_integrity_score < self.MASK_INTEGRITY_THRESHOLD:
                        result.add_issue(
                            f"Mask integrity too low: {result.mask_integrity_score:.2f} "
                            f"(threshold: {self.MASK_INTEGRITY_THRESHOLD})"
                        )
            except Exception as e:
                result.add_issue(f"Failed to apply transform: {str(e)[:50]}")
        
        # Domain-specific checks
        if self.domain.name == "ocr":
            result.legibility_score = self._estimate_ocr_impact(transform)
            if result.legibility_score < self.LEGIBILITY_THRESHOLD:
                result.add_issue(
                    f"Transform may reduce OCR legibility: {result.legibility_score:.2f}"
                )
        
        if self.domain.name == "medical":
            # Check for transforms that might distort anatomy
            if self._is_distortion_transform(transform):
                result.add_issue(
                    f"Transform '{transform.name}' may distort anatomical structures"
                )
        
        return result
    
    def test_policy(
        self,
        policy: Policy,
        sample_images: list[np.ndarray],
        sample_masks: list[np.ndarray] | None = None,
        apply_fn: Callable | None = None,
    ) -> PolicyTestResult:
        """
        Test a complete policy for safety.
        
        Args:
            policy: The policy to test
            sample_images: List of sample images
            sample_masks: Optional list of corresponding masks
            apply_fn: Optional function to apply transforms
        
        Returns:
            PolicyTestResult with all transform results
        """
        result = PolicyTestResult()
        
        masks = sample_masks or [None] * len(sample_images)
        
        for transform in policy.transforms:
            # Test on first sample image
            sample_img = sample_images[0] if sample_images else np.zeros((100, 100, 3), dtype=np.uint8)
            sample_mask = masks[0] if masks else None
            
            transform_result = self.test_transform(
                transform,
                sample_img,
                sample_mask,
                apply_fn
            )
            
            result.transform_results.append(transform_result)
            
            if not transform_result.passed:
                result.passed = False
                result.flagged_transforms.append(transform.name)
        
        # Compute overall scores
        if result.transform_results:
            result.overall_mask_integrity = np.mean([
                r.mask_integrity_score for r in result.transform_results
            ])
            result.overall_legibility = np.mean([
                r.legibility_score for r in result.transform_results
            ])
        
        return result
    
    def _compute_mask_integrity(
        self,
        original_mask: np.ndarray,
        augmented_mask: np.ndarray,
    ) -> float:
        """
        Compute mask integrity score using IoU.
        
        Args:
            original_mask: Original mask
            augmented_mask: Augmented mask
            
        Returns:
            IoU score between 0 and 1
        """
        if original_mask.shape != augmented_mask.shape:
            return 0.0
        
        # Binarize masks if needed
        orig_binary = original_mask > 0
        aug_binary = augmented_mask > 0
        
        intersection = np.logical_and(orig_binary, aug_binary).sum()
        union = np.logical_or(orig_binary, aug_binary).sum()
        
        if union == 0:
            return 1.0  # Both empty, no mask to preserve
        
        return intersection / union
    
    def _estimate_ocr_impact(self, transform: Transform) -> float:
        """
        Estimate the impact of a transform on OCR legibility.
        
        Returns a score from 0 (destroys text) to 1 (preserves text).
        """
        # Transforms that strongly impact OCR
        high_impact = {
            "MotionBlur": 0.3,
            "GaussianBlur": 0.5,
            "ElasticTransform": 0.2,
            "GridDistortion": 0.3,
            "OpticalDistortion": 0.4,
            "Defocus": 0.3,
        }
        
        # Transforms with moderate impact
        medium_impact = {
            "GaussNoise": 0.7,
            "RandomBrightnessContrast": 0.85,
            "CLAHE": 0.9,
            "Sharpen": 0.95,
        }
        
        # Safe transforms
        safe_transforms = {
            "HorizontalFlip", "VerticalFlip", "Rotate", "ShiftScaleRotate",
            "Normalize", "ToGray",
        }
        
        if transform.name in high_impact:
            return high_impact[transform.name]
        elif transform.name in medium_impact:
            return medium_impact[transform.name]
        elif transform.name in safe_transforms:
            return 1.0
        
        # Default: assume moderate impact
        return 0.75
    
    def _is_distortion_transform(self, transform: Transform) -> bool:
        """Check if transform applies geometric distortion."""
        distortion_transforms = {
            "ElasticTransform",
            "GridDistortion",
            "OpticalDistortion",
            "Morphological",
        }
        return transform.name in distortion_transforms
    
    def get_safe_transforms(self, policy: Policy) -> list[Transform]:
        """
        Get only the safe transforms from a policy.
        
        Args:
            policy: The policy to filter
            
        Returns:
            List of transforms that passed safety testing
        """
        safe = []
        sample = np.zeros((100, 100, 3), dtype=np.uint8)  # Dummy sample
        
        for transform in policy.transforms:
            result = self.test_transform(transform, sample)
            if result.passed:
                safe.append(transform)
        
        return safe

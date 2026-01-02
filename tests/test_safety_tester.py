"""Tests for the augmentation safety tester module."""

import numpy as np
import pytest

from augmentai.core.policy import Policy, Transform
from augmentai.domains import get_domain
from augmentai.rules.safety_tester import (
    AugmentationSafetyTester,
    SafetyTestResult,
    PolicyTestResult,
)


class TestSafetyTestResult:
    """Test SafetyTestResult dataclass."""
    
    def test_default_passes(self):
        """Default result should pass."""
        result = SafetyTestResult()
        assert result.passed is True
        assert result.mask_integrity_score == 1.0
    
    def test_add_issue_fails(self):
        """Adding an issue should mark result as failed."""
        result = SafetyTestResult(transform_name="TestTransform")
        result.add_issue("Test issue")
        
        assert result.passed is False
        assert len(result.issues) == 1
    
    def test_summary_passed(self):
        """Summary shows passed status."""
        result = SafetyTestResult(transform_name="HorizontalFlip")
        summary = result.summary()
        
        assert "✓" in summary
        assert "HorizontalFlip" in summary
    
    def test_summary_failed(self):
        """Summary shows failed status."""
        result = SafetyTestResult(transform_name="BadTransform")
        result.add_issue("Problem 1")
        result.add_issue("Problem 2")
        summary = result.summary()
        
        assert "✗" in summary
        assert "2 issues" in summary


class TestAugmentationSafetyTester:
    """Test AugmentationSafetyTester class."""
    
    def test_forbidden_transform_fails(self):
        """Forbidden transforms should fail safety test."""
        domain = get_domain("medical")
        tester = AugmentationSafetyTester(domain)
        
        # ElasticTransform is forbidden in medical domain
        transform = Transform("ElasticTransform", 0.5)
        sample = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = tester.test_transform(transform, sample)
        
        assert result.passed is False
        assert "forbidden" in result.issues[0].lower()
    
    def test_safe_transform_passes(self):
        """Safe transforms should pass."""
        domain = get_domain("medical")
        tester = AugmentationSafetyTester(domain)
        
        # HorizontalFlip is safe for medical
        transform = Transform("HorizontalFlip", 0.5)
        sample = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = tester.test_transform(transform, sample)
        
        assert result.passed is True
    
    def test_ocr_legibility_scoring(self):
        """OCR domain should check legibility or mark as forbidden."""
        domain = get_domain("ocr")
        tester = AugmentationSafetyTester(domain)
        
        # MotionBlur is forbidden in OCR domain
        transform = Transform("MotionBlur", 0.5)
        sample = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = tester.test_transform(transform, sample)
        
        # Should fail because MotionBlur is forbidden in OCR
        assert result.passed is False
        assert len(result.issues) > 0
    
    def test_mask_integrity_computation(self):
        """Test mask integrity scoring."""
        domain = get_domain("natural")
        tester = AugmentationSafetyTester(domain)
        
        # Create test masks
        original = np.zeros((100, 100), dtype=np.uint8)
        original[25:75, 25:75] = 1  # Center square
        
        # Perfect match
        perfect_score = tester._compute_mask_integrity(original, original.copy())
        assert perfect_score == 1.0
        
        # Partial overlap
        shifted = np.zeros((100, 100), dtype=np.uint8)
        shifted[30:80, 30:80] = 1  # Shifted square
        partial_score = tester._compute_mask_integrity(original, shifted)
        assert 0 < partial_score < 1.0
        
        # No overlap
        no_overlap = np.zeros((100, 100), dtype=np.uint8)
        no_overlap[0:10, 0:10] = 1  # Corner square
        no_score = tester._compute_mask_integrity(original, no_overlap)
        assert no_score == 0.0
    
    def test_policy_testing(self):
        """Test complete policy testing."""
        domain = get_domain("natural")
        tester = AugmentationSafetyTester(domain)
        
        policy = Policy(
            name="test_policy",
            domain="natural",
            transforms=[
                Transform("HorizontalFlip", 0.5),
                Transform("Rotate", 0.5, parameters={"limit": 15}),
            ]
        )
        
        samples = [np.zeros((100, 100, 3), dtype=np.uint8)]
        result = tester.test_policy(policy, samples)
        
        assert result.passed is True
        assert len(result.transform_results) == 2
    
    def test_policy_with_forbidden_transform(self):
        """Policy with forbidden transform fails testing."""
        domain = get_domain("medical")
        tester = AugmentationSafetyTester(domain)
        
        policy = Policy(
            name="bad_policy",
            domain="medical",
            transforms=[
                Transform("HorizontalFlip", 0.5),
                Transform("ElasticTransform", 0.5),  # Forbidden!
            ]
        )
        
        samples = [np.zeros((100, 100, 3), dtype=np.uint8)]
        result = tester.test_policy(policy, samples)
        
        assert result.passed is False
        assert "ElasticTransform" in result.flagged_transforms
    
    def test_get_safe_transforms(self):
        """Get only safe transforms from policy."""
        domain = get_domain("medical")
        tester = AugmentationSafetyTester(domain)
        
        policy = Policy(
            name="mixed_policy",
            domain="medical",
            transforms=[
                Transform("HorizontalFlip", 0.5),
                Transform("ElasticTransform", 0.5),  # Will be filtered
                Transform("Rotate", 0.5),
            ]
        )
        
        safe = tester.get_safe_transforms(policy)
        
        assert len(safe) == 2
        assert all(t.name != "ElasticTransform" for t in safe)
    
    def test_strict_mode(self):
        """Strict mode uses tighter thresholds."""
        domain = get_domain("natural")
        
        normal_tester = AugmentationSafetyTester(domain, strict=False)
        strict_tester = AugmentationSafetyTester(domain, strict=True)
        
        assert strict_tester.MASK_INTEGRITY_THRESHOLD > normal_tester.MASK_INTEGRITY_THRESHOLD
        assert strict_tester.LEGIBILITY_THRESHOLD > normal_tester.LEGIBILITY_THRESHOLD

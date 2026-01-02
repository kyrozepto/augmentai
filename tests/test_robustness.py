"""Tests for the robustness metrics module."""

import numpy as np
import pytest

from augmentai.core.policy import Policy, Transform
from augmentai.metrics import RobustnessEvaluator, RobustnessScore, RobustnessReport


class TestRobustnessScore:
    """Test RobustnessScore dataclass."""
    
    def test_fragile_detection(self):
        """Detects fragile transforms."""
        fragile = RobustnessScore(
            transform_name="BadTransform",
            sensitivity=0.5,
            consistency=0.4,
            invariance_score=0.3,
        )
        
        robust = RobustnessScore(
            transform_name="GoodTransform",
            sensitivity=0.1,
            consistency=0.9,
            invariance_score=0.85,
        )
        
        assert fragile.is_fragile is True
        assert robust.is_fragile is False
    
    def test_robustness_labels(self):
        """Assigns correct robustness labels."""
        highly_robust = RobustnessScore("t1", 0.05, 0.95, 0.92)
        robust = RobustnessScore("t2", 0.15, 0.85, 0.75)
        moderate = RobustnessScore("t3", 0.30, 0.70, 0.55)
        fragile = RobustnessScore("t4", 0.50, 0.50, 0.35)
        very_fragile = RobustnessScore("t5", 0.80, 0.30, 0.20)
        
        assert highly_robust.robustness_label == "highly robust"
        assert robust.robustness_label == "robust"
        assert moderate.robustness_label == "moderately robust"
        assert fragile.robustness_label == "fragile"
        assert very_fragile.robustness_label == "very fragile"
    
    def test_to_dict(self):
        """Can convert to dictionary."""
        score = RobustnessScore(
            transform_name="Rotate",
            sensitivity=0.2,
            consistency=0.8,
            invariance_score=0.7,
            n_samples_tested=10,
        )
        
        d = score.to_dict()
        assert d["transform_name"] == "Rotate"
        assert d["sensitivity"] == 0.2
        assert "is_fragile" in d


class TestRobustnessReport:
    """Test RobustnessReport dataclass."""
    
    def test_computes_summary(self):
        """Computes summary statistics."""
        scores = [
            RobustnessScore("T1", 0.1, 0.9, 0.85),  # robust
            RobustnessScore("T2", 0.2, 0.8, 0.75),  # robust
            RobustnessScore("T3", 0.5, 0.5, 0.40),  # fragile
        ]
        
        report = RobustnessReport(
            policy_name="test",
            domain="natural",
            scores=scores,
        )
        
        assert len(report.robust_transforms) == 2
        assert len(report.fragile_transforms) == 1
        assert "T1" in report.robust_transforms
        assert "T3" in report.fragile_transforms
    
    def test_overall_robustness(self):
        """Computes overall robustness score."""
        scores = [
            RobustnessScore("T1", 0.1, 0.9, 0.80),
            RobustnessScore("T2", 0.2, 0.8, 0.60),
        ]
        
        report = RobustnessReport(
            policy_name="test",
            domain="natural",
            scores=scores,
        )
        
        assert report.overall_robustness == 0.70  # (0.80 + 0.60) / 2
    
    def test_to_json(self):
        """Can export to JSON."""
        report = RobustnessReport(
            policy_name="test_policy",
            domain="medical",
            scores=[
                RobustnessScore("Flip", 0.1, 0.9, 0.85),
            ],
        )
        
        json_str = report.to_json()
        assert "test_policy" in json_str
        assert "Flip" in json_str


class TestRobustnessEvaluator:
    """Test RobustnessEvaluator class."""
    
    def test_evaluate_robust_model(self):
        """Test evaluation with a robust model."""
        policy = Policy(
            name="test_policy",
            domain="natural",
            transforms=[
                Transform("HorizontalFlip", 0.5),
                Transform("Rotate", 0.5),
            ]
        )
        
        # Mock model that always returns same prediction
        def robust_model(image: np.ndarray) -> int:
            return 1
        
        evaluator = RobustnessEvaluator(model_fn=robust_model, n_variations=3)
        
        images = [np.zeros((100, 100, 3), dtype=np.uint8)]
        report = evaluator.evaluate(images, policy)
        
        # All transforms should show high robustness
        assert report.policy_name == "test_policy"
        assert len(report.scores) == 2
        for score in report.scores:
            assert score.consistency == 1.0
    
    def test_evaluate_fragile_model(self):
        """Test evaluation with a fragile model."""
        policy = Policy(
            name="test_policy",
            domain="natural",
            transforms=[
                Transform("HorizontalFlip", 0.5),
            ]
        )
        
        # Mock model that returns different predictions each time
        call_count = 0
        def fragile_model(image: np.ndarray) -> int:
            nonlocal call_count
            call_count += 1
            return call_count  # Different each time
        
        evaluator = RobustnessEvaluator(model_fn=fragile_model, n_variations=3)
        
        images = [np.zeros((100, 100, 3), dtype=np.uint8)]
        report = evaluator.evaluate(images, policy)
        
        # Should show low robustness
        assert report.scores[0].consistency < 1.0
    
    def test_custom_compare_function(self):
        """Test with custom comparison function."""
        policy = Policy(
            name="test",
            domain="natural",
            transforms=[Transform("T", 0.5)]
        )
        
        def model(image: np.ndarray) -> float:
            return np.mean(image)
        
        # Custom compare: close values are similar
        def custom_compare(a: float, b: float) -> float:
            return max(0, 1 - abs(a - b) / 100)
        
        evaluator = RobustnessEvaluator(
            model_fn=model,
            compare_fn=custom_compare,
            n_variations=2,
        )
        
        images = [np.ones((10, 10, 3), dtype=np.uint8) * 128]
        report = evaluator.evaluate(images, policy)
        
        assert len(report.scores) == 1
    
    def test_default_compare_classification(self):
        """Test default comparison for classification."""
        evaluator = RobustnessEvaluator(model_fn=lambda x: 0)
        
        # Same class
        assert evaluator._default_compare(1, 1) == 1.0
        # Different class
        assert evaluator._default_compare(1, 2) == 0.0
    
    def test_default_compare_arrays(self):
        """Test default comparison for array predictions."""
        evaluator = RobustnessEvaluator(model_fn=lambda x: x)
        
        a = np.array([1, 0, 0])
        b = np.array([1, 0, 0])
        c = np.array([0, 1, 0])
        
        assert evaluator._default_compare(a, b) == 1.0
        assert evaluator._default_compare(a, c) == 0.0
    
    def test_generate_html_report(self, tmp_path):
        """Can generate HTML report."""
        policy = Policy(
            name="test_policy",
            domain="natural",
            transforms=[Transform("HorizontalFlip", 0.5)]
        )
        
        def model(image: np.ndarray) -> int:
            return 1
        
        evaluator = RobustnessEvaluator(model_fn=model, n_variations=2)
        report = evaluator.evaluate([np.zeros((10, 10, 3), dtype=np.uint8)], policy)
        
        html_path = evaluator.generate_html_report(report, tmp_path)
        
        assert html_path.exists()
        content = html_path.read_text(encoding="utf-8")
        assert "test_policy" in content
        assert "Robustness" in content
        
        # JSON should also be created
        json_path = tmp_path / "robustness_report.json"
        assert json_path.exists()

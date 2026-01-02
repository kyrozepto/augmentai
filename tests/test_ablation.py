"""Tests for the augmentation ablation module."""

import pytest

from augmentai.ablation import AugmentationAblation, AblationResult, AblationReport
from augmentai.core.policy import Policy, Transform


class TestAblationResult:
    """Test AblationResult dataclass."""
    
    def test_contribution_calculated(self):
        """Contribution is calculated from baseline and ablated scores."""
        result = AblationResult(
            transform_name="HorizontalFlip",
            baseline_score=0.85,
            ablated_score=0.80,
        )
        
        assert result.contribution == pytest.approx(0.05)
        assert result.is_helpful is True
    
    def test_harmful_transform(self):
        """Detect harmful transforms (negative contribution)."""
        result = AblationResult(
            transform_name="BadTransform",
            baseline_score=0.70,
            ablated_score=0.80,  # Better without it
        )
        
        assert result.contribution == pytest.approx(-0.10)
        assert result.is_helpful is False
    
    def test_impact_labels(self):
        """Impact labels are assigned correctly."""
        very_helpful = AblationResult("t1", baseline_score=0.80, ablated_score=0.70)
        helpful = AblationResult("t2", baseline_score=0.80, ablated_score=0.77)
        neutral = AblationResult("t3", baseline_score=0.80, ablated_score=0.80)
        harmful = AblationResult("t4", baseline_score=0.70, ablated_score=0.80)
        
        assert very_helpful.impact_label == "very helpful"
        assert helpful.impact_label == "helpful"
        assert neutral.impact_label == "neutral"
        assert harmful.impact_label == "harmful"
    
    def test_to_dict(self):
        """Can convert to dictionary."""
        result = AblationResult(
            transform_name="Rotate",
            baseline_score=0.85,
            ablated_score=0.82,
        )
        
        d = result.to_dict()
        assert d["transform_name"] == "Rotate"
        assert d["contribution"] == pytest.approx(0.03)
        assert "is_helpful" in d


class TestAblationReport:
    """Test AblationReport dataclass."""
    
    def test_ranks_results(self):
        """Results are ranked by contribution."""
        results = [
            AblationResult("A", baseline_score=0.80, ablated_score=0.70),  # +0.10
            AblationResult("B", baseline_score=0.80, ablated_score=0.75),  # +0.05
            AblationResult("C", baseline_score=0.80, ablated_score=0.85),  # -0.05
        ]
        
        report = AblationReport(
            policy_name="test",
            domain="natural",
            baseline_score=0.80,
            results=results,
        )
        
        # Should be sorted by contribution (highest first)
        assert report.results[0].transform_name == "A"
        assert report.results[0].rank == 1
        assert report.results[1].transform_name == "B"
        assert report.results[2].transform_name == "C"
    
    def test_recommendations(self):
        """Computes recommendations correctly."""
        results = [
            AblationResult("Good1", baseline_score=0.80, ablated_score=0.70),
            AblationResult("Good2", baseline_score=0.80, ablated_score=0.75),
            AblationResult("Bad", baseline_score=0.70, ablated_score=0.80),
        ]
        
        report = AblationReport(
            policy_name="test",
            domain="natural",
            baseline_score=0.80,
            results=results,
        )
        
        assert "Good1" in report.recommended_keeps
        assert "Good2" in report.recommended_keeps
        assert "Bad" in report.recommended_removes
    
    def test_to_json(self):
        """Can export to JSON."""
        report = AblationReport(
            policy_name="test_policy",
            domain="medical",
            baseline_score=0.85,
            results=[
                AblationResult("Flip", baseline_score=0.85, ablated_score=0.80),
            ],
        )
        
        json_str = report.to_json()
        assert "test_policy" in json_str
        assert "Flip" in json_str


class TestAugmentationAblation:
    """Test AugmentationAblation class."""
    
    def test_ablate_policy(self):
        """Can ablate a policy."""
        policy = Policy(
            name="test_policy",
            domain="natural",
            transforms=[
                Transform("HorizontalFlip", 0.5),
                Transform("Rotate", 0.5, parameters={"limit": 15}),
                Transform("GaussNoise", 0.3),
            ]
        )
        
        # Mock eval function: score decreases when transforms are removed
        def mock_eval(p: Policy) -> float:
            return 0.5 + len(p.transforms) * 0.1
        
        ablation = AugmentationAblation(eval_fn=mock_eval)
        report = ablation.ablate(policy)
        
        assert report.policy_name == "test_policy"
        assert report.baseline_score == 0.8  # 0.5 + 3*0.1
        assert len(report.results) == 3
        
        # All transforms should be helpful (score drops when removed)
        for result in report.results:
            assert result.is_helpful is True
    
    def test_lower_is_better(self):
        """Can handle lower-is-better metrics (e.g., loss)."""
        policy = Policy(
            name="test",
            domain="natural",
            transforms=[
                Transform("HorizontalFlip", 0.5),
                Transform("BadTransform", 0.5),
            ]
        )
        
        # Mock: HorizontalFlip reduces loss, BadTransform increases it
        def mock_eval(p: Policy) -> float:
            loss = 1.0
            for t in p.transforms:
                if t.name == "HorizontalFlip":
                    loss -= 0.1
                elif t.name == "BadTransform":
                    loss += 0.2
            return loss
        
        ablation = AugmentationAblation(eval_fn=mock_eval, higher_is_better=False)
        report = ablation.ablate(policy)
        
        # In lower-is-better mode, transforms that increase loss when removed are helpful
        # HorizontalFlip baseline=1.1, ablated=1.2 (without it loss goes up)-> helpful
        flip_result = next(r for r in report.results if r.transform_name == "HorizontalFlip")
        # Contribution in lower-is-better: ablated - baseline (positive if ablated is worse)
        assert flip_result.contribution > 0  # Removing it increases loss
        
        # BadTransform: baseline=1.1, ablated=0.9 (without it loss goes down) -> harmful
        bad_result = next(r for r in report.results if r.transform_name == "BadTransform")
        assert bad_result.contribution < 0  # Removing it decreases loss
    
    def test_multiple_runs(self):
        """Can average over multiple evaluation runs."""
        policy = Policy(
            name="test",
            domain="natural",
            transforms=[Transform("Flip", 0.5)]
        )
        
        call_count = 0
        def counting_eval(p: Policy) -> float:
            nonlocal call_count
            call_count += 1
            return 0.8
        
        ablation = AugmentationAblation(eval_fn=counting_eval, n_runs=3)
        ablation.ablate(policy)
        
        # Should call eval 3 times for baseline + 3 times for ablated
        assert call_count == 6
    
    def test_generate_html_report(self, tmp_path):
        """Can generate HTML report."""
        policy = Policy(
            name="test_policy",
            domain="natural",
            transforms=[
                Transform("HorizontalFlip", 0.5),
                Transform("Rotate", 0.5),
            ]
        )
        
        def mock_eval(p: Policy) -> float:
            return 0.5 + len(p.transforms) * 0.1
        
        ablation = AugmentationAblation(eval_fn=mock_eval)
        report = ablation.ablate(policy)
        
        html_path = ablation.generate_html_report(report, tmp_path)
        
        assert html_path.exists()
        content = html_path.read_text(encoding="utf-8")
        assert "test_policy" in content
        assert "HorizontalFlip" in content
        
        # JSON should also be created
        json_path = tmp_path / "ablation_report.json"
        assert json_path.exists()

"""Tests for the model-guided data repair module."""

import pytest
from pathlib import Path

from augmentai.repair import (
    SampleAnalysis,
    SampleAnalyzer,
    RepairSuggestion,
    DataRepair,
    RepairReport,
    RepairReportGenerator,
)
from augmentai.repair.repair_suggestions import RepairAction


class TestSampleAnalysis:
    """Test SampleAnalysis dataclass."""
    
    def test_quality_score_high(self):
        """High quality sample gets high score."""
        analysis = SampleAnalysis(
            sample_id="test_001",
            file_path=Path("/test/image.jpg"),
            uncertainty=0.1,
            loss=0.5,
            confidence=0.95,
            predicted_label="cat",
            true_label="cat",
        )
        
        assert analysis.is_misclassified is False
        assert analysis.quality_score > 0.7
    
    def test_quality_score_low(self):
        """Low quality (misclassified, high uncertainty) gets low score."""
        analysis = SampleAnalysis(
            sample_id="test_002",
            file_path=Path("/test/image.jpg"),
            uncertainty=0.9,
            loss=3.0,
            confidence=0.8,
            predicted_label="dog",
            true_label="cat",
        )
        
        assert analysis.is_misclassified is True
        assert analysis.quality_score < 0.5
    
    def test_to_dict(self):
        """Can convert to dictionary."""
        analysis = SampleAnalysis(
            sample_id="test_003",
            file_path=Path("/test/image.jpg"),
            uncertainty=0.5,
            loss=1.0,
            confidence=0.7,
            predicted_label="cat",
            true_label="cat",
        )
        
        d = analysis.to_dict()
        assert d["sample_id"] == "test_003"
        assert "quality_score" in d
        assert "is_misclassified" in d


class TestSampleAnalyzer:
    """Test SampleAnalyzer class."""
    
    def test_analyze_sample(self):
        """Can analyze a single sample."""
        def mock_uncertainty(path):
            return 0.3
        
        def mock_loss(path, label):
            return 0.8
        
        def mock_predict(path):
            return ("cat", 0.9)
        
        analyzer = SampleAnalyzer(
            uncertainty_fn=mock_uncertainty,
            loss_fn=mock_loss,
            predict_fn=mock_predict,
        )
        
        analysis = analyzer.analyze_sample(
            Path("/test/image.jpg"),
            true_label="cat",
        )
        
        assert analysis.uncertainty == 0.3
        assert analysis.loss == 0.8
        assert analysis.predicted_label == "cat"
        assert analysis.is_misclassified is False
    
    def test_analyze_dataset(self):
        """Can analyze multiple samples."""
        def mock_uncertainty(path):
            return 0.2
        
        def mock_loss(path, label):
            return 0.5
        
        def mock_predict(path):
            return (path.parent.name, 0.9)
        
        analyzer = SampleAnalyzer(
            uncertainty_fn=mock_uncertainty,
            loss_fn=mock_loss,
            predict_fn=mock_predict,
        )
        
        samples = [
            (Path("/test/cat/img1.jpg"), "cat"),
            (Path("/test/dog/img2.jpg"), "dog"),
        ]
        
        analyses = analyzer.analyze_dataset(samples)
        
        assert len(analyses) == 2
        assert all(a.is_misclassified is False for a in analyses)


class TestRepairSuggestion:
    """Test RepairSuggestion dataclass."""
    
    def test_to_dict(self):
        """Can convert to dictionary."""
        suggestion = RepairSuggestion(
            sample_id="test_001",
            action=RepairAction.RELABEL,
            reason="High confidence wrong prediction",
            confidence=0.9,
            suggested_label="dog",
        )
        
        d = suggestion.to_dict()
        assert d["sample_id"] == "test_001"
        assert d["action"] == "relabel"
        assert d["suggested_label"] == "dog"


class TestDataRepair:
    """Test DataRepair class."""
    
    def test_suggest_removal(self):
        """Suggests removal for high uncertainty + high loss."""
        # Need multiple samples so percentile calculation works
        analyses = [
            SampleAnalysis(
                sample_id="bad_sample",
                file_path=Path("/test/bad.jpg"),
                uncertainty=0.95,  # Very high uncertainty
                loss=100.0,  # Extremely high loss
                confidence=0.3,
                predicted_label="cat",
                true_label="dog",
            ),
            # Add normal samples to establish a reasonable percentile baseline
            SampleAnalysis(
                sample_id="good_sample1",
                file_path=Path("/test/good1.jpg"),
                uncertainty=0.1,
                loss=0.5,
                confidence=0.95,
                predicted_label="cat",
                true_label="cat",
            ),
            SampleAnalysis(
                sample_id="good_sample2",
                file_path=Path("/test/good2.jpg"),
                uncertainty=0.15,
                loss=0.6,
                confidence=0.9,
                predicted_label="dog",
                true_label="dog",
            ),
            SampleAnalysis(
                sample_id="good_sample3",
                file_path=Path("/test/good3.jpg"),
                uncertainty=0.2,
                loss=0.7,
                confidence=0.88,
                predicted_label="bird",
                true_label="bird",
            ),
        ]
        
        repair = DataRepair()
        suggestions = repair.suggest_repairs(analyses)
        
        bad_suggestion = next(
            (s for s in suggestions if s.sample_id == "bad_sample"),
            None
        )
        assert bad_suggestion is not None
        assert bad_suggestion.action == RepairAction.REMOVE
    
    def test_suggest_relabel(self):
        """Suggests relabeling for high-confidence wrong predictions."""
        analyses = [
            SampleAnalysis(
                sample_id="mislabeled",
                file_path=Path("/test/mislabeled.jpg"),
                uncertainty=0.2,
                loss=1.0,
                confidence=0.95,  # High confidence in wrong prediction
                predicted_label="dog",
                true_label="cat",
            ),
        ]
        
        repair = DataRepair()
        suggestions = repair.suggest_repairs(analyses)
        
        assert len(suggestions) == 1
        assert suggestions[0].action == RepairAction.RELABEL
        assert suggestions[0].suggested_label == "dog"
    
    def test_apply_reweighting(self):
        """Can generate weight mapping."""
        suggestions = [
            RepairSuggestion(
                sample_id="s1",
                action=RepairAction.REWEIGHT,
                reason="moderate issues",
                confidence=0.7,
                suggested_weight=0.5,
            ),
            RepairSuggestion(
                sample_id="s2",
                action=RepairAction.REMOVE,
                reason="corrupt",
                confidence=0.9,
            ),
        ]
        
        repair = DataRepair()
        weights = repair.apply_reweighting(suggestions)
        
        assert weights["s1"] == 0.5
        assert weights["s2"] == 0.0  # Removed samples get 0 weight
    
    def test_get_relabel_mapping(self):
        """Can generate relabel mapping."""
        suggestions = [
            RepairSuggestion(
                sample_id="s1",
                action=RepairAction.RELABEL,
                reason="mislabeled",
                confidence=0.9,
                suggested_label="dog",
            ),
            RepairSuggestion(
                sample_id="s2",
                action=RepairAction.KEEP,
                reason="ok",
                confidence=0.9,
            ),
        ]
        
        repair = DataRepair()
        mapping = repair.get_relabel_mapping(suggestions)
        
        assert mapping == {"s1": "dog"}


class TestRepairReport:
    """Test RepairReport dataclass."""
    
    def test_computes_counts(self):
        """Computes action counts from suggestions."""
        suggestions = [
            RepairSuggestion("s1", RepairAction.REMOVE, "bad", 0.9),
            RepairSuggestion("s2", RepairAction.RELABEL, "mislabeled", 0.9, suggested_label="x"),
            RepairSuggestion("s3", RepairAction.REWEIGHT, "noisy", 0.7, suggested_weight=0.5),
        ]
        
        report = RepairReport(n_samples=10, suggestions=suggestions)
        
        assert report.n_samples == 10
        assert report.n_remove == 1
        assert report.n_relabel == 1
        assert report.n_reweight == 1
        assert report.n_keep == 7  # 10 - 3 suggestions
    
    def test_repair_rate(self):
        """Computes repair rate correctly."""
        suggestions = [
            RepairSuggestion("s1", RepairAction.REMOVE, "bad", 0.9),
        ]
        
        report = RepairReport(n_samples=10, suggestions=suggestions)
        
        assert report.repair_rate == 0.1
    
    def test_to_json(self):
        """Can export to JSON."""
        report = RepairReport(
            n_samples=5,
            suggestions=[
                RepairSuggestion("s1", RepairAction.REMOVE, "corrupt", 0.9),
            ],
        )
        
        json_str = report.to_json()
        assert "s1" in json_str
        assert "remove" in json_str


class TestRepairReportGenerator:
    """Test RepairReportGenerator class."""
    
    def test_generate_report(self, tmp_path):
        """Can generate HTML and JSON reports."""
        report = RepairReport(
            n_samples=100,
            suggestions=[
                RepairSuggestion("s1", RepairAction.REMOVE, "corrupt data", 0.9),
                RepairSuggestion("s2", RepairAction.RELABEL, "mislabeled", 0.85, suggested_label="dog"),
            ],
        )
        
        generator = RepairReportGenerator()
        html_path = generator.generate(report, tmp_path)
        
        assert html_path.exists()
        assert (tmp_path / "repair_report.json").exists()
        
        # Check HTML content
        html_content = html_path.read_text()
        assert "Data Repair Report" in html_content
        assert "s1" in html_content
        assert "s2" in html_content

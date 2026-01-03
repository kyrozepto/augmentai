"""Tests for the domain shift simulation module."""

import pytest
from pathlib import Path

from augmentai.shift import (
    ShiftConfig,
    ShiftGenerator,
    ShiftResult,
    ShiftReport,
    ShiftEvaluator,
)
from augmentai.core.policy import Transform


class TestShiftConfig:
    """Test ShiftConfig dataclass."""
    
    def test_severity_clamping(self):
        """Severity is clamped to [0, 1]."""
        low = ShiftConfig("test", severity=-0.5)
        high = ShiftConfig("test", severity=1.5)
        
        assert low.severity == 0.0
        assert high.severity == 1.0
    
    def test_with_severity(self):
        """Can create copy with different severity."""
        config = ShiftConfig(
            name="brightness",
            severity=0.5,
            transforms=[Transform("RandomBrightnessContrast", 1.0, 
                                 parameters={"brightness_limit": 0.4})],
        )
        
        scaled = config.with_severity(0.8)
        
        assert scaled.severity == 0.8
        assert scaled.name == "brightness"
    
    def test_to_dict(self):
        """Can convert to dictionary."""
        config = ShiftConfig(
            name="blur",
            shift_type="covariate",
            severity=0.5,
        )
        
        d = config.to_dict()
        assert d["name"] == "blur"
        assert d["severity"] == 0.5


class TestShiftGenerator:
    """Test ShiftGenerator class."""
    
    def test_list_shifts(self):
        """Can list available shifts."""
        generator = ShiftGenerator()
        shifts = generator.list_shifts()
        
        assert "brightness" in shifts
        assert "blur" in shifts
        assert "noise" in shifts
        assert len(shifts) >= 5
    
    def test_get_shift(self):
        """Can get predefined shift."""
        generator = ShiftGenerator()
        
        brightness = generator.get_shift("brightness")
        assert brightness.name == "brightness"
        assert len(brightness.transforms) > 0
    
    def test_get_unknown_shift_raises(self):
        """Unknown shift raises error."""
        generator = ShiftGenerator()
        
        with pytest.raises(ValueError):
            generator.get_shift("nonexistent_shift")


class TestShiftResult:
    """Test ShiftResult dataclass."""
    
    def test_degradation_calculated(self):
        """Degradation is computed automatically."""
        result = ShiftResult(
            shift_name="blur",
            severity=0.5,
            original_accuracy=0.90,
            shifted_accuracy=0.75,
        )
        
        assert result.degradation == pytest.approx(0.15)
    
    def test_robustness_score(self):
        """Robustness score is computed correctly."""
        result = ShiftResult(
            shift_name="blur",
            severity=0.5,
            original_accuracy=0.90,
            shifted_accuracy=0.81,  # 90% of original
        )
        
        assert result.robustness_score == pytest.approx(0.9)
    
    def test_is_fragile(self):
        """Fragility detection works."""
        robust = ShiftResult("test", 0.5, 0.90, 0.85)
        fragile = ShiftResult("test", 0.5, 0.90, 0.50)
        
        assert robust.is_fragile is False
        assert fragile.is_fragile is True
    
    def test_severity_labels(self):
        """Severity labels are assigned correctly."""
        mild = ShiftResult("test", 0.2, 0.9, 0.8)
        moderate = ShiftResult("test", 0.5, 0.9, 0.8)
        severe = ShiftResult("test", 0.8, 0.9, 0.8)
        
        assert mild.severity_label == "mild"
        assert moderate.severity_label == "moderate"
        assert severe.severity_label == "severe"


class TestShiftReport:
    """Test ShiftReport dataclass."""
    
    def test_computes_summary(self):
        """Summary metrics are computed."""
        results = [
            ShiftResult("brightness", 0.5, 0.90, 0.85),
            ShiftResult("blur", 0.5, 0.90, 0.60),
            ShiftResult("noise", 0.5, 0.90, 0.75),
        ]
        
        report = ShiftReport(results=results)
        
        assert report.most_fragile_shift == "blur"
        assert report.overall_robustness > 0
    
    def test_get_fragile_shifts(self):
        """Can get list of fragile shifts."""
        results = [
            ShiftResult("robust", 0.5, 0.90, 0.88),  # Not fragile
            ShiftResult("fragile", 0.5, 0.90, 0.50),  # Fragile
        ]
        
        report = ShiftReport(results=results)
        fragile = report.get_fragile_shifts()
        
        assert len(fragile) == 1
        assert fragile[0].shift_name == "fragile"
    
    def test_to_json(self):
        """Can export to JSON."""
        report = ShiftReport(
            results=[
                ShiftResult("blur", 0.5, 0.90, 0.75),
            ],
        )
        
        json_str = report.to_json()
        assert "blur" in json_str
        assert "overall_robustness" in json_str


class TestShiftEvaluator:
    """Test ShiftEvaluator class."""
    
    def test_evaluate_shift(self, tmp_path):
        """Can evaluate a single shift."""
        # Create mock samples
        sample_dir = tmp_path / "samples"
        sample_dir.mkdir()
        (sample_dir / "img1.jpg").touch()
        (sample_dir / "img2.jpg").touch()
        
        shifted_dir = tmp_path / "shifted"
        shifted_dir.mkdir()
        (shifted_dir / "img1_shifted.jpg").touch()
        (shifted_dir / "img2_shifted.jpg").touch()
        
        # Mock predict function
        def mock_predict(path):
            name = path.stem.replace("_shifted", "")
            if "shifted" in str(path):
                return "wrong", 0.9  # Wrong on shifted
            return name, 0.9  # Correct on original
        
        labels = {"img1": "img1", "img2": "img2"}
        evaluator = ShiftEvaluator(predict_fn=mock_predict, true_labels=labels)
        
        shift = ShiftConfig("test", severity=0.5)
        result = evaluator.evaluate_shift(
            list(sample_dir.iterdir()),
            list(shifted_dir.iterdir()),
            shift,
        )
        
        assert result.original_accuracy == 1.0  # All correct on original
        assert result.shifted_accuracy == 0.0  # All wrong on shifted
        assert result.degradation == 1.0

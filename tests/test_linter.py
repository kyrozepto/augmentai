"""Tests for the dataset linting module."""

import tempfile
from pathlib import Path

import pytest

from augmentai.linting import DatasetLinter, LintReport, LintIssue, LintSeverity, LintCategory


class TestLintReport:
    """Test LintReport dataclass."""
    
    def test_empty_report_passes(self):
        """Empty report should pass."""
        report = LintReport()
        assert report.passed is True
        assert report.error_count == 0
        assert report.warning_count == 0
    
    def test_warning_still_passes(self):
        """Report with only warnings should still pass."""
        report = LintReport()
        report.add_issue(LintIssue(
            severity=LintSeverity.WARNING,
            category=LintCategory.IMBALANCE,
            message="Test warning"
        ))
        assert report.passed is True
        assert report.warning_count == 1
    
    def test_error_fails_report(self):
        """Report with errors should fail."""
        report = LintReport()
        report.add_issue(LintIssue(
            severity=LintSeverity.ERROR,
            category=LintCategory.CORRUPT,
            message="Test error"
        ))
        assert report.passed is False
        assert report.error_count == 1
    
    def test_summary_passed(self):
        """Summary shows passed status."""
        report = LintReport(files_checked=10)
        summary = report.summary()
        assert "✓" in summary
        assert "10 files" in summary
    
    def test_summary_failed(self):
        """Summary shows failed status."""
        report = LintReport()
        report.add_issue(LintIssue(
            severity=LintSeverity.ERROR,
            category=LintCategory.CORRUPT,
            message="Bad file"
        ))
        summary = report.summary()
        assert "✗" in summary
        assert "1 error" in summary


class TestDatasetLinter:
    """Test DatasetLinter class."""
    
    def test_lint_nonexistent_path_raises(self):
        """Linting non-existent path raises ValueError."""
        linter = DatasetLinter()
        with pytest.raises(ValueError, match="does not exist"):
            linter.lint(Path("/nonexistent/path"))
    
    def test_lint_file_raises(self, tmp_path):
        """Linting a file (not directory) raises ValueError."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test")
        
        linter = DatasetLinter()
        with pytest.raises(ValueError, match="not a directory"):
            linter.lint(file_path)
    
    def test_lint_empty_directory_warns(self, tmp_path):
        """Linting empty directory produces warning."""
        linter = DatasetLinter()
        report = linter.lint(tmp_path)
        
        assert report.files_checked == 0
        assert report.warning_count == 1
        assert any("No image files" in i.message for i in report.issues)
    
    def test_lint_valid_images_passes(self, tmp_path):
        """Linting directory with valid images passes."""
        # Create fake image files
        from PIL import Image
        
        img1 = Image.new("RGB", (100, 100), color="red")
        img2 = Image.new("RGB", (100, 100), color="blue")  # Different color = different hash
        img1.save(tmp_path / "img1.jpg")
        img2.save(tmp_path / "img2.png")
        
        linter = DatasetLinter()
        report = linter.lint(tmp_path)
        
        assert report.files_checked >= 2
        assert report.passed is True
    
    def test_detect_duplicate_images(self, tmp_path):
        """Linter detects duplicate images."""
        from PIL import Image
        
        # Create identical images
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(tmp_path / "img1.jpg")
        img.save(tmp_path / "img2.jpg")  # Duplicate
        
        linter = DatasetLinter()
        report = linter.lint(tmp_path)
        
        assert report.duplicates_found >= 1
        assert any(i.category == LintCategory.DUPLICATE for i in report.issues)
    
    def test_detect_corrupt_image(self, tmp_path):
        """Linter detects corrupt images."""
        # Create a fake corrupt image
        corrupt_file = tmp_path / "corrupt.jpg"
        corrupt_file.write_bytes(b"not a real image")
        
        linter = DatasetLinter()
        report = linter.lint(tmp_path)
        
        assert report.corrupt_found >= 1
        assert any(i.category == LintCategory.CORRUPT for i in report.issues)
        assert report.passed is False  # Corrupt images are errors
    
    def test_detect_class_imbalance(self, tmp_path):
        """Linter detects class imbalance."""
        from PIL import Image
        
        # Create imbalanced dataset
        (tmp_path / "class_a").mkdir()
        (tmp_path / "class_b").mkdir()
        
        img = Image.new("RGB", (10, 10), color="red")
        
        # Class A: 10 images
        for i in range(10):
            img.save(tmp_path / "class_a" / f"img_{i}.jpg")
        
        # Class B: 1 image (10:1 imbalance)
        img.save(tmp_path / "class_b" / "img_0.jpg")
        
        linter = DatasetLinter(imbalance_threshold=5.0)
        report = linter.lint(tmp_path)
        
        assert any(i.category == LintCategory.IMBALANCE for i in report.issues)
    
    def test_detect_label_leakage(self, tmp_path):
        """Linter detects potential label leakage in filenames."""
        from PIL import Image
        
        img = Image.new("RGB", (10, 10))
        img.save(tmp_path / "sample_positive_001.jpg")
        img.save(tmp_path / "sample_negative_002.jpg")
        
        linter = DatasetLinter()
        report = linter.lint(tmp_path)
        
        assert any(i.category == LintCategory.LEAKAGE for i in report.issues)
    
    def test_skip_checks(self, tmp_path):
        """Can skip individual checks."""
        # Create corrupt file
        (tmp_path / "corrupt.jpg").write_bytes(b"not real")
        
        # Skip corrupt check
        linter = DatasetLinter(check_corrupt=False)
        report = linter.lint(tmp_path)
        
        assert report.corrupt_found == 0
    
    def test_max_scan_files_limit(self, tmp_path):
        """Linter respects max_scan_files limit."""
        from PIL import Image
        
        img = Image.new("RGB", (10, 10))
        for i in range(20):
            img.save(tmp_path / f"img_{i}.jpg")
        
        linter = DatasetLinter(max_scan_files=5)
        report = linter.lint(tmp_path)
        
        assert report.files_checked == 5

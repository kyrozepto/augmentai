"""
Dataset analyzer.

Computes statistics and generates reports about datasets.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from augmentai.inspection.detector import DatasetDetector, DetectionResult, DatasetFormat


@dataclass
class DatasetIssue:
    """An issue found in the dataset."""
    severity: str  # "error", "warning", "info"
    message: str
    file_path: Path | None = None


@dataclass
class DatasetReport:
    """Complete report about a dataset."""
    
    # Detection results
    detection: DetectionResult
    
    # Counts
    total_files: int = 0
    image_count: int = 0
    
    # Structure
    class_distribution: dict[str, int] = field(default_factory=dict)
    split_distribution: dict[str, int] = field(default_factory=dict)
    
    # Image properties (from sampling)
    image_sizes: list[tuple[int, int]] = field(default_factory=list)
    size_range: tuple[tuple[int, int], tuple[int, int]] | None = None  # (min, max)
    
    # Issues
    issues: list[DatasetIssue] = field(default_factory=list)
    
    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Dataset Report",
            f"=" * 40,
            f"Format: {self.detection.format.value}",
            f"Image Type: {self.detection.image_type.value}",
            f"Total Images: {self.image_count}",
            f"Suggested Domain: {self.detection.suggested_domain}",
        ]
        
        if self.class_distribution:
            lines.append(f"\nClass Distribution:")
            for cls, count in sorted(self.class_distribution.items()):
                lines.append(f"  {cls}: {count}")
        
        if self.split_distribution:
            lines.append(f"\nSplit Distribution:")
            for split, count in self.split_distribution.items():
                lines.append(f"  {split}: {count}")
        
        if self.issues:
            lines.append(f"\nIssues Found: {len(self.issues)}")
            for issue in self.issues[:5]:  # Show first 5
                lines.append(f"  [{issue.severity.upper()}] {issue.message}")
        
        if self.recommendations:
            lines.append(f"\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")
        
        return "\n".join(lines)


class DatasetAnalyzer:
    """Analyze datasets and generate reports."""
    
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    
    def __init__(self, sample_size: int = 100) -> None:
        """
        Initialize analyzer.
        
        Args:
            sample_size: Number of images to sample for size analysis
        """
        self.sample_size = sample_size
        self.detector = DatasetDetector()
    
    def analyze(self, path: Path) -> DatasetReport:
        """
        Analyze a dataset and generate a report.
        
        Args:
            path: Path to dataset directory
            
        Returns:
            DatasetReport with analysis results
        """
        path = Path(path)
        
        # First, detect format
        detection = self.detector.detect(path)
        
        # Create report
        report = DatasetReport(detection=detection)
        
        # Count images and analyze structure
        self._count_images(path, detection, report)
        
        # Sample image sizes
        self._sample_sizes(path, report)
        
        # Check for issues
        self._check_issues(report)
        
        # Generate recommendations
        self._generate_recommendations(report)
        
        return report
    
    def _count_images(
        self, 
        path: Path, 
        detection: DetectionResult, 
        report: DatasetReport
    ) -> None:
        """Count images and analyze class/split distribution."""
        class_counts: Counter[str] = Counter()
        split_counts: Counter[str] = Counter()
        
        all_extensions = detection.image_extensions or self.IMAGE_EXTENSIONS
        
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in all_extensions:
                report.image_count += 1
                
                # Get relative path parts
                rel_path = file_path.relative_to(path)
                parts = rel_path.parts
                
                if len(parts) >= 2:
                    # Could be split/class or class/file structure
                    if parts[0].lower() in {"train", "val", "validation", "test", "dev"}:
                        split_counts[parts[0]] += 1
                        if len(parts) >= 3:
                            class_counts[parts[1]] += 1
                    else:
                        # Assume first dir is class
                        class_counts[parts[0]] += 1
        
        report.total_files = report.image_count
        report.class_distribution = dict(class_counts)
        report.split_distribution = dict(split_counts)
    
    def _sample_sizes(self, path: Path, report: DatasetReport) -> None:
        """Sample image sizes from dataset."""
        try:
            from PIL import Image
        except ImportError:
            report.issues.append(DatasetIssue(
                severity="warning",
                message="PIL not installed, cannot analyze image sizes"
            ))
            return
        
        image_files = []
        all_extensions = report.detection.image_extensions or self.IMAGE_EXTENSIONS
        
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in all_extensions:
                image_files.append(file_path)
                if len(image_files) >= self.sample_size:
                    break
        
        sizes = []
        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    sizes.append(img.size)  # (width, height)
            except Exception:
                report.issues.append(DatasetIssue(
                    severity="warning",
                    message=f"Could not read image",
                    file_path=img_path
                ))
        
        if sizes:
            report.image_sizes = sizes
            widths = [s[0] for s in sizes]
            heights = [s[1] for s in sizes]
            report.size_range = (
                (min(widths), min(heights)),
                (max(widths), max(heights))
            )
    
    def _check_issues(self, report: DatasetReport) -> None:
        """Check for common issues."""
        # Check for class imbalance
        if report.class_distribution:
            counts = list(report.class_distribution.values())
            if counts:
                max_count = max(counts)
                min_count = min(counts)
                if max_count > 0 and min_count > 0:
                    ratio = max_count / min_count
                    if ratio > 10:
                        report.issues.append(DatasetIssue(
                            severity="warning",
                            message=f"Severe class imbalance detected (ratio {ratio:.1f}:1)"
                        ))
                    elif ratio > 5:
                        report.issues.append(DatasetIssue(
                            severity="info",
                            message=f"Moderate class imbalance detected (ratio {ratio:.1f}:1)"
                        ))
        
        # Check for small dataset
        if report.image_count < 100:
            report.issues.append(DatasetIssue(
                severity="warning",
                message=f"Small dataset ({report.image_count} images) - consider using stronger augmentation"
            ))
        
        # Check for size inconsistency
        if report.size_range:
            (min_w, min_h), (max_w, max_h) = report.size_range
            if max_w / min_w > 2 or max_h / min_h > 2:
                report.issues.append(DatasetIssue(
                    severity="info",
                    message="Variable image sizes detected - consider resize preprocessing"
                ))
    
    def _generate_recommendations(self, report: DatasetReport) -> None:
        """Generate recommendations based on analysis."""
        # Domain recommendation
        report.recommendations.append(
            f"Recommended domain: {report.detection.suggested_domain} "
            f"({report.detection.domain_reason})"
        )
        
        # Split recommendation if not pre-split
        if not report.detection.is_presplit:
            if report.class_distribution:
                report.recommendations.append(
                    "Use stratified split to maintain class balance"
                )
            else:
                report.recommendations.append(
                    "Use random split (80% train, 10% val, 10% test)"
                )
        
        # Augmentation recommendations based on size
        if report.image_count < 500:
            report.recommendations.append(
                "Small dataset: use aggressive augmentation (higher probabilities)"
            )
        elif report.image_count > 10000:
            report.recommendations.append(
                "Large dataset: use moderate augmentation (lower probabilities)"
            )
        
        # Mask sync recommendation
        if report.detection.has_masks:
            report.recommendations.append(
                "Masks detected: ensure geometric transforms are synchronized"
            )

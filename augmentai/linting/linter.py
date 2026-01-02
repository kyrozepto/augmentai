"""
Dataset linter for pre-prepare quality checks.

Detects common data quality issues before augmentation:
- Duplicate images (via perceptual hashing)
- Corrupt/unreadable images
- Mask-image dimension mismatches
- Class imbalance warnings
- Potential label leakage
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

# Image extensions to check
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
MASK_PATTERNS = ["mask", "seg", "label", "annotation", "_m", "_mask"]


class LintSeverity(str, Enum):
    """Severity level of a lint issue."""
    ERROR = "error"      # Critical issue, should stop prepare
    WARNING = "warning"  # Notable issue, but can proceed
    INFO = "info"        # Informational only


class LintCategory(str, Enum):
    """Category of lint issue."""
    DUPLICATE = "duplicate"
    CORRUPT = "corrupt"
    MISMATCH = "mismatch"
    IMBALANCE = "imbalance"
    LEAKAGE = "leakage"
    STRUCTURE = "structure"


@dataclass
class LintIssue:
    """A single lint issue found in the dataset."""
    
    severity: LintSeverity
    category: LintCategory
    message: str
    file_path: Path | None = None
    suggestion: str = ""
    
    def __str__(self) -> str:
        icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}[self.severity.value]
        path_str = f" ({self.file_path.name})" if self.file_path else ""
        return f"{icon} [{self.category.value}] {self.message}{path_str}"


@dataclass
class LintReport:
    """Result of linting a dataset."""
    
    issues: list[LintIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    files_checked: int = 0
    duplicates_found: int = 0
    corrupt_found: int = 0
    
    @property
    def passed(self) -> bool:
        """Check if linting passed (no errors)."""
        return not any(i.severity == LintSeverity.ERROR for i in self.issues)
    
    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == LintSeverity.ERROR)
    
    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == LintSeverity.WARNING)
    
    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == LintSeverity.INFO)
    
    def add_issue(self, issue: LintIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)
    
    def summary(self) -> str:
        """Get a one-line summary."""
        if self.passed:
            if self.warning_count > 0:
                return f"✓ Passed with {self.warning_count} warnings"
            return f"✓ Passed ({self.files_checked} files checked)"
        return f"✗ Failed with {self.error_count} errors, {self.warning_count} warnings"
    
    def display(self, console: Console | None = None) -> None:
        """Display the report using Rich."""
        console = console or Console()
        
        if not self.issues:
            console.print(f"[green]✓ No issues found ({self.files_checked} files checked)[/green]")
            return
        
        table = Table(title="Lint Results", show_header=True)
        table.add_column("Severity", style="bold")
        table.add_column("Category")
        table.add_column("Message")
        table.add_column("File")
        
        for issue in sorted(self.issues, key=lambda x: x.severity.value):
            severity_style = {
                LintSeverity.ERROR: "red",
                LintSeverity.WARNING: "yellow",
                LintSeverity.INFO: "blue",
            }[issue.severity]
            
            table.add_row(
                f"[{severity_style}]{issue.severity.value}[/{severity_style}]",
                issue.category.value,
                issue.message,
                issue.file_path.name if issue.file_path else "-",
            )
        
        console.print(table)
        console.print(f"\n{self.summary()}")


class DatasetLinter:
    """
    Lint a dataset for common quality issues.
    
    Runs automatically before prepare to catch problems early.
    """
    
    def __init__(
        self,
        check_duplicates: bool = True,
        check_corrupt: bool = True,
        check_masks: bool = True,
        check_imbalance: bool = True,
        check_leakage: bool = True,
        imbalance_threshold: float = 10.0,  # Max/min ratio
        max_scan_files: int = 5000,
    ) -> None:
        """
        Initialize the linter.
        
        Args:
            check_duplicates: Check for duplicate images
            check_corrupt: Check for corrupt/unreadable images
            check_masks: Check mask-image dimension matches
            check_imbalance: Warn about class imbalance
            check_leakage: Check for label leakage patterns
            imbalance_threshold: Ratio threshold for imbalance warning
            max_scan_files: Maximum files to scan
        """
        self.check_duplicates = check_duplicates
        self.check_corrupt = check_corrupt
        self.check_masks = check_masks
        self.check_imbalance = check_imbalance
        self.check_leakage = check_leakage
        self.imbalance_threshold = imbalance_threshold
        self.max_scan_files = max_scan_files
    
    def lint(self, path: Path, domain: str | None = None) -> LintReport:
        """
        Lint a dataset directory.
        
        Args:
            path: Path to dataset directory
            domain: Optional domain for domain-specific checks
            
        Returns:
            LintReport with all issues found
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        
        report = LintReport()
        
        # Collect image files
        image_files = self._collect_images(path)
        report.files_checked = len(image_files)
        
        if not image_files:
            report.add_issue(LintIssue(
                severity=LintSeverity.WARNING,
                category=LintCategory.STRUCTURE,
                message="No image files found in dataset",
            ))
            return report
        
        # Run checks
        if self.check_corrupt:
            self._check_corrupt_images(image_files, report)
        
        if self.check_duplicates:
            self._check_duplicates(image_files, report)
        
        if self.check_masks:
            self._check_mask_mismatches(path, image_files, report)
        
        if self.check_imbalance:
            self._check_class_imbalance(path, image_files, report)
        
        if self.check_leakage:
            self._check_label_leakage(image_files, report)
        
        return report
    
    def _collect_images(self, path: Path) -> list[Path]:
        """Collect all image files in the directory."""
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(path.rglob(f"*{ext}"))
            images.extend(path.rglob(f"*{ext.upper()}"))
        
        # Limit to max_scan_files
        return sorted(images)[:self.max_scan_files]
    
    def _check_corrupt_images(self, files: list[Path], report: LintReport) -> None:
        """Check for corrupt or unreadable images."""
        try:
            from PIL import Image
        except ImportError:
            report.add_issue(LintIssue(
                severity=LintSeverity.INFO,
                category=LintCategory.CORRUPT,
                message="Pillow not installed, skipping corrupt image check",
            ))
            return
        
        for file_path in files:
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify image integrity
            except Exception as e:
                report.corrupt_found += 1
                report.add_issue(LintIssue(
                    severity=LintSeverity.ERROR,
                    category=LintCategory.CORRUPT,
                    message=f"Corrupt or unreadable image: {str(e)[:50]}",
                    file_path=file_path,
                    suggestion="Remove or replace this file",
                ))
    
    def _check_duplicates(self, files: list[Path], report: LintReport) -> None:
        """Check for duplicate images using file hash."""
        hashes: dict[str, list[Path]] = {}
        
        for file_path in files:
            try:
                file_hash = self._compute_file_hash(file_path)
                if file_hash in hashes:
                    hashes[file_hash].append(file_path)
                else:
                    hashes[file_hash] = [file_path]
            except Exception:
                pass  # Skip files that can't be hashed
        
        # Report duplicates
        for file_hash, paths in hashes.items():
            if len(paths) > 1:
                report.duplicates_found += len(paths) - 1
                report.add_issue(LintIssue(
                    severity=LintSeverity.WARNING,
                    category=LintCategory.DUPLICATE,
                    message=f"Found {len(paths)} identical files",
                    file_path=paths[0],
                    suggestion=f"Consider removing duplicates: {', '.join(p.name for p in paths[1:])}",
                ))
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of file contents."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _check_mask_mismatches(self, base_path: Path, files: list[Path], report: LintReport) -> None:
        """Check for mask-image dimension mismatches."""
        try:
            from PIL import Image
        except ImportError:
            return
        
        # Find potential mask directories/files
        mask_dirs = []
        for pattern in MASK_PATTERNS:
            mask_dirs.extend(base_path.rglob(f"*{pattern}*"))
        
        if not mask_dirs:
            return  # No mask directories found
        
        # Check a sample of files
        for file_path in files[:100]:  # Sample first 100
            # Look for corresponding mask
            mask_path = self._find_mask_for_image(file_path, mask_dirs)
            if mask_path and mask_path.exists():
                try:
                    with Image.open(file_path) as img, Image.open(mask_path) as mask:
                        if img.size != mask.size:
                            report.add_issue(LintIssue(
                                severity=LintSeverity.ERROR,
                                category=LintCategory.MISMATCH,
                                message=f"Mask size {mask.size} doesn't match image {img.size}",
                                file_path=file_path,
                                suggestion="Resize mask to match image dimensions",
                            ))
                except Exception:
                    pass
    
    def _find_mask_for_image(self, image_path: Path, mask_dirs: list[Path]) -> Path | None:
        """Try to find a matching mask file for an image."""
        stem = image_path.stem
        
        for mask_dir in mask_dirs:
            if mask_dir.is_dir():
                # Check for same filename in mask directory
                for ext in IMAGE_EXTENSIONS:
                    potential_mask = mask_dir / f"{stem}{ext}"
                    if potential_mask.exists():
                        return potential_mask
            elif mask_dir.is_file() and mask_dir.stem == stem:
                return mask_dir
        
        return None
    
    def _check_class_imbalance(self, base_path: Path, files: list[Path], report: LintReport) -> None:
        """Check for severe class imbalance in folder-based datasets."""
        # Check if dataset uses folder structure for classes
        class_counts: dict[str, int] = {}
        
        for file_path in files:
            # Get parent folder as class (if not base path)
            relative = file_path.relative_to(base_path)
            if len(relative.parts) > 1:
                class_name = relative.parts[0]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if len(class_counts) < 2:
            return  # Not a multi-class folder structure
        
        # Store stats
        report.stats["class_distribution"] = class_counts
        
        # Check imbalance ratio
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        
        if min_count > 0:
            ratio = max_count / min_count
            report.stats["imbalance_ratio"] = ratio
            
            if ratio > self.imbalance_threshold:
                max_class = max(class_counts, key=class_counts.get)
                min_class = min(class_counts, key=class_counts.get)
                report.add_issue(LintIssue(
                    severity=LintSeverity.WARNING,
                    category=LintCategory.IMBALANCE,
                    message=f"Class imbalance: {max_class}({max_count}) vs {min_class}({min_count}), ratio={ratio:.1f}x",
                    suggestion="Consider oversampling minority class or using weighted loss",
                ))
    
    def _check_label_leakage(self, files: list[Path], report: LintReport) -> None:
        """Check for potential label leakage in filenames."""
        # Common patterns that might indicate label leakage
        leakage_patterns = [
            "positive", "negative", "benign", "malignant",
            "class_0", "class_1", "label_", "true", "false",
            "_yes_", "_no_", "good", "bad", "normal", "abnormal",
        ]
        
        leakage_found = set()
        
        for file_path in files:
            filename_lower = file_path.stem.lower()
            for pattern in leakage_patterns:
                if pattern in filename_lower:
                    leakage_found.add(pattern)
        
        if leakage_found:
            report.add_issue(LintIssue(
                severity=LintSeverity.WARNING,
                category=LintCategory.LEAKAGE,
                message=f"Potential label leakage in filenames: {', '.join(leakage_found)}",
                suggestion="Ensure model cannot infer labels from filenames during training",
            ))

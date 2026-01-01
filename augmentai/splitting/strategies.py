"""
Dataset splitting strategies.

Provides random, stratified, and group-based splitting.
"""

from __future__ import annotations

import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SplitStrategy(Enum):
    """Available splitting strategies."""
    RANDOM = "random"  # Pure random split
    STRATIFIED = "stratified"  # Maintain class distribution
    GROUP = "group"  # Keep groups together (e.g., patients)


@dataclass
class SplitResult:
    """Result of dataset splitting."""
    
    success: bool
    
    # Split statistics
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    
    # Paths (if files were copied)
    train_dir: Path | None = None
    val_dir: Path | None = None
    test_dir: Path | None = None
    
    # File lists (for reference)
    train_files: list[Path] = field(default_factory=list)
    val_files: list[Path] = field(default_factory=list)
    test_files: list[Path] = field(default_factory=list)
    
    # Issues
    errors: list[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Get human-readable summary."""
        total = self.train_count + self.val_count + self.test_count
        if total == 0:
            return "No files split"
        
        return (
            f"Split: {self.train_count} train ({self.train_count/total*100:.1f}%), "
            f"{self.val_count} val ({self.val_count/total*100:.1f}%), "
            f"{self.test_count} test ({self.test_count/total*100:.1f}%)"
        )


@dataclass
class SplitConfig:
    """Configuration for splitting."""
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    strategy: SplitStrategy = SplitStrategy.STRATIFIED
    seed: int = 42
    
    # For stratified split
    class_key: str | None = None  # How to get class from path
    
    # For group split (e.g., keep patient data together)
    group_pattern: str | None = None  # Regex to extract group ID
    
    def __post_init__(self) -> None:
        """Validate config."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Ratios must sum to 1.0, got {total}")


class DatasetSplitter:
    """Split datasets into train/val/test partitions."""
    
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    
    def __init__(self, config: SplitConfig | None = None) -> None:
        """
        Initialize splitter.
        
        Args:
            config: Split configuration
        """
        self.config = config or SplitConfig()
    
    def split(
        self,
        source_path: Path,
        output_path: Path | None = None,
        copy_files: bool = True,
    ) -> SplitResult:
        """
        Split a dataset.
        
        Args:
            source_path: Path to source dataset
            output_path: Where to write split data (if copy_files=True)
            copy_files: Whether to copy files to output directories
            
        Returns:
            SplitResult with split information
        """
        source_path = Path(source_path)
        
        # Set random seed for reproducibility
        random.seed(self.config.seed)
        
        # Collect all image files
        files = self._collect_files(source_path)
        
        if not files:
            return SplitResult(
                success=False,
                errors=["No image files found in dataset"]
            )
        
        # Split based on strategy
        if self.config.strategy == SplitStrategy.STRATIFIED:
            train, val, test = self._stratified_split(files, source_path)
        elif self.config.strategy == SplitStrategy.GROUP:
            train, val, test = self._group_split(files)
        else:
            train, val, test = self._random_split(files)
        
        result = SplitResult(
            success=True,
            train_count=len(train),
            val_count=len(val),
            test_count=len(test),
            train_files=train,
            val_files=val,
            test_files=test,
        )
        
        # Copy files if requested
        if copy_files and output_path:
            output_path = Path(output_path)
            self._copy_split_files(source_path, output_path, result)
        
        return result
    
    def _collect_files(self, path: Path) -> list[Path]:
        """Collect all image files from directory."""
        files = []
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                files.append(file_path)
        return sorted(files)  # Sort for determinism
    
    def _random_split(
        self, files: list[Path]
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Simple random split."""
        shuffled = files.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)
        
        return (
            shuffled[:train_end],
            shuffled[train_end:val_end],
            shuffled[val_end:]
        )
    
    def _stratified_split(
        self, files: list[Path], base_path: Path
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Stratified split maintaining class distribution."""
        # Group by class (parent directory name)
        class_files: dict[str, list[Path]] = defaultdict(list)
        
        for file_path in files:
            rel_path = file_path.relative_to(base_path)
            parts = rel_path.parts
            
            # Use parent directory as class
            if len(parts) >= 2:
                class_name = parts[-2]  # Directory containing the file
            else:
                class_name = "_default"
            
            class_files[class_name].append(file_path)
        
        # Split each class
        train_all, val_all, test_all = [], [], []
        
        for class_name, class_file_list in class_files.items():
            random.shuffle(class_file_list)
            
            n = len(class_file_list)
            train_end = max(1, int(n * self.config.train_ratio))
            val_end = train_end + max(0, int(n * self.config.val_ratio))
            
            train_all.extend(class_file_list[:train_end])
            val_all.extend(class_file_list[train_end:val_end])
            test_all.extend(class_file_list[val_end:])
        
        return train_all, val_all, test_all
    
    def _group_split(
        self, files: list[Path]
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Group-based split (keeps related files together)."""
        import re
        
        # Group files by pattern or filename prefix
        groups: dict[str, list[Path]] = defaultdict(list)
        
        pattern = self.config.group_pattern or r"^([^_]+)"  # Default: prefix before underscore
        
        for file_path in files:
            match = re.match(pattern, file_path.stem)
            group_id = match.group(1) if match else file_path.stem
            groups[group_id].append(file_path)
        
        # Split groups
        group_ids = list(groups.keys())
        random.shuffle(group_ids)
        
        n = len(group_ids)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)
        
        train_groups = group_ids[:train_end]
        val_groups = group_ids[train_end:val_end]
        test_groups = group_ids[val_end:]
        
        train_files = [f for g in train_groups for f in groups[g]]
        val_files = [f for g in val_groups for f in groups[g]]
        test_files = [f for g in test_groups for f in groups[g]]
        
        return train_files, val_files, test_files
    
    def _copy_split_files(
        self,
        source_path: Path,
        output_path: Path,
        result: SplitResult
    ) -> None:
        """Copy files to split directories."""
        # Create directories
        train_dir = output_path / "train"
        val_dir = output_path / "val"  
        test_dir = output_path / "test"
        
        for split_dir in [train_dir, val_dir, test_dir]:
            split_dir.mkdir(parents=True, exist_ok=True)
        
        result.train_dir = train_dir
        result.val_dir = val_dir
        result.test_dir = test_dir
        
        # Copy files preserving class structure
        for files, dest_dir in [
            (result.train_files, train_dir),
            (result.val_files, val_dir),
            (result.test_files, test_dir)
        ]:
            for file_path in files:
                # Get relative class path
                rel_path = file_path.relative_to(source_path)
                
                # Create class subdirectory if needed
                dest_file = dest_dir / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(file_path, dest_file)

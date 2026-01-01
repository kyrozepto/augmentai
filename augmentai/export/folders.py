"""
Folder structure generator.

Creates standardized output directories for train-ready datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from augmentai.core.manifest import ReproducibilityManifest


@dataclass
class FolderStructure:
    """Manages the output folder structure."""
    
    base_path: Path
    
    # Standard directories
    data_dir: Path = field(init=False)
    train_dir: Path = field(init=False)
    val_dir: Path = field(init=False)
    test_dir: Path = field(init=False)
    augmented_dir: Path = field(init=False)
    
    # Files
    script_path: Path = field(init=False)
    config_path: Path = field(init=False)
    manifest_path: Path = field(init=False)
    requirements_path: Path = field(init=False)
    
    def __post_init__(self) -> None:
        """Initialize paths."""
        self.base_path = Path(self.base_path)
        
        # Data directories
        self.data_dir = self.base_path / "data"
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.test_dir = self.data_dir / "test"
        
        # Output for augmented data
        self.augmented_dir = self.base_path / "augmented"
        
        # Files
        self.script_path = self.base_path / "augment.py"
        self.config_path = self.base_path / "config.yaml"
        self.manifest_path = self.base_path / "manifest.json"
        self.requirements_path = self.base_path / "requirements.txt"
    
    def create(self, include_augmented: bool = True) -> None:
        """
        Create the folder structure.
        
        Args:
            include_augmented: Whether to create augmented output dirs
        """
        # Create data directories
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create augmented directories
        if include_augmented:
            for split in ["train", "val", "test"]:
                (self.augmented_dir / split).mkdir(parents=True, exist_ok=True)
    
    def save_script(self, content: str) -> Path:
        """Save the augmentation script."""
        self.script_path.write_text(content, encoding="utf-8")
        return self.script_path
    
    def save_config(self, content: str) -> Path:
        """Save the configuration file."""
        self.config_path.write_text(content, encoding="utf-8")
        return self.config_path
    
    def save_manifest(self, manifest: ReproducibilityManifest) -> Path:
        """Save the reproducibility manifest."""
        manifest.save(self.manifest_path)
        return self.manifest_path
    
    def save_requirements(self, content: str) -> Path:
        """Save requirements.txt."""
        self.requirements_path.write_text(content, encoding="utf-8")
        return self.requirements_path
    
    def get_readme_content(self, policy_name: str, domain: str) -> str:
        """Generate README content for the output directory."""
        return f"""\
# {policy_name}

This directory contains a train-ready dataset prepared by AugmentAI.

## Domain
{domain}

## Structure
```
{self.base_path.name}/
+-- data/
|   +-- train/     # Training images
|   +-- val/       # Validation images
|   +-- test/      # Test images
+-- augmented/     # Augmented output (after running augment.py)
+-- augment.py     # Augmentation script
+-- config.yaml    # Pipeline configuration
+-- manifest.json  # Reproducibility manifest
+-- requirements.txt
```

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run augmentation:
   ```bash
   python augment.py --input data/train --output augmented/train
   ```

3. For custom seed:
   ```bash
   python augment.py --input data/train --output augmented/train --seed 123
   ```

## Reproducibility

This dataset was prepared with fixed random seeds. See `manifest.json` for 
complete reproducibility information including versions and configuration.
"""
    
    def save_readme(self, policy_name: str, domain: str) -> Path:
        """Save README file."""
        readme_path = self.base_path / "README.md"
        readme_path.write_text(self.get_readme_content(policy_name, domain), encoding="utf-8")
        return readme_path
    
    def summary(self) -> str:
        """Get summary of folder structure."""
        lines = [
            f"Output directory: {self.base_path}",
            f"  data/",
            f"    train/",
            f"    val/",
            f"    test/",
            f"  augmented/",
            f"  augment.py",
            f"  config.yaml",
            f"  manifest.json",
        ]
        return "\n".join(lines)

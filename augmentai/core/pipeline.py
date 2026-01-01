"""
Pipeline definition for end-to-end data preparation.

Orchestrates: inspect → split → augment → export
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from augmentai.core.policy import Policy
from augmentai.core.manifest import ReproducibilityManifest


@dataclass
class PipelineConfig:
    """Configuration for a data preparation pipeline."""
    
    # Input/Output
    dataset_path: Path
    output_path: Path
    
    # Domain and task
    domain: str = "auto"  # "auto" for detection, or specific domain name
    task_description: str = ""  # Natural language task description for LLM
    
    # Splitting
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_strategy: str = "stratified"  # "random", "stratified", "group"
    
    # Reproducibility
    seed: int = 42
    
    # Backend
    backend: str = "albumentations"
    
    # Options
    dry_run: bool = False
    overwrite: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        self.dataset_path = Path(self.dataset_path)
        self.output_path = Path(self.output_path)
        
        # Validate split ratios
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")


@dataclass
class PipelineResult:
    """Result of running a data preparation pipeline."""
    
    success: bool
    manifest: ReproducibilityManifest | None = None
    
    # Paths to outputs
    output_dir: Path | None = None
    train_dir: Path | None = None
    val_dir: Path | None = None
    test_dir: Path | None = None
    
    # Generated files
    augment_script: Path | None = None
    config_file: Path | None = None
    manifest_file: Path | None = None
    
    # Statistics
    stats: dict[str, Any] = field(default_factory=dict)
    
    # Issues
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = []
        
        if self.success:
            lines.append("✓ Pipeline completed successfully")
            if self.output_dir:
                lines.append(f"  Output: {self.output_dir}")
            if self.stats:
                lines.append(f"  Train: {self.stats.get('train_count', 0)} files")
                lines.append(f"  Val: {self.stats.get('val_count', 0)} files")
                lines.append(f"  Test: {self.stats.get('test_count', 0)} files")
        else:
            lines.append("✗ Pipeline failed")
            for error in self.errors:
                lines.append(f"  Error: {error}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")
        
        return "\n".join(lines)


@dataclass 
class CompiledPipeline:
    """A compiled, ready-to-execute augmentation pipeline."""
    
    policy: Policy
    backend: str
    seed: int
    
    # Generated code
    code: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    
    def save_script(self, path: Path) -> None:
        """Save the augmentation script."""
        path.write_text(self.code)
    
    def save_config(self, path: Path) -> None:
        """Save the configuration as YAML."""
        import yaml
        path.write_text(yaml.dump(self.config, default_flow_style=False))

"""
Reproducibility manifest for tracking pipeline execution.

Captures all information needed to reproduce a data preparation run.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ReproducibilityManifest:
    """Everything needed to reproduce a data preparation run."""
    
    # Core identifiers
    seed: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Version info
    augmentai_version: str = "0.1.0"
    python_version: str = field(default_factory=lambda: platform.python_version())
    
    # Dataset info
    dataset_path: str = ""
    dataset_hash: str = ""
    file_count: int = 0
    
    # Configuration
    domain: str = "natural"
    backend: str = "albumentations"
    
    # Split info
    split_ratios: dict[str, float] = field(default_factory=lambda: {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1
    })
    
    # Policy info
    policy_name: str = ""
    policy_hash: str = ""
    transforms: list[dict[str, Any]] = field(default_factory=list)
    
    # Output info
    output_path: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: Path) -> None:
        """Save manifest to file."""
        path.write_text(self.to_json())
    
    @classmethod
    def from_json(cls, json_str: str) -> "ReproducibilityManifest":
        """Load from JSON string."""
        data = json.loads(json_str)
        return cls(**data)
    
    @classmethod
    def from_file(cls, path: Path) -> "ReproducibilityManifest":
        """Load manifest from file."""
        return cls.from_json(path.read_text())
    
    @staticmethod
    def hash_directory(path: Path, extensions: set[str] | None = None) -> str:
        """
        Compute a hash of directory contents for reproducibility tracking.
        
        Args:
            path: Directory path
            extensions: File extensions to include (e.g., {".jpg", ".png"})
            
        Returns:
            SHA256 hash of sorted file paths and sizes
        """
        if extensions is None:
            extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        
        hasher = hashlib.sha256()
        
        files = []
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                # Hash relative path and size (not content for speed)
                rel_path = file_path.relative_to(path)
                size = file_path.stat().st_size
                files.append(f"{rel_path}:{size}")
        
        hasher.update("\n".join(files).encode())
        return hasher.hexdigest()[:16]  # Short hash for readability
    
    @staticmethod
    def hash_policy(policy_dict: dict[str, Any]) -> str:
        """Compute hash of policy for tracking."""
        hasher = hashlib.sha256()
        hasher.update(json.dumps(policy_dict, sort_keys=True).encode())
        return hasher.hexdigest()[:16]

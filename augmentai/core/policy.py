"""
Core policy data structures for augmentation policies.

Defines the fundamental building blocks: Transform and Policy,
which represent individual augmentations and complete augmentation pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import json
import yaml


class TransformCategory(str, Enum):
    """Categories of image transforms."""
    
    GEOMETRIC = "geometric"
    COLOR = "color"
    BLUR = "blur"
    NOISE = "noise"
    DISTORTION = "distortion"
    CROP = "crop"
    FLIP = "flip"
    ROTATE = "rotate"
    SCALE = "scale"
    OTHER = "other"


@dataclass
class Transform:
    """
    Represents a single augmentation transform with its parameters.
    
    Attributes:
        name: The transform name (e.g., "HorizontalFlip", "RandomBrightnessContrast")
        probability: Probability of applying this transform (0.0 to 1.0)
        parameters: Transform-specific parameters as key-value pairs
        category: The category of transform (geometric, color, etc.)
        magnitude: Optional magnitude level for RandAugment-style policies (0-10)
    """
    
    name: str
    probability: float = 0.5
    parameters: dict[str, Any] = field(default_factory=dict)
    category: TransformCategory = TransformCategory.OTHER
    magnitude: int | None = None
    
    def __post_init__(self) -> None:
        """Validate transform after initialization."""
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"Probability must be between 0 and 1, got {self.probability}")
        if self.magnitude is not None and not 0 <= self.magnitude <= 10:
            raise ValueError(f"Magnitude must be between 0 and 10, got {self.magnitude}")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert transform to dictionary representation."""
        # Convert any tuples to lists for YAML compatibility
        def sanitize_params(obj: Any) -> Any:
            if isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: sanitize_params(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_params(item) for item in obj]
            return obj
        
        return {
            "name": self.name,
            "probability": self.probability,
            "parameters": sanitize_params(self.parameters),
            "category": self.category.value,
            "magnitude": self.magnitude,
        }

    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transform:
        """Create a Transform from a dictionary."""
        category = data.get("category", "other")
        if isinstance(category, str):
            category = TransformCategory(category)
        
        return cls(
            name=data["name"],
            probability=data.get("probability", 0.5),
            parameters=data.get("parameters", {}),
            category=category,
            magnitude=data.get("magnitude"),
        )


@dataclass
class Policy:
    """
    Represents a complete augmentation policy with multiple transforms.
    
    A Policy is a collection of transforms that together define an augmentation
    strategy. It includes metadata about the domain, creation time, and can be
    exported to various backend formats.
    
    Attributes:
        name: Human-readable policy name
        domain: The domain this policy is designed for (e.g., "medical", "ocr")
        transforms: List of transforms in this policy
        description: Optional description of the policy's purpose
        magnitude_bins: Number of magnitude bins for RandAugment-style policies
        num_ops: Number of operations to apply per image (for RandAugment)
        created_at: When the policy was created
        metadata: Additional metadata for reproducibility
    """
    
    name: str
    domain: str
    transforms: list[Transform] = field(default_factory=list)
    description: str = ""
    magnitude_bins: int = 10
    num_ops: int = 2
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def add_transform(self, transform: Transform) -> None:
        """Add a transform to the policy."""
        self.transforms.append(transform)
    
    def remove_transform(self, name: str) -> bool:
        """Remove a transform by name. Returns True if removed."""
        for i, t in enumerate(self.transforms):
            if t.name == name:
                self.transforms.pop(i)
                return True
        return False
    
    def get_transform(self, name: str) -> Transform | None:
        """Get a transform by name."""
        for t in self.transforms:
            if t.name == name:
                return t
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert policy to dictionary representation."""
        return {
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "magnitude_bins": self.magnitude_bins,
            "num_ops": self.num_ops,
            "created_at": self.created_at.isoformat(),
            "transforms": [t.to_dict() for t in self.transforms],
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Policy:
        """Create a Policy from a dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()
        
        transforms = [
            Transform.from_dict(t) for t in data.get("transforms", [])
        ]
        
        return cls(
            name=data["name"],
            domain=data["domain"],
            transforms=transforms,
            description=data.get("description", ""),
            magnitude_bins=data.get("magnitude_bins", 10),
            num_ops=data.get("num_ops", 2),
            created_at=created_at,
            metadata=data.get("metadata", {}),
        )
    
    def to_yaml(self) -> str:
        """Export policy to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
    
    def to_json(self, indent: int = 2) -> str:
        """Export policy to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> Policy:
        """Load policy from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> Policy:
        """Load policy from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def __len__(self) -> int:
        """Return the number of transforms in the policy."""
        return len(self.transforms)
    
    def __repr__(self) -> str:
        return f"Policy(name='{self.name}', domain='{self.domain}', transforms={len(self.transforms)})"

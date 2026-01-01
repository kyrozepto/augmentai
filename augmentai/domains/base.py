"""
Base domain class and constraint definitions.

Domains define what augmentations are safe and appropriate for specific
types of image data (medical, OCR, satellite, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from augmentai.core.policy import Policy, Transform, TransformCategory


class ConstraintLevel(str, Enum):
    """Severity level of a constraint."""
    
    FORBIDDEN = "forbidden"  # HARD constraint - cannot be overridden
    DISCOURAGED = "discouraged"  # Soft constraint - warn but allow
    RECOMMENDED = "recommended"  # Suggested to use
    REQUIRED = "required"  # Must be included


@dataclass
class DomainConstraint:
    """
    A constraint on what transforms/parameters are allowed.
    
    Attributes:
        transform_name: Name of the transform this applies to (or "*" for all)
        level: How strict this constraint is
        reason: Human-readable explanation of why this constraint exists
        parameter_limits: Override parameter ranges for this domain
    """
    
    transform_name: str
    level: ConstraintLevel
    reason: str
    parameter_limits: dict[str, tuple[float, float]] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "transform_name": self.transform_name,
            "level": self.level.value,
            "reason": self.reason,
            "parameter_limits": self.parameter_limits,
        }


@dataclass
class ValidationResult:
    """Result of validating a policy against domain constraints."""
    
    is_valid: bool
    errors: list[str] = field(default_factory=list)  # Critical issues
    warnings: list[str] = field(default_factory=list)  # Non-critical issues
    suggestions: list[str] = field(default_factory=list)  # Recommendations
    
    def add_error(self, message: str) -> None:
        """Add an error (makes result invalid)."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning (doesn't affect validity)."""
        self.warnings.append(message)
    
    def add_suggestion(self, message: str) -> None:
        """Add a suggestion."""
        self.suggestions.append(message)
    
    def merge(self, other: ValidationResult) -> None:
        """Merge another result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.suggestions.extend(other.suggestions)
        if not other.is_valid:
            self.is_valid = False


class Domain(ABC):
    """
    Abstract base class for domain definitions.
    
    A domain defines the rules and constraints for augmentations
    in a specific image type (medical, OCR, satellite, etc.).
    """
    
    name: str
    description: str
    constraints: list[DomainConstraint]
    
    # Categories of transforms that are generally safe/unsafe
    allowed_categories: set[TransformCategory]
    forbidden_categories: set[TransformCategory]
    
    # Specific transforms that are always forbidden
    forbidden_transforms: set[str]
    
    # Specific transforms that are recommended
    recommended_transforms: set[str]
    
    def __init__(self) -> None:
        """Initialize the domain."""
        self.constraints = []
        self.allowed_categories = set(TransformCategory)
        self.forbidden_categories = set()
        self.forbidden_transforms = set()
        self.recommended_transforms = set()
        self._setup()
    
    @abstractmethod
    def _setup(self) -> None:
        """Set up domain-specific constraints. Called by __init__."""
        pass
    
    def add_constraint(self, constraint: DomainConstraint) -> None:
        """Add a constraint to this domain."""
        self.constraints.append(constraint)
        
        # Update forbidden/recommended sets based on constraint level
        if constraint.level == ConstraintLevel.FORBIDDEN:
            self.forbidden_transforms.add(constraint.transform_name)
        elif constraint.level == ConstraintLevel.RECOMMENDED:
            self.recommended_transforms.add(constraint.transform_name)
    
    def validate_transform(self, transform: Transform) -> ValidationResult:
        """Validate a single transform against domain constraints."""
        result = ValidationResult(is_valid=True)
        
        # Check if transform category is forbidden
        if transform.category in self.forbidden_categories:
            result.add_error(
                f"Transform '{transform.name}' category '{transform.category.value}' "
                f"is forbidden in {self.name} domain"
            )
        
        # Check if specific transform is forbidden
        if transform.name in self.forbidden_transforms:
            # Find the constraint to get the reason
            reason = "Not allowed in this domain"
            for c in self.constraints:
                if c.transform_name == transform.name and c.level == ConstraintLevel.FORBIDDEN:
                    reason = c.reason
                    break
            result.add_error(f"Transform '{transform.name}' is FORBIDDEN: {reason}")
        
        # Check parameter limits
        for constraint in self.constraints:
            if constraint.transform_name == transform.name and constraint.parameter_limits:
                for param_name, (min_val, max_val) in constraint.parameter_limits.items():
                    if param_name in transform.parameters:
                        value = transform.parameters[param_name]
                        # Handle list values (e.g., brightness_limit=[0.0, 0.2])
                        if isinstance(value, (list, tuple)):
                            # For range values, check both bounds
                            for v in value:
                                if isinstance(v, (int, float)) and not (min_val <= v <= max_val):
                                    result.add_error(
                                        f"Parameter '{param_name}' value {v} out of safe range "
                                        f"[{min_val}, {max_val}] for {self.name} domain"
                                    )
                        elif isinstance(value, (int, float)):
                            if not min_val <= value <= max_val:
                                result.add_error(
                                    f"Parameter '{param_name}' value {value} out of safe range "
                                    f"[{min_val}, {max_val}] for {self.name} domain"
                                )
        
        # Check for discouraged transforms
        for constraint in self.constraints:
            if constraint.transform_name == transform.name and constraint.level == ConstraintLevel.DISCOURAGED:
                result.add_warning(f"Transform '{transform.name}' is discouraged: {constraint.reason}")
        
        return result
    
    def validate_policy(self, policy: Policy) -> ValidationResult:
        """Validate a complete policy against domain constraints."""
        result = ValidationResult(is_valid=True)
        
        # Validate each transform
        for transform in policy.transforms:
            transform_result = self.validate_transform(transform)
            result.merge(transform_result)
        
        # Check for recommended transforms
        used_transforms = {t.name for t in policy.transforms}
        for recommended in self.recommended_transforms:
            if recommended not in used_transforms:
                result.add_suggestion(
                    f"Consider adding '{recommended}' - recommended for {self.name} domain"
                )
        
        return result
    
    def get_context_for_llm(self) -> str:
        """
        Get domain context to include in LLM prompts.
        
        Returns a string describing the domain constraints for the LLM.
        """
        lines = [
            f"## Domain: {self.name}",
            f"Description: {self.description}",
            "",
            "### FORBIDDEN Transforms (NEVER use these):",
        ]
        
        for constraint in self.constraints:
            if constraint.level == ConstraintLevel.FORBIDDEN:
                lines.append(f"- {constraint.transform_name}: {constraint.reason}")
        
        lines.append("")
        lines.append("### Discouraged Transforms (use with caution):")
        for constraint in self.constraints:
            if constraint.level == ConstraintLevel.DISCOURAGED:
                lines.append(f"- {constraint.transform_name}: {constraint.reason}")
        
        lines.append("")
        lines.append("### Recommended Transforms:")
        for transform in self.recommended_transforms:
            lines.append(f"- {transform}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert domain to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "constraints": [c.to_dict() for c in self.constraints],
            "forbidden_transforms": list(self.forbidden_transforms),
            "recommended_transforms": list(self.recommended_transforms),
        }
    
    @classmethod
    def from_yaml(cls, path: Path | str) -> Domain:
        """Load a custom domain from a YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return CustomDomain.from_dict(data)


class CustomDomain(Domain):
    """A domain loaded from a YAML configuration file."""
    
    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize from dictionary data."""
        self._data = data
        self.name = data.get("name", "custom")
        self.description = data.get("description", "Custom domain")
        super().__init__()
    
    def _setup(self) -> None:
        """Set up constraints from the loaded data."""
        # Load constraints
        for constraint_data in self._data.get("constraints", []):
            constraint = DomainConstraint(
                transform_name=constraint_data["transform_name"],
                level=ConstraintLevel(constraint_data["level"]),
                reason=constraint_data.get("reason", ""),
                parameter_limits=constraint_data.get("parameter_limits"),
            )
            self.add_constraint(constraint)
        
        # Load forbidden transforms
        for transform in self._data.get("forbidden_transforms", []):
            if isinstance(transform, str):
                self.forbidden_transforms.add(transform)
        
        # Load recommended transforms
        for transform in self._data.get("recommended_transforms", []):
            if isinstance(transform, str):
                self.recommended_transforms.add(transform)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CustomDomain:
        """Create a CustomDomain from a dictionary."""
        return cls(data)

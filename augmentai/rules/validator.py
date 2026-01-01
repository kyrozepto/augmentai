"""
Safety validator for augmentation policies.

This is the final line of defense - it validates policies against
domain constraints and ensures no unsafe augmentations slip through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augmentai.core.policy import Policy, Transform
from augmentai.core.schema import PolicySchema, DEFAULT_SCHEMA
from augmentai.domains.base import Domain, ValidationResult

if TYPE_CHECKING:
    pass


@dataclass
class SafetyResult:
    """Result of safety validation."""
    
    is_safe: bool
    policy: Policy | None = None  # The validated (possibly modified) policy
    original_policy: Policy | None = None  # The original policy before validation
    removed_transforms: list[Transform] = field(default_factory=list)
    modified_transforms: list[tuple[Transform, Transform]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Get a human-readable summary of the validation."""
        lines = []
        
        if self.is_safe:
            lines.append("✓ Policy is safe for the specified domain")
        else:
            lines.append("✗ Policy has safety issues")
        
        if self.removed_transforms:
            lines.append(f"\nRemoved {len(self.removed_transforms)} forbidden transforms:")
            for t in self.removed_transforms:
                lines.append(f"  - {t.name}")
        
        if self.modified_transforms:
            lines.append(f"\nModified {len(self.modified_transforms)} transforms to safe ranges:")
            for original, modified in self.modified_transforms:
                lines.append(f"  - {original.name}: parameters adjusted")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        
        if self.errors:
            lines.append("\nErrors:")
            for e in self.errors:
                lines.append(f"  ✗ {e}")
        
        return "\n".join(lines)


class SafetyValidator:
    """
    Validate and enforce safety constraints on policies.
    
    This validator is the final safety net. It:
    1. Removes any FORBIDDEN transforms
    2. Adjusts parameters to safe ranges
    3. Warns about discouraged transforms
    4. Suggests recommended transforms
    """
    
    def __init__(
        self,
        domain: Domain,
        schema: PolicySchema | None = None,
        strict: bool = True,
    ) -> None:
        """
        Initialize the validator.
        
        Args:
            domain: The domain to validate against
            schema: Transform schema for parameter validation
            strict: If True, remove forbidden transforms. If False, only warn.
        """
        self.domain = domain
        self.schema = schema or DEFAULT_SCHEMA
        self.strict = strict
    
    def validate(self, policy: Policy) -> SafetyResult:
        """
        Validate a policy against domain constraints.
        
        In strict mode, forbidden transforms are removed.
        In non-strict mode, they generate errors but remain.
        
        Args:
            policy: The policy to validate
            
        Returns:
            SafetyResult with the validated policy
        """
        result = SafetyResult(is_safe=True, original_policy=policy)
        
        # Create a copy of the policy to modify
        validated_transforms = []
        
        for transform in policy.transforms:
            # Check domain constraints
            domain_result = self.domain.validate_transform(transform)
            
            if domain_result.errors:
                # This transform has errors
                if self.strict:
                    # Remove the transform
                    result.removed_transforms.append(transform)
                    result.warnings.append(
                        f"Removed '{transform.name}': {domain_result.errors[0]}"
                    )
                else:
                    # Keep it but record the error
                    result.errors.extend(domain_result.errors)
                    result.is_safe = False
                    validated_transforms.append(transform)
            else:
                # Transform is allowed, but may need parameter adjustments
                adjusted = self._adjust_parameters(transform)
                
                if adjusted != transform:
                    result.modified_transforms.append((transform, adjusted))
                
                validated_transforms.append(adjusted)
                
                # Add any warnings
                result.warnings.extend(domain_result.warnings)
        
        # Check schema constraints
        for transform in validated_transforms:
            schema_result = self._validate_schema(transform)
            result.warnings.extend(schema_result)
        
        # Create the validated policy
        result.policy = Policy(
            name=policy.name,
            domain=policy.domain,
            transforms=validated_transforms,
            description=policy.description,
            magnitude_bins=policy.magnitude_bins,
            num_ops=policy.num_ops,
            metadata=policy.metadata,
        )
        
        # Get domain suggestions
        domain_validation = self.domain.validate_policy(result.policy)
        result.warnings.extend([f"Suggestion: {s}" for s in domain_validation.suggestions])
        
        return result
    
    def _adjust_parameters(self, transform: Transform) -> Transform:
        """
        Adjust transform parameters to safe ranges.
        
        Args:
            transform: The transform to adjust
            
        Returns:
            New transform with adjusted parameters (or the same if no changes)
        """
        # Find domain-specific parameter limits
        for constraint in self.domain.constraints:
            if constraint.transform_name == transform.name and constraint.parameter_limits:
                adjusted_params = dict(transform.parameters)
                modified = False
                
                for param_name, (min_val, max_val) in constraint.parameter_limits.items():
                    if param_name in adjusted_params:
                        value = adjusted_params[param_name]
                        
                        # Handle list/tuple values (e.g., brightness_limit=[0.0, 0.2])
                        if isinstance(value, (list, tuple)):
                            clamped = [
                                max(min_val, min(max_val, v)) if isinstance(v, (int, float)) else v
                                for v in value
                            ]
                            if clamped != list(value):
                                adjusted_params[param_name] = clamped
                                modified = True
                        elif isinstance(value, (int, float)):
                            clamped = max(min_val, min(max_val, value))
                            if clamped != value:
                                adjusted_params[param_name] = clamped
                                modified = True
                
                if modified:
                    return Transform(
                        name=transform.name,
                        probability=transform.probability,
                        parameters=adjusted_params,
                        category=transform.category,
                        magnitude=transform.magnitude,
                    )
        
        return transform
    
    def _validate_schema(self, transform: Transform) -> list[str]:
        """
        Validate transform against the schema.
        
        Args:
            transform: The transform to validate
            
        Returns:
            List of warnings
        """
        warnings = []
        
        spec = self.schema.get(transform.name)
        if spec is None:
            warnings.append(f"Unknown transform '{transform.name}' - not in schema")
            return warnings
        
        # Validate parameters
        is_valid, errors = spec.validate_parameters(transform.parameters)
        for error in errors:
            warnings.append(f"{transform.name}: {error}")
        
        return warnings
    
    def quick_check(self, transform_name: str) -> tuple[bool, str]:
        """
        Quickly check if a transform is allowed in this domain.
        
        Args:
            transform_name: Name of the transform to check
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        if transform_name in self.domain.forbidden_transforms:
            # Find the reason
            for constraint in self.domain.constraints:
                if constraint.transform_name == transform_name:
                    return False, constraint.reason
            return False, "Forbidden in this domain"
        
        return True, "Allowed"

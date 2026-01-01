"""
Rule enforcement engine.

Coordinates between LLM suggestions and domain constraints to
produce safe, validated augmentation policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augmentai.core.policy import Policy
from augmentai.core.schema import PolicySchema, DEFAULT_SCHEMA
from augmentai.domains.base import Domain
from augmentai.rules.validator import SafetyValidator, SafetyResult

if TYPE_CHECKING:
    from augmentai.llm.parser import ParseResult


@dataclass
class EnforcementResult:
    """Result of rule enforcement on an LLM suggestion."""
    
    success: bool
    policy: Policy | None = None
    safety_result: SafetyResult | None = None
    llm_reasoning: str = ""
    enforcer_notes: list[str] = field(default_factory=list)
    
    def get_summary(self) -> str:
        """Get a human-readable summary."""
        lines = []
        
        if self.success and self.policy:
            lines.append(f"✓ Policy '{self.policy.name}' created successfully")
            lines.append(f"  Domain: {self.policy.domain}")
            lines.append(f"  Transforms: {len(self.policy.transforms)}")
        else:
            lines.append("✗ Failed to create policy")
        
        if self.llm_reasoning:
            lines.append(f"\nLLM Reasoning: {self.llm_reasoning[:200]}...")
        
        if self.enforcer_notes:
            lines.append("\nEnforcer Notes:")
            for note in self.enforcer_notes:
                lines.append(f"  • {note}")
        
        if self.safety_result:
            lines.append(f"\n{self.safety_result.summary()}")
        
        return "\n".join(lines)


class RuleEnforcer:
    """
    Enforce domain rules on LLM-generated policies.
    
    This is the central coordinator that:
    1. Takes parsed LLM output
    2. Applies domain constraints
    3. Validates safety
    4. Produces a final, safe policy
    """
    
    def __init__(
        self,
        domain: Domain,
        schema: PolicySchema | None = None,
        strict: bool = True,
    ) -> None:
        """
        Initialize the enforcer.
        
        Args:
            domain: The domain to enforce
            schema: Transform schema
            strict: If True, remove forbidden transforms
        """
        self.domain = domain
        self.schema = schema or DEFAULT_SCHEMA
        self.validator = SafetyValidator(domain, schema, strict)
    
    def enforce(self, parse_result: "ParseResult") -> EnforcementResult:
        """
        Enforce rules on a parsed LLM response.
        
        Args:
            parse_result: The parsed LLM output
            
        Returns:
            EnforcementResult with the final policy
        """
        result = EnforcementResult(
            success=False,
            llm_reasoning=parse_result.reasoning,
        )
        
        # Check if parsing succeeded
        if not parse_result.success or parse_result.policy is None:
            result.enforcer_notes.append("LLM response could not be parsed")
            result.enforcer_notes.extend(parse_result.errors)
            return result
        
        # Validate the policy
        safety_result = self.validator.validate(parse_result.policy)
        result.safety_result = safety_result
        
        # Record what happened
        if safety_result.removed_transforms:
            result.enforcer_notes.append(
                f"Removed {len(safety_result.removed_transforms)} forbidden transforms"
            )
        
        if safety_result.modified_transforms:
            result.enforcer_notes.append(
                f"Adjusted {len(safety_result.modified_transforms)} transforms to safe ranges"
            )
        
        # Check if we still have a valid policy
        if safety_result.policy and len(safety_result.policy.transforms) > 0:
            result.success = True
            result.policy = safety_result.policy
        else:
            result.enforcer_notes.append(
                "All transforms were removed - no valid policy could be created"
            )
        
        return result
    
    def enforce_policy(self, policy: Policy) -> EnforcementResult:
        """
        Enforce rules on an existing policy.
        
        Args:
            policy: The policy to validate and enforce
            
        Returns:
            EnforcementResult with the validated policy
        """
        result = EnforcementResult(success=False, llm_reasoning="Direct policy validation")
        
        safety_result = self.validator.validate(policy)
        result.safety_result = safety_result
        
        if safety_result.removed_transforms:
            result.enforcer_notes.append(
                f"Removed {len(safety_result.removed_transforms)} forbidden transforms"
            )
        
        if safety_result.modified_transforms:
            result.enforcer_notes.append(
                f"Adjusted {len(safety_result.modified_transforms)} transforms to safe ranges"
            )
        
        if safety_result.policy and len(safety_result.policy.transforms) > 0:
            result.success = True
            result.policy = safety_result.policy
        
        return result
    
    def suggest_alternatives(self, forbidden_transform: str) -> list[str]:
        """
        Suggest alternative transforms for a forbidden one.
        
        Args:
            forbidden_transform: The forbidden transform name
            
        Returns:
            List of allowed alternative transform names
        """
        # Map common forbidden transforms to safer alternatives
        alternatives_map = {
            "ElasticTransform": ["ShiftScaleRotate", "Affine", "Rotate"],
            "GridDistortion": ["ShiftScaleRotate", "Affine"],
            "OpticalDistortion": ["RandomScale", "Affine"],
            "ColorJitter": ["RandomBrightnessContrast"],
            "HueSaturationValue": ["RandomBrightnessContrast"],
            "MotionBlur": ["GaussianBlur"],
            "Cutout": [],  # No safe alternative for medical
            "CoarseDropout": [],
        }
        
        base_alternatives = alternatives_map.get(forbidden_transform, [])
        
        # Filter to only allowed transforms
        allowed = []
        for alt in base_alternatives:
            if alt not in self.domain.forbidden_transforms:
                allowed.append(alt)
        
        return allowed
    
    def get_domain_summary(self) -> str:
        """
        Get a summary of domain constraints.
        
        Returns:
            Human-readable summary of what's allowed/forbidden
        """
        lines = [
            f"Domain: {self.domain.name}",
            f"Description: {self.domain.description}",
            "",
            "Forbidden Transforms:",
        ]
        
        for t in sorted(self.domain.forbidden_transforms):
            lines.append(f"  ✗ {t}")
        
        lines.append("")
        lines.append("Recommended Transforms:")
        for t in sorted(self.domain.recommended_transforms):
            lines.append(f"  ✓ {t}")
        
        return "\n".join(lines)

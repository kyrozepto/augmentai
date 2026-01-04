"""Policies API endpoints."""

from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

router = APIRouter()


class TransformConfig(BaseModel):
    """Single transform configuration."""
    name: str
    probability: float = 0.5
    parameters: dict = {}


class PolicyCreate(BaseModel):
    """Request body for creating a policy."""
    name: str
    domain: str
    transforms: List[TransformConfig] = []


class PolicyResponse(BaseModel):
    """Policy response model."""
    id: str
    name: str
    domain: str
    transforms: List[TransformConfig]
    is_valid: bool = True
    validation_errors: List[str] = []


class ValidationResult(BaseModel):
    """Policy validation result."""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    blocked_transforms: List[str] = []


# In-memory policy storage (for MVP)
_policies: dict[str, PolicyResponse] = {}


@router.get("", response_model=List[PolicyResponse])
async def list_policies():
    """List all policies."""
    return list(_policies.values())


@router.post("", response_model=PolicyResponse)
async def create_policy(policy: PolicyCreate):
    """Create a new augmentation policy."""
    policy_id = policy.name.replace(" ", "_").lower()
    
    # Validate against domain rules
    validation = await validate_policy_internal(policy.domain, policy.transforms)
    
    response = PolicyResponse(
        id=policy_id,
        name=policy.name,
        domain=policy.domain,
        transforms=policy.transforms,
        is_valid=validation.is_valid,
        validation_errors=validation.errors,
    )
    
    _policies[policy_id] = response
    return response


@router.get("/{policy_id}", response_model=PolicyResponse)
async def get_policy(policy_id: str):
    """Get a specific policy."""
    if policy_id not in _policies:
        raise HTTPException(status_code=404, detail="Policy not found")
    return _policies[policy_id]


@router.post("/{policy_id}/validate", response_model=ValidationResult)
async def validate_policy(policy_id: str):
    """Validate a policy against its domain rules."""
    if policy_id not in _policies:
        raise HTTPException(status_code=404, detail="Policy not found")
    
    policy = _policies[policy_id]
    return await validate_policy_internal(policy.domain, policy.transforms)


@router.post("/{policy_id}/export")
async def export_policy(policy_id: str, format: str = "yaml"):
    """Export policy to YAML, JSON, or Python."""
    if policy_id not in _policies:
        raise HTTPException(status_code=404, detail="Policy not found")
    
    policy = _policies[policy_id]
    
    if format == "yaml":
        import yaml
        content = yaml.dump({
            "name": policy.name,
            "domain": policy.domain,
            "transforms": [t.dict() for t in policy.transforms],
        }, default_flow_style=False)
        return {"format": "yaml", "content": content}
    
    elif format == "json":
        import json
        content = json.dumps({
            "name": policy.name,
            "domain": policy.domain,
            "transforms": [t.dict() for t in policy.transforms],
        }, indent=2)
        return {"format": "json", "content": content}
    
    elif format == "python":
        # Generate Python script
        lines = [
            "import albumentations as A",
            "",
            f"# Policy: {policy.name}",
            f"# Domain: {policy.domain}",
            "",
            "transform = A.Compose([",
        ]
        for t in policy.transforms:
            params = ", ".join(f"{k}={v}" for k, v in t.parameters.items())
            if params:
                lines.append(f"    A.{t.name}({params}, p={t.probability}),")
            else:
                lines.append(f"    A.{t.name}(p={t.probability}),")
        lines.append("])")
        content = "\n".join(lines)
        return {"format": "python", "content": content}
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown format: {format}")


async def validate_policy_internal(domain: str, transforms: List[TransformConfig]) -> ValidationResult:
    """Internal validation logic using AugmentAI rules."""
    errors = []
    warnings = []
    blocked = []
    
    try:
        from augmentai.domains import get_domain
        from augmentai.rules.enforcement import RuleEnforcer
        from augmentai.core.policy import Policy, Transform
        
        domain_obj = get_domain(domain)
        enforcer = RuleEnforcer(domain_obj)
        
        # Convert to internal policy format
        policy = Policy(
            name="validation_check",
            domain=domain,
            transforms=[
                Transform(name=t.name, probability=t.probability, parameters=t.parameters)
                for t in transforms
            ]
        )
        
        result = enforcer.enforce_policy(policy)
        
        return ValidationResult(
            is_valid=result.is_valid,
            errors=result.errors,
            warnings=result.warnings,
            blocked_transforms=result.blocked_transforms,
        )
    except ImportError:
        # Fallback if rules not available
        return ValidationResult(
            is_valid=True,
            warnings=["Validation rules not available"],
        )
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            errors=[str(e)],
        )

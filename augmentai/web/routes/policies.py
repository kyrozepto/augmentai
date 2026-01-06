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


@router.post("/preview-transform")
async def preview_transform(
    transform_name: str = Body(...),
    probability: float = Body(default=1.0),
    parameters: dict = Body(default={}),
    image_base64: str = Body(default=None),
):
    """
    Preview a transform on a sample image.
    Returns before and after images as base64.
    If no image provided, uses a generated sample image.
    """
    import io
    import base64
    
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PIL and numpy required for preview"
        )
    
    # Get or create sample image
    if image_base64:
        try:
            # Decode provided image
            image_data = base64.b64decode(image_base64.split(',')[-1])
            img = Image.open(io.BytesIO(image_data))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image data")
    else:
        # Create a colorful sample image
        img = create_sample_image()
    
    # Convert to numpy array for albumentations
    img_array = np.array(img.convert('RGB'))
    
    # Apply transform
    try:
        import albumentations as A
        
        # Get transform class
        if hasattr(A, transform_name):
            transform_class = getattr(A, transform_name)
            transform = A.Compose([
                transform_class(p=probability, **parameters)
            ])
            augmented = transform(image=img_array)
            augmented_array = augmented['image']
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown transform: {transform_name}"
            )
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="albumentations required for preview"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Transform error: {str(e)}"
        )
    
    # Convert back to PIL and to base64
    before_img = Image.fromarray(img_array)
    after_img = Image.fromarray(augmented_array)
    
    def img_to_base64(img):
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    
    return {
        "before": img_to_base64(before_img),
        "after": img_to_base64(after_img),
        "transform": transform_name,
        "parameters": parameters,
    }


def create_sample_image(size: tuple = (256, 256)) -> 'Image':
    """Create a sample image with geometric patterns and colors."""
    from PIL import Image, ImageDraw
    
    img = Image.new('RGB', size, color=(40, 40, 40))
    draw = ImageDraw.Draw(img)
    
    # Draw colorful shapes
    # Grid lines
    for i in range(0, size[0], 32):
        draw.line([(i, 0), (i, size[1])], fill=(60, 60, 60), width=1)
        draw.line([(0, i), (size[0], i)], fill=(60, 60, 60), width=1)
    
    # Colored rectangles
    draw.rectangle([20, 20, 80, 80], fill=(220, 60, 60))
    draw.rectangle([100, 30, 180, 90], fill=(60, 180, 60))
    draw.rectangle([50, 120, 130, 200], fill=(60, 100, 220))
    
    # Circle
    draw.ellipse([160, 120, 230, 190], fill=(220, 180, 60))
    
    # Diagonal line
    draw.line([(10, 230), (240, 100)], fill=(200, 100, 200), width=3)
    
    # Text-like pattern
    for y in range(210, 250, 10):
        width = 50 + (y % 30) * 2
        draw.rectangle([20, y, 20 + width, y + 6], fill=(180, 180, 180))
    
    return img


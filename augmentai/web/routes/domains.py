"""Domains API endpoints."""

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class DomainInfo(BaseModel):
    """Domain information."""
    id: str
    name: str
    description: str
    forbidden_transforms: List[str] = []
    recommended_transforms: List[str] = []


class TransformInfo(BaseModel):
    """Available transform information."""
    name: str
    description: str
    parameters: dict = {}
    allowed_in: List[str] = []  # Domain IDs


@router.get("", response_model=List[DomainInfo])
async def list_domains():
    """List all available domains."""
    try:
        from augmentai.domains import DOMAINS
        
        return [
            DomainInfo(
                id=domain_id,
                name=domain.name,
                description=domain.description,
                forbidden_transforms=list(domain.forbidden_transforms),
                recommended_transforms=list(domain.recommended_transforms),
            )
            for domain_id, domain in DOMAINS.items()
        ]
    except ImportError:
        # Fallback with built-in domain definitions
        return [
            DomainInfo(
                id="medical",
                name="Medical Imaging",
                description="CT, MRI, X-ray - preserves anatomical structures",
                forbidden_transforms=["ElasticTransform", "GridDistortion", "ColorJitter"],
                recommended_transforms=["HorizontalFlip", "Rotate", "CLAHE", "GaussNoise"],
            ),
            DomainInfo(
                id="ocr",
                name="OCR / Documents",
                description="Text recognition - preserves legibility",
                forbidden_transforms=["MotionBlur", "ElasticTransform"],
                recommended_transforms=["Rotate", "Brightness", "Contrast"],
            ),
            DomainInfo(
                id="satellite",
                name="Satellite / Remote Sensing",
                description="Multi-spectral imagery - preserves spectral relationships",
                forbidden_transforms=["ColorJitter", "HueSaturationValue", "ChannelShuffle"],
                recommended_transforms=["Rotate", "Flip", "Scale"],
            ),
            DomainInfo(
                id="natural",
                name="Natural Images",
                description="General photos - maximum flexibility",
                forbidden_transforms=[],
                recommended_transforms=["HorizontalFlip", "Rotate", "ColorJitter", "RandomCrop"],
            ),
        ]


@router.get("/{domain_id}", response_model=DomainInfo)
async def get_domain(domain_id: str):
    """Get detailed information about a specific domain."""
    domains = await list_domains()
    for domain in domains:
        if domain.id == domain_id:
            return domain
    raise HTTPException(status_code=404, detail=f"Domain not found: {domain_id}")


@router.get("/{domain_id}/transforms", response_model=List[TransformInfo])
async def get_domain_transforms(domain_id: str):
    """Get available transforms for a specific domain."""
    # Common transforms with their parameters
    all_transforms = [
        TransformInfo(
            name="HorizontalFlip",
            description="Flip the image horizontally",
            parameters={"p": {"type": "float", "min": 0, "max": 1, "default": 0.5}},
        ),
        TransformInfo(
            name="VerticalFlip",
            description="Flip the image vertically",
            parameters={"p": {"type": "float", "min": 0, "max": 1, "default": 0.5}},
        ),
        TransformInfo(
            name="Rotate",
            description="Rotate the image by a random angle",
            parameters={
                "limit": {"type": "int", "min": 0, "max": 180, "default": 15},
                "p": {"type": "float", "min": 0, "max": 1, "default": 0.5},
            },
        ),
        TransformInfo(
            name="RandomBrightnessContrast",
            description="Adjust brightness and contrast",
            parameters={
                "brightness_limit": {"type": "float", "min": 0, "max": 0.5, "default": 0.2},
                "contrast_limit": {"type": "float", "min": 0, "max": 0.5, "default": 0.2},
                "p": {"type": "float", "min": 0, "max": 1, "default": 0.5},
            },
        ),
        TransformInfo(
            name="GaussNoise",
            description="Add gaussian noise",
            parameters={
                "var_limit": {"type": "tuple", "default": "(10, 50)"},
                "p": {"type": "float", "min": 0, "max": 1, "default": 0.3},
            },
        ),
        TransformInfo(
            name="CLAHE",
            description="Contrast Limited Adaptive Histogram Equalization",
            parameters={
                "clip_limit": {"type": "float", "min": 1, "max": 8, "default": 4},
                "p": {"type": "float", "min": 0, "max": 1, "default": 0.5},
            },
        ),
        TransformInfo(
            name="ElasticTransform",
            description="Elastic deformation (NOT recommended for medical)",
            parameters={
                "alpha": {"type": "float", "min": 0, "max": 200, "default": 120},
                "sigma": {"type": "float", "min": 0, "max": 20, "default": 6},
                "p": {"type": "float", "min": 0, "max": 1, "default": 0.5},
            },
        ),
        TransformInfo(
            name="GridDistortion",
            description="Grid-based distortion",
            parameters={
                "distort_limit": {"type": "float", "min": 0, "max": 1, "default": 0.3},
                "p": {"type": "float", "min": 0, "max": 1, "default": 0.5},
            },
        ),
        TransformInfo(
            name="ColorJitter",
            description="Random color adjustments",
            parameters={
                "brightness": {"type": "float", "min": 0, "max": 1, "default": 0.2},
                "contrast": {"type": "float", "min": 0, "max": 1, "default": 0.2},
                "saturation": {"type": "float", "min": 0, "max": 1, "default": 0.2},
                "hue": {"type": "float", "min": 0, "max": 0.5, "default": 0.1},
                "p": {"type": "float", "min": 0, "max": 1, "default": 0.5},
            },
        ),
        TransformInfo(
            name="RandomCrop",
            description="Randomly crop a portion of the image",
            parameters={
                "height": {"type": "int", "min": 32, "max": 1024, "default": 224},
                "width": {"type": "int", "min": 32, "max": 1024, "default": 224},
                "p": {"type": "float", "min": 0, "max": 1, "default": 0.5},
            },
        ),
    ]
    
    # Filter based on domain
    domain = await get_domain(domain_id)
    forbidden = set(domain.forbidden_transforms)
    
    return [t for t in all_transforms if t.name not in forbidden]

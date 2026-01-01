"""
RandAugment-style schema definitions for transforms.

Defines the available transforms, their parameter ranges, and categories
that can be used in augmentation policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from augmentai.core.policy import TransformCategory


@dataclass
class ParameterRange:
    """Defines the valid range for a transform parameter."""
    
    name: str
    min_value: float | int
    max_value: float | int
    default: float | int
    dtype: str = "float"  # "float", "int", "bool"
    description: str = ""
    
    def validate(self, value: Any) -> bool:
        """Check if a value is within the valid range."""
        if self.dtype == "bool":
            return isinstance(value, bool)
        try:
            num_value = float(value)
            return self.min_value <= num_value <= self.max_value
        except (TypeError, ValueError):
            return False
    
    def clamp(self, value: float | int) -> float | int:
        """Clamp a value to the valid range."""
        if self.dtype == "int":
            return int(max(self.min_value, min(self.max_value, value)))
        return max(self.min_value, min(self.max_value, value))


@dataclass
class TransformSpec:
    """
    Specification for an augmentation transform.
    
    Defines everything needed to create and validate a transform:
    name, category, parameters, and their valid ranges.
    """
    
    name: str
    category: TransformCategory
    description: str
    parameters: dict[str, ParameterRange] = field(default_factory=dict)
    supports_masks: bool = True
    supports_bboxes: bool = True
    is_safe_for_segmentation: bool = True
    
    def validate_parameters(self, params: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate a set of parameters against this spec.
        
        Returns (is_valid, list_of_errors).
        """
        errors = []
        
        for param_name, param_value in params.items():
            if param_name not in self.parameters:
                errors.append(f"Unknown parameter: {param_name}")
                continue
            
            param_spec = self.parameters[param_name]
            if not param_spec.validate(param_value):
                errors.append(
                    f"Parameter '{param_name}' value {param_value} out of range "
                    f"[{param_spec.min_value}, {param_spec.max_value}]"
                )
        
        return len(errors) == 0, errors


@dataclass 
class PolicySchema:
    """
    Schema defining all available transforms and their specifications.
    
    This is the master registry of transforms that can be used in policies.
    """
    
    transforms: dict[str, TransformSpec] = field(default_factory=dict)
    
    def register(self, spec: TransformSpec) -> None:
        """Register a transform specification."""
        self.transforms[spec.name] = spec
    
    def get(self, name: str) -> TransformSpec | None:
        """Get a transform specification by name."""
        return self.transforms.get(name)
    
    def list_by_category(self, category: TransformCategory) -> list[TransformSpec]:
        """Get all transforms in a category."""
        return [t for t in self.transforms.values() if t.category == category]
    
    def list_safe_for_segmentation(self) -> list[TransformSpec]:
        """Get all transforms that are safe for segmentation tasks."""
        return [t for t in self.transforms.values() if t.is_safe_for_segmentation]


# ============================================================================
# Default Transform Specifications (Albumentations-compatible)
# ============================================================================

def create_default_schema() -> PolicySchema:
    """Create the default schema with common transforms."""
    schema = PolicySchema()
    
    # Flip transforms
    schema.register(TransformSpec(
        name="HorizontalFlip",
        category=TransformCategory.FLIP,
        description="Flip the image horizontally (left-right)",
        parameters={},
    ))
    
    schema.register(TransformSpec(
        name="VerticalFlip",
        category=TransformCategory.FLIP,
        description="Flip the image vertically (top-bottom)",
        parameters={},
    ))
    
    # Rotation transforms
    schema.register(TransformSpec(
        name="Rotate",
        category=TransformCategory.ROTATE,
        description="Rotate the image by a random angle",
        parameters={
            "limit": ParameterRange("limit", -180, 180, 45, "int", "Maximum rotation angle"),
        },
    ))
    
    schema.register(TransformSpec(
        name="RandomRotate90",
        category=TransformCategory.ROTATE,
        description="Rotate the image by 90 degrees (0, 1, 2, or 3 times)",
        parameters={},
    ))
    
    # Color transforms
    schema.register(TransformSpec(
        name="RandomBrightnessContrast",
        category=TransformCategory.COLOR,
        description="Randomly adjust brightness and contrast",
        parameters={
            "brightness_limit": ParameterRange("brightness_limit", 0.0, 0.5, 0.2, "float"),
            "contrast_limit": ParameterRange("contrast_limit", 0.0, 0.5, 0.2, "float"),
        },
    ))
    
    schema.register(TransformSpec(
        name="HueSaturationValue",
        category=TransformCategory.COLOR,
        description="Randomly adjust hue, saturation, and value",
        parameters={
            "hue_shift_limit": ParameterRange("hue_shift_limit", 0, 180, 20, "int"),
            "sat_shift_limit": ParameterRange("sat_shift_limit", 0, 100, 30, "int"),
            "val_shift_limit": ParameterRange("val_shift_limit", 0, 100, 20, "int"),
        },
        is_safe_for_segmentation=True,
    ))
    
    schema.register(TransformSpec(
        name="ColorJitter",
        category=TransformCategory.COLOR,
        description="Apply random color jittering",
        parameters={
            "brightness": ParameterRange("brightness", 0.0, 1.0, 0.2, "float"),
            "contrast": ParameterRange("contrast", 0.0, 1.0, 0.2, "float"),
            "saturation": ParameterRange("saturation", 0.0, 1.0, 0.2, "float"),
            "hue": ParameterRange("hue", 0.0, 0.5, 0.1, "float"),
        },
    ))
    
    # Blur transforms
    schema.register(TransformSpec(
        name="GaussianBlur",
        category=TransformCategory.BLUR,
        description="Apply Gaussian blur",
        parameters={
            "blur_limit": ParameterRange("blur_limit", 3, 15, 7, "int", "Maximum kernel size (must be odd)"),
        },
    ))
    
    schema.register(TransformSpec(
        name="MotionBlur",
        category=TransformCategory.BLUR,
        description="Apply motion blur",
        parameters={
            "blur_limit": ParameterRange("blur_limit", 3, 15, 7, "int"),
        },
    ))
    
    # Noise transforms
    schema.register(TransformSpec(
        name="GaussNoise",
        category=TransformCategory.NOISE,
        description="Add Gaussian noise",
        parameters={
            "var_limit": ParameterRange("var_limit", 0.0, 0.1, 0.02, "float", "Variance range"),
        },
    ))
    
    schema.register(TransformSpec(
        name="ISONoise",
        category=TransformCategory.NOISE,
        description="Apply camera sensor noise (ISO noise)",
        parameters={
            "intensity": ParameterRange("intensity", 0.0, 1.0, 0.5, "float"),
        },
    ))
    
    # Geometric transforms
    schema.register(TransformSpec(
        name="ShiftScaleRotate",
        category=TransformCategory.GEOMETRIC,
        description="Randomly shift, scale, and rotate the image",
        parameters={
            "shift_limit": ParameterRange("shift_limit", 0.0, 0.3, 0.1, "float"),
            "scale_limit": ParameterRange("scale_limit", 0.0, 0.3, 0.1, "float"),
            "rotate_limit": ParameterRange("rotate_limit", 0, 180, 45, "int"),
        },
    ))
    
    schema.register(TransformSpec(
        name="Affine",
        category=TransformCategory.GEOMETRIC,
        description="Apply affine transformation",
        parameters={
            "scale": ParameterRange("scale", 0.5, 1.5, 1.0, "float"),
            "rotate": ParameterRange("rotate", -180, 180, 0, "int"),
            "shear": ParameterRange("shear", -30, 30, 0, "int"),
        },
    ))
    
    # Distortion transforms
    schema.register(TransformSpec(
        name="ElasticTransform",
        category=TransformCategory.DISTORTION,
        description="Apply elastic deformation",
        parameters={
            "alpha": ParameterRange("alpha", 1, 500, 120, "int"),
            "sigma": ParameterRange("sigma", 1, 50, 12, "int"),
        },
        is_safe_for_segmentation=False,  # Can break anatomical structures
    ))
    
    schema.register(TransformSpec(
        name="GridDistortion",
        category=TransformCategory.DISTORTION,
        description="Apply grid-based distortion",
        parameters={
            "distort_limit": ParameterRange("distort_limit", 0.0, 0.5, 0.3, "float"),
        },
        is_safe_for_segmentation=False,
    ))
    
    schema.register(TransformSpec(
        name="OpticalDistortion",
        category=TransformCategory.DISTORTION,
        description="Apply optical (barrel/pincushion) distortion",
        parameters={
            "distort_limit": ParameterRange("distort_limit", 0.0, 1.0, 0.5, "float"),
        },
    ))
    
    # Crop transforms
    schema.register(TransformSpec(
        name="RandomCrop",
        category=TransformCategory.CROP,
        description="Randomly crop a region from the image",
        parameters={
            "height": ParameterRange("height", 32, 1024, 256, "int"),
            "width": ParameterRange("width", 32, 1024, 256, "int"),
        },
    ))
    
    schema.register(TransformSpec(
        name="CenterCrop",
        category=TransformCategory.CROP,
        description="Crop the center region of the image",
        parameters={
            "height": ParameterRange("height", 32, 1024, 256, "int"),
            "width": ParameterRange("width", 32, 1024, 256, "int"),
        },
    ))
    
    # Scale transforms
    schema.register(TransformSpec(
        name="RandomScale",
        category=TransformCategory.SCALE,
        description="Randomly scale the image",
        parameters={
            "scale_limit": ParameterRange("scale_limit", 0.0, 0.5, 0.1, "float"),
        },
    ))
    
    schema.register(TransformSpec(
        name="Resize",
        category=TransformCategory.SCALE,
        description="Resize image to specified dimensions",
        parameters={
            "height": ParameterRange("height", 32, 2048, 256, "int"),
            "width": ParameterRange("width", 32, 2048, 256, "int"),
        },
    ))
    
    return schema


# Global default schema instance
DEFAULT_SCHEMA = create_default_schema()

"""
Distribution shift generator for stress-testing generalization.

Generates controlled distribution shifts to test model robustness
before deployment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from copy import deepcopy

import numpy as np

from augmentai.core.policy import Transform


@dataclass
class ShiftConfig:
    """Configuration for a distribution shift.
    
    Attributes:
        name: Human-readable shift name
        shift_type: Type of shift (covariate, label, domain)
        severity: Shift intensity (0=none, 1=extreme)
        transforms: List of transforms to apply
        description: Description of the shift
    """
    name: str
    shift_type: Literal["covariate", "label", "domain"] = "covariate"
    severity: float = 0.5
    transforms: list[Transform] = field(default_factory=list)
    description: str = ""
    
    def __post_init__(self) -> None:
        """Validate severity."""
        self.severity = max(0.0, min(1.0, self.severity))
    
    def with_severity(self, severity: float) -> "ShiftConfig":
        """Create a copy with different severity."""
        new = deepcopy(self)
        new.severity = max(0.0, min(1.0, severity))
        # Scale transform parameters
        for t in new.transforms:
            new._scale_transform(t, severity)
        return new
    
    def _scale_transform(self, transform: Transform, severity: float) -> None:
        """Scale transform parameters by severity."""
        scalable = {"limit", "var_limit", "blur_limit", "brightness_limit", 
                   "contrast_limit", "sigma", "alpha"}
        
        for key in scalable:
            if key in transform.parameters:
                val = transform.parameters[key]
                if isinstance(val, (int, float)):
                    transform.parameters[key] = val * severity
                elif isinstance(val, tuple) and len(val) == 2:
                    transform.parameters[key] = (val[0] * severity, val[1] * severity)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "shift_type": self.shift_type,
            "severity": self.severity,
            "transforms": [t.to_dict() for t in self.transforms],
            "description": self.description,
        }


class ShiftGenerator:
    """Generate controlled distribution shifts for robustness testing.
    
    Provides predefined shift configurations for common distribution shifts
    and tools to apply them to images.
    
    Example:
        generator = ShiftGenerator(domain="natural")
        
        # Apply brightness shift at 50% severity
        shifted_images = generator.generate_shifted_samples(
            images,
            generator.get_shift("brightness"),
            output_dir,
        )
        
        # Sweep across severities
        severity_results = generator.generate_severity_sweep(
            images,
            "blur",
            severities=[0.1, 0.3, 0.5, 0.7, 0.9],
            output_dir,
        )
    """
    
    # Predefined shift configurations
    SHIFTS = {
        "brightness": ShiftConfig(
            name="brightness",
            shift_type="covariate",
            severity=0.5,
            transforms=[
                Transform("RandomBrightnessContrast", 1.0, 
                         parameters={"brightness_limit": 0.4, "contrast_limit": 0}),
            ],
            description="Brightness changes simulating lighting conditions",
        ),
        "contrast": ShiftConfig(
            name="contrast",
            shift_type="covariate",
            severity=0.5,
            transforms=[
                Transform("RandomBrightnessContrast", 1.0,
                         parameters={"brightness_limit": 0, "contrast_limit": 0.4}),
            ],
            description="Contrast changes simulating camera/scanner differences",
        ),
        "noise": ShiftConfig(
            name="noise",
            shift_type="covariate",
            severity=0.5,
            transforms=[
                Transform("GaussNoise", 1.0, parameters={"var_limit": (50, 100)}),
            ],
            description="Gaussian noise simulating sensor noise",
        ),
        "blur": ShiftConfig(
            name="blur",
            shift_type="covariate",
            severity=0.5,
            transforms=[
                Transform("GaussianBlur", 1.0, parameters={"blur_limit": (7, 15)}),
            ],
            description="Blur simulating focus issues or motion",
        ),
        "compression": ShiftConfig(
            name="compression",
            shift_type="covariate",
            severity=0.5,
            transforms=[
                Transform("ImageCompression", 1.0, 
                         parameters={"quality_lower": 20, "quality_upper": 50}),
            ],
            description="JPEG compression artifacts",
        ),
        "color": ShiftConfig(
            name="color",
            shift_type="covariate",
            severity=0.5,
            transforms=[
                Transform("HueSaturationValue", 1.0,
                         parameters={"hue_shift_limit": 30, "sat_shift_limit": 40}),
            ],
            description="Color shifts simulating different lighting/cameras",
        ),
        "combined_mild": ShiftConfig(
            name="combined_mild",
            shift_type="domain",
            severity=0.3,
            transforms=[
                Transform("RandomBrightnessContrast", 0.5,
                         parameters={"brightness_limit": 0.2, "contrast_limit": 0.2}),
                Transform("GaussNoise", 0.3, parameters={"var_limit": (10, 30)}),
                Transform("GaussianBlur", 0.2, parameters={"blur_limit": (3, 5)}),
            ],
            description="Mild combined shift simulating real-world variations",
        ),
        "combined_severe": ShiftConfig(
            name="combined_severe",
            shift_type="domain",
            severity=0.7,
            transforms=[
                Transform("RandomBrightnessContrast", 0.8,
                         parameters={"brightness_limit": 0.5, "contrast_limit": 0.5}),
                Transform("GaussNoise", 0.6, parameters={"var_limit": (50, 100)}),
                Transform("GaussianBlur", 0.5, parameters={"blur_limit": (7, 15)}),
                Transform("ImageCompression", 0.4, 
                         parameters={"quality_lower": 20, "quality_upper": 40}),
            ],
            description="Severe combined shift for stress testing",
        ),
    }
    
    def __init__(
        self,
        domain: str = "natural",
        seed: int = 42,
    ):
        """Initialize shift generator.
        
        Args:
            domain: Domain context (affects which shifts are appropriate)
            seed: Random seed for reproducibility
        """
        self.domain = domain
        self.seed = seed
        self._rng = np.random.default_rng(seed)
    
    def get_shift(self, name: str) -> ShiftConfig:
        """Get a predefined shift configuration."""
        if name not in self.SHIFTS:
            available = ", ".join(self.SHIFTS.keys())
            raise ValueError(f"Unknown shift: {name}. Available: {available}")
        return deepcopy(self.SHIFTS[name])
    
    def list_shifts(self) -> list[str]:
        """List available shift names."""
        return list(self.SHIFTS.keys())
    
    def generate_shifted_samples(
        self,
        samples: list[Path],
        shift: ShiftConfig,
        output_dir: Path,
    ) -> list[Path]:
        """Apply shift to samples and save to output directory.
        
        Args:
            samples: Paths to input images
            shift: Shift configuration to apply
            output_dir: Directory to save shifted images
            
        Returns:
            Paths to shifted images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        
        for sample_path in samples:
            output_path = output_dir / f"{sample_path.stem}_shifted{sample_path.suffix}"
            
            try:
                # Apply shift
                shifted = self._apply_shift(sample_path, shift)
                
                # Save
                self._save_image(shifted, output_path)
                output_paths.append(output_path)
            except Exception as e:
                # Skip failed samples
                print(f"Warning: Failed to shift {sample_path}: {e}")
        
        return output_paths
    
    def generate_severity_sweep(
        self,
        samples: list[Path],
        shift_name: str,
        severities: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
        output_dir: Path = Path("./shift_sweep"),
    ) -> dict[float, list[Path]]:
        """Generate samples at multiple severity levels.
        
        Args:
            samples: Input image paths
            shift_name: Name of shift to apply
            severities: List of severity levels to test
            output_dir: Base output directory
            
        Returns:
            Dict mapping severity to list of shifted image paths
        """
        output_dir = Path(output_dir)
        base_shift = self.get_shift(shift_name)
        
        results = {}
        for severity in severities:
            shift = base_shift.with_severity(severity)
            sev_dir = output_dir / f"severity_{severity:.1f}"
            shifted_paths = self.generate_shifted_samples(samples, shift, sev_dir)
            results[severity] = shifted_paths
        
        return results
    
    def _apply_shift(self, image_path: Path, shift: ShiftConfig) -> np.ndarray:
        """Apply shift transforms to an image."""
        try:
            import albumentations as A
            from PIL import Image
            
            # Load image
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Build albumentations pipeline
            transform_list = []
            for t in shift.transforms:
                alb_transform = self._get_albumentations_transform(t)
                if alb_transform is not None:
                    transform_list.append(alb_transform)
            
            if transform_list:
                pipeline = A.Compose(transform_list)
                result = pipeline(image=img_array)
                return result["image"]
            
            return img_array
            
        except ImportError:
            # Fallback: just return original if albumentations not available
            from PIL import Image
            img = Image.open(image_path)
            return np.array(img)
    
    def _get_albumentations_transform(self, transform: Transform):
        """Convert Transform to albumentations transform."""
        try:
            import albumentations as A
            
            mapping = {
                "RandomBrightnessContrast": A.RandomBrightnessContrast,
                "GaussNoise": A.GaussNoise,
                "GaussianBlur": A.GaussianBlur,
                "ImageCompression": A.ImageCompression,
                "HueSaturationValue": A.HueSaturationValue,
            }
            
            if transform.name in mapping:
                return mapping[transform.name](
                    p=transform.probability,
                    **transform.parameters,
                )
            return None
        except ImportError:
            return None
    
    def _save_image(self, img_array: np.ndarray, path: Path) -> None:
        """Save image array to file."""
        from PIL import Image
        img = Image.fromarray(img_array.astype(np.uint8))
        img.save(path)
    
    def save_shift_config(
        self,
        shift: ShiftConfig,
        output_path: Path,
    ) -> Path:
        """Save shift configuration to JSON."""
        output_path = Path(output_path)
        output_path.write_text(
            json.dumps(shift.to_dict(), indent=2),
            encoding="utf-8"
        )
        return output_path

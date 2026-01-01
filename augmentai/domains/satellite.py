"""
Satellite and aerial imagery domain.

Constraints for remote sensing data including multi-spectral
and geospatial considerations.
"""

from augmentai.core.policy import TransformCategory
from augmentai.domains.base import Domain, DomainConstraint, ConstraintLevel


class SatelliteDomain(Domain):
    """
    Domain for satellite and aerial imagery.
    
    Satellite imagery has unique properties:
    - Any rotation angle is valid (no "up" direction)
    - Multi-spectral data (more than RGB channels)
    - Spectral band relationships must be preserved
    - Scale variation is common and valid
    
    Constraints focus on preserving spectral integrity
    while allowing geometric flexibility.
    """
    
    name = "satellite"
    description = "Satellite and aerial imagery with spectral band preservation"
    
    def _setup(self) -> None:
        """Configure satellite domain constraints."""
        
        # === FORBIDDEN TRANSFORMS ===
        
        self.add_constraint(DomainConstraint(
            transform_name="ColorJitter",
            level=ConstraintLevel.FORBIDDEN,
            reason="Color jitter breaks spectral band relationships needed for "
                   "vegetation indices, water detection, etc."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="HueSaturationValue",
            level=ConstraintLevel.FORBIDDEN,
            reason="HSV transforms are meaningless for multi-spectral data and "
                   "break band calibration."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="RGBShift",
            level=ConstraintLevel.FORBIDDEN,
            reason="RGB shift destroys carefully calibrated spectral relationships."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="ChannelShuffle",
            level=ConstraintLevel.FORBIDDEN,
            reason="Channel shuffling makes spectral indices impossible to compute."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="ToGray",
            level=ConstraintLevel.FORBIDDEN,
            reason="Grayscale conversion loses all multi-spectral information."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="ToSepia",
            level=ConstraintLevel.FORBIDDEN,
            reason="Color transformation destroys spectral calibration."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="FancyPCA",
            level=ConstraintLevel.FORBIDDEN,
            reason="PCA color augmentation is not valid for calibrated spectral data."
        ))
        
        # === DISCOURAGED TRANSFORMS ===
        
        self.add_constraint(DomainConstraint(
            transform_name="CLAHE",
            level=ConstraintLevel.DISCOURAGED,
            reason="CLAHE changes per-channel statistics independently, "
                   "potentially breaking spectral ratios."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="Equalize",
            level=ConstraintLevel.DISCOURAGED,
            reason="Histogram equalization destroys calibrated reflectance values."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="Posterize",
            level=ConstraintLevel.DISCOURAGED,
            reason="Posterization destroys continuous spectral values."
        ))
        
        # === SAFE TRANSFORMS ===
        # Satellite data is very flexible for geometric augmentations
        
        self.add_constraint(DomainConstraint(
            transform_name="Rotate",
            level=ConstraintLevel.RECOMMENDED,
            reason="Any rotation is valid - satellite imagery has no fixed orientation.",
            parameter_limits={"limit": (-180, 180)}  # Full rotation range
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="HorizontalFlip",
            level=ConstraintLevel.RECOMMENDED,
            reason="Flipping is always valid for overhead imagery."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="VerticalFlip",
            level=ConstraintLevel.RECOMMENDED,
            reason="Flipping is always valid for overhead imagery."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="RandomRotate90",
            level=ConstraintLevel.RECOMMENDED,
            reason="90-degree rotations are safe and efficient."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="RandomScale",
            level=ConstraintLevel.RECOMMENDED,
            reason="Scale augmentation simulates different ground sampling distances.",
            parameter_limits={"scale_limit": (0.0, 0.5)}
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="ShiftScaleRotate",
            level=ConstraintLevel.RECOMMENDED,
            reason="Combined geometric transforms are all valid.",
            parameter_limits={
                "shift_limit": (0.0, 0.2),
                "scale_limit": (0.0, 0.3),
                "rotate_limit": (-180, 180)
            }
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="RandomBrightnessContrast",
            level=ConstraintLevel.RECOMMENDED,
            reason="Simulates atmospheric and sensor variations.",
            parameter_limits={
                "brightness_limit": (0.0, 0.2),
                "contrast_limit": (0.0, 0.2)
            }
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="GaussNoise",
            level=ConstraintLevel.RECOMMENDED,
            reason="Simulates sensor noise.",
            parameter_limits={"var_limit": (0.0, 0.05)}
        ))
        
        # === RECOMMENDED TRANSFORMS ===
        
        self.recommended_transforms.add("HorizontalFlip")
        self.recommended_transforms.add("VerticalFlip")
        self.recommended_transforms.add("RandomRotate90")
        self.recommended_transforms.add("Rotate")
        self.recommended_transforms.add("RandomScale")
        self.recommended_transforms.add("ShiftScaleRotate")
        self.recommended_transforms.add("RandomBrightnessContrast")
        self.recommended_transforms.add("GaussNoise")


class MultiSpectralDomain(SatelliteDomain):
    """
    Specialized domain for multi-spectral satellite data (>3 channels).
    
    Extra constraints for handling more than RGB channels, such as
    NIR, SWIR, thermal bands.
    """
    
    name = "multispectral"
    description = "Multi-spectral satellite imagery (4+ bands) with band relationship preservation"
    
    def _setup(self) -> None:
        """Set up multi-spectral specific constraints."""
        super()._setup()
        
        # Additional multi-spectral constraints
        self.add_constraint(DomainConstraint(
            transform_name="Normalize",
            level=ConstraintLevel.DISCOURAGED,
            reason="Per-channel normalization must preserve band relationships. "
                   "Use per-image or dataset-level statistics."
        ))
        
        # Even stricter on brightness/contrast
        for i, c in enumerate(self.constraints):
            if c.transform_name == "RandomBrightnessContrast":
                self.constraints[i] = DomainConstraint(
                    transform_name="RandomBrightnessContrast",
                    level=ConstraintLevel.RECOMMENDED,
                    reason="Must apply uniformly across all bands to preserve relationships.",
                    parameter_limits={
                        "brightness_limit": (0.0, 0.1),
                        "contrast_limit": (0.0, 0.1)
                    }
                )
                break

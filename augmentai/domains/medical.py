"""
Medical imaging domain with strict safety constraints.

This domain enforces hard constraints that protect anatomical integrity
and prevent scientifically invalid augmentations for medical images.
"""

from augmentai.core.policy import TransformCategory
from augmentai.domains.base import Domain, DomainConstraint, ConstraintLevel


class MedicalDomain(Domain):
    """
    Domain for medical imaging (CT, MRI, X-ray, pathology).
    
    This domain has the strictest constraints because incorrect augmentations
    can lead to anatomically invalid training data, potentially affecting
    diagnostic accuracy of trained models.
    
    Hard Constraints:
    - No heavy geometric distortions (breaks anatomy)
    - No arbitrary color changes (HU values, contrast values are meaningful)
    - Conservative rotation limits
    - Must preserve label synchronization for segmentation
    """
    
    name = "medical"
    description = "Medical imaging (CT, MRI, X-ray, pathology) with strict anatomical constraints"
    
    def _setup(self) -> None:
        """Configure medical domain constraints."""
        
        # === FORBIDDEN TRANSFORMS ===
        # These can never be used in medical imaging
        
        self.add_constraint(DomainConstraint(
            transform_name="ElasticTransform",
            level=ConstraintLevel.FORBIDDEN,
            reason="Elastic deformation can break anatomical structures and create "
                   "physically impossible organ shapes. Never use for medical imaging."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="GridDistortion",
            level=ConstraintLevel.FORBIDDEN,
            reason="Grid distortion creates non-anatomical deformations that can "
                   "mislead diagnostic models."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="OpticalDistortion",
            level=ConstraintLevel.FORBIDDEN,
            reason="Optical distortion is not physically valid for medical scans "
                   "and can alter apparent lesion sizes."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="ColorJitter",
            level=ConstraintLevel.FORBIDDEN,
            reason="Color values in medical images (HU for CT, signal intensity "
                   "for MRI) have diagnostic meaning. Arbitrary color changes are invalid."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="HueSaturationValue",
            level=ConstraintLevel.FORBIDDEN,
            reason="HSV changes are not applicable to grayscale medical images "
                   "and can corrupt multi-channel data."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="RGBShift",
            level=ConstraintLevel.FORBIDDEN,
            reason="RGB channel operations are invalid for grayscale medical scans."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="ChannelShuffle",
            level=ConstraintLevel.FORBIDDEN,
            reason="Channel shuffling destroys the meaning of multi-sequence MRI or spectral data."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="Posterize",
            level=ConstraintLevel.FORBIDDEN,
            reason="Posterization destroys the continuous intensity values needed for diagnosis."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="Solarize",
            level=ConstraintLevel.FORBIDDEN,
            reason="Solarization inverts intensity values, making images diagnostically invalid."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="CoarseDropout",
            level=ConstraintLevel.FORBIDDEN,
            reason="Dropping regions can hide pathology and create misleading training data."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="Cutout",
            level=ConstraintLevel.FORBIDDEN,
            reason="Cutout can remove diagnostic regions from medical images."
        ))
        
        # === DISCOURAGED TRANSFORMS ===
        # These should be used with extreme caution
        
        self.add_constraint(DomainConstraint(
            transform_name="MotionBlur",
            level=ConstraintLevel.DISCOURAGED,
            reason="Motion blur is rarely appropriate for medical imaging. "
                   "Only use if simulating specific acquisition artifacts."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="GaussianBlur",
            level=ConstraintLevel.DISCOURAGED,
            reason="Blur can hide fine diagnostic details. Use very conservatively.",
            parameter_limits={"blur_limit": (3, 5)}  # Restrict to minimal blur
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="Sharpen",
            level=ConstraintLevel.DISCOURAGED,
            reason="Over-sharpening can create artifacts that look like pathology."
        ))
        
        # === SAFE TRANSFORMS WITH PARAMETER LIMITS ===
        
        self.add_constraint(DomainConstraint(
            transform_name="Rotate",
            level=ConstraintLevel.RECOMMENDED,
            reason="Mild rotation is safe for most medical images.",
            parameter_limits={"limit": (-15, 15)}  # Conservative rotation
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="RandomBrightnessContrast",
            level=ConstraintLevel.RECOMMENDED,
            reason="Mild intensity variations simulate scanner differences.",
            parameter_limits={
                "brightness_limit": (0.0, 0.1),
                "contrast_limit": (0.0, 0.1)
            }
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="GaussNoise",
            level=ConstraintLevel.RECOMMENDED,
            reason="Mild Gaussian noise simulates scanner noise.",
            parameter_limits={"var_limit": (0.0, 0.02)}
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="RandomScale",
            level=ConstraintLevel.RECOMMENDED,
            reason="Mild scaling simulates patient size variability.",
            parameter_limits={"scale_limit": (0.0, 0.1)}
        ))
        
        # === RECOMMENDED TRANSFORMS ===
        
        self.recommended_transforms.add("HorizontalFlip")
        self.recommended_transforms.add("VerticalFlip")  # For axial slices
        self.recommended_transforms.add("RandomRotate90")  # Safe for square slices
        self.recommended_transforms.add("RandomBrightnessContrast")
        self.recommended_transforms.add("GaussNoise")
        
        # === FORBIDDEN CATEGORIES ===
        
        self.forbidden_categories.add(TransformCategory.DISTORTION)


class CTSegmentationDomain(MedicalDomain):
    """
    Specialized domain for CT segmentation tasks.
    
    Inherits all medical constraints plus additional restrictions
    specific to CT Hounsfield unit preservation and segmentation.
    """
    
    name = "ct_segmentation"
    description = "CT scan segmentation with HU preservation and mask synchronization"
    
    def _setup(self) -> None:
        """Set up CT segmentation specific constraints."""
        # Inherit base medical constraints
        super()._setup()
        
        # Additional CT-specific constraints
        self.add_constraint(DomainConstraint(
            transform_name="CLAHE",
            level=ConstraintLevel.DISCOURAGED,
            reason="CLAHE can shift HU values and affect windowing-dependent features."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="Equalize",
            level=ConstraintLevel.FORBIDDEN,
            reason="Histogram equalization destroys HU calibration."
        ))


class MRIDomain(MedicalDomain):
    """
    Specialized domain for MRI images.
    
    MRI has different constraints than CT due to different
    acquisition physics and intensity meanings.
    """
    
    name = "mri"
    description = "MRI imaging with intensity and multi-sequence constraints"
    
    def _setup(self) -> None:
        """Set up MRI-specific constraints."""
        # Inherit base medical constraints
        super()._setup()
        
        # MRI-specific additions
        self.add_constraint(DomainConstraint(
            transform_name="ISONoise",
            level=ConstraintLevel.RECOMMENDED,
            reason="ISO noise can simulate MRI acquisition noise patterns.",
            parameter_limits={"intensity": (0.0, 0.3)}
        ))
        
        self.recommended_transforms.add("ISONoise")

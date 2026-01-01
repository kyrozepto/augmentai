"""
Natural image domain.

The most permissive domain - standard RGB photographs with
minimal constraints. Good for general classification tasks.
"""

from augmentai.core.policy import TransformCategory
from augmentai.domains.base import Domain, DomainConstraint, ConstraintLevel


class NaturalDomain(Domain):
    """
    Domain for natural RGB photographs.
    
    This is the most permissive domain, suitable for general
    image classification, object detection, and similar tasks
    on standard photographs.
    
    Most augmentations are allowed since natural images don't
    have the strict constraints of medical or scientific imagery.
    """
    
    name = "natural"
    description = "Natural RGB photographs with minimal constraints"
    
    def _setup(self) -> None:
        """Configure natural image domain constraints."""
        
        # Natural images allow almost everything
        # Only constrain things that could break training
        
        # === DISCOURAGED TRANSFORMS ===
        # These might hurt training if overused
        
        self.add_constraint(DomainConstraint(
            transform_name="Solarize",
            level=ConstraintLevel.DISCOURAGED,
            reason="Solarization creates unnatural images that may not generalize well."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="Posterize",
            level=ConstraintLevel.DISCOURAGED,
            reason="Heavy posterization loses natural color gradients."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="Superpixels",
            level=ConstraintLevel.DISCOURAGED,
            reason="Superpixel conversion can create artifacts."
        ))
        
        # === RECOMMENDED TRANSFORMS ===
        # Standard augmentations that work well for natural images
        
        self.add_constraint(DomainConstraint(
            transform_name="HorizontalFlip",
            level=ConstraintLevel.RECOMMENDED,
            reason="Standard augmentation for most natural image tasks."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="Rotate",
            level=ConstraintLevel.RECOMMENDED,
            reason="Rotation adds rotational invariance.",
            parameter_limits={"limit": (-45, 45)}
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="RandomBrightnessContrast",
            level=ConstraintLevel.RECOMMENDED,
            reason="Simulates different lighting conditions.",
            parameter_limits={
                "brightness_limit": (0.0, 0.3),
                "contrast_limit": (0.0, 0.3)
            }
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="HueSaturationValue",
            level=ConstraintLevel.RECOMMENDED,
            reason="Color variation is beneficial for RGB images.",
            parameter_limits={
                "hue_shift_limit": (0, 30),
                "sat_shift_limit": (0, 40),
                "val_shift_limit": (0, 30)
            }
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="ColorJitter",
            level=ConstraintLevel.RECOMMENDED,
            reason="Standard color augmentation for natural images."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="RandomScale",
            level=ConstraintLevel.RECOMMENDED,
            reason="Scale invariance is useful for object detection.",
            parameter_limits={"scale_limit": (0.0, 0.3)}
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="ShiftScaleRotate",
            level=ConstraintLevel.RECOMMENDED,
            reason="Efficient combined geometric augmentation."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="GaussNoise",
            level=ConstraintLevel.RECOMMENDED,
            reason="Adds robustness to sensor noise."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="GaussianBlur",
            level=ConstraintLevel.RECOMMENDED,
            reason="Adds robustness to focus variation."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="RandomCrop",
            level=ConstraintLevel.RECOMMENDED,
            reason="Standard cropping for data diversity."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="CoarseDropout",
            level=ConstraintLevel.RECOMMENDED,
            reason="Cutout/dropout regularization helps generalization."
        ))
        
        # Add to recommended set
        self.recommended_transforms.add("HorizontalFlip")
        self.recommended_transforms.add("Rotate")
        self.recommended_transforms.add("RandomBrightnessContrast")
        self.recommended_transforms.add("HueSaturationValue")
        self.recommended_transforms.add("ColorJitter")
        self.recommended_transforms.add("RandomScale")
        self.recommended_transforms.add("ShiftScaleRotate")
        self.recommended_transforms.add("GaussNoise")
        self.recommended_transforms.add("GaussianBlur")
        self.recommended_transforms.add("RandomCrop")
        self.recommended_transforms.add("CoarseDropout")


class ObjectDetectionDomain(NaturalDomain):
    """
    Specialized domain for object detection tasks.
    
    Similar to natural images but with bounding box awareness.
    """
    
    name = "object_detection"
    description = "Object detection with bounding box preservation"
    
    def _setup(self) -> None:
        """Set up object detection specific constraints."""
        super()._setup()
        
        # Additional safety note for bbox transforms
        self.add_constraint(DomainConstraint(
            transform_name="RandomCrop",
            level=ConstraintLevel.RECOMMENDED,
            reason="Ensure minimum overlap with bounding boxes is enforced.",
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="Cutout",
            level=ConstraintLevel.DISCOURAGED,
            reason="Cutout may occlude small objects entirely."
        ))


class SegmentationDomain(NaturalDomain):
    """
    Specialized domain for semantic segmentation tasks.
    
    Ensures mask synchronization and segment-safe transforms.
    """
    
    name = "segmentation"
    description = "Semantic segmentation with mask synchronization"
    
    def _setup(self) -> None:
        """Set up segmentation specific constraints."""
        super()._setup()
        
        # Transforms that might not sync well with masks
        self.add_constraint(DomainConstraint(
            transform_name="ElasticTransform",
            level=ConstraintLevel.DISCOURAGED,
            reason="Elastic transform can create thin segments that are hard to predict."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="GridDistortion",
            level=ConstraintLevel.DISCOURAGED,
            reason="May create fragmented segments."
        ))

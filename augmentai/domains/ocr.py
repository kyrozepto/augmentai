"""
OCR and document imaging domain.

Constraints for text recognition and document analysis tasks,
where character integrity must be preserved.
"""

from augmentai.core.policy import TransformCategory
from augmentai.domains.base import Domain, DomainConstraint, ConstraintLevel


class OCRDomain(Domain):
    """
    Domain for OCR and document analysis.
    
    The primary constraint is preserving text legibility and character
    structure. Geometric transforms must not break character shapes.
    
    Hard Constraints:
    - No heavy blur that destroys text
    - No elastic distortion that warps characters
    - Limited rotation to prevent diagonal text (hard to recognize)
    """
    
    name = "ocr"
    description = "OCR and document analysis with character preservation constraints"
    
    def _setup(self) -> None:
        """Configure OCR domain constraints."""
        
        # === FORBIDDEN TRANSFORMS ===
        
        self.add_constraint(DomainConstraint(
            transform_name="ElasticTransform",
            level=ConstraintLevel.FORBIDDEN,
            reason="Elastic deformation warps character shapes, making them unrecognizable."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="GridDistortion",
            level=ConstraintLevel.FORBIDDEN,
            reason="Grid distortion breaks text line structure and character shapes."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="MotionBlur",
            level=ConstraintLevel.FORBIDDEN,
            reason="Motion blur makes text illegible by smearing characters."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="Defocus",
            level=ConstraintLevel.FORBIDDEN,
            reason="Defocus destroys fine character details needed for recognition."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="ZoomBlur",
            level=ConstraintLevel.FORBIDDEN,
            reason="Zoom blur makes text completely unreadable."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="Morphological",
            level=ConstraintLevel.FORBIDDEN,
            reason="Morphological operations can close or erode character strokes."
        ))
        
        # === DISCOURAGED TRANSFORMS ===
        
        self.add_constraint(DomainConstraint(
            transform_name="GaussianBlur",
            level=ConstraintLevel.DISCOURAGED,
            reason="Even mild blur can affect character edge detection.",
            parameter_limits={"blur_limit": (3, 5)}  # Only very mild if used
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="MedianBlur",
            level=ConstraintLevel.DISCOURAGED,
            reason="Median blur can affect thin character strokes."
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="Downscale",
            level=ConstraintLevel.DISCOURAGED,
            reason="Downscaling must be careful not to lose character detail."
        ))
        
        # === SAFE TRANSFORMS WITH PARAMETER LIMITS ===
        
        self.add_constraint(DomainConstraint(
            transform_name="Rotate",
            level=ConstraintLevel.RECOMMENDED,
            reason="Small rotations simulate scanned document skew.",
            parameter_limits={"limit": (-10, 10)}  # Keep text readable
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="Perspective",
            level=ConstraintLevel.RECOMMENDED,
            reason="Mild perspective simulates camera capture angle.",
            parameter_limits={"scale": (0.02, 0.08)}
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="RandomBrightnessContrast",
            level=ConstraintLevel.RECOMMENDED,
            reason="Simulates different lighting and paper conditions.",
            parameter_limits={
                "brightness_limit": (0.0, 0.3),
                "contrast_limit": (0.0, 0.3)
            }
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="GaussNoise",
            level=ConstraintLevel.RECOMMENDED,
            reason="Simulates scanner noise and paper texture.",
            parameter_limits={"var_limit": (0.0, 0.03)}
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="ShiftScaleRotate",
            level=ConstraintLevel.RECOMMENDED,
            reason="Combined transform for document variability.",
            parameter_limits={
                "shift_limit": (0.0, 0.1),
                "scale_limit": (0.0, 0.15),
                "rotate_limit": (-10, 10)
            }
        ))
        
        # === RECOMMENDED TRANSFORMS ===
        
        self.recommended_transforms.add("RandomBrightnessContrast")
        self.recommended_transforms.add("GaussNoise")
        self.recommended_transforms.add("Rotate")
        self.recommended_transforms.add("ShiftScaleRotate")
        self.recommended_transforms.add("ToGray")  # For color document handling
        
        # OCR benefits from text-specific augmentations
        self.recommended_transforms.add("CLAHE")  # Improves text contrast
        self.recommended_transforms.add("Sharpen")  # Can improve text edges


class HandwritingDomain(OCRDomain):
    """
    Specialized domain for handwriting recognition.
    
    Slightly more tolerant than printed OCR because handwriting
    has natural variability.
    """
    
    name = "handwriting"
    description = "Handwriting recognition with natural stroke variation tolerance"
    
    def _setup(self) -> None:
        """Set up handwriting-specific constraints."""
        super()._setup()
        
        # Handwriting can tolerate slightly more geometric variation
        self.add_constraint(DomainConstraint(
            transform_name="Rotate",
            level=ConstraintLevel.RECOMMENDED,
            reason="Handwriting naturally varies in slant.",
            parameter_limits={"limit": (-20, 20)}  # More tolerance than printed
        ))
        
        self.add_constraint(DomainConstraint(
            transform_name="Affine",
            level=ConstraintLevel.RECOMMENDED,
            reason="Affine transforms simulate natural writing variation.",
            parameter_limits={
                "scale": (0.9, 1.1),
                "shear": (-10, 10)
            }
        ))
        
        # Very mild elastic can simulate pen pressure variation
        # Override the FORBIDDEN status from parent
        for i, c in enumerate(self.constraints):
            if c.transform_name == "ElasticTransform":
                self.constraints[i] = DomainConstraint(
                    transform_name="ElasticTransform",
                    level=ConstraintLevel.DISCOURAGED,
                    reason="Very mild elastic can simulate pen pressure, but use carefully.",
                    parameter_limits={"alpha": (1, 50), "sigma": (5, 10)}
                )
                self.forbidden_transforms.discard("ElasticTransform")
                break

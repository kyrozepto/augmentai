"""Domain-specific rule definitions for augmentation policies."""

from augmentai.domains.base import Domain, DomainConstraint, ValidationResult
from augmentai.domains.medical import MedicalDomain
from augmentai.domains.ocr import OCRDomain
from augmentai.domains.satellite import SatelliteDomain
from augmentai.domains.natural import NaturalDomain

__all__ = [
    "Domain",
    "DomainConstraint", 
    "ValidationResult",
    "MedicalDomain",
    "OCRDomain",
    "SatelliteDomain",
    "NaturalDomain",
]

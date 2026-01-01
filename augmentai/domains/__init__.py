"""Domain-specific rule definitions for augmentation policies."""

from augmentai.domains.base import Domain, DomainConstraint, ValidationResult
from augmentai.domains.medical import MedicalDomain
from augmentai.domains.ocr import OCRDomain
from augmentai.domains.satellite import SatelliteDomain
from augmentai.domains.natural import NaturalDomain


# Registry of available domains
_DOMAINS = {
    "medical": MedicalDomain,
    "ocr": OCRDomain,
    "satellite": SatelliteDomain,
    "natural": NaturalDomain,
}


def get_domain(name: str) -> Domain:
    """
    Get a domain instance by name.
    
    Args:
        name: Domain name (medical, ocr, satellite, natural)
        
    Returns:
        Domain instance
        
    Raises:
        ValueError: If domain name is unknown
    """
    name_lower = name.lower()
    if name_lower not in _DOMAINS:
        available = ", ".join(_DOMAINS.keys())
        raise ValueError(f"Unknown domain: {name}. Available: {available}")
    return _DOMAINS[name_lower]()


def list_domains() -> list[str]:
    """Get list of available domain names."""
    return list(_DOMAINS.keys())


__all__ = [
    "Domain",
    "DomainConstraint", 
    "ValidationResult",
    "MedicalDomain",
    "OCRDomain",
    "SatelliteDomain",
    "NaturalDomain",
    "get_domain",
    "list_domains",
]


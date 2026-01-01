"""Tests for domain constraints and validation."""

import pytest

from augmentai.core.policy import Policy, Transform, TransformCategory
from augmentai.domains import MedicalDomain, OCRDomain, SatelliteDomain, NaturalDomain
from augmentai.domains.base import ConstraintLevel
from augmentai.rules.validator import SafetyValidator


class TestMedicalDomain:
    """Test medical domain constraints."""
    
    def test_forbids_elastic_transform(self):
        """ElasticTransform must be forbidden in medical domain."""
        domain = MedicalDomain()
        assert "ElasticTransform" in domain.forbidden_transforms
    
    def test_forbids_color_jitter(self):
        """ColorJitter must be forbidden in medical domain."""
        domain = MedicalDomain()
        assert "ColorJitter" in domain.forbidden_transforms
    
    def test_forbids_grid_distortion(self):
        """GridDistortion must be forbidden in medical domain."""
        domain = MedicalDomain()
        assert "GridDistortion" in domain.forbidden_transforms
    
    def test_recommends_horizontal_flip(self):
        """HorizontalFlip should be recommended."""
        domain = MedicalDomain()
        assert "HorizontalFlip" in domain.recommended_transforms
    
    def test_validate_forbidden_transform(self):
        """Validation should fail for forbidden transforms."""
        domain = MedicalDomain()
        
        transform = Transform(
            name="ElasticTransform",
            probability=0.5,
            parameters={"alpha": 120},
            category=TransformCategory.DISTORTION,
        )
        
        result = domain.validate_transform(transform)
        assert not result.is_valid
        assert len(result.errors) > 0
        assert "forbidden" in result.errors[0].lower()
    
    def test_validate_allowed_transform(self):
        """Validation should pass for allowed transforms."""
        domain = MedicalDomain()
        
        transform = Transform(
            name="HorizontalFlip",
            probability=0.5,
            category=TransformCategory.FLIP,
        )
        
        result = domain.validate_transform(transform)
        assert result.is_valid
    
    def test_validate_policy_removes_forbidden(self):
        """Safety validator should remove forbidden transforms in strict mode."""
        domain = MedicalDomain()
        validator = SafetyValidator(domain, strict=True)
        
        policy = Policy(
            name="test_policy",
            domain="medical",
            transforms=[
                Transform("HorizontalFlip", 0.5, category=TransformCategory.FLIP),
                Transform("ElasticTransform", 0.5, category=TransformCategory.DISTORTION),
                Transform("GaussNoise", 0.3, category=TransformCategory.NOISE),
            ]
        )
        
        result = validator.validate(policy)
        
        # Should still be valid after removing forbidden
        assert result.is_safe
        assert result.policy is not None
        assert len(result.policy.transforms) == 2  # ElasticTransform removed
        assert len(result.removed_transforms) == 1
        assert result.removed_transforms[0].name == "ElasticTransform"


class TestOCRDomain:
    """Test OCR domain constraints."""
    
    def test_forbids_elastic_transform(self):
        """ElasticTransform must be forbidden in OCR domain."""
        domain = OCRDomain()
        assert "ElasticTransform" in domain.forbidden_transforms
    
    def test_forbids_motion_blur(self):
        """MotionBlur must be forbidden in OCR domain."""
        domain = OCRDomain()
        assert "MotionBlur" in domain.forbidden_transforms
    
    def test_allows_rotate(self):
        """Rotate should be allowed with limits."""
        domain = OCRDomain()
        assert "Rotate" not in domain.forbidden_transforms


class TestSatelliteDomain:
    """Test satellite domain constraints."""
    
    def test_forbids_color_jitter(self):
        """ColorJitter must be forbidden for spectral integrity."""
        domain = SatelliteDomain()
        assert "ColorJitter" in domain.forbidden_transforms
    
    def test_forbids_channel_shuffle(self):
        """ChannelShuffle must be forbidden for multi-spectral."""
        domain = SatelliteDomain()
        assert "ChannelShuffle" in domain.forbidden_transforms
    
    def test_allows_full_rotation(self):
        """Satellite allows any rotation angle."""
        domain = SatelliteDomain()
        assert "Rotate" in domain.recommended_transforms


class TestNaturalDomain:
    """Test natural image domain constraints."""
    
    def test_most_permissive(self):
        """Natural domain should have few forbidden transforms."""
        domain = NaturalDomain()
        # Natural domain should allow most things
        assert len(domain.forbidden_transforms) == 0
    
    def test_recommends_common_augmentations(self):
        """Should recommend standard augmentations."""
        domain = NaturalDomain()
        assert "HorizontalFlip" in domain.recommended_transforms
        assert "RandomBrightnessContrast" in domain.recommended_transforms


class TestPolicy:
    """Test Policy class."""
    
    def test_create_policy(self):
        """Test creating a basic policy."""
        policy = Policy(
            name="test_policy",
            domain="medical",
            transforms=[
                Transform("HorizontalFlip", 0.5),
                Transform("Rotate", 0.3, parameters={"limit": 15}),
            ]
        )
        
        assert policy.name == "test_policy"
        assert policy.domain == "medical"
        assert len(policy.transforms) == 2
    
    def test_yaml_serialization(self):
        """Test YAML export/import."""
        policy = Policy(
            name="test_policy",
            domain="medical",
            transforms=[
                Transform("HorizontalFlip", 0.5),
            ]
        )
        
        yaml_str = policy.to_yaml()
        loaded = Policy.from_yaml(yaml_str)
        
        assert loaded.name == policy.name
        assert loaded.domain == policy.domain
        assert len(loaded.transforms) == 1
    
    def test_json_serialization(self):
        """Test JSON export/import."""
        policy = Policy(
            name="test_policy",
            domain="ocr",
            transforms=[
                Transform("Rotate", 0.5, parameters={"limit": 10}),
            ]
        )
        
        json_str = policy.to_json()
        loaded = Policy.from_json(json_str)
        
        assert loaded.name == policy.name
        assert loaded.transforms[0].parameters["limit"] == 10


class TestTransform:
    """Test Transform class."""
    
    def test_probability_validation(self):
        """Probability must be between 0 and 1."""
        with pytest.raises(ValueError):
            Transform("Test", probability=1.5)
        
        with pytest.raises(ValueError):
            Transform("Test", probability=-0.1)
    
    def test_magnitude_validation(self):
        """Magnitude must be between 0 and 10."""
        with pytest.raises(ValueError):
            Transform("Test", magnitude=15)
        
        # Valid magnitudes should work
        t = Transform("Test", magnitude=5)
        assert t.magnitude == 5

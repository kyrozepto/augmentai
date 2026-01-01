"""Tests for the Albumentations compiler."""

import pytest

from augmentai.core.policy import Policy, Transform, TransformCategory
from augmentai.compilers.albumentations import AlbumentationsCompiler


class TestAlbumentationsCompiler:
    """Test Albumentations code generation."""
    
    @pytest.fixture
    def compiler(self):
        return AlbumentationsCompiler()
    
    @pytest.fixture
    def sample_policy(self):
        return Policy(
            name="test_policy",
            domain="natural",
            description="Test policy for unit tests",
            transforms=[
                Transform("HorizontalFlip", 0.5, category=TransformCategory.FLIP),
                Transform("Rotate", 0.3, {"limit": 30}, category=TransformCategory.ROTATE),
                Transform("RandomBrightnessContrast", 0.4, 
                         {"brightness_limit": 0.2, "contrast_limit": 0.2},
                         category=TransformCategory.COLOR),
            ]
        )
    
    def test_generate_code(self, compiler, sample_policy):
        """Test Python code generation."""
        code = compiler.generate_code(sample_policy)
        
        assert "import albumentations as A" in code
        assert "A.HorizontalFlip" in code
        assert "A.Rotate" in code
        assert "p=0.5" in code
        assert "limit=30" in code
    
    def test_generate_config(self, compiler, sample_policy):
        """Test YAML config generation."""
        config = compiler.generate_config(sample_policy)
        
        assert "policy_name: test_policy" in config
        assert "HorizontalFlip" in config
        assert "Rotate" in config
    
    def test_compile_creates_pipeline(self, compiler, sample_policy):
        """Test that compilation creates an Albumentations pipeline."""
        # Skip if albumentations not installed or has import issues
        try:
            available, _ = compiler.validate_backend_available()
        except Exception as e:
            pytest.skip(f"Albumentations has import issues: {e}")
        
        if not available:
            pytest.skip("Albumentations not installed")
        
        result = compiler.compile(sample_policy)
        
        assert result.success
        assert result.pipeline is not None
        assert len(result.code) > 0
        assert len(result.config) > 0
    
    def test_transform_mapping(self, compiler):
        """Test transform name mapping."""
        assert compiler.TRANSFORM_MAPPING["HorizontalFlip"] == "A.HorizontalFlip"
        assert compiler.TRANSFORM_MAPPING["GaussNoise"] == "A.GaussNoise"
        assert compiler.TRANSFORM_MAPPING["ElasticTransform"] == "A.ElasticTransform"
    
    def test_unknown_transform_warning(self, compiler):
        """Unknown transforms should generate warnings."""
        policy = Policy(
            name="test",
            domain="natural",
            transforms=[
                Transform("UnknownTransform", 0.5),
            ]
        )
        
        # Skip if albumentations not installed or has import issues
        try:
            available, _ = compiler.validate_backend_available()
        except Exception as e:
            pytest.skip(f"Albumentations has import issues: {e}")
        
        if not available:
            pytest.skip("Albumentations not installed")
        
        result = compiler.compile(policy)
        
        # Should have warnings about unknown transform
        assert len(result.warnings) > 0 or not result.success

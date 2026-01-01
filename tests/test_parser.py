"""Tests for LLM response parsing."""

import pytest

from augmentai.llm.parser import PolicyParser
from augmentai.core.schema import DEFAULT_SCHEMA


class TestPolicyParser:
    """Test LLM response parsing."""
    
    @pytest.fixture
    def parser(self):
        return PolicyParser(DEFAULT_SCHEMA)
    
    def test_parse_valid_json(self, parser):
        """Parse a valid JSON response."""
        response = '''
        {
            "reasoning": "Conservative policy for medical imaging",
            "policy_name": "medical_ct_v1",
            "transforms": [
                {
                    "name": "HorizontalFlip",
                    "probability": 0.5,
                    "parameters": {}
                },
                {
                    "name": "Rotate",
                    "probability": 0.3,
                    "parameters": {"limit": 15}
                }
            ],
            "warnings": [],
            "alternatives": []
        }
        '''
        
        result = parser.parse(response, "medical")
        
        assert result.success
        assert result.policy is not None
        assert result.policy.name == "medical_ct_v1"
        assert len(result.policy.transforms) == 2
    
    def test_parse_markdown_code_block(self, parser):
        """Parse JSON from markdown code block."""
        response = '''
        Here's my recommendation:
        
        ```json
        {
            "reasoning": "Test policy",
            "policy_name": "test",
            "transforms": [
                {"name": "HorizontalFlip", "probability": 0.5}
            ]
        }
        ```
        '''
        
        result = parser.parse(response, "natural")
        
        assert result.success
        assert result.policy is not None
    
    def test_normalize_transform_names(self, parser):
        """Test transform name normalization."""
        assert parser._normalize_transform_name("horizontal_flip") == "HorizontalFlip"
        assert parser._normalize_transform_name("hflip") == "HorizontalFlip"
        assert parser._normalize_transform_name("gaussian_blur") == "GaussianBlur"
        assert parser._normalize_transform_name("HorizontalFlip") == "HorizontalFlip"
    
    def test_parse_with_aliases(self, parser):
        """Parse response with aliased transform names."""
        response = '''
        {
            "reasoning": "Using common aliases",
            "policy_name": "alias_test",
            "transforms": [
                {"name": "hflip", "probability": 0.5},
                {"name": "gaussian_noise", "probability": 0.3}
            ]
        }
        '''
        
        result = parser.parse(response, "natural")
        
        assert result.success
        assert result.policy.transforms[0].name == "HorizontalFlip"
        assert result.policy.transforms[1].name == "GaussNoise"
    
    def test_extract_transforms_from_text(self, parser):
        """Extract transform names from natural language."""
        text = "I want to add rotation and horizontal flip, maybe some blur too"
        
        found = parser.extract_transforms_from_text(text)
        
        assert "Rotate" in found or "RandomRotate90" in found
        assert "HorizontalFlip" in found
        assert "GaussianBlur" in found or "MotionBlur" in found
    
    def test_handle_invalid_json(self, parser):
        """Handle completely invalid response."""
        response = "This is not JSON at all, just plain text."
        
        result = parser.parse(response, "natural")
        
        assert not result.success
        assert len(result.errors) > 0
    
    def test_clamp_probability(self, parser):
        """Probabilities outside [0,1] should be clamped."""
        response = '''
        {
            "reasoning": "Test",
            "policy_name": "test",
            "transforms": [
                {"name": "HorizontalFlip", "probability": 1.5}
            ]
        }
        '''
        
        result = parser.parse(response, "natural")
        
        assert result.success
        assert result.policy.transforms[0].probability == 1.0

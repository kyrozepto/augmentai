"""Tests for the augmentation preview module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from augmentai.core.policy import Policy, Transform
from augmentai.preview import AugmentationPreview, PreviewConfig, PreviewResult


class TestPreviewResult:
    """Test PreviewResult dataclass."""
    
    def test_to_dict(self, tmp_path):
        """PreviewResult can be converted to dict."""
        result = PreviewResult(
            original_path=tmp_path / "orig.jpg",
            augmented_path=tmp_path / "aug.jpg",
            transform_applied="HorizontalFlip",
            parameters={"p": 0.5},
        )
        
        d = result.to_dict()
        assert "original" in d
        assert "augmented" in d
        assert d["transform"] == "HorizontalFlip"
        assert d["parameters"] == {"p": 0.5}


class TestPreviewConfig:
    """Test PreviewConfig dataclass."""
    
    def test_default_values(self):
        """Check default configuration values."""
        config = PreviewConfig()
        
        assert config.n_samples == 5
        assert config.n_variations == 3
        assert config.save_diffs is True
        assert config.output_format == "html"


class TestAugmentationPreview:
    """Test AugmentationPreview class."""
    
    def test_creates_directories(self, tmp_path):
        """Preview creates required directory structure."""
        previewer = AugmentationPreview(tmp_path)
        
        assert previewer.preview_dir.exists()
        assert previewer.originals_dir.exists()
        assert previewer.augmented_dir.exists()
        assert previewer.diffs_dir.exists()
    
    def test_generate_diff_same_image(self, tmp_path):
        """Diff of same image should be minimal."""
        previewer = AugmentationPreview(tmp_path)
        
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        diff = previewer.generate_diff(image, image.copy())
        
        # Diff should be mostly zeros (no change)
        assert diff.max() < 10
    
    def test_generate_diff_different_images(self, tmp_path):
        """Diff of different images should show changes."""
        previewer = AugmentationPreview(tmp_path)
        
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        augmented = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        diff = previewer.generate_diff(original, augmented)
        
        # Diff should show significant changes
        assert diff.max() > 100
    
    def test_generate_json_report(self, tmp_path):
        """Can generate JSON report."""
        previewer = AugmentationPreview(tmp_path)
        
        policy = Policy(
            name="test_policy",
            domain="natural",
            transforms=[Transform("HorizontalFlip", 0.5)]
        )
        
        results = [
            PreviewResult(
                original_path=tmp_path / "orig.jpg",
                augmented_path=tmp_path / "aug.jpg",
                transform_applied="HorizontalFlip",
            )
        ]
        
        json_path = previewer.generate_json_report(results, policy)
        
        assert json_path.exists()
        content = json_path.read_text()
        assert "test_policy" in content
        assert "HorizontalFlip" in content
    
    def test_generate_html_report(self, tmp_path):
        """Can generate HTML report."""
        previewer = AugmentationPreview(tmp_path)
        
        policy = Policy(
            name="test_policy",
            domain="natural",
            transforms=[Transform("HorizontalFlip", 0.5)]
        )
        
        results = [
            PreviewResult(
                original_path=tmp_path / "orig.jpg",
                augmented_path=tmp_path / "aug.jpg",
                transform_applied="HorizontalFlip",
            )
        ]
        
        html_path = previewer.generate_html_report(results, policy)
        
        assert html_path.exists()
        content = html_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "test_policy" in content
        assert "HorizontalFlip" in content
    
    def test_generate_samples_with_images(self, tmp_path):
        """Can generate samples from real images."""
        from PIL import Image
        
        # Create test images
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        
        img = Image.new("RGB", (100, 100), color="red")
        img_path = img_dir / "test.jpg"
        img.save(img_path)
        
        # Create previewer and policy
        output_dir = tmp_path / "output"
        previewer = AugmentationPreview(output_dir, PreviewConfig(n_samples=1, n_variations=1))
        
        policy = Policy(
            name="test_policy",
            domain="natural",
            transforms=[Transform("HorizontalFlip", 0.5)]
        )
        
        results = previewer.generate_samples([img_path], policy)
        
        assert len(results) >= 1
        assert results[0].original_path.exists()
    
    def test_custom_config(self, tmp_path):
        """Custom config is respected."""
        config = PreviewConfig(
            n_samples=3,
            n_variations=2,
            save_diffs=False,
            seed=42,
        )
        previewer = AugmentationPreview(tmp_path, config)
        
        assert previewer.config.n_samples == 3
        assert previewer.config.n_variations == 2
        assert previewer.config.save_diffs is False
        assert previewer.config.seed == 42

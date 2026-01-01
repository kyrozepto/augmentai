"""Tests for the prepare command and new modules."""

import tempfile
from pathlib import Path

import pytest

from augmentai.core.manifest import ReproducibilityManifest
from augmentai.core.policy import Policy, Transform
from augmentai.inspection import DatasetAnalyzer, DatasetDetector
from augmentai.inspection.detector import DatasetFormat, ImageType
from augmentai.splitting import DatasetSplitter, SplitStrategy
from augmentai.splitting.strategies import SplitConfig
from augmentai.export import ScriptGenerator, FolderStructure
from augmentai.domains import get_domain, list_domains


class TestGetDomain:
    """Test domain lookup helpers."""
    
    def test_get_medical_domain(self):
        """Can get medical domain by name."""
        domain = get_domain("medical")
        assert domain.name == "medical"
        assert "ElasticTransform" in domain.forbidden_transforms
    
    def test_get_domain_case_insensitive(self):
        """Domain lookup is case insensitive."""
        domain = get_domain("MEDICAL")
        assert domain.name == "medical"
    
    def test_get_invalid_domain(self):
        """Invalid domain raises ValueError."""
        with pytest.raises(ValueError, match="Unknown domain"):
            get_domain("invalid_domain")
    
    def test_list_domains(self):
        """List all available domains."""
        domains = list_domains()
        assert "medical" in domains
        assert "ocr" in domains
        assert "satellite" in domains
        assert "natural" in domains


class TestReproducibilityManifest:
    """Test reproducibility manifest."""
    
    def test_create_manifest(self):
        """Can create a manifest with seed."""
        manifest = ReproducibilityManifest(seed=42)
        assert manifest.seed == 42
        assert manifest.augmentai_version == "0.1.0"
    
    def test_manifest_to_json(self):
        """Manifest can be serialized to JSON."""
        manifest = ReproducibilityManifest(seed=123, domain="medical")
        json_str = manifest.to_json()
        assert '"seed": 123' in json_str
        assert '"domain": "medical"' in json_str
    
    def test_manifest_roundtrip(self):
        """Manifest can be serialized and deserialized."""
        manifest = ReproducibilityManifest(
            seed=42,
            domain="medical",
            policy_name="test_policy"
        )
        json_str = manifest.to_json()
        loaded = ReproducibilityManifest.from_json(json_str)
        
        assert loaded.seed == manifest.seed
        assert loaded.domain == manifest.domain
        assert loaded.policy_name == manifest.policy_name


class TestDatasetDetector:
    """Test dataset format detection."""
    
    def test_detect_empty_raises(self):
        """Detecting non-existent path raises."""
        detector = DatasetDetector()
        with pytest.raises(ValueError, match="does not exist"):
            detector.detect(Path("/non/existent/path"))
    
    def test_detect_imagefolder_format(self, tmp_path):
        """Detect image folder structure."""
        # Create class folders with images
        (tmp_path / "cat").mkdir()
        (tmp_path / "dog").mkdir()
        (tmp_path / "cat" / "img1.jpg").write_bytes(b"fake")
        (tmp_path / "dog" / "img2.jpg").write_bytes(b"fake")
        
        detector = DatasetDetector()
        result = detector.detect(tmp_path)
        
        assert result.format == DatasetFormat.IMAGEFOLDER
    
    def test_detect_presplit_structure(self, tmp_path):
        """Detect pre-split dataset."""
        (tmp_path / "train").mkdir()
        (tmp_path / "val").mkdir()
        (tmp_path / "test").mkdir()
        
        detector = DatasetDetector()
        result = detector.detect(tmp_path)
        
        assert result.is_presplit is True
        assert result.format == DatasetFormat.PRESPLIT


class TestDatasetSplitter:
    """Test dataset splitting."""
    
    def test_random_split(self, tmp_path):
        """Random split creates correct ratios."""
        # Create test images
        for i in range(10):
            (tmp_path / f"img_{i}.jpg").write_bytes(b"fake")
        
        config = SplitConfig(
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            strategy=SplitStrategy.RANDOM,
            seed=42,
        )
        splitter = DatasetSplitter(config)
        result = splitter.split(tmp_path, copy_files=False)
        
        assert result.success
        assert result.train_count == 6
        assert result.val_count == 2
        assert result.test_count == 2
    
    def test_split_deterministic(self, tmp_path):
        """Same seed produces same split."""
        for i in range(20):
            (tmp_path / f"img_{i}.jpg").write_bytes(b"fake")
        
        config = SplitConfig(seed=123)
        splitter = DatasetSplitter(config)
        
        result1 = splitter.split(tmp_path, copy_files=False)
        result2 = splitter.split(tmp_path, copy_files=False)
        
        assert result1.train_files == result2.train_files


class TestScriptGenerator:
    """Test script generation."""
    
    def test_generate_script(self):
        """Generate augmentation script."""
        policy = Policy(
            name="test_policy",
            domain="natural",
            transforms=[
                Transform("HorizontalFlip", 0.5),
                Transform("Rotate", 0.3, parameters={"limit": 15}),
            ]
        )
        
        generator = ScriptGenerator()
        script = generator.generate_augment_script(policy, seed=42)
        
        assert "import albumentations as A" in script
        assert "A.HorizontalFlip" in script
        assert "A.Rotate" in script
        assert "seed" in script
    
    def test_generate_config_yaml(self):
        """Generate YAML config."""
        policy = Policy(
            name="test_policy",
            domain="medical",
            transforms=[Transform("HorizontalFlip", 0.5)]
        )
        
        generator = ScriptGenerator()
        config = generator.generate_config_yaml(policy)
        
        assert "name: test_policy" in config
        assert "domain: medical" in config


class TestFolderStructure:
    """Test folder structure generation."""
    
    def test_create_folders(self, tmp_path):
        """Create folder structure."""
        folders = FolderStructure(tmp_path / "output")
        folders.create()
        
        assert folders.train_dir.exists()
        assert folders.val_dir.exists()
        assert folders.test_dir.exists()
        assert folders.augmented_dir.exists()
    
    def test_save_script(self, tmp_path):
        """Save script file."""
        folders = FolderStructure(tmp_path / "output")
        folders.create()
        
        path = folders.save_script("print('hello')")
        assert path.exists()
        assert path.read_text() == "print('hello')"

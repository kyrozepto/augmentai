"""
Dataset format detector.

Automatically detects the format and structure of image datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DatasetFormat(Enum):
    """Recognized dataset formats."""
    IMAGEFOLDER = "imagefolder"  # class_name/images structure
    FLAT = "flat"  # All images in one directory
    COCO = "coco"  # COCO JSON annotation format
    YOLO = "yolo"  # YOLO txt annotations
    PRESPLIT = "presplit"  # train/val/test already split
    UNKNOWN = "unknown"


class ImageType(Enum):
    """Types of images detected."""
    NATURAL = "natural"  # Standard RGB photos
    GRAYSCALE = "grayscale"  # Single channel
    MEDICAL = "medical"  # DICOM, NIfTI
    MULTISPECTRAL = "multispectral"  # Satellite/spectral


@dataclass
class DetectionResult:
    """Result of format detection."""
    format: DatasetFormat
    image_type: ImageType
    confidence: float  # 0.0 to 1.0
    
    # Structure info
    has_labels: bool = False
    has_masks: bool = False
    is_presplit: bool = False
    
    # Detected paths
    image_extensions: set[str] | None = None
    class_folders: list[str] | None = None
    annotation_file: Path | None = None
    
    # Suggested domain
    suggested_domain: str = "natural"
    domain_reason: str = ""


class DatasetDetector:
    """Detect dataset format and structure."""
    
    # Common image extensions
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    MEDICAL_EXTENSIONS = {".dcm", ".dicom", ".nii", ".nii.gz", ".nrrd"}
    MASK_PATTERNS = {"mask", "seg", "label", "annotation", "_mask", "_seg"}
    
    def __init__(self, max_scan_files: int = 1000) -> None:
        """
        Initialize detector.
        
        Args:
            max_scan_files: Maximum files to scan for detection
        """
        self.max_scan_files = max_scan_files
    
    def detect(self, path: Path) -> DetectionResult:
        """
        Detect dataset format and structure.
        
        Args:
            path: Path to dataset directory
            
        Returns:
            DetectionResult with format info
        """
        path = Path(path)
        
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        
        # Check for pre-split structure
        is_presplit = self._check_presplit(path)
        
        # Detect image type and extensions
        image_extensions, image_type = self._detect_image_type(path)
        
        # Detect format
        format_result = self._detect_format(path, is_presplit)
        
        # Check for labels/masks
        has_labels = self._check_labels(path)
        has_masks = self._check_masks(path)
        
        # Suggest domain
        suggested_domain, domain_reason = self._suggest_domain(
            image_type, has_masks, format_result
        )
        
        return DetectionResult(
            format=format_result,
            image_type=image_type,
            confidence=0.9 if format_result != DatasetFormat.UNKNOWN else 0.5,
            has_labels=has_labels,
            has_masks=has_masks,
            is_presplit=is_presplit,
            image_extensions=image_extensions,
            class_folders=self._get_class_folders(path) if format_result == DatasetFormat.IMAGEFOLDER else None,
            suggested_domain=suggested_domain,
            domain_reason=domain_reason,
        )
    
    def _check_presplit(self, path: Path) -> bool:
        """Check if dataset is already split into train/val/test."""
        split_names = {"train", "val", "validation", "test", "dev"}
        subdirs = {d.name.lower() for d in path.iterdir() if d.is_dir()}
        return len(subdirs & split_names) >= 2
    
    def _detect_image_type(self, path: Path) -> tuple[set[str], ImageType]:
        """Detect image type and extensions used."""
        extensions: set[str] = set()
        
        # Scan for files
        count = 0
        for file_path in path.rglob("*"):
            if count >= self.max_scan_files:
                break
            
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in self.IMAGE_EXTENSIONS:
                    extensions.add(ext)
                    count += 1
                elif ext in self.MEDICAL_EXTENSIONS or file_path.name.endswith(".nii.gz"):
                    extensions.add(ext)
                    return extensions, ImageType.MEDICAL
        
        # Default to natural
        return extensions, ImageType.NATURAL
    
    def _detect_format(self, path: Path, is_presplit: bool) -> DatasetFormat:
        """Detect the dataset organization format."""
        if is_presplit:
            return DatasetFormat.PRESPLIT
        
        # Check for COCO format (annotations/*.json)
        coco_files = list(path.glob("*.json")) + list(path.glob("annotations/*.json"))
        for json_file in coco_files:
            if self._is_coco_format(json_file):
                return DatasetFormat.COCO
        
        # Check for YOLO format (labels/*.txt)
        labels_dir = path / "labels"
        if labels_dir.exists() and any(labels_dir.glob("*.txt")):
            return DatasetFormat.YOLO
        
        # Check for image folder format (subdirs with images)
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        if subdirs:
            # Check if subdirs contain images (class folders)
            for subdir in subdirs[:5]:  # Check first 5
                has_images = any(
                    f for f in subdir.iterdir() 
                    if f.suffix.lower() in self.IMAGE_EXTENSIONS
                )
                if has_images:
                    return DatasetFormat.IMAGEFOLDER
        
        # Check for flat structure
        direct_images = [
            f for f in path.iterdir() 
            if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS
        ]
        if direct_images:
            return DatasetFormat.FLAT
        
        return DatasetFormat.UNKNOWN
    
    def _is_coco_format(self, json_path: Path) -> bool:
        """Check if JSON file is COCO format."""
        import json
        try:
            with open(json_path) as f:
                data = json.load(f)
            return "images" in data and "annotations" in data
        except (json.JSONDecodeError, KeyError):
            return False
    
    def _check_labels(self, path: Path) -> bool:
        """Check if dataset has labels."""
        # Check for common label patterns
        label_patterns = ["*.json", "*.xml", "*.txt", "labels/*", "annotations/*"]
        for pattern in label_patterns:
            if list(path.glob(pattern)):
                return True
        return False
    
    def _check_masks(self, path: Path) -> bool:
        """Check if dataset has segmentation masks."""
        # Look for mask directories or files
        for subdir in path.rglob("*"):
            if subdir.is_dir():
                name_lower = subdir.name.lower()
                if any(pattern in name_lower for pattern in self.MASK_PATTERNS):
                    return True
        return False
    
    def _get_class_folders(self, path: Path) -> list[str]:
        """Get list of class folder names."""
        classes = []
        for subdir in sorted(path.iterdir()):
            if subdir.is_dir():
                # Skip common non-class folders
                if subdir.name.lower() not in {"train", "val", "test", "validation", "annotations", "labels", "masks"}:
                    classes.append(subdir.name)
        return classes
    
    def _suggest_domain(
        self, 
        image_type: ImageType, 
        has_masks: bool,
        format_result: DatasetFormat
    ) -> tuple[str, str]:
        """Suggest appropriate domain based on detection."""
        if image_type == ImageType.MEDICAL:
            return "medical", "Medical image format detected (DICOM/NIfTI)"
        
        if image_type == ImageType.MULTISPECTRAL:
            return "satellite", "Multi-spectral/satellite imagery detected"
        
        # Default to natural for general images
        return "natural", "Standard image format, using permissive domain"

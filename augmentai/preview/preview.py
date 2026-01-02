"""
Augmentation preview and diff visualization.

Generates visual comparisons of augmentation effects for review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from augmentai.core.policy import Policy, Transform


@dataclass
class PreviewConfig:
    """Configuration for preview generation."""
    
    n_samples: int = 5          # Number of sample images to preview
    n_variations: int = 3       # Number of augmentation variations per image
    save_diffs: bool = True     # Generate diff images
    output_format: str = "html" # html, json, or both
    seed: int | None = None     # For reproducible previews


@dataclass
class PreviewResult:
    """Result of a single preview generation."""
    
    original_path: Path
    augmented_path: Path
    transform_applied: str
    diff_path: Path | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original": str(self.original_path),
            "augmented": str(self.augmented_path),
            "transform": self.transform_applied,
            "diff": str(self.diff_path) if self.diff_path else None,
            "parameters": self.parameters,
        }


@dataclass
class PreviewReport:
    """Complete preview report for a policy."""
    
    policy_name: str
    domain: str
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    results: list[PreviewResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "policy_name": self.policy_name,
            "domain": self.domain,
            "generated_at": self.generated_at,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
        }


class AugmentationPreview:
    """
    Generate visual previews of augmentation effects.
    
    Creates before/after comparisons and diff visualizations
    to help users understand and validate augmentation pipelines.
    """
    
    def __init__(
        self,
        output_dir: Path,
        config: PreviewConfig | None = None,
    ) -> None:
        """
        Initialize the preview generator.
        
        Args:
            output_dir: Directory to save preview outputs
            config: Preview configuration
        """
        self.output_dir = Path(output_dir)
        self.config = config or PreviewConfig()
        self._setup_directories()
    
    def _setup_directories(self) -> None:
        """Create output directory structure."""
        self.preview_dir = self.output_dir / "preview"
        self.originals_dir = self.preview_dir / "originals"
        self.augmented_dir = self.preview_dir / "augmented"
        self.diffs_dir = self.preview_dir / "diffs"
        
        for dir_path in [self.preview_dir, self.originals_dir, 
                         self.augmented_dir, self.diffs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_samples(
        self,
        images: list[Path],
        policy: Policy,
        apply_fn: Any = None,
    ) -> list[PreviewResult]:
        """
        Generate preview samples for a set of images.
        
        Args:
            images: List of image paths to preview
            policy: The augmentation policy to apply
            apply_fn: Optional albumentations pipeline to apply
            
        Returns:
            List of PreviewResult objects
        """
        results = []
        
        # Sample images (limit to n_samples)
        sample_images = images[:self.config.n_samples]
        
        for idx, img_path in enumerate(sample_images):
            try:
                img_array = self._load_image(img_path)
                
                # Save original
                orig_save_path = self.originals_dir / f"sample_{idx:03d}{img_path.suffix}"
                self._save_image(img_array, orig_save_path)
                
                # Generate augmented variations
                for var_idx in range(self.config.n_variations):
                    # Apply each transform
                    for transform in policy.transforms:
                        aug_array = self._apply_transform(
                            img_array, transform, apply_fn,
                            seed=self.config.seed + var_idx if self.config.seed else None
                        )
                        
                        aug_filename = f"sample_{idx:03d}_{transform.name}_v{var_idx}{img_path.suffix}"
                        aug_save_path = self.augmented_dir / aug_filename
                        self._save_image(aug_array, aug_save_path)
                        
                        # Generate diff if enabled
                        diff_path = None
                        if self.config.save_diffs:
                            diff_array = self.generate_diff(img_array, aug_array)
                            diff_filename = f"diff_{idx:03d}_{transform.name}_v{var_idx}.png"
                            diff_path = self.diffs_dir / diff_filename
                            self._save_image(diff_array, diff_path)
                        
                        results.append(PreviewResult(
                            original_path=orig_save_path,
                            augmented_path=aug_save_path,
                            transform_applied=transform.name,
                            diff_path=diff_path,
                            parameters=transform.parameters or {},
                        ))
                        
            except Exception as e:
                # Log error but continue with other images
                print(f"Warning: Could not process {img_path}: {e}")
        
        return results
    
    def generate_diff(
        self,
        original: np.ndarray,
        augmented: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a diff visualization between original and augmented images.
        
        Args:
            original: Original image array (H, W, C)
            augmented: Augmented image array (H, W, C)
            
        Returns:
            Diff image as numpy array
        """
        # Ensure same shape
        if original.shape != augmented.shape:
            # Resize augmented to match original for comparison
            augmented = self._resize_to_match(augmented, original.shape)
        
        # Compute absolute difference
        diff = np.abs(original.astype(np.int16) - augmented.astype(np.int16))
        
        # Normalize and convert to uint8
        diff = (diff * 2).clip(0, 255).astype(np.uint8)
        
        # Apply colormap for better visualization (differences in red)
        if len(diff.shape) == 3:
            # Convert to grayscale diff
            gray_diff = np.mean(diff, axis=2).astype(np.uint8)
        else:
            gray_diff = diff
        
        # Create RGB diff with red highlighting
        rgb_diff = np.zeros((*gray_diff.shape, 3), dtype=np.uint8)
        rgb_diff[:, :, 0] = gray_diff  # Red channel shows differences
        rgb_diff[:, :, 1] = gray_diff // 2  # Slight green for visibility
        
        return rgb_diff
    
    def generate_html_report(
        self,
        results: list[PreviewResult],
        policy: Policy,
    ) -> Path:
        """
        Generate an HTML report with all previews.
        
        Args:
            results: List of preview results
            policy: The policy that was previewed
            
        Returns:
            Path to the generated HTML file
        """
        html_path = self.preview_dir / "report.html"
        
        # Group results by transform
        by_transform: dict[str, list[PreviewResult]] = {}
        for result in results:
            if result.transform_applied not in by_transform:
                by_transform[result.transform_applied] = []
            by_transform[result.transform_applied].append(result)
        
        html_content = self._generate_html(by_transform, policy)
        html_path.write_text(html_content, encoding="utf-8")
        
        return html_path
    
    def generate_json_report(
        self,
        results: list[PreviewResult],
        policy: Policy,
    ) -> Path:
        """
        Generate a JSON report with all previews.
        
        Args:
            results: List of preview results
            policy: The policy that was previewed
            
        Returns:
            Path to the generated JSON file
        """
        json_path = self.preview_dir / "report.json"
        
        report = PreviewReport(
            policy_name=policy.name,
            domain=policy.domain,
            results=results,
            summary={
                "total_samples": len(results),
                "transforms": list(set(r.transform_applied for r in results)),
            }
        )
        
        json_path.write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8"
        )
        
        return json_path
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load an image as numpy array."""
        try:
            from PIL import Image
            with Image.open(path) as img:
                return np.array(img.convert("RGB"))
        except ImportError:
            # Fallback: create dummy array
            return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def _save_image(self, array: np.ndarray, path: Path) -> None:
        """Save numpy array as image."""
        try:
            from PIL import Image
            img = Image.fromarray(array)
            img.save(path)
        except ImportError:
            pass  # Can't save without PIL
    
    def _apply_transform(
        self,
        image: np.ndarray,
        transform: Transform,
        apply_fn: Any = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Apply a transform to an image."""
        if apply_fn is not None:
            # Use provided apply function
            return apply_fn(image, transform, seed)
        
        # Fallback: try to use albumentations directly
        try:
            import albumentations as A
            
            if seed is not None:
                np.random.seed(seed)
            
            # Build simple transform
            albu_transform = getattr(A, transform.name, None)
            if albu_transform is not None:
                params = transform.parameters or {}
                aug = A.Compose([albu_transform(p=1.0, **params)])
                result = aug(image=image)
                return result["image"]
        except (ImportError, Exception):
            pass
        
        # If no augmentation can be applied, return copy
        return image.copy()
    
    def _resize_to_match(self, image: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Resize image to match target shape."""
        try:
            from PIL import Image
            img = Image.fromarray(image)
            img = img.resize((target_shape[1], target_shape[0]))
            return np.array(img)
        except ImportError:
            return image
    
    def _generate_html(
        self,
        by_transform: dict[str, list[PreviewResult]],
        policy: Policy,
    ) -> str:
        """Generate HTML content for the report."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Augmentation Preview - {policy.name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e; color: #eee; padding: 2rem;
        }}
        h1 {{ color: #00d4ff; margin-bottom: 0.5rem; }}
        h2 {{ color: #7c3aed; margin: 2rem 0 1rem; border-bottom: 2px solid #7c3aed; padding-bottom: 0.5rem; }}
        .meta {{ color: #888; margin-bottom: 2rem; }}
        .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1.5rem; }}
        .card {{ 
            background: #16213e; border-radius: 12px; overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .card img {{ width: 100%; height: 200px; object-fit: cover; }}
        .card-body {{ padding: 1rem; }}
        .card-title {{ font-size: 0.9rem; color: #00d4ff; }}
        .comparison {{ display: flex; gap: 0.5rem; }}
        .comparison img {{ flex: 1; height: 150px; object-fit: cover; border-radius: 8px; }}
        .label {{ font-size: 0.75rem; color: #888; text-align: center; margin-top: 0.25rem; }}
    </style>
</head>
<body>
    <h1>🎨 Augmentation Preview</h1>
    <p class="meta">Policy: {policy.name} | Domain: {policy.domain} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
"""
        
        for transform_name, results in by_transform.items():
            html += f"""
    <h2>{transform_name}</h2>
    <div class="gallery">
"""
            for result in results[:6]:  # Limit to 6 per transform
                orig_rel = result.original_path.name
                aug_rel = result.augmented_path.name
                
                html += f"""
        <div class="card">
            <div class="comparison">
                <div>
                    <img src="originals/{orig_rel}" alt="Original">
                    <p class="label">Original</p>
                </div>
                <div>
                    <img src="augmented/{aug_rel}" alt="Augmented">
                    <p class="label">Augmented</p>
                </div>
            </div>
            <div class="card-body">
                <p class="card-title">{transform_name}</p>
            </div>
        </div>
"""
            html += "    </div>\n"
        
        html += """
</body>
</html>
"""
        return html

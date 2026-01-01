"""
Albumentations backend compiler.

Compiles policies to Albumentations augmentation pipelines.
"""

from __future__ import annotations

from typing import Any

import yaml

from augmentai.compilers.base import BaseCompiler, CompileResult
from augmentai.core.policy import Policy, Transform


class AlbumentationsCompiler(BaseCompiler):
    """
    Compile policies to Albumentations pipelines.
    
    Generates executable Python code and YAML configurations
    that can be used directly with Albumentations.
    """
    
    backend_name = "albumentations"
    
    # Mapping from our transform names to Albumentations classes
    TRANSFORM_MAPPING: dict[str, str] = {
        "HorizontalFlip": "A.HorizontalFlip",
        "VerticalFlip": "A.VerticalFlip",
        "Rotate": "A.Rotate",
        "RandomRotate90": "A.RandomRotate90",
        "RandomBrightnessContrast": "A.RandomBrightnessContrast",
        "HueSaturationValue": "A.HueSaturationValue",
        "ColorJitter": "A.ColorJitter",
        "GaussianBlur": "A.GaussianBlur",
        "MotionBlur": "A.MotionBlur",
        "GaussNoise": "A.GaussNoise",
        "ISONoise": "A.ISONoise",
        "ShiftScaleRotate": "A.ShiftScaleRotate",
        "Affine": "A.Affine",
        "ElasticTransform": "A.ElasticTransform",
        "GridDistortion": "A.GridDistortion",
        "OpticalDistortion": "A.OpticalDistortion",
        "RandomCrop": "A.RandomCrop",
        "CenterCrop": "A.CenterCrop",
        "RandomScale": "A.RandomScale",
        "Resize": "A.Resize",
        "CLAHE": "A.CLAHE",
        "Equalize": "A.Equalize",
        "Sharpen": "A.Sharpen",
        "MedianBlur": "A.MedianBlur",
        "CoarseDropout": "A.CoarseDropout",
        "Normalize": "A.Normalize",
        "ToGray": "A.ToGray",
        "Perspective": "A.Perspective",
    }
    
    # Parameter name mappings (if different between our schema and Albumentations)
    PARAM_MAPPING: dict[str, dict[str, str]] = {
        "GaussNoise": {"var_limit": "var_limit"},
        "Rotate": {"limit": "limit"},
        "RandomScale": {"scale_limit": "scale_limit"},
    }
    
    def validate_backend_available(self) -> tuple[bool, str]:
        """Check if Albumentations is installed."""
        try:
            import albumentations as A
            version = A.__version__
            return True, f"Albumentations {version} is available"
        except ImportError:
            return False, "Albumentations not installed. Run: pip install albumentations"
    
    def compile(self, policy: Policy) -> CompileResult:
        """
        Compile a policy to an Albumentations pipeline.
        
        Args:
            policy: The policy to compile
            
        Returns:
            CompileResult with the pipeline and generated code
        """
        result = CompileResult(success=False, backend=self.backend_name)
        
        # Check if Albumentations is available
        is_available, message = self.validate_backend_available()
        if not is_available:
            result.errors.append(message)
            return result
        
        import albumentations as A
        
        # Build the transform list
        transforms = []
        
        for transform in policy.transforms:
            try:
                alb_transform = self._build_transform(transform, A)
                if alb_transform is not None:
                    transforms.append(alb_transform)
            except Exception as e:
                result.warnings.append(f"Could not build {transform.name}: {e}")
        
        if not transforms:
            result.errors.append("No transforms could be compiled")
            return result
        
        # Create the pipeline
        try:
            pipeline = A.Compose(transforms)
            result.pipeline = pipeline
            result.success = True
        except Exception as e:
            result.errors.append(f"Failed to create pipeline: {e}")
            return result
        
        # Generate code and config
        result.code = self.generate_code(policy)
        result.config = self.generate_config(policy)
        
        return result
    
    def _build_transform(self, transform: Transform, A: Any) -> Any:
        """
        Build a single Albumentations transform.
        
        Args:
            transform: Our Transform object
            A: The albumentations module
            
        Returns:
            Albumentations transform object
        """
        alb_name = self.TRANSFORM_MAPPING.get(transform.name)
        if alb_name is None:
            raise ValueError(f"Unknown transform: {transform.name}")
        
        # Get the Albumentations class
        class_name = alb_name.split(".")[-1]
        if not hasattr(A, class_name):
            raise ValueError(f"Albumentations does not have {class_name}")
        
        transform_class = getattr(A, class_name)
        
        # Map parameters
        params = dict(transform.parameters)
        params["p"] = transform.probability
        
        # Apply parameter name mappings
        if transform.name in self.PARAM_MAPPING:
            for our_name, alb_name in self.PARAM_MAPPING[transform.name].items():
                if our_name in params and our_name != alb_name:
                    params[alb_name] = params.pop(our_name)
        
        return transform_class(**params)
    
    def generate_code(self, policy: Policy) -> str:
        """
        Generate Python code for the policy.
        
        Args:
            policy: The policy to generate code for
            
        Returns:
            Python code as a string
        """
        lines = [
            '"""',
            f'Augmentation policy: {policy.name}',
            f'Domain: {policy.domain}',
            f'Description: {policy.description}',
            '',
            'Generated by AugmentAI',
            '"""',
            '',
            'import albumentations as A',
            'import cv2',
            'import numpy as np',
            '',
            '',
            'def get_augmentation_pipeline():',
            '    """Get the augmentation pipeline."""',
            '    return A.Compose([',
        ]
        
        for transform in policy.transforms:
            transform_line = self._transform_to_code(transform)
            lines.append(f'        {transform_line},')
        
        lines.extend([
            '    ])',
            '',
            '',
            '# Usage example:',
            '# pipeline = get_augmentation_pipeline()',
            '# augmented = pipeline(image=image, mask=mask)',
            '# aug_image = augmented["image"]',
            '# aug_mask = augmented["mask"]',
        ])
        
        return '\n'.join(lines)
    
    def _transform_to_code(self, transform: Transform) -> str:
        """
        Convert a transform to a Python code string.
        
        Args:
            transform: The transform to convert
            
        Returns:
            Python code for this transform
        """
        alb_name = self.TRANSFORM_MAPPING.get(transform.name, f"A.{transform.name}")
        
        # Build parameter string
        params = []
        
        for name, value in transform.parameters.items():
            if isinstance(value, str):
                params.append(f'{name}="{value}"')
            elif isinstance(value, bool):
                params.append(f'{name}={value}')
            elif isinstance(value, (int, float)):
                params.append(f'{name}={value}')
            else:
                params.append(f'{name}={repr(value)}')
        
        params.append(f'p={transform.probability}')
        
        param_str = ', '.join(params)
        return f'{alb_name}({param_str})'
    
    def generate_config(self, policy: Policy) -> str:
        """
        Generate YAML config for the policy.
        
        This config can be loaded with A.from_dict() or saved for reproducibility.
        
        Args:
            policy: The policy to generate config for
            
        Returns:
            YAML configuration as a string
        """
        config = {
            "__version__": "1.0.0",
            "policy_name": policy.name,
            "domain": policy.domain,
            "description": policy.description,
            "transform": {
                "__class_fullname__": "albumentations.core.composition.Compose",
                "transforms": []
            }
        }
        
        for transform in policy.transforms:
            alb_name = self.TRANSFORM_MAPPING.get(
                transform.name, 
                f"albumentations.augmentations.transforms.{transform.name}"
            )
            
            # Convert A.X to full class name
            if alb_name.startswith("A."):
                class_name = alb_name[2:]
                alb_name = f"albumentations.augmentations.transforms.{class_name}"
            
            transform_config = {
                "__class_fullname__": alb_name,
                "p": transform.probability,
            }
            transform_config.update(transform.parameters)
            
            config["transform"]["transforms"].append(transform_config)
        
        return yaml.dump(config, default_flow_style=False, sort_keys=False)
    
    def from_config(self, config_path: str) -> Any:
        """
        Load a pipeline from a saved config.
        
        Args:
            config_path: Path to the YAML config file
            
        Returns:
            Albumentations pipeline
        """
        is_available, message = self.validate_backend_available()
        if not is_available:
            raise ImportError(message)
        
        import albumentations as A
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Note: A.from_dict expects a specific format
        # This is a simplified version
        return A.load(config_path)

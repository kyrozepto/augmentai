"""
Parse LLM responses into structured policy objects.

Handles JSON extraction, validation, and error recovery from
potentially malformed LLM outputs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from augmentai.core.policy import Policy, Transform, TransformCategory
from augmentai.core.schema import PolicySchema, DEFAULT_SCHEMA


@dataclass
class ParseResult:
    """Result of parsing an LLM response."""
    
    success: bool
    policy: Policy | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reasoning: str = ""
    alternatives: list[str] = field(default_factory=list)


class PolicyParser:
    """
    Parse LLM responses into Policy objects.
    
    Handles various edge cases:
    - JSON embedded in markdown code blocks
    - Partial or malformed JSON
    - Missing or invalid fields
    - Transform name mapping/correction
    """
    
    def __init__(self, schema: PolicySchema | None = None) -> None:
        """
        Initialize the parser.
        
        Args:
            schema: Transform schema for validation. Uses default if not provided.
        """
        self.schema = schema or DEFAULT_SCHEMA
        
        # Common aliases for transform names
        self.transform_aliases: dict[str, str] = {
            "horizontal_flip": "HorizontalFlip",
            "horizontalflip": "HorizontalFlip",
            "hflip": "HorizontalFlip",
            "vertical_flip": "VerticalFlip",
            "verticalflip": "VerticalFlip",
            "vflip": "VerticalFlip",
            "rotate": "Rotate",
            "rotation": "Rotate",
            "random_rotate_90": "RandomRotate90",
            "randomrotate90": "RandomRotate90",
            "rot90": "RandomRotate90",
            "brightness_contrast": "RandomBrightnessContrast",
            "brightcontrast": "RandomBrightnessContrast",
            "randombrightcontrast": "RandomBrightnessContrast",
            "gaussian_noise": "GaussNoise",
            "gaussiannoise": "GaussNoise",
            "gauss_noise": "GaussNoise",
            "gaussian_blur": "GaussianBlur",
            "gaussianblur": "GaussianBlur",
            "gauss_blur": "GaussianBlur",
            "elastic": "ElasticTransform",
            "elastic_transform": "ElasticTransform",
            "elastictransform": "ElasticTransform",
            "grid_distort": "GridDistortion",
            "griddistortion": "GridDistortion",
            "grid_distortion": "GridDistortion",
            "hsv": "HueSaturationValue",
            "hue_saturation": "HueSaturationValue",
            "huesaturationvalue": "HueSaturationValue",
            "color_jitter": "ColorJitter",
            "colorjitter": "ColorJitter",
            "shift_scale_rotate": "ShiftScaleRotate",
            "shiftscalerotate": "ShiftScaleRotate",
            "ssr": "ShiftScaleRotate",
            "random_crop": "RandomCrop",
            "randomcrop": "RandomCrop",
            "random_scale": "RandomScale",
            "randomscale": "RandomScale",
            "motion_blur": "MotionBlur",
            "motionblur": "MotionBlur",
            "iso_noise": "ISONoise",
            "isonoise": "ISONoise",
            "coarse_dropout": "CoarseDropout",
            "coarsedropout": "CoarseDropout",
            "cutout": "CoarseDropout",
        }
    
    def parse(self, response: str, domain_name: str = "custom") -> ParseResult:
        """
        Parse an LLM response into a Policy.
        
        Args:
            response: The raw LLM response text
            domain_name: The domain for this policy
            
        Returns:
            ParseResult with the parsed policy or errors
        """
        result = ParseResult(success=False)
        
        # Extract JSON from the response
        json_data = self._extract_json(response)
        if json_data is None:
            result.errors.append("Could not find valid JSON in response")
            return result
        
        result.raw_data = json_data
        
        # Extract metadata
        result.reasoning = json_data.get("reasoning", "")
        result.alternatives = json_data.get("alternatives", [])
        
        # Extract warnings from LLM
        llm_warnings = json_data.get("warnings", [])
        if isinstance(llm_warnings, list):
            result.warnings.extend(llm_warnings)
        
        # Parse transforms
        transforms_data = json_data.get("transforms", [])
        if not isinstance(transforms_data, list):
            result.errors.append("'transforms' must be a list")
            return result
        
        transforms = []
        for i, t_data in enumerate(transforms_data):
            transform, errors = self._parse_transform(t_data, i)
            if transform:
                transforms.append(transform)
            result.errors.extend(errors)
        
        if not transforms:
            result.errors.append("No valid transforms found in response")
            return result
        
        # Create the policy
        policy_name = json_data.get("policy_name", f"{domain_name}_policy")
        
        result.policy = Policy(
            name=policy_name,
            domain=domain_name,
            transforms=transforms,
            description=result.reasoning,
        )
        
        result.success = True
        return result
    
    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """
        Extract JSON from text, handling markdown code blocks.
        
        Args:
            text: The text that may contain JSON
            
        Returns:
            Parsed JSON dictionary or None if not found
        """
        # Try to find JSON in markdown code block
        patterns = [
            r"```json\s*(.*?)\s*```",  # ```json ... ```
            r"```\s*(.*?)\s*```",       # ``` ... ```
            r"\{[\s\S]*\}",              # Raw JSON object
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # Handle both group match and full match
                    json_str = match if isinstance(match, str) else match[0]
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _parse_transform(
        self, 
        data: dict[str, Any], 
        index: int
    ) -> tuple[Transform | None, list[str]]:
        """
        Parse a single transform from the response data.
        
        Args:
            data: Dictionary with transform data
            index: Index in the transforms list (for error messages)
            
        Returns:
            Tuple of (Transform or None, list of errors)
        """
        errors = []
        
        if not isinstance(data, dict):
            errors.append(f"Transform {index}: expected dict, got {type(data).__name__}")
            return None, errors
        
        # Get and normalize transform name
        name = data.get("name")
        if not name:
            errors.append(f"Transform {index}: missing 'name' field")
            return None, errors
        
        normalized_name = self._normalize_transform_name(str(name))
        if normalized_name is None:
            errors.append(f"Transform {index}: unknown transform '{name}'")
            return None, errors
        
        # Get probability
        probability = data.get("probability", 0.5)
        try:
            probability = float(probability)
            probability = max(0.0, min(1.0, probability))
        except (TypeError, ValueError):
            errors.append(f"Transform {index}: invalid probability '{probability}', using 0.5")
            probability = 0.5
        
        # Get parameters
        parameters = data.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        
        # Determine category from schema
        spec = self.schema.get(normalized_name)
        category = spec.category if spec else TransformCategory.OTHER
        
        # Get magnitude if specified
        magnitude = data.get("magnitude")
        if magnitude is not None:
            try:
                magnitude = int(magnitude)
                magnitude = max(0, min(10, magnitude))
            except (TypeError, ValueError):
                magnitude = None
        
        transform = Transform(
            name=normalized_name,
            probability=probability,
            parameters=parameters,
            category=category,
            magnitude=magnitude,
        )
        
        return transform, errors
    
    def _normalize_transform_name(self, name: str) -> str | None:
        """
        Normalize a transform name to its canonical form.
        
        Args:
            name: The input transform name (possibly aliased)
            
        Returns:
            Canonical transform name or None if unknown
        """
        # Check if already canonical
        if name in self.schema.transforms:
            return name
        
        # Check aliases
        lower_name = name.lower().replace(" ", "_").replace("-", "_")
        if lower_name in self.transform_aliases:
            return self.transform_aliases[lower_name]
        
        # Try case-insensitive match against schema
        for schema_name in self.schema.transforms:
            if schema_name.lower() == lower_name:
                return schema_name
        
        return None
    
    def extract_transforms_from_text(self, text: str) -> list[str]:
        """
        Extract transform names mentioned in natural language text.
        
        Useful for understanding what transforms the user is asking about.
        
        Args:
            text: Natural language text
            
        Returns:
            List of recognized transform names
        """
        found = []
        text_lower = text.lower()
        
        # Check schema names
        for name in self.schema.transforms:
            if name.lower() in text_lower:
                found.append(name)
        
        # Check aliases
        for alias, canonical in self.transform_aliases.items():
            if alias.replace("_", " ") in text_lower or alias in text_lower:
                if canonical not in found:
                    found.append(canonical)
        
        # Check common keywords
        keyword_mapping = {
            "flip": ["HorizontalFlip", "VerticalFlip"],
            "rotate": ["Rotate", "RandomRotate90"],
            "blur": ["GaussianBlur", "MotionBlur"],
            "noise": ["GaussNoise", "ISONoise"],
            "scale": ["RandomScale"],
            "crop": ["RandomCrop", "CenterCrop"],
            "distort": ["ElasticTransform", "GridDistortion", "OpticalDistortion"],
            "color": ["ColorJitter", "HueSaturationValue", "RandomBrightnessContrast"],
            "brightness": ["RandomBrightnessContrast"],
            "contrast": ["RandomBrightnessContrast"],
        }
        
        for keyword, transforms in keyword_mapping.items():
            if keyword in text_lower:
                for t in transforms:
                    if t not in found:
                        found.append(t)
        
        return found

"""
Augmentation-aware robustness evaluation.

Measures how sensitive/invariant a model is to specific augmentations.
Helps identify fragile invariances where the model may fail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from augmentai.core.policy import Policy, Transform


@dataclass
class RobustnessScore:
    """Robustness score for a single transform."""
    
    transform_name: str
    sensitivity: float        # Variance in predictions (0-1, lower = more robust)
    consistency: float        # Prediction agreement rate (0-1, higher = more consistent)
    invariance_score: float   # Combined invariance score (0-1, higher = more invariant)
    n_samples_tested: int = 0
    
    @property
    def is_fragile(self) -> bool:
        """Check if model is fragile to this transform."""
        return self.invariance_score < 0.7
    
    @property
    def robustness_label(self) -> str:
        """Get human-readable robustness label."""
        if self.invariance_score >= 0.9:
            return "highly robust"
        elif self.invariance_score >= 0.7:
            return "robust"
        elif self.invariance_score >= 0.5:
            return "moderately robust"
        elif self.invariance_score >= 0.3:
            return "fragile"
        else:
            return "very fragile"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_name": self.transform_name,
            "sensitivity": self.sensitivity,
            "consistency": self.consistency,
            "invariance_score": self.invariance_score,
            "n_samples_tested": self.n_samples_tested,
            "is_fragile": self.is_fragile,
            "robustness_label": self.robustness_label,
        }


@dataclass
class RobustnessReport:
    """Complete robustness evaluation report."""
    
    policy_name: str
    domain: str
    scores: list[RobustnessScore] = field(default_factory=list)
    fragile_transforms: list[str] = field(default_factory=list)
    robust_transforms: list[str] = field(default_factory=list)
    overall_robustness: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self._compute_summary()
    
    def _compute_summary(self):
        """Compute summary statistics."""
        if self.scores:
            self.fragile_transforms = [s.transform_name for s in self.scores if s.is_fragile]
            self.robust_transforms = [s.transform_name for s in self.scores if not s.is_fragile]
            self.overall_robustness = np.mean([s.invariance_score for s in self.scores])
    
    @property
    def summary(self) -> str:
        """Get one-line summary."""
        fragile = len(self.fragile_transforms)
        robust = len(self.robust_transforms)
        return f"Robustness: {robust} robust, {fragile} fragile transforms (overall: {self.overall_robustness:.2f})"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "domain": self.domain,
            "scores": [s.to_dict() for s in self.scores],
            "fragile_transforms": self.fragile_transforms,
            "robust_transforms": self.robust_transforms,
            "overall_robustness": self.overall_robustness,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class RobustnessEvaluator:
    """
    Evaluate model robustness to augmentation transforms.
    
    For each transform, applies it multiple times to each image
    and measures prediction consistency/variance.
    
    High invariance = model gives same predictions regardless of augmentation.
    Low invariance (fragile) = model predictions change with augmentation.
    """
    
    DEFAULT_N_VARIATIONS = 5
    
    def __init__(
        self,
        model_fn: Callable[[np.ndarray], Any],
        compare_fn: Callable[[Any, Any], float] | None = None,
        n_variations: int = 5,
        seed: int = 42,
    ) -> None:
        """
        Initialize robustness evaluator.
        
        Args:
            model_fn: Function that takes an image array and returns predictions
            compare_fn: Function to compare two predictions (returns 0-1 similarity)
                        If None, uses default comparison based on prediction type
            n_variations: Number of augmented variations per image
            seed: Random seed for reproducibility
        """
        self.model_fn = model_fn
        self.compare_fn = compare_fn or self._default_compare
        self.n_variations = n_variations
        self.seed = seed
    
    def evaluate(
        self,
        images: list[np.ndarray],
        policy: Policy,
        apply_fn: Callable[[np.ndarray, Transform], np.ndarray] | None = None,
    ) -> RobustnessReport:
        """
        Evaluate model robustness to each transform in the policy.
        
        Args:
            images: List of image arrays to test
            policy: The policy containing transforms to evaluate
            apply_fn: Optional function to apply transforms
                      Signature: apply_fn(image, transform) -> augmented_image
        
        Returns:
            RobustnessReport with scores for each transform
        """
        scores = []
        
        for transform in policy.transforms:
            score = self._evaluate_transform(images, transform, apply_fn)
            scores.append(score)
        
        report = RobustnessReport(
            policy_name=policy.name,
            domain=policy.domain,
            scores=scores,
        )
        
        return report
    
    def _evaluate_transform(
        self,
        images: list[np.ndarray],
        transform: Transform,
        apply_fn: Callable | None,
    ) -> RobustnessScore:
        """Evaluate robustness to a single transform."""
        all_consistencies = []
        all_sensitivities = []
        
        for image in images:
            # Get original prediction
            original_pred = self.model_fn(image)
            
            # Generate variations and get predictions
            variation_preds = []
            for i in range(self.n_variations):
                if apply_fn:
                    augmented = apply_fn(image, transform)
                else:
                    augmented = self._apply_transform_default(image, transform, seed=self.seed + i)
                
                pred = self.model_fn(augmented)
                variation_preds.append(pred)
            
            # Compute consistency with original
            consistencies = [
                self.compare_fn(original_pred, var_pred) 
                for var_pred in variation_preds
            ]
            avg_consistency = np.mean(consistencies)
            all_consistencies.append(avg_consistency)
            
            # Compute sensitivity (variance among variations)
            if len(variation_preds) > 1:
                pairwise_sims = []
                for i, p1 in enumerate(variation_preds):
                    for j, p2 in enumerate(variation_preds):
                        if i < j:
                            pairwise_sims.append(self.compare_fn(p1, p2))
                sensitivity = 1.0 - np.mean(pairwise_sims) if pairwise_sims else 0.0
            else:
                sensitivity = 0.0
            all_sensitivities.append(sensitivity)
        
        # Aggregate scores
        avg_consistency = float(np.mean(all_consistencies)) if all_consistencies else 1.0
        avg_sensitivity = float(np.mean(all_sensitivities)) if all_sensitivities else 0.0
        
        # Invariance score: high consistency + low sensitivity = high invariance
        invariance_score = avg_consistency * (1 - avg_sensitivity)
        
        return RobustnessScore(
            transform_name=transform.name,
            sensitivity=avg_sensitivity,
            consistency=avg_consistency,
            invariance_score=invariance_score,
            n_samples_tested=len(images),
        )
    
    def _default_compare(self, pred1: Any, pred2: Any) -> float:
        """Default comparison function for predictions."""
        # Handle different prediction types
        if isinstance(pred1, (int, str)):
            # Classification: exact match
            return 1.0 if pred1 == pred2 else 0.0
        elif isinstance(pred1, np.ndarray):
            # Numeric array: cosine similarity
            if pred1.shape != pred2.shape:
                return 0.0
            norm1 = np.linalg.norm(pred1)
            norm2 = np.linalg.norm(pred2)
            if norm1 == 0 or norm2 == 0:
                return 1.0 if norm1 == norm2 else 0.0
            return float(np.dot(pred1.flatten(), pred2.flatten()) / (norm1 * norm2))
        elif isinstance(pred1, (list, tuple)):
            # List of predictions: average similarity
            if len(pred1) != len(pred2):
                return 0.0
            sims = [self._default_compare(p1, p2) for p1, p2 in zip(pred1, pred2)]
            return np.mean(sims)
        elif isinstance(pred1, float):
            # Regression: relative difference
            max_val = max(abs(pred1), abs(pred2), 1e-8)
            return 1.0 - min(abs(pred1 - pred2) / max_val, 1.0)
        else:
            # Unknown type: try equality
            return 1.0 if pred1 == pred2 else 0.0
    
    def _apply_transform_default(
        self,
        image: np.ndarray,
        transform: Transform,
        seed: int | None = None,
    ) -> np.ndarray:
        """Apply transform using albumentations if available."""
        try:
            import albumentations as A
            
            if seed is not None:
                np.random.seed(seed)
            
            albu_transform = getattr(A, transform.name, None)
            if albu_transform is not None:
                params = transform.parameters or {}
                aug = A.Compose([albu_transform(p=1.0, **params)])
                result = aug(image=image)
                return result["image"]
        except (ImportError, Exception):
            pass
        
        # Fallback: return copy
        return image.copy()
    
    def generate_html_report(self, report: RobustnessReport, output_dir: Path) -> Path:
        """Generate HTML report for robustness evaluation."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        html_path = output_dir / "robustness_report.html"
        html_content = self._generate_html(report)
        html_path.write_text(html_content, encoding="utf-8")
        
        json_path = output_dir / "robustness_report.json"
        json_path.write_text(report.to_json(), encoding="utf-8")
        
        return html_path
    
    def _generate_html(self, report: RobustnessReport) -> str:
        """Generate HTML content."""
        rows = ""
        for score in sorted(report.scores, key=lambda s: s.invariance_score, reverse=True):
            color = "green" if not score.is_fragile else "orange" if score.invariance_score > 0.5 else "red"
            bar_width = int(score.invariance_score * 100)
            
            rows += f"""
            <tr>
                <td><strong>{score.transform_name}</strong></td>
                <td>{score.sensitivity:.3f}</td>
                <td>{score.consistency:.3f}</td>
                <td>
                    <div class="bar-container">
                        <div class="bar" style="width: {bar_width}%; background: {color};"></div>
                    </div>
                    {score.invariance_score:.3f}
                </td>
                <td style="color: {color};">{score.robustness_label}</td>
            </tr>
            """
        
        fragile_list = ", ".join(report.fragile_transforms) or "None"
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Robustness Report - {report.policy_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee; padding: 2rem; min-height: 100vh;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #00d4ff; margin-bottom: 0.5rem; }}
        .meta {{ color: #888; margin-bottom: 2rem; }}
        .card {{ background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }}
        h2 {{ color: #7c3aed; margin-bottom: 1rem; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #333; }}
        th {{ color: #00d4ff; }}
        .bar-container {{ width: 100px; height: 8px; background: #333; border-radius: 4px; display: inline-block; margin-right: 8px; }}
        .bar {{ height: 100%; border-radius: 4px; }}
        .warning {{ background: rgba(245, 158, 11, 0.2); border: 1px solid #f59e0b; padding: 1rem; border-radius: 8px; margin-top: 1rem; }}
        .overall {{ font-size: 2rem; color: #00d4ff; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Robustness Analysis Report</h1>
        <p class="meta">Policy: {report.policy_name} | Domain: {report.domain}</p>
        
        <div class="card">
            <h2>Overall Robustness</h2>
            <p class="overall">{report.overall_robustness:.2f}</p>
        </div>
        
        <div class="card">
            <h2>Per-Transform Robustness</h2>
            <table>
                <tr>
                    <th>Transform</th>
                    <th>Sensitivity</th>
                    <th>Consistency</th>
                    <th>Invariance</th>
                    <th>Status</th>
                </tr>
                {rows}
            </table>
        </div>
        
        {"<div class='card warning'><h2>⚠️ Fragile Transforms</h2><p>Model is not robust to: " + fragile_list + "</p></div>" if report.fragile_transforms else ""}
        
        <p class="meta">Generated: {report.timestamp}</p>
    </div>
</body>
</html>
"""

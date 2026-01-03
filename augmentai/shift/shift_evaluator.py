"""
Shift evaluator for measuring model robustness under distribution shifts.

Evaluates model performance on original vs shifted data to quantify
robustness and identify vulnerable conditions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from augmentai.shift.shift_generator import ShiftConfig


@dataclass
class ShiftResult:
    """Evaluation result for one shift type/severity.
    
    Attributes:
        shift_name: Name of the shift applied
        severity: Severity level (0-1)
        original_accuracy: Accuracy on original samples
        shifted_accuracy: Accuracy on shifted samples
        degradation: Absolute performance drop
        robustness_score: Normalized robustness (0=fragile, 1=robust)
        n_samples: Number of samples tested
    """
    shift_name: str
    severity: float
    original_accuracy: float
    shifted_accuracy: float
    degradation: float = 0.0
    robustness_score: float = 0.0
    n_samples: int = 0
    
    def __post_init__(self) -> None:
        """Compute derived metrics."""
        self.degradation = self.original_accuracy - self.shifted_accuracy
        # Robustness: 1 if no degradation, 0 if total degradation
        if self.original_accuracy > 0:
            self.robustness_score = max(0, self.shifted_accuracy / self.original_accuracy)
        else:
            self.robustness_score = 1.0 if self.shifted_accuracy >= 0 else 0.0
    
    @property
    def is_fragile(self) -> bool:
        """Check if model is fragile to this shift."""
        return self.robustness_score < 0.8
    
    @property
    def severity_label(self) -> str:
        """Get human-readable severity label."""
        if self.severity < 0.3:
            return "mild"
        elif self.severity < 0.6:
            return "moderate"
        else:
            return "severe"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "shift_name": self.shift_name,
            "severity": self.severity,
            "severity_label": self.severity_label,
            "original_accuracy": self.original_accuracy,
            "shifted_accuracy": self.shifted_accuracy,
            "degradation": self.degradation,
            "robustness_score": self.robustness_score,
            "is_fragile": self.is_fragile,
            "n_samples": self.n_samples,
        }


@dataclass
class ShiftReport:
    """Complete shift evaluation report.
    
    Attributes:
        results: List of shift evaluation results
        most_fragile_shift: Shift type causing most degradation
        most_robust_shift: Shift type with best robustness
        overall_robustness: Average robustness across all shifts
        created_at: Report creation timestamp
        metadata: Additional metadata
    """
    results: list[ShiftResult] = field(default_factory=list)
    most_fragile_shift: str = ""
    most_robust_shift: str = ""
    overall_robustness: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Compute summary metrics."""
        self._compute_summary()
    
    def _compute_summary(self) -> None:
        """Compute summary statistics."""
        if not self.results:
            return
        
        # Find most fragile (lowest robustness)
        sorted_by_robustness = sorted(self.results, key=lambda r: r.robustness_score)
        self.most_fragile_shift = sorted_by_robustness[0].shift_name
        self.most_robust_shift = sorted_by_robustness[-1].shift_name
        
        # Average robustness
        self.overall_robustness = np.mean([r.robustness_score for r in self.results])
    
    def get_fragile_shifts(self) -> list[ShiftResult]:
        """Get list of shifts where model is fragile."""
        return [r for r in self.results if r.is_fragile]
    
    def summary(self) -> str:
        """Get one-line summary."""
        n_fragile = len(self.get_fragile_shifts())
        return (
            f"Tested {len(self.results)} shifts: "
            f"overall robustness {self.overall_robustness:.1%}, "
            f"{n_fragile} fragile conditions"
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary(),
            "overall_robustness": self.overall_robustness,
            "most_fragile_shift": self.most_fragile_shift,
            "most_robust_shift": self.most_robust_shift,
            "n_shifts_tested": len(self.results),
            "n_fragile": len(self.get_fragile_shifts()),
            "results": [r.to_dict() for r in self.results],
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export to JSON."""
        return json.dumps(self.to_dict(), indent=indent)


class ShiftEvaluator:
    """Evaluate model robustness under distribution shifts.
    
    Compares model performance on original vs shifted samples
    to measure robustness and identify vulnerable conditions.
    
    Example:
        def predict_fn(image_path):
            img = load_image(image_path)
            pred_class, confidence = model.predict(img)
            return pred_class, confidence
        
        evaluator = ShiftEvaluator(
            predict_fn=predict_fn,
            true_labels={"img1": "cat", "img2": "dog", ...},
        )
        
        report = evaluator.evaluate_all_shifts(
            original_samples,
            shifts=["brightness", "blur", "noise"],
            output_dir=Path("./shift_report"),
        )
        print(report.summary())
    """
    
    def __init__(
        self,
        predict_fn: Callable[[Path], tuple[str, float]],
        true_labels: dict[str, str],
    ):
        """Initialize shift evaluator.
        
        Args:
            predict_fn: Function that takes image path, returns (pred_label, confidence)
            true_labels: Dict mapping sample ID to ground truth label
        """
        self.predict_fn = predict_fn
        self.true_labels = true_labels
    
    def evaluate_shift(
        self,
        original_samples: list[Path],
        shifted_samples: list[Path],
        shift: ShiftConfig,
    ) -> ShiftResult:
        """Evaluate model on original vs shifted samples.
        
        Args:
            original_samples: Paths to original images
            shifted_samples: Paths to shifted versions
            shift: The shift configuration applied
            
        Returns:
            ShiftResult with comparison metrics
        """
        # Evaluate original
        original_correct = 0
        for path in original_samples:
            sample_id = path.stem
            if sample_id not in self.true_labels:
                continue
            
            pred_label, _ = self.predict_fn(path)
            if pred_label == self.true_labels[sample_id]:
                original_correct += 1
        
        n_samples = len(original_samples)
        original_accuracy = original_correct / n_samples if n_samples > 0 else 0
        
        # Evaluate shifted
        shifted_correct = 0
        for path in shifted_samples:
            # Extract original sample ID from shifted filename
            sample_id = path.stem.replace("_shifted", "")
            if sample_id not in self.true_labels:
                continue
            
            pred_label, _ = self.predict_fn(path)
            if pred_label == self.true_labels[sample_id]:
                shifted_correct += 1
        
        shifted_accuracy = shifted_correct / n_samples if n_samples > 0 else 0
        
        return ShiftResult(
            shift_name=shift.name,
            severity=shift.severity,
            original_accuracy=original_accuracy,
            shifted_accuracy=shifted_accuracy,
            n_samples=n_samples,
        )
    
    def evaluate_all_shifts(
        self,
        samples: list[Path],
        shifts: list[ShiftConfig],
        output_dir: Path,
    ) -> ShiftReport:
        """Evaluate model across multiple shifts.
        
        Args:
            samples: Original image paths
            shifts: List of shift configurations to test
            output_dir: Directory for shifted images and reports
            
        Returns:
            ShiftReport with all results
        """
        from augmentai.shift.shift_generator import ShiftGenerator
        
        output_dir = Path(output_dir)
        generator = ShiftGenerator()
        
        results = []
        for shift in shifts:
            # Generate shifted samples
            shift_dir = output_dir / shift.name
            shifted_samples = generator.generate_shifted_samples(
                samples, shift, shift_dir
            )
            
            # Evaluate
            result = self.evaluate_shift(samples, shifted_samples, shift)
            results.append(result)
        
        report = ShiftReport(
            results=results,
            metadata={
                "n_original_samples": len(samples),
                "shifts_tested": [s.name for s in shifts],
            },
        )
        
        # Save report
        self.save_report(report, output_dir)
        
        return report
    
    def save_report(self, report: ShiftReport, output_dir: Path) -> Path:
        """Save report to JSON and generate HTML."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_dir / "shift_report.json"
        json_path.write_text(report.to_json(), encoding="utf-8")
        
        # Generate HTML
        html_path = output_dir / "shift_report.html"
        html_path.write_text(self._generate_html(report), encoding="utf-8")
        
        return html_path
    
    def _generate_html(self, report: ShiftReport) -> str:
        """Generate HTML report."""
        results_html = ""
        for result in report.results:
            robustness_color = "#10b981" if result.robustness_score > 0.8 else (
                "#f59e0b" if result.robustness_score > 0.6 else "#ef4444"
            )
            results_html += f"""
            <tr>
                <td>{result.shift_name}</td>
                <td>{result.severity_label} ({result.severity:.1f})</td>
                <td>{result.original_accuracy:.1%}</td>
                <td>{result.shifted_accuracy:.1%}</td>
                <td style="color: {'#ef4444' if result.degradation > 0.1 else '#666'}">
                    {result.degradation:+.1%}
                </td>
                <td style="color: {robustness_color}; font-weight: bold;">
                    {result.robustness_score:.1%}
                </td>
            </tr>
            """
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Distribution Shift Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{ color: #888; margin-bottom: 2rem; }}
        .summary {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}
        .summary-stat {{
            display: inline-block;
            margin-right: 2rem;
        }}
        .summary-value {{ font-size: 1.5rem; font-weight: bold; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            overflow: hidden;
        }}
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        th {{
            background: rgba(255,255,255,0.05);
            font-weight: 600;
            color: #aaa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Distribution Shift Report</h1>
        <p class="subtitle">Generated on {report.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <div class="summary-stat">
                <div class="summary-value">{report.overall_robustness:.1%}</div>
                <div style="color: #888;">Overall Robustness</div>
            </div>
            <div class="summary-stat">
                <div class="summary-value">{len(report.results)}</div>
                <div style="color: #888;">Shifts Tested</div>
            </div>
            <div class="summary-stat">
                <div class="summary-value" style="color: #ef4444;">{len(report.get_fragile_shifts())}</div>
                <div style="color: #888;">Fragile Conditions</div>
            </div>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Shift</th>
                    <th>Severity</th>
                    <th>Original Acc</th>
                    <th>Shifted Acc</th>
                    <th>Degradation</th>
                    <th>Robustness</th>
                </tr>
            </thead>
            <tbody>
                {results_html if results_html else '<tr><td colspan="6" style="text-align:center;color:#666;">No results</td></tr>'}
            </tbody>
        </table>
        
        <div style="margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.03); border-radius: 8px;">
            <strong>Most Fragile:</strong> {report.most_fragile_shift} &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Most Robust:</strong> {report.most_robust_shift}
        </div>
    </div>
</body>
</html>"""

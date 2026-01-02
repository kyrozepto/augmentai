"""
Augmentation ablation analysis.

Measures the contribution of each transform through leave-one-out analysis.
Helps identify which augmentations help or hurt model performance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from augmentai.core.policy import Policy, Transform


@dataclass
class AblationResult:
    """Result of ablating a single transform."""
    
    transform_name: str
    baseline_score: float      # Score with all transforms
    ablated_score: float       # Score without this transform
    contribution: float | None = None  # baseline - ablated (positive = helpful)
    rank: int = 0              # Rank by contribution (1 = most helpful)
    parameters: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Only calculate if not explicitly provided
        if self.contribution is None:
            self.contribution = self.baseline_score - self.ablated_score
    
    @property
    def is_helpful(self) -> bool:
        """Check if this transform improves performance."""
        return self.contribution > 0
    
    @property
    def impact_label(self) -> str:
        """Get human-readable impact label."""
        if self.contribution > 0.05:
            return "very helpful"
        elif self.contribution > 0.01:
            return "helpful"
        elif self.contribution > -0.01:
            return "neutral"
        elif self.contribution > -0.05:
            return "slightly harmful"
        else:
            return "harmful"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_name": self.transform_name,
            "baseline_score": self.baseline_score,
            "ablated_score": self.ablated_score,
            "contribution": self.contribution,
            "rank": self.rank,
            "parameters": self.parameters,
            "is_helpful": self.is_helpful,
            "impact_label": self.impact_label,
        }


@dataclass
class AblationReport:
    """Complete ablation report for a policy."""
    
    policy_name: str
    domain: str
    baseline_score: float
    results: list[AblationResult] = field(default_factory=list)
    recommended_removes: list[str] = field(default_factory=list)
    recommended_keeps: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self._rank_results()
        self._compute_recommendations()
    
    def _rank_results(self):
        """Rank results by contribution (highest first)."""
        sorted_results = sorted(self.results, key=lambda r: r.contribution, reverse=True)
        for i, result in enumerate(sorted_results):
            result.rank = i + 1
        self.results = sorted_results
    
    def _compute_recommendations(self):
        """Compute recommended removes/keeps."""
        self.recommended_keeps = [r.transform_name for r in self.results if r.is_helpful]
        self.recommended_removes = [r.transform_name for r in self.results if not r.is_helpful]
    
    @property
    def summary(self) -> str:
        """Get one-line summary."""
        helpful = len(self.recommended_keeps)
        harmful = len(self.recommended_removes)
        return f"Ablation: {helpful} helpful, {harmful} harmful transforms"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "domain": self.domain,
            "baseline_score": self.baseline_score,
            "results": [r.to_dict() for r in self.results],
            "recommended_removes": self.recommended_removes,
            "recommended_keeps": self.recommended_keeps,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class AugmentationAblation:
    """
    Perform ablation analysis on augmentation policies.
    
    Uses leave-one-out analysis to measure each transform's contribution:
    1. Evaluate baseline with all transforms
    2. For each transform, evaluate without it
    3. Contribution = baseline_score - ablated_score
    
    Higher contribution means the transform helps more.
    """
    
    def __init__(
        self,
        eval_fn: Callable[[Policy], float],
        higher_is_better: bool = True,
        n_runs: int = 1,
        seed: int = 42,
    ) -> None:
        """
        Initialize ablation analyzer.
        
        Args:
            eval_fn: Function that takes a Policy and returns a score
            higher_is_better: If True, higher scores are better
            n_runs: Number of evaluation runs to average (for stability)
            seed: Random seed for reproducibility
        """
        self.eval_fn = eval_fn
        self.higher_is_better = higher_is_better
        self.n_runs = n_runs
        self.seed = seed
    
    def ablate(self, policy: Policy) -> AblationReport:
        """
        Perform leave-one-out ablation analysis.
        
        Args:
            policy: The policy to analyze
            
        Returns:
            AblationReport with results for each transform
        """
        # Evaluate baseline with all transforms
        baseline_score = self._evaluate(policy)
        
        results = []
        
        for transform in policy.transforms:
            # Create policy without this transform
            ablated_policy = self._create_ablated_policy(policy, transform.name)
            
            # Evaluate ablated policy
            ablated_score = self._evaluate(ablated_policy)
            
            # Calculate contribution
            if self.higher_is_better:
                contribution = baseline_score - ablated_score
            else:
                contribution = ablated_score - baseline_score
            
            result = AblationResult(
                transform_name=transform.name,
                baseline_score=baseline_score,
                ablated_score=ablated_score,
                contribution=contribution,
                parameters=transform.parameters or {},
            )
            results.append(result)
        
        return AblationReport(
            policy_name=policy.name,
            domain=policy.domain,
            baseline_score=baseline_score,
            results=results,
        )
    
    def _evaluate(self, policy: Policy) -> float:
        """Evaluate a policy, averaging over multiple runs if specified."""
        if self.n_runs == 1:
            return self.eval_fn(policy)
        
        scores = []
        for _ in range(self.n_runs):
            scores.append(self.eval_fn(policy))
        
        return sum(scores) / len(scores)
    
    def _create_ablated_policy(self, policy: Policy, remove_transform: str) -> Policy:
        """Create a copy of policy without the specified transform."""
        ablated_transforms = [
            t for t in policy.transforms if t.name != remove_transform
        ]
        
        return Policy(
            name=f"{policy.name}_ablate_{remove_transform}",
            domain=policy.domain,
            transforms=ablated_transforms,
            description=f"Ablated: removed {remove_transform}",
            magnitude_bins=policy.magnitude_bins,
            num_ops=policy.num_ops,
        )
    
    def generate_html_report(self, report: AblationReport, output_dir: Path) -> Path:
        """
        Generate an HTML report for the ablation results.
        
        Args:
            report: The ablation report
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated HTML file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        html_path = output_dir / "ablation_report.html"
        html_content = self._generate_html(report)
        html_path.write_text(html_content, encoding="utf-8")
        
        # Also save JSON
        json_path = output_dir / "ablation_report.json"
        json_path.write_text(report.to_json(), encoding="utf-8")
        
        return html_path
    
    def _generate_html(self, report: AblationReport) -> str:
        """Generate HTML content for the report."""
        rows = ""
        for result in report.results:
            color = "green" if result.is_helpful else "red"
            arrow = "↑" if result.is_helpful else "↓"
            
            rows += f"""
            <tr>
                <td>{result.rank}</td>
                <td><strong>{result.transform_name}</strong></td>
                <td>{result.baseline_score:.4f}</td>
                <td>{result.ablated_score:.4f}</td>
                <td style="color: {color};">{arrow} {result.contribution:+.4f}</td>
                <td>{result.impact_label}</td>
            </tr>
            """
        
        keeps = ", ".join(report.recommended_keeps) or "None"
        removes = ", ".join(report.recommended_removes) or "None"
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ablation Report - {report.policy_name}</title>
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
        .card {{ 
            background: rgba(255,255,255,0.05); border-radius: 12px; 
            padding: 1.5rem; margin-bottom: 1.5rem;
            backdrop-filter: blur(10px);
        }}
        h2 {{ color: #7c3aed; margin-bottom: 1rem; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #333; }}
        th {{ color: #00d4ff; }}
        .highlight {{ color: #10b981; }}
        .warning {{ color: #f59e0b; }}
        .recommendation {{ display: flex; gap: 2rem; margin-top: 1rem; }}
        .rec-box {{ flex: 1; padding: 1rem; border-radius: 8px; }}
        .keep {{ background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; }}
        .remove {{ background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Ablation Analysis Report</h1>
        <p class="meta">Policy: {report.policy_name} | Domain: {report.domain} | Baseline: {report.baseline_score:.4f}</p>
        
        <div class="card">
            <h2>Transform Contributions</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Transform</th>
                    <th>Baseline</th>
                    <th>Ablated</th>
                    <th>Contribution</th>
                    <th>Impact</th>
                </tr>
                {rows}
            </table>
        </div>
        
        <div class="card">
            <h2>Recommendations</h2>
            <div class="recommendation">
                <div class="rec-box keep">
                    <strong>✅ Keep These</strong>
                    <p>{keeps}</p>
                </div>
                <div class="rec-box remove">
                    <strong>❌ Consider Removing</strong>
                    <p>{removes}</p>
                </div>
            </div>
        </div>
        
        <p class="meta">Generated: {report.timestamp}</p>
    </div>
</body>
</html>
"""

"""
Repair report generation for model-guided data repair.

Generates HTML and JSON reports summarizing repair suggestions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from augmentai.repair.repair_suggestions import RepairSuggestion, RepairAction


@dataclass
class RepairReport:
    """Complete data repair report.
    
    Attributes:
        n_samples: Total number of samples analyzed
        n_keep: Samples to keep as-is
        n_remove: Samples suggested for removal
        n_relabel: Samples suggested for relabeling
        n_reweight: Samples suggested for reweighting
        n_review: Samples needing manual review
        suggestions: List of all repair suggestions
        created_at: Report creation timestamp
        metadata: Additional metadata
    """
    n_samples: int
    n_keep: int = 0
    n_remove: int = 0
    n_relabel: int = 0
    n_reweight: int = 0
    n_review: int = 0
    suggestions: list[RepairSuggestion] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Compute counts from suggestions."""
        self._compute_counts()
    
    def _compute_counts(self) -> None:
        """Count suggestions by action type."""
        counts = {action: 0 for action in RepairAction}
        for suggestion in self.suggestions:
            counts[suggestion.action] += 1
        
        self.n_remove = counts[RepairAction.REMOVE]
        self.n_relabel = counts[RepairAction.RELABEL]
        self.n_reweight = counts[RepairAction.REWEIGHT]
        self.n_review = counts[RepairAction.REVIEW]
        self.n_keep = self.n_samples - len(self.suggestions)
    
    @property
    def repair_rate(self) -> float:
        """Percentage of samples needing some action."""
        if self.n_samples == 0:
            return 0.0
        return len(self.suggestions) / self.n_samples
    
    def summary(self) -> str:
        """Get one-line summary."""
        return (
            f"Analyzed {self.n_samples} samples: "
            f"{self.n_keep} keep, {self.n_remove} remove, "
            f"{self.n_relabel} relabel, {self.n_reweight} reweight, "
            f"{self.n_review} review"
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "n_samples": self.n_samples,
            "n_keep": self.n_keep,
            "n_remove": self.n_remove,
            "n_relabel": self.n_relabel,
            "n_reweight": self.n_reweight,
            "n_review": self.n_review,
            "repair_rate": self.repair_rate,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class RepairReportGenerator:
    """Generate HTML and JSON reports for data repair.
    
    Creates visual reports showing repair suggestions with:
    - Summary statistics
    - Action breakdown charts
    - Detailed suggestion tables
    - Export files for automated processing
    """
    
    def generate(
        self,
        report: RepairReport,
        output_dir: Path,
    ) -> Path:
        """Generate repair report files.
        
        Args:
            report: The repair report
            output_dir: Directory to save reports
            
        Returns:
            Path to the generated HTML file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_path = output_dir / "repair_report.json"
        json_path.write_text(report.to_json(), encoding="utf-8")
        
        # Generate and save HTML report
        html_content = self._generate_html(report)
        html_path = output_dir / "repair_report.html"
        html_path.write_text(html_content, encoding="utf-8")
        
        # Save action-specific CSVs
        self._save_action_csvs(report, output_dir)
        
        return html_path
    
    def _save_action_csvs(
        self,
        report: RepairReport,
        output_dir: Path,
    ) -> None:
        """Save CSV files for each action type."""
        action_suggestions: dict[str, list[RepairSuggestion]] = {
            action.value: [] for action in RepairAction
        }
        
        for suggestion in report.suggestions:
            action_suggestions[suggestion.action.value].append(suggestion)
        
        for action, suggestions in action_suggestions.items():
            if not suggestions:
                continue
            
            csv_path = output_dir / f"{action}_samples.csv"
            lines = ["sample_id,confidence,reason,suggested_label,suggested_weight"]
            for s in suggestions:
                lines.append(
                    f"{s.sample_id},{s.confidence:.3f},"
                    f"\"{s.reason}\",{s.suggested_label or ''},{s.suggested_weight:.3f}"
                )
            csv_path.write_text("\n".join(lines), encoding="utf-8")
    
    def _generate_html(self, report: RepairReport) -> str:
        """Generate HTML content for the report."""
        # Color scheme for actions
        colors = {
            "keep": "#10b981",     # Green
            "remove": "#ef4444",   # Red
            "relabel": "#f59e0b",  # Amber
            "reweight": "#3b82f6", # Blue
            "review": "#8b5cf6",   # Purple
        }
        
        suggestions_html = ""
        for suggestion in report.suggestions[:100]:  # Show top 100
            color = colors.get(suggestion.action.value, "#666")
            suggestions_html += f"""
            <tr>
                <td class="sample-id">{suggestion.sample_id}</td>
                <td><span class="action-badge" style="background: {color}20; color: {color}; border: 1px solid {color}40;">{suggestion.action.value}</span></td>
                <td>{suggestion.confidence:.2f}</td>
                <td class="reason">{suggestion.reason}</td>
                <td>{suggestion.suggested_label or '-'}</td>
                <td>{suggestion.suggested_weight:.2f}</td>
            </tr>
            """
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Repair Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{ color: #888; margin-bottom: 2rem; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }}
        .stat-label {{ color: #888; font-size: 0.9rem; }}
        .stat-keep .stat-value {{ color: {colors['keep']}; }}
        .stat-remove .stat-value {{ color: {colors['remove']}; }}
        .stat-relabel .stat-value {{ color: {colors['relabel']}; }}
        .stat-reweight .stat-value {{ color: {colors['reweight']}; }}
        .stat-review .stat-value {{ color: {colors['review']}; }}
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
        .sample-id {{ font-family: monospace; color: #888; }}
        .reason {{ font-size: 0.85rem; color: #aaa; max-width: 300px; }}
        .action-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 500;
            text-transform: uppercase;
        }}
        .summary-bar {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin-bottom: 2rem;
            font-size: 0.95rem;
            color: #aaa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 Data Repair Report</h1>
        <p class="subtitle">Generated on {report.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary-bar">
            {report.summary()}
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{report.n_samples}</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat-card stat-keep">
                <div class="stat-value">{report.n_keep}</div>
                <div class="stat-label">Keep</div>
            </div>
            <div class="stat-card stat-remove">
                <div class="stat-value">{report.n_remove}</div>
                <div class="stat-label">Remove</div>
            </div>
            <div class="stat-card stat-relabel">
                <div class="stat-value">{report.n_relabel}</div>
                <div class="stat-label">Relabel</div>
            </div>
            <div class="stat-card stat-reweight">
                <div class="stat-value">{report.n_reweight}</div>
                <div class="stat-label">Reweight</div>
            </div>
            <div class="stat-card stat-review">
                <div class="stat-value">{report.n_review}</div>
                <div class="stat-label">Review</div>
            </div>
        </div>
        
        <h2 style="margin: 2rem 0 1rem; font-size: 1.25rem;">Suggestions</h2>
        <table>
            <thead>
                <tr>
                    <th>Sample ID</th>
                    <th>Action</th>
                    <th>Confidence</th>
                    <th>Reason</th>
                    <th>Suggested Label</th>
                    <th>Weight</th>
                </tr>
            </thead>
            <tbody>
                {suggestions_html if suggestions_html else '<tr><td colspan="6" style="text-align:center;color:#666;">No repairs needed</td></tr>'}
            </tbody>
        </table>
    </div>
</body>
</html>"""
        
        return html

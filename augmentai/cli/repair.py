"""
CLI command for model-guided data repair.

Usage:
    augmentai repair ./dataset --eval-script repair_eval.py --output report/
    augmentai repair ./dataset --mock  # For testing
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from augmentai.repair import (
    SampleAnalyzer,
    DataRepair,
    RepairReport,
    RepairReportGenerator,
)
from augmentai.repair.repair_suggestions import RepairAction


console = Console()


def repair(
    dataset_path: Path = typer.Argument(
        ...,
        help="Path to dataset directory to analyze",
        exists=True,
    ),
    eval_script: Optional[Path] = typer.Option(
        None,
        "--eval-script", "-e",
        help="Python script with uncertainty_fn, loss_fn, predict_fn functions",
    ),
    output: Path = typer.Option(
        Path("./repair_report"),
        "--output", "-o",
        help="Output directory for repair report",
    ),
    uncertainty_threshold: float = typer.Option(
        0.7,
        "--uncertainty-threshold",
        help="Threshold for high uncertainty samples",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use mock evaluation for testing",
    ),
) -> None:
    """
    Analyze dataset samples for potential data quality issues.
    
    Uses model feedback to suggest repair actions:
    - REMOVE: Corrupt or extremely noisy samples
    - RELABEL: Likely mislabeled samples
    - REWEIGHT: Samples that should have different training weight
    - REVIEW: Samples needing manual inspection
    
    Examples:
        augmentai repair ./dataset --eval-script repair_eval.py
        augmentai repair ./dataset --mock --output repair/
    """
    console.print(Panel.fit(
        "[bold blue]AugmentAI[/bold blue] - Model-Guided Data Repair",
        subtitle="Analyze and repair data quality issues"
    ))
    
    # Discover samples
    samples = _discover_samples(dataset_path)
    if not samples:
        console.print(f"[red]Error:[/red] No samples found in {dataset_path}")
        raise typer.Exit(1)
    
    console.print(f"[green]✓[/green] Found {len(samples)} samples")
    
    # Set up evaluation functions
    if mock:
        uncertainty_fn, loss_fn, predict_fn = _create_mock_functions()
        console.print("[yellow]Using mock evaluation functions[/yellow]")
    elif eval_script:
        uncertainty_fn, loss_fn, predict_fn = _load_eval_functions(eval_script)
        console.print(f"[green]✓[/green] Loaded evaluation functions from {eval_script}")
    else:
        console.print("[red]Error:[/red] Must provide --eval-script or use --mock")
        raise typer.Exit(1)
    
    # Analyze samples
    console.print("\n[bold]Analyzing samples...[/bold]")
    analyzer = SampleAnalyzer(
        uncertainty_fn=uncertainty_fn,
        loss_fn=loss_fn,
        predict_fn=predict_fn,
    )
    
    with console.status("Computing model feedback..."):
        analyses = analyzer.analyze_dataset(samples)
    
    console.print(f"[green]✓[/green] Analyzed {len(analyses)} samples")
    
    # Generate repair suggestions
    console.print("\n[bold]Generating repair suggestions...[/bold]")
    repair_suggester = DataRepair(
        uncertainty_threshold=uncertainty_threshold,
    )
    suggestions = repair_suggester.suggest_repairs(analyses)
    
    # Create report
    report = RepairReport(
        n_samples=len(samples),
        suggestions=suggestions,
    )
    
    # Display results
    _display_results(report)
    
    # Generate report files
    generator = RepairReportGenerator()
    html_path = generator.generate(report, output)
    console.print(f"\n[green]✓[/green] Report saved to: {html_path}")


def _discover_samples(dataset_path: Path) -> list[tuple[Path, str]]:
    """Discover samples in dataset directory.
    
    Expects structure: dataset/class_name/image.jpg
    """
    samples = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    
    for class_dir in dataset_path.iterdir():
        if not class_dir.is_dir():
            continue
        if class_dir.name.startswith("."):
            continue
        
        label = class_dir.name
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in extensions:
                samples.append((img_file, label))
    
    return samples


def _create_mock_functions():
    """Create mock evaluation functions for testing."""
    import random
    import hashlib
    
    def uncertainty_fn(path: Path) -> float:
        # Deterministic based on filename
        h = int(hashlib.md5(str(path).encode()).hexdigest(), 16)
        return (h % 100) / 100
    
    def loss_fn(path: Path, label: str) -> float:
        h = int(hashlib.md5(f"{path}{label}".encode()).hexdigest(), 16)
        return (h % 500) / 100  # 0.0 to 5.0
    
    def predict_fn(path: Path) -> tuple[str, float]:
        h = int(hashlib.md5(str(path).encode()).hexdigest(), 16)
        confidence = 0.5 + (h % 50) / 100
        # Sometimes predict wrong label
        if h % 7 == 0:
            return "wrong_class", confidence
        return path.parent.name, confidence
    
    return uncertainty_fn, loss_fn, predict_fn


def _load_eval_functions(script_path: Path):
    """Load evaluation functions from a Python script."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("eval_module", script_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from {script_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    required_fns = ["uncertainty_fn", "loss_fn", "predict_fn"]
    for fn_name in required_fns:
        if not hasattr(module, fn_name):
            raise ValueError(f"Script must define '{fn_name}' function")
    
    return module.uncertainty_fn, module.loss_fn, module.predict_fn


def _display_results(report: RepairReport) -> None:
    """Display repair results in a table."""
    console.print(f"\n[bold]Summary:[/bold] {report.summary()}")
    console.print(f"Repair rate: {report.repair_rate:.1%}")
    
    if not report.suggestions:
        console.print("\n[green]✓ No repairs needed![/green]")
        return
    
    table = Table(title="Top Repair Suggestions", show_header=True)
    table.add_column("Sample ID", style="dim")
    table.add_column("Action", style="bold")
    table.add_column("Confidence")
    table.add_column("Reason")
    
    colors = {
        RepairAction.REMOVE: "red",
        RepairAction.RELABEL: "yellow",
        RepairAction.REWEIGHT: "blue",
        RepairAction.REVIEW: "magenta",
        RepairAction.KEEP: "green",
    }
    
    for suggestion in report.suggestions[:15]:  # Show top 15
        color = colors.get(suggestion.action, "white")
        table.add_row(
            suggestion.sample_id,
            f"[{color}]{suggestion.action.value}[/{color}]",
            f"{suggestion.confidence:.2f}",
            suggestion.reason[:50] + "..." if len(suggestion.reason) > 50 else suggestion.reason,
        )
    
    console.print(table)
    
    if len(report.suggestions) > 15:
        console.print(f"\n[dim]... and {len(report.suggestions) - 15} more suggestions[/dim]")

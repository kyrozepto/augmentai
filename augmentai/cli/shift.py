"""
CLI command for domain shift simulation and robustness testing.

Usage:
    augmentai shift ./dataset --eval-script model.py --output shift_report/
    augmentai shift ./dataset --mock --shifts brightness,blur,noise
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from augmentai.shift import (
    ShiftGenerator,
    ShiftEvaluator,
)


console = Console()


def shift(
    dataset_path: Path = typer.Argument(
        ...,
        help="Path to dataset directory to test",
        exists=True,
    ),
    eval_script: Optional[Path] = typer.Option(
        None,
        "--eval-script", "-e",
        help="Python script with predict_fn(path) -> (label, confidence)",
    ),
    output: Path = typer.Option(
        Path("./shift_report"),
        "--output", "-o",
        help="Output directory for shift report",
    ),
    shifts: str = typer.Option(
        "brightness,contrast,noise,blur",
        "--shifts", "-s",
        help="Comma-separated list of shifts to test",
    ),
    severity: float = typer.Option(
        0.5,
        "--severity",
        help="Shift severity (0.0-1.0)",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use mock evaluation for testing",
    ),
) -> None:
    """
    Test model robustness under distribution shifts.
    
    Generates shifted versions of samples and evaluates performance degradation.
    
    Examples:
        augmentai shift ./dataset --eval-script model.py
        augmentai shift ./dataset --mock --shifts brightness,blur,noise
    """
    console.print(Panel.fit(
        "[bold blue]AugmentAI[/bold blue] - Domain Shift Simulation",
        subtitle="Test model robustness under distribution shifts"
    ))
    
    # Discover samples
    samples, labels = _discover_samples(dataset_path)
    if not samples:
        console.print(f"[red]Error:[/red] No samples found in {dataset_path}")
        raise typer.Exit(1)
    
    console.print(f"[green]✓[/green] Found {len(samples)} samples")
    
    # Set up prediction function
    if mock:
        predict_fn = _create_mock_predict_fn(labels)
        console.print("[yellow]Using mock prediction function[/yellow]")
    elif eval_script:
        predict_fn = _load_predict_fn(eval_script)
        console.print(f"[green]✓[/green] Loaded prediction function from {eval_script}")
    else:
        console.print("[red]Error:[/red] Must provide --eval-script or use --mock")
        raise typer.Exit(1)
    
    # Parse shifts
    shift_names = [s.strip() for s in shifts.split(",")]
    generator = ShiftGenerator()
    
    # Display available shifts
    available = generator.list_shifts()
    invalid = [s for s in shift_names if s not in available]
    if invalid:
        console.print(f"[yellow]Warning:[/yellow] Unknown shifts: {invalid}")
        console.print(f"Available: {', '.join(available)}")
        shift_names = [s for s in shift_names if s in available]
    
    if not shift_names:
        console.print("[red]Error:[/red] No valid shifts to test")
        raise typer.Exit(1)
    
    console.print(f"\n[bold]Testing {len(shift_names)} shifts:[/bold] {', '.join(shift_names)}")
    
    # Create shift configs with specified severity
    shift_configs = []
    for name in shift_names:
        config = generator.get_shift(name)
        config = config.with_severity(severity)
        shift_configs.append(config)
    
    # Run evaluation
    console.print("\n[bold]Running shift evaluation...[/bold]")
    evaluator = ShiftEvaluator(
        predict_fn=predict_fn,
        true_labels=labels,
    )
    
    with console.status("Generating shifts and evaluating..."):
        report = evaluator.evaluate_all_shifts(
            samples,
            shift_configs,
            Path(output),
        )
    
    # Display results
    _display_results(report)
    
    console.print(f"\n[green]✓[/green] Report saved to: {output}")


def _discover_samples(dataset_path: Path) -> tuple[list[Path], dict[str, str]]:
    """Discover samples and build label mapping."""
    samples = []
    labels = {}
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    
    for class_dir in dataset_path.iterdir():
        if not class_dir.is_dir():
            continue
        if class_dir.name.startswith("."):
            continue
        
        label = class_dir.name
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in extensions:
                samples.append(img_file)
                labels[img_file.stem] = label
    
    return samples, labels


def _create_mock_predict_fn(true_labels: dict[str, str]):
    """Create mock prediction function for testing."""
    import hashlib
    
    def predict_fn(path: Path) -> tuple[str, float]:
        sample_id = path.stem.replace("_shifted", "")
        h = int(hashlib.md5(str(path).encode()).hexdigest(), 16)
        
        # Usually predict correctly, sometimes wrong
        if h % 10 < 7:  # 70% accuracy baseline
            pred = true_labels.get(sample_id, "unknown")
        else:
            pred = "wrong_class"
        
        confidence = 0.5 + (h % 50) / 100
        return pred, confidence
    
    return predict_fn


def _load_predict_fn(script_path: Path):
    """Load prediction function from a Python script."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("eval_module", script_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from {script_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, "predict_fn"):
        raise ValueError("Script must define 'predict_fn(path) -> (label, confidence)' function")
    
    return module.predict_fn


def _display_results(report) -> None:
    """Display shift evaluation results."""
    console.print(f"\n[bold]Summary:[/bold] {report.summary()}")
    console.print(f"Overall Robustness: {report.overall_robustness:.1%}")
    
    table = Table(title="Shift Results", show_header=True)
    table.add_column("Shift", style="bold")
    table.add_column("Severity")
    table.add_column("Original Acc")
    table.add_column("Shifted Acc")
    table.add_column("Degradation")
    table.add_column("Robustness")
    
    for result in report.results:
        robustness_style = (
            "green" if result.robustness_score > 0.8 else
            "yellow" if result.robustness_score > 0.6 else
            "red"
        )
        
        table.add_row(
            result.shift_name,
            f"{result.severity_label} ({result.severity:.1f})",
            f"{result.original_accuracy:.1%}",
            f"{result.shifted_accuracy:.1%}",
            f"{result.degradation:+.1%}",
            f"[{robustness_style}]{result.robustness_score:.1%}[/{robustness_style}]",
        )
    
    console.print(table)
    
    if report.get_fragile_shifts():
        fragile_names = [r.shift_name for r in report.get_fragile_shifts()]
        console.print(f"\n[yellow]⚠ Fragile to:[/yellow] {', '.join(fragile_names)}")
    else:
        console.print("\n[green]✓ Model is robust to all tested shifts![/green]")

"""
CLI command for curriculum-aware dataset preparation.

Usage:
    augmentai curriculum ./dataset --eval-script scorer.py --output curriculum/
    augmentai curriculum ./dataset --mock --epochs 100
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from augmentai.curriculum import (
    DifficultyScorer,
    CurriculumScheduler,
)


console = Console()


def curriculum(
    dataset_path: Path = typer.Argument(
        ...,
        help="Path to dataset directory to analyze",
        exists=True,
    ),
    eval_script: Optional[Path] = typer.Option(
        None,
        "--eval-script", "-e",
        help="Python script with loss_fn and optional margin_fn functions",
    ),
    output: Path = typer.Option(
        Path("./curriculum_output"),
        "--output", "-o",
        help="Output directory for curriculum schedule",
    ),
    epochs: int = typer.Option(
        100,
        "--epochs", "-n",
        help="Number of training epochs",
    ),
    pacing: str = typer.Option(
        "linear",
        "--pacing", "-p",
        help="Pacing function: linear, quadratic, exponential, step",
    ),
    warmup: int = typer.Option(
        5,
        "--warmup", "-w",
        help="Number of warmup epochs with easy samples",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use mock evaluation for testing",
    ),
) -> None:
    """
    Create a curriculum learning schedule for your dataset.
    
    Scores samples by difficulty and creates an easy→hard training schedule.
    
    Examples:
        augmentai curriculum ./dataset --eval-script scorer.py --epochs 100
        augmentai curriculum ./dataset --mock --pacing quadratic
    """
    console.print(Panel.fit(
        "[bold blue]AugmentAI[/bold blue] - Curriculum Learning",
        subtitle="Score samples and create easy→hard schedule"
    ))
    
    # Discover samples
    samples = _discover_samples(dataset_path)
    if not samples:
        console.print(f"[red]Error:[/red] No samples found in {dataset_path}")
        raise typer.Exit(1)
    
    console.print(f"[green]✓[/green] Found {len(samples)} samples")
    
    # Set up evaluation functions
    if mock:
        loss_fn, margin_fn = _create_mock_functions()
        console.print("[yellow]Using mock evaluation functions[/yellow]")
    elif eval_script:
        loss_fn, margin_fn = _load_eval_functions(eval_script)
        console.print(f"[green]✓[/green] Loaded evaluation functions from {eval_script}")
    else:
        console.print("[red]Error:[/red] Must provide --eval-script or use --mock")
        raise typer.Exit(1)
    
    # Score samples
    console.print("\n[bold]Scoring sample difficulty...[/bold]")
    scorer = DifficultyScorer(
        loss_fn=loss_fn,
        margin_fn=margin_fn,
    )
    
    with console.status("Computing difficulty scores..."):
        scores = scorer.score_dataset(samples)
    
    # Display summary
    summary = scorer.summary(scores)
    _display_difficulty_summary(summary)
    
    # Create schedule
    console.print("\n[bold]Creating curriculum schedule...[/bold]")
    scheduler = CurriculumScheduler(
        pacing=pacing,
        warmup_epochs=warmup,
    )
    schedule = scheduler.create_schedule(scores, epochs)
    
    # Save output
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    
    # Save schedule
    schedule_path = scheduler.save_schedule(schedule, output / "schedule.json")
    console.print(f"[green]✓[/green] Schedule saved to: {schedule_path}")
    
    # Save difficulty scores
    import json
    scores_path = output / "difficulty_scores.json"
    scores_data = [s.to_dict() for s in scores]
    scores_path.write_text(json.dumps(scores_data, indent=2), encoding="utf-8")
    console.print(f"[green]✓[/green] Difficulty scores saved to: {scores_path}")
    
    # Show schedule preview
    _display_schedule_preview(schedule, epochs)


def _discover_samples(dataset_path: Path) -> list[tuple[Path, str]]:
    """Discover samples in dataset directory."""
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
    import hashlib
    
    def loss_fn(path: Path, label: str) -> float:
        h = int(hashlib.md5(f"{path}{label}".encode()).hexdigest(), 16)
        return (h % 500) / 100  # 0.0 to 5.0
    
    def margin_fn(path: Path) -> float:
        h = int(hashlib.md5(str(path).encode()).hexdigest(), 16)
        return (h % 100) / 100  # 0.0 to 1.0
    
    return loss_fn, margin_fn


def _load_eval_functions(script_path: Path):
    """Load evaluation functions from a Python script."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("eval_module", script_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from {script_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, "loss_fn"):
        raise ValueError("Script must define 'loss_fn(path, label) -> float' function")
    
    margin_fn = getattr(module, "margin_fn", None)
    
    return module.loss_fn, margin_fn


def _display_difficulty_summary(summary: dict) -> None:
    """Display difficulty distribution summary."""
    console.print(f"\n[bold]Difficulty Summary:[/bold]")
    console.print(f"  Samples: {summary['n_samples']}")
    console.print(f"  Mean difficulty: {summary['mean_difficulty']:.3f}")
    console.print(f"  Std difficulty: {summary['std_difficulty']:.3f}")
    
    if "distribution" in summary:
        console.print(f"\n  Distribution:")
        for level, count in summary["distribution"].items():
            bar = "█" * min(30, int(count / summary['n_samples'] * 30))
            console.print(f"    {level:10s}: {count:5d} {bar}")


def _display_schedule_preview(schedule, total_epochs: int) -> None:
    """Display curriculum schedule preview."""
    table = Table(title="Curriculum Schedule Preview", show_header=True)
    table.add_column("Epoch", style="dim")
    table.add_column("Samples Included")
    table.add_column("% of Dataset")
    
    # Show a few key epochs
    epochs_to_show = [0, total_epochs // 4, total_epochs // 2, 3 * total_epochs // 4, total_epochs - 1]
    epochs_to_show = sorted(set(e for e in epochs_to_show if e >= 0 and e < total_epochs))
    
    for epoch in epochs_to_show:
        samples = schedule.get_samples_for_epoch(epoch)
        n_total = schedule.metadata.get("n_total_samples", len(samples))
        pct = len(samples) / n_total * 100 if n_total > 0 else 100
        table.add_row(
            str(epoch),
            str(len(samples)),
            f"{pct:.1f}%",
        )
    
    console.print(table)

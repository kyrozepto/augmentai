"""
CLI command for augmentation ablation analysis.

Usage:
    augmentai ablate policy.yaml --eval-script eval.py --output report/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from augmentai.ablation import AugmentationAblation, AblationReport
from augmentai.core.policy import Policy


console = Console()


def ablate(
    policy_file: Path = typer.Argument(
        ...,
        help="Path to policy YAML/JSON file to analyze",
        exists=True,
    ),
    eval_script: Optional[Path] = typer.Option(
        None,
        "--eval-script", "-e",
        help="Python script with evaluate(policy) -> float function",
    ),
    output: Path = typer.Option(
        Path("./ablation_report"),
        "--output", "-o",
        help="Output directory for ablation report",
    ),
    n_runs: int = typer.Option(
        1,
        "--n-runs", "-n",
        help="Number of evaluation runs to average",
    ),
    higher_is_better: bool = typer.Option(
        True,
        "--higher-is-better/--lower-is-better",
        help="Whether higher scores are better",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use mock evaluation for testing",
    ),
) -> None:
    """
    Perform ablation analysis on an augmentation policy.
    
    Measures the contribution of each transform through leave-one-out analysis.
    
    Examples:
        augmentai ablate policy.yaml --eval-script train_eval.py
        augmentai ablate policy.yaml --mock --output ablation/
    """
    console.print(Panel.fit(
        "[bold blue]AugmentAI[/bold blue] - Augmentation Ablation Analysis",
        subtitle="Measure transform contributions"
    ))
    
    # Load policy
    try:
        policy_content = policy_file.read_text()
        if policy_file.suffix in [".yaml", ".yml"]:
            policy = Policy.from_yaml(policy_content)
        else:
            policy = Policy.from_json(policy_content)
        console.print(f"[green]✓[/green] Loaded policy: {policy.name} ({len(policy.transforms)} transforms)")
    except Exception as e:
        console.print(f"[red]Error loading policy:[/red] {e}")
        raise typer.Exit(1)
    
    # Set up evaluation function
    if mock:
        eval_fn = _create_mock_eval_fn()
        console.print("[yellow]Using mock evaluation function[/yellow]")
    elif eval_script:
        eval_fn = _load_eval_fn(eval_script)
        console.print(f"[green]✓[/green] Loaded evaluation function from {eval_script}")
    else:
        console.print("[red]Error:[/red] Must provide --eval-script or use --mock")
        raise typer.Exit(1)
    
    # Run ablation
    console.print("\n[bold]Running ablation analysis...[/bold]")
    ablation = AugmentationAblation(
        eval_fn=eval_fn,
        higher_is_better=higher_is_better,
        n_runs=n_runs,
    )
    
    with console.status("Evaluating transforms..."):
        report = ablation.ablate(policy)
    
    # Display results
    _display_results(report)
    
    # Generate report
    html_path = ablation.generate_html_report(report, output)
    console.print(f"\n[green]✓[/green] Report saved to: {html_path}")


def _create_mock_eval_fn():
    """Create a mock evaluation function for testing."""
    import random
    
    def mock_eval(policy: Policy) -> float:
        # Mock: score based on number of transforms
        base = 0.7
        bonus = len(policy.transforms) * 0.02
        noise = random.uniform(-0.05, 0.05)
        return min(1.0, base + bonus + noise)
    
    return mock_eval


def _load_eval_fn(script_path: Path):
    """Load evaluation function from a Python script."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("eval_module", script_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from {script_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, "evaluate"):
        raise ValueError(f"Script {script_path} must define an 'evaluate(policy) -> float' function")
    
    return module.evaluate


def _display_results(report: AblationReport) -> None:
    """Display ablation results in a table."""
    console.print(f"\n[bold]Baseline Score:[/bold] {report.baseline_score:.4f}")
    
    table = Table(title="Transform Contributions", show_header=True)
    table.add_column("Rank", style="dim")
    table.add_column("Transform", style="bold")
    table.add_column("Ablated Score")
    table.add_column("Contribution")
    table.add_column("Impact")
    
    for result in report.results:
        color = "green" if result.is_helpful else "red"
        contribution_str = f"[{color}]{result.contribution:+.4f}[/{color}]"
        
        table.add_row(
            str(result.rank),
            result.transform_name,
            f"{result.ablated_score:.4f}",
            contribution_str,
            result.impact_label,
        )
    
    console.print(table)
    
    # Recommendations
    if report.recommended_removes:
        console.print(f"\n[yellow]⚠ Consider removing:[/yellow] {', '.join(report.recommended_removes)}")
    if report.recommended_keeps:
        console.print(f"[green]✓ Most helpful:[/green] {', '.join(report.recommended_keeps[:3])}")

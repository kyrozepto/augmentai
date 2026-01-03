"""
CLI command for AutoSearch.

Provides the `augmentai search` command for automated policy optimization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from augmentai.search import PolicyOptimizer, SearchResult
from augmentai.search.optimizer import OptimizerConfig
from augmentai.utils.progress import print_info, print_success, print_warning


console = Console()


def search(
    dataset: Path = typer.Argument(
        ...,
        help="Path to dataset directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    domain: str = typer.Option(
        "auto",
        "--domain", "-d",
        help="Domain (medical, ocr, satellite, natural) or 'auto' to detect",
    ),
    budget: int = typer.Option(
        50,
        "--budget", "-b",
        help="Maximum number of policy evaluations",
    ),
    output: Path = typer.Option(
        Path("./search_results"),
        "--output", "-o",
        help="Output directory for search results",
    ),
    population: int = typer.Option(
        10,
        "--population", "-p",
        help="Population size per generation",
    ),
    seed: int = typer.Option(
        42,
        "--seed", "-s",
        help="Random seed for reproducibility",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be searched without running",
    ),
) -> None:
    """
    Search for optimal augmentation policy automatically.
    
    Uses evolutionary optimization to find the best policy for your dataset.
    
    Examples:
        augmentai search ./dataset --domain medical --budget 100
        augmentai search ./images --budget 50 --output ./results
    """
    console.print(Panel.fit(
        "[bold blue]AugmentAI AutoSearch[/bold blue] - Automated Policy Optimization",
        subtitle="Find the best augmentations for your data"
    ))
    
    # Auto-detect domain if needed
    if domain == "auto":
        from augmentai.inspection import DatasetAnalyzer
        
        console.print("[dim]Detecting domain...[/dim]")
        analyzer = DatasetAnalyzer()
        try:
            report = analyzer.analyze(dataset)
            domain = report.detection.suggested_domain
            console.print(f"[cyan]Auto-detected domain:[/cyan] {domain}")
        except Exception:
            domain = "natural"
            console.print(f"[yellow]Could not detect domain, using:[/yellow] {domain}")
    
    # Show search configuration
    table = Table(title="Search Configuration", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    
    table.add_row("Dataset", str(dataset))
    table.add_row("Domain", domain)
    table.add_row("Budget", str(budget))
    table.add_row("Population", str(population))
    table.add_row("Seed", str(seed))
    table.add_row("Output", str(output))
    
    console.print(table)
    
    if dry_run:
        console.print("\n[yellow]Dry run - no search performed.[/yellow]")
        return
    
    # Configure optimizer
    config = OptimizerConfig(
        population_size=population,
        generations=max(3, budget // population),
        seed=seed,
    )
    
    # Run search
    console.print("\n[bold]Starting search...[/bold]\n")
    
    optimizer = PolicyOptimizer(config)
    result = optimizer.search(domain, budget=budget)
    
    # Display results
    _show_search_results(result)
    
    # Save results
    output.mkdir(parents=True, exist_ok=True)
    result_path = result.save(output)
    
    console.print(Panel.fit(
        f"[bold green]✓ Search complete![/bold green]\n\n"
        f"Best score: [cyan]{result.best_score:.4f}[/cyan]\n"
        f"Evaluations: {result.budget_used}\n"
        f"Time: {result.search_time:.1f}s\n\n"
        f"Results saved to:\n"
        f"  [dim]{result_path}[/dim]\n"
        f"  [dim]{output / 'best_policy.yaml'}[/dim]",
        title="Complete",
    ))


def _show_search_results(result: SearchResult) -> None:
    """Display search results."""
    console.print(f"\n[bold]Best Policy:[/bold] {result.best_policy.name}")
    console.print(f"[dim]Domain: {result.domain}[/dim]\n")
    
    # Show transforms
    table = Table(title="Optimal Transforms", show_header=True)
    table.add_column("Transform")
    table.add_column("Probability")
    table.add_column("Parameters")
    
    for t in result.best_policy.transforms:
        params = ", ".join(f"{k}={v}" for k, v in (t.parameters or {}).items())
        table.add_row(t.name, f"{t.probability:.1%}", params or "-")
    
    console.print(table)
    
    # Show search history summary
    if result.history:
        console.print("\n[bold]Search Progress:[/bold]")
        first_score = result.history[0].get("best_score", 0)
        last_score = result.history[-1].get("best_score", 0)
        improvement = last_score - first_score
        
        console.print(
            f"  Generations: {len(result.history)} | "
            f"Initial: {first_score:.4f} | "
            f"Final: {last_score:.4f} | "
            f"Improvement: {improvement:+.4f}"
        )

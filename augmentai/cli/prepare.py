"""
One-command data preparation CLI.

Orchestrates: inspect → split → augment policy → export

Usage:
    augmentai prepare ./dataset --domain medical --task "segmentation"
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from augmentai.core.config import AugmentAIConfig
from augmentai.core.manifest import ReproducibilityManifest
from augmentai.core.pipeline import PipelineConfig, PipelineResult
from augmentai.core.policy import Policy, Transform
from augmentai.domains import get_domain
from augmentai.inspection import DatasetAnalyzer
from augmentai.splitting import DatasetSplitter, SplitStrategy, SplitResult
from augmentai.splitting.strategies import SplitConfig
from augmentai.export import ScriptGenerator, FolderStructure
from augmentai.rules.enforcement import RuleEnforcer


console = Console()


def prepare(
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
    task: Optional[str] = typer.Option(
        None,
        "--task", "-t",
        help="Task description for LLM policy generation",
    ),
    output: Path = typer.Option(
        Path("./prepared"),
        "--output", "-o",
        help="Output directory for prepared dataset",
    ),
    seed: int = typer.Option(
        42,
        "--seed", "-s",
        help="Random seed for reproducibility",
    ),
    split: str = typer.Option(
        "80/10/10",
        "--split",
        help="Train/val/test split ratios (e.g., '80/10/10')",
    ),
    strategy: str = typer.Option(
        "stratified",
        "--strategy",
        help="Split strategy: random, stratified, group",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without writing files",
    ),
    skip_split: bool = typer.Option(
        False,
        "--skip-split",
        help="Skip splitting if dataset is already split",
    ),
) -> None:
    """
    Prepare a dataset for training with one command.
    
    Inspects the dataset, splits it, generates an augmentation policy,
    and exports executable scripts.
    
    Examples:
        augmentai prepare ./images --domain medical
        augmentai prepare ./data --domain auto --task "classification"
        augmentai prepare ./dataset --split 70/15/15 --seed 123
    """
    console.print(Panel.fit(
        "[bold blue]AugmentAI[/bold blue] - One-Command Data Preparation",
        subtitle="inspect → split → policy → export"
    ))
    
    # Parse split ratios
    try:
        train_ratio, val_ratio, test_ratio = _parse_split(split)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    
    # Parse strategy
    try:
        split_strategy = SplitStrategy(strategy)
    except ValueError:
        console.print(f"[red]Error:[/red] Invalid strategy '{strategy}'. Use: random, stratified, group")
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Step 1: Inspect dataset
        task_id = progress.add_task("Inspecting dataset...", total=None)
        analyzer = DatasetAnalyzer()
        report = analyzer.analyze(dataset)
        progress.update(task_id, completed=True)
        
        # Show inspection results
        _show_inspection_results(report)
        
        # Determine domain
        if domain == "auto":
            domain = report.detection.suggested_domain
            console.print(f"[cyan]Auto-detected domain:[/cyan] {domain}")
        
        # Check if already split
        if report.detection.is_presplit and not skip_split:
            console.print("[yellow]Dataset appears to be pre-split. Use --skip-split to skip splitting.[/yellow]")
        
        # Step 2: Split dataset (if not pre-split)
        split_result: SplitResult | None = None
        if not report.detection.is_presplit and not skip_split:
            progress.update(task_id, description="Splitting dataset...")
            
            split_config = SplitConfig(
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                strategy=split_strategy,
                seed=seed,
            )
            
            splitter = DatasetSplitter(split_config)
            
            if dry_run:
                # Dry run: just compute split without copying
                split_result = splitter.split(dataset, output_path=None, copy_files=False)
            else:
                # Create output structure and copy files
                folders = FolderStructure(output)
                folders.create(include_augmented=True)
                split_result = splitter.split(
                    dataset, 
                    output_path=folders.data_dir,
                    copy_files=True
                )
            
            progress.update(task_id, completed=True)
            console.print(f"[green]✓[/green] {split_result.summary()}")
        
        # Step 3: Generate augmentation policy
        progress.update(task_id, description="Generating augmentation policy...")
        
        policy = _generate_policy(domain, task, seed)
        
        # Validate against domain rules
        domain_obj = get_domain(domain)
        enforcer = RuleEnforcer(domain_obj)
        enforcement = enforcer.enforce_policy(policy)
        
        if enforcement.success and enforcement.policy:
            policy = enforcement.policy
        
        progress.update(task_id, completed=True)
        _show_policy(policy, enforcement)
        
        # Step 4: Export scripts and configs
        if not dry_run:
            progress.update(task_id, description="Exporting scripts...")
            
            folders = FolderStructure(output)
            generator = ScriptGenerator(backend="albumentations")
            
            # Generate and save files
            script_content = generator.generate_augment_script(
                policy, 
                input_dir="data/train",
                output_dir="augmented/train",
                seed=seed
            )
            folders.save_script(script_content)
            
            config_content = generator.generate_config_yaml(policy, seed)
            folders.save_config(config_content)
            
            requirements = generator.generate_requirements()
            folders.save_requirements(requirements)
            
            # Create and save manifest
            manifest = ReproducibilityManifest(
                seed=seed,
                dataset_path=str(dataset.absolute()),
                dataset_hash=ReproducibilityManifest.hash_directory(dataset),
                file_count=report.image_count,
                domain=domain,
                backend="albumentations",
                split_ratios={
                    "train": train_ratio,
                    "val": val_ratio,
                    "test": test_ratio,
                },
                policy_name=policy.name,
                policy_hash=ReproducibilityManifest.hash_policy(policy.to_dict()),
                transforms=[t.to_dict() for t in policy.transforms],
                output_path=str(output.absolute()),
            )
            folders.save_manifest(manifest)
            folders.save_readme(policy.name, domain)
            
            progress.update(task_id, completed=True)
    
    # Final summary
    if dry_run:
        console.print("\n[yellow]Dry run complete. No files written.[/yellow]")
    else:
        _show_final_summary(output, policy, split_result)


def _parse_split(split_str: str) -> tuple[float, float, float]:
    """Parse split string like '80/10/10' into ratios."""
    parts = split_str.split("/")
    if len(parts) != 3:
        raise ValueError(f"Split must be in format 'train/val/test' (e.g., '80/10/10'), got '{split_str}'")
    
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"Split values must be numbers, got '{split_str}'")
    
    total = sum(values)
    if total <= 0:
        raise ValueError("Split values must be positive")
    
    # Normalize to ratios
    ratios = [v / total for v in values]
    return ratios[0], ratios[1], ratios[2]


def _show_inspection_results(report) -> None:
    """Display inspection results."""
    table = Table(title="Dataset Inspection", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    
    table.add_row("Format", report.detection.format.value)
    table.add_row("Image Type", report.detection.image_type.value)
    table.add_row("Total Images", str(report.image_count))
    table.add_row("Has Labels", "Yes" if report.detection.has_labels else "No")
    table.add_row("Has Masks", "Yes" if report.detection.has_masks else "No")
    table.add_row("Pre-split", "Yes" if report.detection.is_presplit else "No")
    
    if report.class_distribution:
        classes = ", ".join(f"{k}({v})" for k, v in list(report.class_distribution.items())[:5])
        if len(report.class_distribution) > 5:
            classes += f" (+{len(report.class_distribution) - 5} more)"
        table.add_row("Classes", classes)
    
    console.print(table)
    
    # Show issues
    for issue in report.issues[:3]:
        if issue.severity == "error":
            console.print(f"[red]✗ {issue.message}[/red]")
        elif issue.severity == "warning":
            console.print(f"[yellow]⚠ {issue.message}[/yellow]")


def _generate_policy(domain: str, task: Optional[str], seed: int) -> Policy:
    """Generate a default policy for the domain."""
    # Default transforms by domain
    domain_transforms = {
        "natural": [
            Transform("HorizontalFlip", 0.5),
            Transform("VerticalFlip", 0.2),
            Transform("Rotate", 0.5, parameters={"limit": 30}),
            Transform("RandomBrightnessContrast", 0.5, parameters={"brightness_limit": 0.2, "contrast_limit": 0.2}),
            Transform("GaussNoise", 0.3, parameters={"var_limit": (10, 50)}),
        ],
        "medical": [
            Transform("HorizontalFlip", 0.5),
            Transform("Rotate", 0.5, parameters={"limit": 15}),
            Transform("RandomBrightnessContrast", 0.3, parameters={"brightness_limit": 0.1, "contrast_limit": 0.1}),
            Transform("GaussNoise", 0.2, parameters={"var_limit": (5, 20)}),
        ],
        "ocr": [
            Transform("Rotate", 0.5, parameters={"limit": 5}),
            Transform("RandomBrightnessContrast", 0.3, parameters={"brightness_limit": 0.1, "contrast_limit": 0.1}),
            Transform("GaussNoise", 0.2, parameters={"var_limit": (5, 15)}),
        ],
        "satellite": [
            Transform("HorizontalFlip", 0.5),
            Transform("VerticalFlip", 0.5),
            Transform("Rotate", 0.5, parameters={"limit": 180}),
            Transform("RandomBrightnessContrast", 0.3, parameters={"brightness_limit": 0.1, "contrast_limit": 0.1}),
        ],
    }
    
    transforms = domain_transforms.get(domain, domain_transforms["natural"])
    
    return Policy(
        name=f"{domain}_augmentation_policy",
        domain=domain,
        transforms=transforms,
    )


def _show_policy(policy: Policy, enforcement) -> None:
    """Display the generated policy."""
    console.print(f"\n[bold]Generated Policy:[/bold] {policy.name}")
    console.print(f"[dim]Domain: {policy.domain}[/dim]")
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Transform")
    table.add_column("Probability")
    table.add_column("Parameters")
    
    for t in policy.transforms:
        params = ", ".join(f"{k}={v}" for k, v in (t.parameters or {}).items())
        table.add_row(t.name, f"{t.probability:.1%}", params or "-")
    
    console.print(table)
    
    # Show what was modified
    if enforcement.safety_result:
        if enforcement.safety_result.removed_transforms:
            console.print("[yellow]Removed (domain rules):[/yellow]")
            for t in enforcement.safety_result.removed_transforms:
                console.print(f"  [dim]✗ {t.name}[/dim]")


def _show_final_summary(output: Path, policy: Policy, split_result: SplitResult | None) -> None:
    """Show final summary of what was created."""
    console.print(Panel.fit(
        f"[bold green]✓ Dataset prepared successfully![/bold green]\n\n"
        f"Output: [cyan]{output.absolute()}[/cyan]\n\n"
        f"Files created:\n"
        f"  • augment.py - Run augmentations\n"
        f"  • config.yaml - Pipeline config\n"
        f"  • manifest.json - Reproducibility info\n"
        f"  • requirements.txt - Dependencies\n"
        f"  • README.md - Usage instructions\n\n"
        f"Next steps:\n"
        f"  [dim]cd {output}[/dim]\n"
        f"  [dim]pip install -r requirements.txt[/dim]\n"
        f"  [dim]python augment.py[/dim]",
        title="Complete",
    ))

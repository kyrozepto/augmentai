"""
CLI command for policy diff and version management.

Usage:
    augmentai diff policy_v1.yaml policy_v2.yaml
    augmentai diff policy.yaml --history
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from augmentai.core.policy import Policy
from augmentai.versioning import PolicyVersionControl, PolicyDiff


console = Console()


def diff(
    policy1: Path = typer.Argument(
        ...,
        help="First policy file (YAML/JSON) or 'history' for version history",
    ),
    policy2: Optional[Path] = typer.Argument(
        None,
        help="Second policy file to compare (optional)",
    ),
    storage_dir: Path = typer.Option(
        Path("./.augmentai-versions"),
        "--storage", "-s",
        help="Version storage directory",
    ),
    history: bool = typer.Option(
        False,
        "--history", "-H",
        help="Show version history for the policy",
    ),
    commit: bool = typer.Option(
        False,
        "--commit", "-c",
        help="Commit the policy as a new version",
    ),
    message: str = typer.Option(
        "",
        "--message", "-m",
        help="Commit message",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
    ),
) -> None:
    """
    Compare augmentation policies and manage versions.
    
    Examples:
        augmentai diff policy_v1.yaml policy_v2.yaml
        augmentai diff policy.yaml --history
        augmentai diff policy.yaml --commit -m "Initial version"
    """
    console.print(Panel.fit(
        "[bold blue]AugmentAI[/bold blue] - Policy Diff & Versioning",
        subtitle="Track policy changes"
    ))
    
    # Initialize version control
    vc = PolicyVersionControl(storage_dir)
    
    # Load first policy
    try:
        policy1_content = policy1.read_text()
        if policy1.suffix in [".yaml", ".yml"]:
            policy1_obj = Policy.from_yaml(policy1_content)
        else:
            policy1_obj = Policy.from_json(policy1_content)
        console.print(f"[green]✓[/green] Loaded: {policy1_obj.name}")
    except Exception as e:
        console.print(f"[red]Error loading policy:[/red] {e}")
        raise typer.Exit(1)
    
    # Commit mode
    if commit:
        version = vc.commit(policy1_obj, message)
        console.print(f"[green]✓[/green] Committed as {version.version} (hash: {version.hash})")
        return
    
    # History mode
    if history:
        _show_history(vc, policy1_obj.name)
        return
    
    # Diff mode
    if policy2 is None:
        # Try to diff with previous version
        latest = vc.get_latest(policy1_obj.name)
        if latest is None:
            console.print("[yellow]No previous version found. Use --commit to create one.[/yellow]")
            raise typer.Exit(0)
        
        policy_diff = vc.diff(latest.policy, policy1_obj)
        console.print(f"\nComparing with {latest.version}:")
    else:
        # Load second policy
        try:
            policy2_content = policy2.read_text()
            if policy2.suffix in [".yaml", ".yml"]:
                policy2_obj = Policy.from_yaml(policy2_content)
            else:
                policy2_obj = Policy.from_json(policy2_content)
        except Exception as e:
            console.print(f"[red]Error loading second policy:[/red] {e}")
            raise typer.Exit(1)
        
        policy_diff = vc.diff(policy1_obj, policy2_obj)
    
    # Output
    if json_output:
        import json
        console.print(json.dumps(policy_diff.to_dict(), indent=2))
    else:
        _display_diff(policy_diff)


def _show_history(vc: PolicyVersionControl, policy_name: str) -> None:
    """Display version history."""
    versions = vc.history(policy_name)
    
    if not versions:
        console.print("[yellow]No version history found.[/yellow]")
        return
    
    table = Table(title=f"Version History: {policy_name}")
    table.add_column("Version", style="cyan")
    table.add_column("Hash", style="dim")
    table.add_column("Timestamp")
    table.add_column("Message")
    table.add_column("Transforms")
    
    for v in reversed(versions):
        table.add_row(
            v.version,
            v.hash,
            v.timestamp[:19],
            v.message or "-",
            str(len(v.policy.transforms)),
        )
    
    console.print(table)


def _display_diff(diff: PolicyDiff) -> None:
    """Display diff in colored format."""
    if not diff.has_changes:
        console.print("[green]No changes detected.[/green]")
        return
    
    console.print(f"\n[bold]Summary:[/bold] {diff.summary}\n")
    
    # Removed transforms
    for t in diff.removed_transforms:
        text = Text()
        text.append("- ", style="red bold")
        text.append(t.name, style="red")
        text.append(f" (p={t.probability})", style="red dim")
        console.print(text)
    
    # Added transforms
    for t in diff.added_transforms:
        text = Text()
        text.append("+ ", style="green bold")
        text.append(t.name, style="green")
        text.append(f" (p={t.probability})", style="green dim")
        console.print(text)
    
    # Modified transforms
    for old, new in diff.modified_transforms:
        text = Text()
        text.append("~ ", style="yellow bold")
        text.append(old.name, style="yellow")
        console.print(text)
        
        if old.probability != new.probability:
            console.print(f"    probability: [red]{old.probability}[/red] → [green]{new.probability}[/green]")
        
        for key in set(list(old.parameters.keys()) + list(new.parameters.keys())):
            old_val = old.parameters.get(key)
            new_val = new.parameters.get(key)
            if old_val != new_val:
                console.print(f"    {key}: [red]{old_val}[/red] → [green]{new_val}[/green]")

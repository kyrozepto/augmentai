"""
Main CLI application entry point.

Provides commands for:
- Starting an interactive chat session
- Validating existing policies
- Listing available domains
- Exporting policies to different formats
- One-command dataset preparation
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from augmentai.core.config import AugmentAIConfig, LLMProvider, AugmentationBackend
from augmentai.cli.prepare import prepare as prepare_command
from augmentai.cli.ablate import ablate as ablate_command
from augmentai.cli.diff import diff as diff_command

app = typer.Typer(
    name="augmentai",
    help="LLM-powered data augmentation policy designer",
    add_completion=False,
)
console = Console()

# Register commands
app.command()(prepare_command)
app.command()(ablate_command)
app.command()(diff_command)



@app.command()
def chat(
    domain: str = typer.Option(
        "natural",
        "--domain", "-d",
        help="Domain for augmentation (medical, ocr, satellite, natural)",
    ),
    provider: str = typer.Option(
        "openai",
        "--provider", "-p",
        help="LLM provider (openai, ollama, lmstudio)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="LLM model name (defaults based on provider)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for exported policies",
    ),
) -> None:
    """Start an interactive chat session to design augmentation policies."""
    from augmentai.cli.chat import ChatSession
    from augmentai.core.config import LLMConfig
    
    # Configure LLM
    try:
        llm_provider = LLMProvider(provider.lower())
    except ValueError:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        console.print("Available: openai, ollama, lmstudio")
        raise typer.Exit(1)
    
    # Set default models
    if model is None:
        if llm_provider == LLMProvider.OPENAI:
            model = "gpt-4o-mini"
        elif llm_provider == LLMProvider.OLLAMA:
            model = "llama3.2"
        elif llm_provider == LLMProvider.LMSTUDIO:
            model = "local-model"
    
    llm_config = LLMConfig(provider=llm_provider, model=model)
    
    config = AugmentAIConfig(
        llm=llm_config,
        output_dir=output or Path.cwd(),
    )
    
    # Show welcome
    _show_welcome(domain, provider, model)
    
    # Start chat session
    session = ChatSession(config=config, domain_name=domain)
    
    try:
        session.run()
    except KeyboardInterrupt:
        console.print("\n[dim]Session ended.[/dim]")


@app.command()
def domains() -> None:
    """List available augmentation domains and their constraints."""
    from rich.table import Table
    
    from augmentai.domains import (
        MedicalDomain,
        OCRDomain, 
        SatelliteDomain,
        NaturalDomain,
    )
    
    table = Table(title="Available Domains")
    table.add_column("Domain", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Forbidden Transforms", style="red")
    table.add_column("Recommended", style="green")
    
    domain_classes = [
        MedicalDomain,
        OCRDomain,
        SatelliteDomain,
        NaturalDomain,
    ]
    
    for domain_cls in domain_classes:
        domain = domain_cls()
        forbidden = ", ".join(list(domain.forbidden_transforms)[:5])
        if len(domain.forbidden_transforms) > 5:
            forbidden += f" (+{len(domain.forbidden_transforms) - 5} more)"
        
        recommended = ", ".join(list(domain.recommended_transforms)[:5])
        if len(domain.recommended_transforms) > 5:
            recommended += f" (+{len(domain.recommended_transforms) - 5} more)"
        
        table.add_row(
            domain.name,
            domain.description[:50] + "..." if len(domain.description) > 50 else domain.description,
            forbidden or "None",
            recommended or "None",
        )
    
    console.print(table)


@app.command()
def validate(
    policy_file: Path = typer.Argument(
        ...,
        help="Path to policy YAML file to validate",
    ),
    domain: str = typer.Option(
        "natural",
        "--domain", "-d",
        help="Domain to validate against",
    ),
) -> None:
    """Validate an existing policy file against domain constraints."""
    from augmentai.core.policy import Policy
    from augmentai.domains import MedicalDomain, OCRDomain, SatelliteDomain, NaturalDomain
    from augmentai.rules.validator import SafetyValidator
    
    if not policy_file.exists():
        console.print(f"[red]File not found: {policy_file}[/red]")
        raise typer.Exit(1)
    
    # Load policy
    try:
        with open(policy_file) as f:
            policy = Policy.from_yaml(f.read())
    except Exception as e:
        console.print(f"[red]Failed to load policy: {e}[/red]")
        raise typer.Exit(1)
    
    # Get domain
    domains_map = {
        "medical": MedicalDomain,
        "ocr": OCRDomain,
        "satellite": SatelliteDomain,
        "natural": NaturalDomain,
    }
    
    domain_cls = domains_map.get(domain.lower())
    if domain_cls is None:
        console.print(f"[red]Unknown domain: {domain}[/red]")
        raise typer.Exit(1)
    
    domain_obj = domain_cls()
    validator = SafetyValidator(domain_obj)
    
    # Validate
    result = validator.validate(policy)
    
    console.print(Panel(
        result.summary(),
        title=f"Validation Result: {policy.name}",
        border_style="green" if result.is_safe else "red",
    ))


@app.command()
def export(
    policy_file: Path = typer.Argument(
        ...,
        help="Path to policy YAML file to export",
    ),
    output: Path = typer.Option(
        Path("."),
        "--output", "-o",
        help="Output directory",
    ),
    format: str = typer.Option(
        "python",
        "--format", "-f",
        help="Export format: python, yaml, json",
    ),
    backend: str = typer.Option(
        "albumentations",
        "--backend", "-b",
        help="Target backend: albumentations",
    ),
) -> None:
    """Export a policy to executable code or config."""
    from augmentai.core.policy import Policy
    from augmentai.compilers import AlbumentationsCompiler
    
    if not policy_file.exists():
        console.print(f"[red]File not found: {policy_file}[/red]")
        raise typer.Exit(1)
    
    # Load policy
    try:
        with open(policy_file) as f:
            policy = Policy.from_yaml(f.read())
    except Exception as e:
        console.print(f"[red]Failed to load policy: {e}[/red]")
        raise typer.Exit(1)
    
    # Get compiler
    if backend.lower() == "albumentations":
        compiler = AlbumentationsCompiler()
    else:
        console.print(f"[red]Unknown backend: {backend}[/red]")
        console.print("Available: albumentations")
        raise typer.Exit(1)
    
    # Compile
    result = compiler.compile(policy)
    
    if not result.success:
        console.print("[red]Compilation failed:[/red]")
        for error in result.errors:
            console.print(f"  • {error}")
        raise typer.Exit(1)
    
    # Export
    output.mkdir(parents=True, exist_ok=True)
    
    if format == "python":
        output_file = output / f"{policy.name}.py"
        output_file.write_text(result.code)
        console.print(f"[green]✓ Exported to {output_file}[/green]")
    
    elif format in ("yaml", "yml"):
        output_file = output / f"{policy.name}.yaml"
        output_file.write_text(result.config)
        console.print(f"[green]✓ Exported to {output_file}[/green]")
    
    elif format == "json":
        output_file = output / f"{policy.name}.json"
        output_file.write_text(policy.to_json())
        console.print(f"[green]✓ Exported to {output_file}[/green]")
    
    else:
        console.print(f"[red]Unknown format: {format}[/red]")
        raise typer.Exit(1)


@app.command()
def init(
    path: Path = typer.Argument(
        Path("."),
        help="Directory to initialize",
    ),
) -> None:
    """Initialize a new AugmentAI project with config file and examples."""
    config_file = path / "augmentai.yaml"
    
    if config_file.exists():
        console.print(f"[yellow]Config already exists: {config_file}[/yellow]")
        if not typer.confirm("Overwrite?"):
            raise typer.Exit(0)
    
    # Create default config
    config = AugmentAIConfig()
    config_file.write_text(config.to_yaml())
    
    console.print(f"[green]✓ Created {config_file}[/green]")
    console.print("\nEdit the config to set your LLM provider and API key.")
    console.print("Then run: augmentai chat")


def _show_welcome(domain: str, provider: str, model: str) -> None:
    """Show welcome message."""
    welcome_text = f"""
# 🎨 AugmentAI - Data Augmentation Policy Designer

Design domain-safe augmentation policies through natural conversation.

**Configuration:**
- Domain: `{domain}`
- LLM Provider: `{provider}`  
- Model: `{model}`

**Commands:**
- Type your requirements in natural language
- `/preview` - Preview current policy
- `/export <filename>` - Export policy
- `/domain` - Show domain constraints
- `/help` - Show all commands
- `/quit` - Exit

---
    """
    console.print(Markdown(welcome_text))


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()

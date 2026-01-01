"""
Interactive chat session for designing augmentation policies.

Provides a REPL interface where users can describe their needs
in natural language and iteratively refine augmentation policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.live import Live
from rich.spinner import Spinner

from augmentai.core.config import AugmentAIConfig
from augmentai.core.policy import Policy
from augmentai.core.schema import DEFAULT_SCHEMA
from augmentai.domains import MedicalDomain, OCRDomain, SatelliteDomain, NaturalDomain
from augmentai.domains.base import Domain
from augmentai.llm.client import LLMClient, Message, MessageRole
from augmentai.llm.prompts import PromptBuilder, QUICK_PROMPTS
from augmentai.llm.parser import PolicyParser
from augmentai.rules.enforcement import RuleEnforcer
from augmentai.compilers import AlbumentationsCompiler


console = Console()


@dataclass
class ChatSession:
    """
    Interactive chat session for policy design.
    
    Manages the conversation state, LLM interaction, and policy refinement.
    """
    
    config: AugmentAIConfig
    domain_name: str = "natural"
    
    # Internal state
    domain: Domain = field(init=False)
    llm_client: LLMClient = field(init=False)
    prompt_builder: PromptBuilder = field(init=False)
    parser: PolicyParser = field(init=False)
    enforcer: RuleEnforcer = field(init=False)
    compiler: AlbumentationsCompiler = field(init=False)
    
    messages: list[Message] = field(default_factory=list)
    current_policy: Policy | None = None
    
    def __post_init__(self) -> None:
        """Initialize session components."""
        # Load domain
        self.domain = self._get_domain(self.domain_name)
        
        # Initialize LLM client
        self.llm_client = LLMClient(self.config.llm)
        
        # Initialize helpers
        self.prompt_builder = PromptBuilder(self.domain, DEFAULT_SCHEMA)
        self.parser = PolicyParser(DEFAULT_SCHEMA)
        self.enforcer = RuleEnforcer(self.domain, DEFAULT_SCHEMA)
        self.compiler = AlbumentationsCompiler()
        
        # Set up system prompt
        system_prompt = self.prompt_builder.build_system_prompt()
        self.messages.append(Message(MessageRole.SYSTEM, system_prompt))
    
    def _get_domain(self, name: str) -> Domain:
        """Get domain by name."""
        domains = {
            "medical": MedicalDomain,
            "ct": MedicalDomain,
            "mri": MedicalDomain,
            "ocr": OCRDomain,
            "document": OCRDomain,
            "satellite": SatelliteDomain,
            "aerial": SatelliteDomain,
            "natural": NaturalDomain,
            "general": NaturalDomain,
        }
        
        domain_cls = domains.get(name.lower(), NaturalDomain)
        return domain_cls()
    
    def run(self) -> None:
        """Run the interactive chat loop."""
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    should_continue = self._handle_command(user_input)
                    if not should_continue:
                        break
                    continue
                
                # Handle quick prompts (shortcuts)
                if user_input.lower() in QUICK_PROMPTS:
                    user_input = QUICK_PROMPTS[user_input.lower()]
                
                # Process with LLM
                self._process_message(user_input)
                
            except KeyboardInterrupt:
                console.print("\n[dim]Use /quit to exit[/dim]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    def _handle_command(self, command: str) -> bool:
        """
        Handle a slash command.
        
        Returns False if the session should end.
        """
        parts = command[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd in ("quit", "q", "exit"):
            console.print("[dim]Goodbye![/dim]")
            return False
        
        elif cmd == "help":
            self._show_help()
        
        elif cmd == "preview":
            self._show_preview()
        
        elif cmd == "export":
            self._export_policy(args)
        
        elif cmd == "domain":
            self._show_domain_info()
        
        elif cmd == "clear":
            self.current_policy = None
            console.print("[dim]Policy cleared[/dim]")
        
        elif cmd == "transforms":
            self._show_available_transforms()
        
        elif cmd == "history":
            self._show_history()
        
        else:
            console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
            console.print("Type /help for available commands")
        
        return True
    
    def _process_message(self, user_input: str) -> None:
        """Process a user message with the LLM."""
        # Build the prompt
        if self.current_policy is None:
            prompt = self.prompt_builder.build_generation_prompt(user_input)
        else:
            prompt = self.prompt_builder.build_refinement_prompt(
                user_input, 
                self.current_policy.to_json()
            )
        
        self.messages.append(Message(MessageRole.USER, prompt))
        
        # Show thinking indicator
        console.print()
        with console.status("[bold blue]Thinking...[/bold blue]", spinner="dots"):
            try:
                response = self.llm_client.chat(
                    self.messages,
                    json_mode=True,
                )
            except Exception as e:
                console.print(f"[red]LLM Error: {e}[/red]")
                self.messages.pop()  # Remove failed message
                return
        
        # Add response to history
        self.messages.append(Message(MessageRole.ASSISTANT, response.content))
        
        # Parse response
        parse_result = self.parser.parse(response.content, self.domain_name)
        
        if not parse_result.success:
            console.print("[yellow]Could not parse policy from response.[/yellow]")
            console.print("Errors:", parse_result.errors)
            
            # Try to show reasoning anyway
            if parse_result.reasoning:
                console.print(Panel(
                    Markdown(parse_result.reasoning),
                    title="LLM Response",
                    border_style="blue",
                ))
            return
        
        # Enforce domain rules
        enforcement = self.enforcer.enforce(parse_result)
        
        if not enforcement.success:
            console.print("[red]Could not create valid policy[/red]")
            for note in enforcement.enforcer_notes:
                console.print(f"  • {note}")
            return
        
        self.current_policy = enforcement.policy
        
        # Show the result
        self._display_policy_update(parse_result, enforcement)
    
    def _display_policy_update(self, parse_result: Any, enforcement: Any) -> None:
        """Display the policy update nicely."""
        console.print()
        
        # Show LLM reasoning
        if parse_result.reasoning:
            console.print(Panel(
                Markdown(parse_result.reasoning),
                title="🤖 AugmentAI",
                border_style="blue",
            ))
        
        # Show warnings from enforcement
        if enforcement.enforcer_notes:
            console.print("[yellow]Safety adjustments:[/yellow]")
            for note in enforcement.enforcer_notes:
                console.print(f"  ⚠ {note}")
        
        # Show policy preview
        self._show_preview()
    
    def _show_preview(self) -> None:
        """Show current policy preview."""
        if self.current_policy is None:
            console.print("[dim]No policy created yet. Describe your task to get started.[/dim]")
            return
        
        table = Table(title=f"Policy: {self.current_policy.name}")
        table.add_column("Transform", style="cyan")
        table.add_column("Probability", style="green")
        table.add_column("Parameters", style="white")
        
        for t in self.current_policy.transforms:
            params = ", ".join(f"{k}={v}" for k, v in t.parameters.items())
            if not params:
                params = "-"
            table.add_row(t.name, f"{t.probability:.2f}", params)
        
        console.print(table)
        console.print(f"[dim]Domain: {self.current_policy.domain} | Transforms: {len(self.current_policy.transforms)}[/dim]")
    
    def _export_policy(self, filename: str) -> None:
        """Export current policy to file."""
        if self.current_policy is None:
            console.print("[yellow]No policy to export[/yellow]")
            return
        
        if not filename:
            filename = f"{self.current_policy.name}.yaml"
        
        output_path = self.config.output_dir / filename
        
        # Determine format from extension
        if filename.endswith(".py"):
            result = self.compiler.compile(self.current_policy)
            if result.success:
                output_path.write_text(result.code)
            else:
                console.print(f"[red]Compilation failed: {result.errors}[/red]")
                return
        elif filename.endswith(".json"):
            output_path.write_text(self.current_policy.to_json())
        else:
            output_path.write_text(self.current_policy.to_yaml())
        
        console.print(f"[green]✓ Exported to {output_path}[/green]")
    
    def _show_domain_info(self) -> None:
        """Show domain constraints."""
        console.print(self.enforcer.get_domain_summary())
    
    def _show_available_transforms(self) -> None:
        """Show transforms available in current domain."""
        table = Table(title=f"Available Transforms ({self.domain.name} domain)")
        table.add_column("Transform", style="cyan")
        table.add_column("Category", style="blue")
        table.add_column("Status", style="green")
        
        for name, spec in DEFAULT_SCHEMA.transforms.items():
            if name in self.domain.forbidden_transforms:
                continue
            
            status = "✓ Recommended" if name in self.domain.recommended_transforms else "Allowed"
            table.add_row(name, spec.category.value, status)
        
        console.print(table)
    
    def _show_history(self) -> None:
        """Show conversation history."""
        for i, msg in enumerate(self.messages):
            if msg.role == MessageRole.SYSTEM:
                continue
            
            role_style = "cyan" if msg.role == MessageRole.USER else "blue"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            console.print(f"[{role_style}]{msg.role.value}[/{role_style}]: {content}")
    
    def _show_help(self) -> None:
        """Show help message."""
        help_text = """
## Available Commands

| Command | Description |
|---------|-------------|
| `/preview` | Show current policy |
| `/export <file>` | Export policy (supports .yaml, .json, .py) |
| `/domain` | Show domain constraints |
| `/transforms` | List available transforms |
| `/clear` | Clear current policy |
| `/history` | Show conversation history |
| `/quit` | Exit the session |

## Quick Prompts

Type any of these to quickly modify your policy:
- `more_aggressive` - Increase probabilities and parameters
- `more_conservative` - Reduce augmentation strength
- `add_color` - Add color augmentations
- `add_geometric` - Add geometric augmentations
- `remove_risky` - Remove potentially problematic transforms

## Tips

- Describe your dataset and task clearly
- Mention any specific constraints (e.g., "medical CT scans")
- Ask to add or remove specific transforms
- Request explanations with "why" questions
        """
        console.print(Markdown(help_text))

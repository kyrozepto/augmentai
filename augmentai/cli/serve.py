"""
CLI command for launching the AugmentAI Web UI.

Usage:
    augmentai serve
    augmentai serve --port 9000
    augmentai serve --no-browser --reload
"""

from typing import Optional
import typer

from augmentai.cli.utils import get_console

app = typer.Typer()
console = get_console()


def serve_command(
    host: str = typer.Option(
        "127.0.0.1",
        "--host", "-h",
        help="Host to bind the server to",
    ),
    port: int = typer.Option(
        8000,
        "--port", "-p",
        help="Port to bind the server to",
    ),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Don't open browser automatically",
    ),
    reload: bool = typer.Option(
        False,
        "--reload", "-r",
        help="Enable auto-reload for development",
    ),
):
    """
    Launch the AugmentAI Web UI.
    
    Starts a local web server with a visual interface for:
    - Dataset browsing and inspection
    - Visual policy builder with drag-and-drop
    - AutoSearch visualization
    - Interactive chat
    - Pipeline execution
    
    Example:
        augmentai serve
        augmentai serve --port 9000
        augmentai serve --reload  # Development mode
    """
    try:
        from augmentai.web import run_server
    except ImportError as e:
        console.print(
            "[red]Error:[/red] Web UI dependencies not installed.\n"
            "Install with: pip install augmentai[web]"
        )
        raise typer.Exit(1)
    
    run_server(
        host=host,
        port=port,
        open_browser=not no_browser,
        reload=reload,
    )

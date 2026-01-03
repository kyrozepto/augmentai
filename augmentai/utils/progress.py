"""
Progress tracking and verbosity utilities for AugmentAI.

Provides consistent progress bars, logging, and verbosity control
across all CLI commands.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from enum import IntEnum
from typing import Any, Callable, Generator, Iterable, TypeVar

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.status import Status


T = TypeVar("T")


class VerbosityLevel(IntEnum):
    """Verbosity levels for CLI output."""
    
    QUIET = 0      # Minimal output (errors only)
    NORMAL = 1     # Standard output (default)
    VERBOSE = 2    # Detailed output (debug info)


# Global verbosity setting
_verbosity: VerbosityLevel = VerbosityLevel.NORMAL
_console: Console | None = None


def get_console() -> Console:
    """Get the global Rich console instance."""
    global _console
    if _console is None:
        _console = Console()
    return _console


def set_verbosity(level: VerbosityLevel) -> None:
    """Set the global verbosity level."""
    global _verbosity
    _verbosity = level


def get_verbosity() -> VerbosityLevel:
    """Get the current verbosity level."""
    return _verbosity


def is_quiet() -> bool:
    """Check if running in quiet mode."""
    return _verbosity == VerbosityLevel.QUIET


def is_verbose() -> bool:
    """Check if running in verbose mode."""
    return _verbosity == VerbosityLevel.VERBOSE


def print_info(message: str, **kwargs: Any) -> None:
    """Print info message (hidden in quiet mode)."""
    if _verbosity >= VerbosityLevel.NORMAL:
        get_console().print(message, **kwargs)


def print_success(message: str, **kwargs: Any) -> None:
    """Print success message (hidden in quiet mode)."""
    if _verbosity >= VerbosityLevel.NORMAL:
        get_console().print(f"[green]✓[/green] {message}", **kwargs)


def print_warning(message: str, **kwargs: Any) -> None:
    """Print warning message (always shown)."""
    get_console().print(f"[yellow]⚠[/yellow] {message}", **kwargs)


def print_error(message: str, **kwargs: Any) -> None:
    """Print error message (always shown)."""
    get_console().print(f"[red]✗[/red] {message}", **kwargs)


def print_debug(message: str, **kwargs: Any) -> None:
    """Print debug message (only in verbose mode)."""
    if _verbosity >= VerbosityLevel.VERBOSE:
        get_console().print(f"[dim]{message}[/dim]", **kwargs)


class ProgressTracker:
    """
    Multi-step progress tracker with automatic verbosity handling.
    
    Example usage:
        tracker = ProgressTracker("Preparing dataset", total_steps=4)
        
        with tracker:
            tracker.update("Inspecting dataset...")
            # do work
            tracker.advance()
            
            tracker.update("Splitting dataset...")
            # do work
            tracker.advance()
    """
    
    def __init__(
        self,
        description: str,
        total_steps: int | None = None,
        show_spinner: bool = True,
    ) -> None:
        """
        Initialize progress tracker.
        
        Args:
            description: Overall operation description
            total_steps: Total number of steps (None for indeterminate)
            show_spinner: Show spinner for indeterminate progress
        """
        self.description = description
        self.total_steps = total_steps
        self.show_spinner = show_spinner
        self.current_step = 0
        self._progress: Progress | None = None
        self._task_id: Any = None
        self._status: Status | None = None
    
    def __enter__(self) -> ProgressTracker:
        if is_quiet():
            return self
        
        console = get_console()
        
        if self.total_steps is not None:
            # Determinate progress bar
            self._progress = Progress(
                SpinnerColumn() if self.show_spinner else TextColumn(""),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=console,
            )
            self._progress.__enter__()
            self._task_id = self._progress.add_task(
                self.description,
                total=self.total_steps,
            )
        else:
            # Indeterminate spinner
            self._status = Status(self.description, console=console)
            self._status.__enter__()
        
        return self
    
    def __exit__(self, *args: Any) -> None:
        if self._progress is not None:
            self._progress.__exit__(*args)
        if self._status is not None:
            self._status.__exit__(*args)
    
    def update(self, description: str) -> None:
        """Update the current step description."""
        if is_quiet():
            return
        
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, description=description)
        elif self._status is not None:
            self._status.update(description)
    
    def advance(self, steps: int = 1) -> None:
        """Advance progress by given number of steps."""
        self.current_step += steps
        
        if is_quiet():
            return
        
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=steps)
    
    def log(self, message: str) -> None:
        """Log a message without disrupting progress display."""
        if is_quiet():
            return
        
        console = get_console()
        if self._progress is not None:
            self._progress.print(message)
        elif self._status is not None:
            self._status.console.print(message)
        else:
            console.print(message)


def track_progress(
    iterable: Iterable[T],
    description: str = "Processing...",
    total: int | None = None,
) -> Generator[T, None, None]:
    """
    Wrap an iterable with a progress bar.
    
    Args:
        iterable: Items to iterate over
        description: Description for the progress bar
        total: Total count (auto-detected for sequences)
    
    Yields:
        Items from the iterable
    """
    if is_quiet():
        yield from iterable
        return
    
    # Try to get length if not provided
    if total is None:
        try:
            total = len(iterable)  # type: ignore
        except TypeError:
            pass
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=get_console(),
    ) as progress:
        task_id = progress.add_task(description, total=total)
        
        for item in iterable:
            yield item
            progress.advance(task_id)


@contextmanager
def spinner(description: str) -> Generator[Callable[[str], None], None, None]:
    """
    Display a spinner for indeterminate operations.
    
    Args:
        description: Initial description
    
    Yields:
        Function to update the spinner description
    
    Example:
        with spinner("Loading...") as update:
            # do work
            update("Still loading...")
            # more work
    """
    if is_quiet():
        yield lambda x: None
        return
    
    with Status(description, console=get_console()) as status:
        yield status.update

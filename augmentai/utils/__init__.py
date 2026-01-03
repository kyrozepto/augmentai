"""
Utility modules for AugmentAI.

Provides shared functionality:
- progress: Progress bars, spinners, and verbosity control
"""

from augmentai.utils.progress import (
    VerbosityLevel,
    ProgressTracker,
    get_verbosity,
    set_verbosity,
    is_quiet,
    is_verbose,
    print_info,
    print_success,
    print_warning,
    print_error,
    print_debug,
    track_progress,
    spinner,
)

__all__ = [
    "VerbosityLevel",
    "ProgressTracker",
    "get_verbosity",
    "set_verbosity",
    "is_quiet",
    "is_verbose",
    "print_info",
    "print_success",
    "print_warning",
    "print_error",
    "print_debug",
    "track_progress",
    "spinner",
]

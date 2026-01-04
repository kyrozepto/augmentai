# Web UI Module for AugmentAI
"""
Local web interface for AugmentAI.
Launch with: augmentai serve
"""

from augmentai.web.server import create_app, run_server

__all__ = ["create_app", "run_server"]

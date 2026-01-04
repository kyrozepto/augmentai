"""
FastAPI server for AugmentAI Web UI.

This module provides a local web server that serves the React frontend
and exposes REST API endpoints for all AugmentAI functionality.
"""

import os
import webbrowser
from pathlib import Path
from typing import Optional

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system env vars

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Import route modules
from augmentai.web.routes import (
    datasets, policies, domains, health, search, chat,
    ablation, diff, repair, curriculum, shift
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AugmentAI",
        description="LLM-Powered Data Augmentation Policy Designer",
        version="1.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # CORS for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routes
    app.include_router(health.router, prefix="/api", tags=["Health"])
    app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])
    app.include_router(policies.router, prefix="/api/policies", tags=["Policies"])
    app.include_router(domains.router, prefix="/api/domains", tags=["Domains"])
    app.include_router(search.router, prefix="/api/search", tags=["Search"])
    app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
    app.include_router(ablation.router, prefix="/api/ablation", tags=["Ablation"])
    app.include_router(diff.router, prefix="/api/diff", tags=["Diff"])
    app.include_router(repair.router, prefix="/api/repair", tags=["Repair"])
    app.include_router(curriculum.router, prefix="/api/curriculum", tags=["Curriculum"])
    app.include_router(shift.router, prefix="/api/shift", tags=["Shift"])

    # Serve static frontend files (production)
    frontend_dist = Path(__file__).parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")
        
        @app.get("/{full_path:path}")
        async def serve_frontend(full_path: str):
            """Serve the React app for all non-API routes."""
            file_path = frontend_dist / full_path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(frontend_dist / "index.html")

    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    reload: bool = False,
) -> None:
    """
    Run the AugmentAI web server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        open_browser: Whether to open browser automatically
        reload: Enable auto-reload for development
    """
    import uvicorn

    url = f"http://{host}:{port}"
    
    if open_browser:
        print(f"\n  ╔══════════════════════════════════════════╗")
        print(f"  ║  AUGMENTAI :: WEB UI                     ║")
        print(f"  ╠══════════════════════════════════════════╣")
        print(f"  ║  Opening browser at: {url:<18} ║")
        print(f"  ║  API docs at: {url}/api/docs            ║")
        print(f"  ║                                          ║")
        print(f"  ║  Press Ctrl+C to stop                    ║")
        print(f"  ╚══════════════════════════════════════════╝\n")
        webbrowser.open(url)

    uvicorn.run(
        "augmentai.web.server:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )

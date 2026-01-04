"""Health check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_available: bool


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the server is running and LLM is available."""
    # Try to detect if LLM is configured
    import os
    llm_available = bool(
        os.getenv("OPENAI_API_KEY") or 
        os.getenv("GOOGLE_API_KEY") or
        os.getenv("OLLAMA_HOST")
    )
    
    return HealthResponse(
        status="ok",
        version="1.1.0",
        llm_available=llm_available,
    )

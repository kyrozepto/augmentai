"""AutoSearch API endpoints with WebSocket for real-time progress."""

import asyncio
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

router = APIRouter()


class SearchConfig(BaseModel):
    """AutoSearch configuration."""
    dataset_path: str
    domain: str = "natural"
    model_type: str = "resnet18"
    num_trials: int = 20
    metric: str = "accuracy"
    
    
class SearchResult(BaseModel):
    """Search result with best policy."""
    id: str
    status: str  # pending, running, completed, failed
    best_score: Optional[float] = None
    best_policy: Optional[dict] = None
    trials_completed: int = 0
    total_trials: int = 20


# In-memory search jobs
_searches: dict[str, SearchResult] = {}
_search_connections: dict[str, list[WebSocket]] = {}


@router.post("", response_model=SearchResult)
async def start_search(config: SearchConfig):
    """Start a new AutoSearch job."""
    import uuid
    search_id = str(uuid.uuid4())[:8]
    
    result = SearchResult(
        id=search_id,
        status="pending",
        total_trials=config.num_trials,
    )
    _searches[search_id] = result
    _search_connections[search_id] = []
    
    # Start async search task
    asyncio.create_task(_run_search(search_id, config))
    
    return result


@router.get("/{search_id}", response_model=SearchResult)
async def get_search(search_id: str):
    """Get search status and results."""
    if search_id not in _searches:
        raise HTTPException(status_code=404, detail="Search not found")
    return _searches[search_id]


@router.websocket("/{search_id}/ws")
async def search_websocket(websocket: WebSocket, search_id: str):
    """WebSocket for real-time search progress updates."""
    if search_id not in _searches:
        await websocket.close(code=4004, reason="Search not found")
        return
        
    await websocket.accept()
    _search_connections[search_id].append(websocket)
    
    try:
        # Send current state
        await websocket.send_json(_searches[search_id].model_dump())
        
        # Keep connection alive and handle disconnects
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
    finally:
        if websocket in _search_connections.get(search_id, []):
            _search_connections[search_id].remove(websocket)


async def _broadcast_update(search_id: str):
    """Broadcast search update to all connected clients."""
    if search_id not in _search_connections:
        return
        
    result = _searches[search_id]
    data = result.model_dump()
    
    for ws in _search_connections[search_id][:]:  # Copy list to avoid mutation issues
        try:
            await ws.send_json(data)
        except:
            _search_connections[search_id].remove(ws)


async def _run_search(search_id: str, config: SearchConfig):
    """Run the actual AutoSearch (or simulate for demo)."""
    result = _searches[search_id]
    result.status = "running"
    await _broadcast_update(search_id)
    
    try:
        # Try to run actual search
        from augmentai.search import quick_search
        
        # For now, simulate since we don't want to run a full search
        # In production, this would call quick_search()
        await _simulate_search(search_id, config)
        
    except ImportError:
        # Fallback simulation
        await _simulate_search(search_id, config)
    except Exception as e:
        result.status = "failed"
        await _broadcast_update(search_id)


async def _simulate_search(search_id: str, config: SearchConfig):
    """Simulate search progress for demo purposes."""
    import random
    
    result = _searches[search_id]
    best_score = 0.0
    
    sample_transforms = [
        {"name": "HorizontalFlip", "probability": 0.5},
        {"name": "Rotate", "probability": 0.3, "parameters": {"limit": 15}},
        {"name": "RandomBrightnessContrast", "probability": 0.4},
        {"name": "GaussNoise", "probability": 0.2},
    ]
    
    for i in range(config.num_trials):
        await asyncio.sleep(0.5)  # Simulate trial time
        
        # Generate random score
        score = random.uniform(0.6, 0.95)
        result.trials_completed = i + 1
        
        if score > best_score:
            best_score = score
            result.best_score = round(score, 4)
            result.best_policy = {
                "name": f"trial_{i+1}_policy",
                "domain": config.domain,
                "transforms": random.sample(sample_transforms, k=random.randint(2, 4)),
            }
        
        await _broadcast_update(search_id)
    
    result.status = "completed"
    await _broadcast_update(search_id)

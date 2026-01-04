"""Datasets API endpoints."""

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

router = APIRouter()


class DatasetInfo(BaseModel):
    """Dataset metadata."""
    id: str
    name: str
    path: str
    image_count: int
    domain: Optional[str] = None
    classes: List[str] = []


class DatasetStats(BaseModel):
    """Detailed dataset statistics."""
    id: str
    name: str
    path: str
    image_count: int
    total_size_bytes: int
    avg_resolution: str
    channels: int
    domain: Optional[str] = None
    domain_confidence: float = 0.0
    classes: List[str] = []
    class_distribution: dict = {}
    warnings: List[str] = []


# In-memory dataset registry (for MVP, replace with persistent storage later)
_datasets: dict[str, DatasetStats] = {}


@router.get("", response_model=List[DatasetInfo])
async def list_datasets():
    """List all registered datasets."""
    return [
        DatasetInfo(
            id=ds.id,
            name=ds.name,
            path=ds.path,
            image_count=ds.image_count,
            domain=ds.domain,
            classes=ds.classes,
        )
        for ds in _datasets.values()
    ]


@router.post("", response_model=DatasetStats)
async def register_dataset(path: str = Form(...)):
    """Register a dataset by path and run inspection."""
    dataset_path = Path(path)
    
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")
    
    if not dataset_path.is_dir():
        raise HTTPException(status_code=400, detail="Path must be a directory")
    
    # Generate ID from path
    dataset_id = dataset_path.name.replace(" ", "_").lower()
    
    # Run inspection using existing AugmentAI functionality
    try:
        from augmentai.inspection.inspector import DatasetInspector
        
        inspector = DatasetInspector(str(dataset_path))
        result = inspector.inspect()
        
        stats = DatasetStats(
            id=dataset_id,
            name=dataset_path.name,
            path=str(dataset_path),
            image_count=result.get("image_count", 0),
            total_size_bytes=result.get("total_size_bytes", 0),
            avg_resolution=result.get("avg_resolution", "unknown"),
            channels=result.get("channels", 3),
            domain=result.get("detected_domain"),
            domain_confidence=result.get("domain_confidence", 0.0),
            classes=result.get("classes", []),
            class_distribution=result.get("class_distribution", {}),
            warnings=result.get("warnings", []),
        )
    except ImportError:
        # Fallback if inspector not available
        import os
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = [
            f for f in dataset_path.rglob("*") 
            if f.suffix.lower() in image_extensions
        ]
        
        stats = DatasetStats(
            id=dataset_id,
            name=dataset_path.name,
            path=str(dataset_path),
            image_count=len(images),
            total_size_bytes=sum(f.stat().st_size for f in images),
            avg_resolution="unknown",
            channels=3,
            warnings=["Inspector not available, basic stats only"],
        )
    
    _datasets[dataset_id] = stats
    return stats


@router.get("/{dataset_id}", response_model=DatasetStats)
async def get_dataset(dataset_id: str):
    """Get detailed stats for a specific dataset."""
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return _datasets[dataset_id]


@router.delete("/{dataset_id}")
async def remove_dataset(dataset_id: str):
    """Remove a dataset from the registry (does not delete files)."""
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    del _datasets[dataset_id]
    return {"status": "removed", "id": dataset_id}

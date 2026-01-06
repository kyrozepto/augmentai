"""Datasets API endpoints."""

from pathlib import Path
from typing import List, Optional
import base64
import io
import random

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
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


class ImageInfo(BaseModel):
    """Basic image info."""
    id: str
    filename: str
    path: str
    size_bytes: int
    class_name: Optional[str] = None


class ClassDistributionItem(BaseModel):
    """Class distribution item."""
    class_name: str
    count: int
    percentage: float


# In-memory dataset registry (for MVP, replace with persistent storage later)
_datasets: dict[str, DatasetStats] = {}


def get_image_extensions():
    return {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}


def get_images_from_path(dataset_path: Path, limit: int = None) -> List[Path]:
    """Get list of images from a dataset path."""
    image_extensions = get_image_extensions()
    images = [
        f for f in dataset_path.rglob("*") 
        if f.suffix.lower() in image_extensions
    ]
    if limit:
        images = images[:limit]
    return images


def create_thumbnail(image_path: Path, size: tuple = (150, 150)) -> bytes:
    """Create a thumbnail from an image file."""
    try:
        from PIL import Image
        
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Create thumbnail maintaining aspect ratio
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Save to bytes
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            return buffer.getvalue()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create thumbnail: {e}")


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
        images = get_images_from_path(dataset_path)
        
        # Try to detect classes from directory structure
        classes = []
        class_distribution = {}
        for img in images:
            parent = img.parent.name
            if parent != dataset_path.name:
                if parent not in class_distribution:
                    class_distribution[parent] = 0
                    classes.append(parent)
                class_distribution[parent] += 1
        
        stats = DatasetStats(
            id=dataset_id,
            name=dataset_path.name,
            path=str(dataset_path),
            image_count=len(images),
            total_size_bytes=sum(f.stat().st_size for f in images),
            avg_resolution="unknown",
            channels=3,
            classes=classes,
            class_distribution=class_distribution,
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


@router.get("/{dataset_id}/images", response_model=List[ImageInfo])
async def list_dataset_images(dataset_id: str, limit: int = 50, offset: int = 0):
    """List images in a dataset with pagination."""
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = _datasets[dataset_id]
    dataset_path = Path(dataset.path)
    
    images = get_images_from_path(dataset_path)
    images = sorted(images, key=lambda x: x.name)
    
    # Paginate
    paginated = images[offset:offset + limit]
    
    result = []
    for img in paginated:
        # Determine class from parent directory
        class_name = None
        if img.parent.name != dataset_path.name:
            class_name = img.parent.name
        
        result.append(ImageInfo(
            id=img.stem,
            filename=img.name,
            path=str(img),
            size_bytes=img.stat().st_size,
            class_name=class_name,
        ))
    
    return result


@router.get("/{dataset_id}/images/{image_id}/thumbnail")
async def get_image_thumbnail(dataset_id: str, image_id: str):
    """Get a thumbnail of a specific image."""
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = _datasets[dataset_id]
    dataset_path = Path(dataset.path)
    
    # Find the image
    images = get_images_from_path(dataset_path)
    target = None
    for img in images:
        if img.stem == image_id:
            target = img
            break
    
    if not target:
        raise HTTPException(status_code=404, detail="Image not found")
    
    thumb_bytes = create_thumbnail(target)
    return Response(content=thumb_bytes, media_type="image/jpeg")


@router.get("/{dataset_id}/sample-thumbnails")
async def get_sample_thumbnails(dataset_id: str, count: int = 6):
    """Get random sample thumbnails from dataset as base64."""
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = _datasets[dataset_id]
    dataset_path = Path(dataset.path)
    
    images = get_images_from_path(dataset_path)
    
    # Random sample
    sample_count = min(count, len(images))
    samples = random.sample(images, sample_count) if images else []
    
    result = []
    for img in samples:
        try:
            thumb_bytes = create_thumbnail(img, size=(200, 200))
            b64 = base64.b64encode(thumb_bytes).decode('utf-8')
            
            class_name = None
            if img.parent.name != dataset_path.name:
                class_name = img.parent.name
            
            result.append({
                "id": img.stem,
                "filename": img.name,
                "class_name": class_name,
                "thumbnail": f"data:image/jpeg;base64,{b64}",
            })
        except Exception:
            continue
    
    return result


@router.get("/{dataset_id}/distribution", response_model=List[ClassDistributionItem])
async def get_class_distribution(dataset_id: str):
    """Get class distribution data for visualization."""
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = _datasets[dataset_id]
    
    if not dataset.class_distribution:
        return []
    
    total = sum(dataset.class_distribution.values())
    result = []
    for class_name, count in dataset.class_distribution.items():
        result.append(ClassDistributionItem(
            class_name=class_name,
            count=count,
            percentage=round((count / total) * 100, 1) if total > 0 else 0,
        ))
    
    # Sort by count descending
    result.sort(key=lambda x: x.count, reverse=True)
    return result


@router.delete("/{dataset_id}")
async def remove_dataset(dataset_id: str):
    """Remove a dataset from the registry (does not delete files)."""
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    del _datasets[dataset_id]
    return {"status": "removed", "id": dataset_id}


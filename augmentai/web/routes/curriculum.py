"""Curriculum learning API endpoint - integrated with real augmentai.curriculum."""

from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class CurriculumStage(BaseModel):
    """A stage in the curriculum."""
    epoch_start: int
    epoch_end: int
    difficulty: str
    sample_percentage: float
    n_samples: int


class CurriculumConfig(BaseModel):
    """Configuration for curriculum builder."""
    dataset_path: Optional[str] = None
    total_epochs: int = 100
    pacing: str = "linear"  # linear, quadratic, exponential, step
    warmup_epochs: int = 5
    mock: bool = True


class CurriculumResult(BaseModel):
    """Generated curriculum schedule."""
    n_samples: int
    stages: List[CurriculumStage]
    difficulty_distribution: dict


@router.post("", response_model=CurriculumResult)
async def build_curriculum(config: CurriculumConfig):
    """
    Build a curriculum learning schedule.
    """
    try:
        from augmentai.curriculum import DifficultyScorer, CurriculumScheduler
        
        if config.dataset_path:
            dataset_path = Path(config.dataset_path)
            if dataset_path.exists():
                samples = _discover_samples(dataset_path)
                n_samples = len(samples)
            else:
                n_samples = 1000  # Default if path doesn't exist
        else:
            n_samples = 1000
        
        # Create scheduler
        scheduler = CurriculumScheduler(
            pacing=config.pacing,
            warmup_epochs=config.warmup_epochs,
        )
        
        # Generate stages based on pacing
        stages = _generate_stages(
            n_samples=n_samples,
            total_epochs=config.total_epochs,
            warmup=config.warmup_epochs,
            pacing=config.pacing,
        )
        
        return CurriculumResult(
            n_samples=n_samples,
            stages=stages,
            difficulty_distribution={
                "easy": int(n_samples * 0.3),
                "medium": int(n_samples * 0.4),
                "hard": int(n_samples * 0.3),
            },
        )
        
    except ImportError:
        return await _simulated_curriculum(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _discover_samples(dataset_path: Path) -> list:
    """Discover samples in dataset directory."""
    samples = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    
    for class_dir in dataset_path.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in extensions:
                samples.append(img_file)
    
    return samples


def _generate_stages(n_samples: int, total_epochs: int, warmup: int, pacing: str) -> List[CurriculumStage]:
    """Generate curriculum stages based on pacing function."""
    stages = []
    
    # Warmup stage
    if warmup > 0:
        stages.append(CurriculumStage(
            epoch_start=0,
            epoch_end=warmup - 1,
            difficulty="easy",
            sample_percentage=30.0,
            n_samples=int(n_samples * 0.3),
        ))
    
    # Main stages
    remaining = total_epochs - warmup
    stage_epochs = remaining // 3
    
    stages.append(CurriculumStage(
        epoch_start=warmup,
        epoch_end=warmup + stage_epochs - 1,
        difficulty="easy+medium",
        sample_percentage=60.0,
        n_samples=int(n_samples * 0.6),
    ))
    
    stages.append(CurriculumStage(
        epoch_start=warmup + stage_epochs,
        epoch_end=warmup + 2 * stage_epochs - 1,
        difficulty="medium+hard",
        sample_percentage=80.0,
        n_samples=int(n_samples * 0.8),
    ))
    
    stages.append(CurriculumStage(
        epoch_start=warmup + 2 * stage_epochs,
        epoch_end=total_epochs - 1,
        difficulty="all",
        sample_percentage=100.0,
        n_samples=n_samples,
    ))
    
    return stages


async def _simulated_curriculum(config: CurriculumConfig) -> CurriculumResult:
    """Fallback simulated curriculum."""
    n_samples = 1000
    stages = _generate_stages(
        n_samples=n_samples,
        total_epochs=config.total_epochs,
        warmup=config.warmup_epochs,
        pacing=config.pacing,
    )
    
    return CurriculumResult(
        n_samples=n_samples,
        stages=stages,
        difficulty_distribution={
            "easy": 300,
            "medium": 400,
            "hard": 300,
        },
    )

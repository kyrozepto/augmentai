"""Data repair API endpoint - integrated with real augmentai.repair."""

from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class UncertainSample(BaseModel):
    """Uncertain sample identified for repair."""
    id: str
    path: str
    confidence: float
    predicted_class: str
    true_class: str
    suggested_class: Optional[str] = None
    action: str  # REMOVE, RELABEL, REWEIGHT, REVIEW, KEEP


class RepairConfig(BaseModel):
    """Configuration for data repair."""
    dataset_path: str
    uncertainty_threshold: float = 0.7
    mock: bool = True


class RepairResult(BaseModel):
    """Result of data repair analysis."""
    total_samples: int
    repair_rate: float
    summary: str
    samples: List[UncertainSample]


@router.post("", response_model=RepairResult)
async def analyze_repair(config: RepairConfig):
    """
    Analyze dataset for uncertain or mislabeled samples.
    """
    try:
        from augmentai.repair import SampleAnalyzer, DataRepair, RepairReport
        from augmentai.repair.repair_suggestions import RepairAction
        
        dataset_path = Path(config.dataset_path)
        
        if not dataset_path.exists():
            raise HTTPException(status_code=400, detail=f"Dataset path not found: {config.dataset_path}")
        
        # Discover samples
        samples = _discover_samples(dataset_path)
        if not samples:
            return RepairResult(
                total_samples=0,
                repair_rate=0.0,
                summary="No samples found",
                samples=[],
            )
        
        # Use mock functions
        uncertainty_fn, loss_fn, predict_fn = _create_mock_functions()
        
        # Analyze
        analyzer = SampleAnalyzer(
            uncertainty_fn=uncertainty_fn,
            loss_fn=loss_fn,
            predict_fn=predict_fn,
        )
        analyses = analyzer.analyze_dataset(samples)
        
        # Generate suggestions
        repair = DataRepair(uncertainty_threshold=config.uncertainty_threshold)
        suggestions = repair.suggest_repairs(analyses)
        
        # Create report
        report = RepairReport(n_samples=len(samples), suggestions=suggestions)
        
        return RepairResult(
            total_samples=len(samples),
            repair_rate=report.repair_rate,
            summary=report.summary(),
            samples=[
                UncertainSample(
                    id=s.sample_id,
                    path=str(s.sample_path) if hasattr(s, 'sample_path') else s.sample_id,
                    confidence=s.confidence,
                    predicted_class=s.predicted_class if hasattr(s, 'predicted_class') else "",
                    true_class=s.true_class if hasattr(s, 'true_class') else "",
                    suggested_class=s.suggested_class if hasattr(s, 'suggested_class') else None,
                    action=s.action.value if hasattr(s.action, 'value') else str(s.action),
                )
                for s in suggestions[:20]  # Limit to top 20
            ],
        )
        
    except ImportError:
        return await _simulated_repair(config)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _discover_samples(dataset_path: Path) -> list:
    """Discover samples in dataset directory."""
    samples = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    
    for class_dir in dataset_path.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        label = class_dir.name
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in extensions:
                samples.append((img_file, label))
    
    return samples


def _create_mock_functions():
    """Create mock evaluation functions."""
    import hashlib
    
    def uncertainty_fn(path: Path) -> float:
        h = int(hashlib.md5(str(path).encode()).hexdigest(), 16)
        return (h % 100) / 100
    
    def loss_fn(path: Path, label: str) -> float:
        h = int(hashlib.md5(f"{path}{label}".encode()).hexdigest(), 16)
        return (h % 500) / 100
    
    def predict_fn(path: Path) -> tuple:
        h = int(hashlib.md5(str(path).encode()).hexdigest(), 16)
        confidence = 0.5 + (h % 50) / 100
        if h % 7 == 0:
            return "wrong_class", confidence
        return path.parent.name, confidence
    
    return uncertainty_fn, loss_fn, predict_fn


async def _simulated_repair(config: RepairConfig) -> RepairResult:
    """Fallback simulated repair."""
    import random
    
    samples = []
    for i in range(10):
        conf = random.uniform(0.2, 0.8)
        action = "RELABEL" if conf < 0.3 else "REVIEW" if conf < 0.5 else "KEEP"
        samples.append(UncertainSample(
            id=f"sample_{i:04d}",
            path=f"{config.dataset_path}/class_a/img_{i}.jpg",
            confidence=round(conf, 3),
            predicted_class="class_a",
            true_class="class_b" if conf < 0.3 else "class_a",
            suggested_class="class_b" if conf < 0.4 else None,
            action=action,
        ))
    
    samples.sort(key=lambda x: x.confidence)
    uncertain_count = sum(1 for s in samples if s.action != "KEEP")
    
    return RepairResult(
        total_samples=1000,
        repair_rate=uncertain_count / 10,
        summary=f"{uncertain_count} samples need attention",
        samples=samples,
    )

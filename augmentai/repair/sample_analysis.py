"""
Sample analysis for model-guided data repair.

Analyzes samples using model feedback to identify problematic data points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class SampleAnalysis:
    """Analysis of a single sample's quality signals.
    
    Attributes:
        sample_id: Unique identifier for the sample
        file_path: Path to the sample file
        uncertainty: Model uncertainty (entropy, MC dropout variance)
        loss: Per-sample loss value
        confidence: Top-1 prediction confidence
        predicted_label: Model's predicted label
        true_label: Ground truth label
        is_misclassified: Whether prediction differs from ground truth
        embedding: Optional feature embedding for similarity analysis
        nearest_neighbors: IDs of most similar training samples
    """
    sample_id: str
    file_path: Path
    uncertainty: float
    loss: float
    confidence: float
    predicted_label: str
    true_label: str
    is_misclassified: bool = False
    embedding: np.ndarray | None = None
    nearest_neighbors: list[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate and compute derived fields."""
        self.is_misclassified = self.predicted_label != self.true_label
        
        # Clamp values to valid ranges
        self.uncertainty = max(0.0, min(1.0, self.uncertainty))
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    @property
    def quality_score(self) -> float:
        """Composite quality score (0=poor, 1=high quality).
        
        High quality: low uncertainty, high confidence, correctly classified.
        """
        base_score = 1.0
        
        # Penalize high uncertainty
        base_score -= 0.3 * self.uncertainty
        
        # Penalize low confidence
        base_score -= 0.3 * (1 - self.confidence)
        
        # Major penalty for misclassification
        if self.is_misclassified:
            base_score -= 0.4
        
        return max(0.0, min(1.0, base_score))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "sample_id": self.sample_id,
            "file_path": str(self.file_path),
            "uncertainty": self.uncertainty,
            "loss": self.loss,
            "confidence": self.confidence,
            "predicted_label": self.predicted_label,
            "true_label": self.true_label,
            "is_misclassified": self.is_misclassified,
            "nearest_neighbors": self.nearest_neighbors,
            "quality_score": self.quality_score,
        }


class SampleAnalyzer:
    """Analyze samples using model feedback.
    
    Uses provided callback functions to compute uncertainty, loss, and predictions
    for each sample. This allows integration with any model framework.
    
    Example:
        def uncertainty_fn(path: Path) -> float:
            # MC dropout or ensemble uncertainty
            return compute_uncertainty(model, load_image(path))
        
        def loss_fn(path: Path, label: str) -> float:
            return compute_loss(model, load_image(path), label)
        
        def predict_fn(path: Path) -> tuple[str, float]:
            logits = model(load_image(path))
            return predicted_class, confidence
        
        analyzer = SampleAnalyzer(uncertainty_fn, loss_fn, predict_fn)
        analyses = analyzer.analyze_dataset(samples)
    """
    
    def __init__(
        self,
        uncertainty_fn: Callable[[Path], float],
        loss_fn: Callable[[Path, str], float],
        predict_fn: Callable[[Path], tuple[str, float]],
        embedding_fn: Callable[[Path], np.ndarray] | None = None,
    ):
        """Initialize sample analyzer.
        
        Args:
            uncertainty_fn: Function that computes uncertainty for a sample
            loss_fn: Function that computes loss given sample and true label
            predict_fn: Function that returns (predicted_label, confidence)
            embedding_fn: Optional function to compute feature embeddings
        """
        self.uncertainty_fn = uncertainty_fn
        self.loss_fn = loss_fn
        self.predict_fn = predict_fn
        self.embedding_fn = embedding_fn
    
    def analyze_sample(
        self,
        sample_path: Path,
        true_label: str,
        sample_id: str | None = None,
    ) -> SampleAnalysis:
        """Analyze a single sample.
        
        Args:
            sample_path: Path to the sample file
            true_label: Ground truth label
            sample_id: Optional ID (defaults to filename)
            
        Returns:
            SampleAnalysis with computed metrics
        """
        if sample_id is None:
            sample_id = sample_path.stem
        
        # Get model predictions
        predicted_label, confidence = self.predict_fn(sample_path)
        
        # Compute uncertainty and loss
        uncertainty = self.uncertainty_fn(sample_path)
        loss = self.loss_fn(sample_path, true_label)
        
        # Optionally compute embedding
        embedding = None
        if self.embedding_fn is not None:
            embedding = self.embedding_fn(sample_path)
        
        return SampleAnalysis(
            sample_id=sample_id,
            file_path=sample_path,
            uncertainty=uncertainty,
            loss=loss,
            confidence=confidence,
            predicted_label=predicted_label,
            true_label=true_label,
            embedding=embedding,
        )
    
    def analyze_dataset(
        self,
        samples: list[tuple[Path, str]],
        compute_neighbors: bool = False,
        k_neighbors: int = 5,
    ) -> list[SampleAnalysis]:
        """Analyze all samples in a dataset.
        
        Args:
            samples: List of (path, true_label) tuples
            compute_neighbors: Whether to compute nearest neighbors
            k_neighbors: Number of neighbors to find
            
        Returns:
            List of SampleAnalysis objects
        """
        analyses = []
        
        for i, (path, label) in enumerate(samples):
            sample_id = f"{i:05d}_{path.stem}"
            analysis = self.analyze_sample(path, label, sample_id)
            analyses.append(analysis)
        
        # Optionally compute nearest neighbors using embeddings
        if compute_neighbors and self.embedding_fn is not None:
            self._compute_neighbors(analyses, k_neighbors)
        
        return analyses
    
    def _compute_neighbors(
        self,
        analyses: list[SampleAnalysis],
        k: int,
    ) -> None:
        """Compute k nearest neighbors for each sample using embeddings."""
        embeddings = []
        valid_indices = []
        
        for i, analysis in enumerate(analyses):
            if analysis.embedding is not None:
                embeddings.append(analysis.embedding)
                valid_indices.append(i)
        
        if len(embeddings) < 2:
            return
        
        embeddings_array = np.array(embeddings)
        
        # Compute pairwise distances
        for i, idx in enumerate(valid_indices):
            dists = np.linalg.norm(embeddings_array - embeddings_array[i], axis=1)
            # Get k nearest (excluding self)
            neighbor_indices = np.argsort(dists)[1:k+1]
            analyses[idx].nearest_neighbors = [
                analyses[valid_indices[j]].sample_id 
                for j in neighbor_indices
            ]
    
    def get_high_uncertainty_samples(
        self,
        analyses: list[SampleAnalysis],
        threshold: float = 0.7,
    ) -> list[SampleAnalysis]:
        """Filter samples with high uncertainty."""
        return [a for a in analyses if a.uncertainty >= threshold]
    
    def get_misclassified_samples(
        self,
        analyses: list[SampleAnalysis],
    ) -> list[SampleAnalysis]:
        """Filter misclassified samples."""
        return [a for a in analyses if a.is_misclassified]
    
    def get_high_loss_samples(
        self,
        analyses: list[SampleAnalysis],
        percentile: float = 95,
    ) -> list[SampleAnalysis]:
        """Filter samples with loss above the given percentile."""
        losses = [a.loss for a in analyses]
        threshold = np.percentile(losses, percentile)
        return [a for a in analyses if a.loss >= threshold]

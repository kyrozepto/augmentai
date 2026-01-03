"""
Repair suggestions for model-guided data repair.

Generates actionable suggestions (relabel, reweight, remove) based on sample analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from augmentai.repair.sample_analysis import SampleAnalysis


class RepairAction(str, Enum):
    """Possible repair actions for a sample."""
    KEEP = "keep"
    RELABEL = "relabel"
    REWEIGHT = "reweight"
    REMOVE = "remove"
    REVIEW = "review"  # Needs manual inspection


@dataclass
class RepairSuggestion:
    """Suggested repair action for a sample.
    
    Attributes:
        sample_id: ID of the sample to repair
        action: Suggested repair action
        reason: Human-readable explanation
        confidence: How confident the system is (0-1)
        suggested_label: For relabel action, the suggested new label
        suggested_weight: For reweight action, the suggested sample weight
        priority: Priority score for repair (higher = more urgent)
    """
    sample_id: str
    action: RepairAction
    reason: str
    confidence: float
    suggested_label: str | None = None
    suggested_weight: float = 1.0
    priority: float = 0.5
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "sample_id": self.sample_id,
            "action": self.action.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "suggested_label": self.suggested_label,
            "suggested_weight": self.suggested_weight,
            "priority": self.priority,
        }


class DataRepair:
    """Generate repair suggestions from sample analyses.
    
    Uses heuristics and thresholds to determine appropriate repair actions:
    - REMOVE: Extremely high uncertainty + high loss (likely corrupt/noisy)
    - RELABEL: High confidence in wrong prediction (likely mislabeled)
    - REWEIGHT: Moderate issues (down-weight during training)
    - REVIEW: Needs manual inspection
    - KEEP: No action needed
    
    Example:
        repair = DataRepair(
            uncertainty_threshold=0.8,
            loss_threshold=2.0,
        )
        suggestions = repair.suggest_repairs(analyses)
        weights = repair.apply_reweighting(suggestions)
    """
    
    def __init__(
        self,
        uncertainty_threshold: float = 0.7,
        loss_percentile: float = 95,
        confidence_threshold: float = 0.9,
        remove_threshold: float = 0.9,
        relabel_confidence: float = 0.8,
    ):
        """Initialize data repair suggester.
        
        Args:
            uncertainty_threshold: Samples above this are flagged
            loss_percentile: Samples above this percentile loss are flagged
            confidence_threshold: High-confidence wrong predictions suggest relabeling
            remove_threshold: Combined score above this suggests removal
            relabel_confidence: Min confidence to suggest relabel over review
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.loss_percentile = loss_percentile
        self.confidence_threshold = confidence_threshold
        self.remove_threshold = remove_threshold
        self.relabel_confidence = relabel_confidence
        
        self._loss_threshold: float | None = None
        self._median_loss: float | None = None
    
    def suggest_repairs(
        self,
        analyses: list[SampleAnalysis],
    ) -> list[RepairSuggestion]:
        """Generate repair suggestions for all samples.
        
        Args:
            analyses: List of sample analyses
            
        Returns:
            List of repair suggestions (only for samples needing action)
        """
        if not analyses:
            return []
        
        # Compute loss statistics for thresholds
        losses = [a.loss for a in analyses]
        import numpy as np
        self._loss_threshold = np.percentile(losses, self.loss_percentile)
        self._median_loss = np.median(losses)
        
        suggestions = []
        for analysis in analyses:
            suggestion = self._suggest_for_sample(analysis)
            if suggestion is not None:
                suggestions.append(suggestion)
        
        # Sort by priority (highest first)
        suggestions.sort(key=lambda s: s.priority, reverse=True)
        
        return suggestions
    
    def _suggest_for_sample(
        self,
        analysis: SampleAnalysis,
    ) -> RepairSuggestion | None:
        """Generate suggestion for a single sample."""
        # Check for removal candidates first
        if self._should_remove(analysis):
            return RepairSuggestion(
                sample_id=analysis.sample_id,
                action=RepairAction.REMOVE,
                reason=f"High uncertainty ({analysis.uncertainty:.2f}) and high loss ({analysis.loss:.2f})",
                confidence=0.8,
                priority=1.0,
            )
        
        # Check for relabeling candidates
        if self._should_relabel(analysis):
            return RepairSuggestion(
                sample_id=analysis.sample_id,
                action=RepairAction.RELABEL,
                reason=f"Misclassified with high confidence ({analysis.confidence:.2f})",
                confidence=analysis.confidence,
                suggested_label=analysis.predicted_label,
                priority=0.9,
            )
        
        # Check for reweighting candidates
        if self._should_reweight(analysis):
            weight = self._compute_weight(analysis)
            return RepairSuggestion(
                sample_id=analysis.sample_id,
                action=RepairAction.REWEIGHT,
                reason=f"Moderate uncertainty ({analysis.uncertainty:.2f}), adjusting weight",
                confidence=0.7,
                suggested_weight=weight,
                priority=0.5,
            )
        
        # Check for manual review
        if self._needs_review(analysis):
            return RepairSuggestion(
                sample_id=analysis.sample_id,
                action=RepairAction.REVIEW,
                reason="Ambiguous signals, needs manual inspection",
                confidence=0.5,
                priority=0.3,
            )
        
        return None  # Keep as-is
    
    def _should_remove(self, analysis: SampleAnalysis) -> bool:
        """Check if sample should be removed.
        
        Removal is suggested for samples with BOTH:
        - Very high uncertainty (>= remove_threshold, default 0.9)
        - Significantly outlier loss (>= 3x median loss, or if loss_threshold exists: >= 1.5x)
        """
        high_uncertainty = analysis.uncertainty >= self.remove_threshold
        
        # Check for high loss
        if self._loss_threshold is not None:
            # If loss is much higher than the typical "bad" threshold
            # or if it's an extreme outlier (> median * 5)
            high_loss = (
                analysis.loss >= self._loss_threshold or  # Above 95th percentile
                analysis.loss >= self._median_loss * 5 if self._median_loss else False
            )
        else:
            high_loss = False
        
        return high_uncertainty and high_loss
    
    def _should_relabel(self, analysis: SampleAnalysis) -> bool:
        """Check if sample should be relabeled."""
        # Misclassified with high confidence in wrong prediction
        return (
            analysis.is_misclassified and
            analysis.confidence >= self.confidence_threshold
        )
    
    def _should_reweight(self, analysis: SampleAnalysis) -> bool:
        """Check if sample should be reweighted."""
        # Moderate uncertainty or loss, but not severe
        moderate_uncertainty = (
            self.uncertainty_threshold * 0.5 <= 
            analysis.uncertainty < 
            self.remove_threshold
        )
        moderate_loss = (
            self._loss_threshold is not None and
            analysis.loss >= self._loss_threshold * 0.8 and
            analysis.loss < self._loss_threshold * 1.5
        )
        return moderate_uncertainty or moderate_loss
    
    def _needs_review(self, analysis: SampleAnalysis) -> bool:
        """Check if sample needs manual review."""
        # Misclassified but low confidence (ambiguous)
        return (
            analysis.is_misclassified and
            analysis.confidence < self.confidence_threshold and
            analysis.uncertainty >= self.uncertainty_threshold * 0.5
        )
    
    def _compute_weight(self, analysis: SampleAnalysis) -> float:
        """Compute suggested weight for reweighting."""
        # Lower weight for higher uncertainty
        base_weight = 1.0 - (analysis.uncertainty * 0.5)
        
        # Further reduce if high loss
        if self._loss_threshold and analysis.loss > self._loss_threshold:
            base_weight *= 0.8
        
        return max(0.1, min(1.0, base_weight))
    
    def apply_reweighting(
        self,
        suggestions: list[RepairSuggestion],
    ) -> dict[str, float]:
        """Generate weight mapping from suggestions.
        
        Args:
            suggestions: List of repair suggestions
            
        Returns:
            Dictionary mapping sample_id to weight (samples not in dict have weight=1.0)
        """
        weights = {}
        for suggestion in suggestions:
            if suggestion.action == RepairAction.REWEIGHT:
                weights[suggestion.sample_id] = suggestion.suggested_weight
            elif suggestion.action == RepairAction.REMOVE:
                weights[suggestion.sample_id] = 0.0
        return weights
    
    def get_relabel_mapping(
        self,
        suggestions: list[RepairSuggestion],
    ) -> dict[str, str]:
        """Get mapping of sample_id to suggested new label.
        
        Args:
            suggestions: List of repair suggestions
            
        Returns:
            Dictionary mapping sample_id to suggested label
        """
        mapping = {}
        for suggestion in suggestions:
            if (
                suggestion.action == RepairAction.RELABEL and 
                suggestion.suggested_label is not None
            ):
                mapping[suggestion.sample_id] = suggestion.suggested_label
        return mapping
    
    def summarize(
        self,
        suggestions: list[RepairSuggestion],
    ) -> dict[str, int]:
        """Summarize suggestions by action type.
        
        Returns:
            Dictionary with counts per action type
        """
        counts = {action.value: 0 for action in RepairAction}
        for suggestion in suggestions:
            counts[suggestion.action.value] += 1
        return counts

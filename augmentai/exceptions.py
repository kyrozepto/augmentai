"""
Custom exception hierarchy for AugmentAI.

Provides user-friendly error messages and structured error handling
across the entire application.
"""

from __future__ import annotations

from typing import Any


class AugmentAIError(Exception):
    """
    Base exception for all AugmentAI errors.
    
    All custom exceptions inherit from this class, allowing
    for catch-all error handling at the CLI level.
    """
    
    def __init__(
        self,
        message: str,
        *,
        suggestion: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            suggestion: Optional suggestion for how to fix the error
            details: Optional dict with additional error context
        """
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion
        self.details = details or {}
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.suggestion:
            parts.append(f"\n💡 Suggestion: {self.suggestion}")
        return "".join(parts)


class DatasetError(AugmentAIError):
    """
    Errors related to dataset operations.
    
    Raised during:
    - Dataset inspection (missing files, corrupted images)
    - Dataset splitting (invalid ratios, insufficient samples)
    - Dataset linting (format issues, label mismatches)
    """
    pass


class DatasetNotFoundError(DatasetError):
    """Dataset path does not exist or is not accessible."""
    
    def __init__(self, path: str) -> None:
        super().__init__(
            f"Dataset not found: {path}",
            suggestion="Check that the path exists and you have read permissions.",
            details={"path": path},
        )


class EmptyDatasetError(DatasetError):
    """Dataset contains no valid images."""
    
    def __init__(self, path: str, extensions_checked: list[str] | None = None) -> None:
        exts = extensions_checked or ["jpg", "jpeg", "png", "bmp", "tiff"]
        super().__init__(
            f"No images found in: {path}",
            suggestion=f"Ensure the directory contains images with extensions: {', '.join(exts)}",
            details={"path": path, "extensions_checked": exts},
        )


class InvalidSplitRatioError(DatasetError):
    """Split ratios are invalid (don't sum to 1.0 or contain invalid values)."""
    
    def __init__(self, train: float, val: float, test: float) -> None:
        total = train + val + test
        super().__init__(
            f"Invalid split ratios: train={train}, val={val}, test={test} (sum={total:.2f})",
            suggestion="Split ratios must be between 0 and 1, and sum to 1.0",
            details={"train": train, "val": val, "test": test, "total": total},
        )


class PolicyError(AugmentAIError):
    """
    Errors related to augmentation policies.
    
    Raised during:
    - Policy validation (forbidden transforms, invalid parameters)
    - Policy parsing (malformed YAML/JSON)
    - Policy enforcement (domain constraint violations)
    """
    pass


class PolicyValidationError(PolicyError):
    """Policy violates domain constraints."""
    
    def __init__(
        self,
        domain: str,
        violations: list[str],
    ) -> None:
        msg = f"Policy violates {domain} domain constraints:\n"
        msg += "\n".join(f"  • {v}" for v in violations)
        super().__init__(
            msg,
            suggestion="Remove forbidden transforms or adjust parameters to within allowed ranges.",
            details={"domain": domain, "violations": violations},
        )


class PolicyParseError(PolicyError):
    """Failed to parse policy file."""
    
    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            f"Failed to parse policy file: {path}",
            suggestion="Ensure the file is valid YAML or JSON format.",
            details={"path": path, "reason": reason},
        )


class InvalidTransformError(PolicyError):
    """Transform name is not recognized."""
    
    def __init__(self, transform_name: str, similar: list[str] | None = None) -> None:
        msg = f"Unknown transform: '{transform_name}'"
        suggestion = "Check the transform name spelling."
        if similar:
            suggestion = f"Did you mean: {', '.join(similar)}?"
        super().__init__(
            msg,
            suggestion=suggestion,
            details={"transform_name": transform_name, "similar": similar or []},
        )


class LLMError(AugmentAIError):
    """
    Errors related to LLM operations.
    
    Raised during:
    - LLM connection (API key issues, network errors)
    - LLM response parsing (malformed responses)
    - LLM timeout (request took too long)
    """
    pass


class LLMConnectionError(LLMError):
    """Failed to connect to LLM provider."""
    
    def __init__(self, provider: str, reason: str) -> None:
        suggestions = {
            "openai": "Check your OPENAI_API_KEY environment variable.",
            "ollama": "Ensure Ollama is running: ollama serve",
            "lmstudio": "Ensure LM Studio is running with local server enabled.",
        }
        super().__init__(
            f"Failed to connect to {provider}: {reason}",
            suggestion=suggestions.get(provider.lower(), "Check your connection settings."),
            details={"provider": provider, "reason": reason},
        )


class LLMResponseError(LLMError):
    """LLM returned an unparseable response."""
    
    def __init__(self, reason: str, raw_response: str | None = None) -> None:
        super().__init__(
            f"Failed to parse LLM response: {reason}",
            suggestion="Try again or use a different model. If the issue persists, file a bug report.",
            details={"reason": reason, "raw_response": raw_response[:500] if raw_response else None},
        )


class LLMUnavailableError(LLMError):
    """LLM provider is not available (no API key, service down)."""
    
    def __init__(self, provider: str) -> None:
        super().__init__(
            f"LLM provider '{provider}' is not available.",
            suggestion=(
                "Set up an LLM provider:\n"
                "  • OpenAI: export OPENAI_API_KEY='your-key'\n"
                "  • Ollama: ollama pull llama3.2 && ollama serve\n"
                "  • Or use --no-llm to skip LLM-powered features"
            ),
            details={"provider": provider},
        )


class ConfigurationError(AugmentAIError):
    """
    Errors related to configuration.
    
    Raised during:
    - Config file parsing
    - Environment variable issues
    - Invalid settings
    """
    pass


class MissingAPIKeyError(ConfigurationError):
    """Required API key is missing."""
    
    def __init__(self, key_name: str) -> None:
        super().__init__(
            f"Missing required API key: {key_name}",
            suggestion=f"Set the {key_name} environment variable or add it to .env file.",
            details={"key_name": key_name},
        )


class ExportError(AugmentAIError):
    """
    Errors during export operations.
    
    Raised during:
    - Script generation
    - Folder structure creation
    - File writing
    """
    pass


class OutputDirectoryError(ExportError):
    """Cannot write to output directory."""
    
    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            f"Cannot write to output directory: {path}",
            suggestion="Check that you have write permissions or use a different output path.",
            details={"path": path, "reason": reason},
        )

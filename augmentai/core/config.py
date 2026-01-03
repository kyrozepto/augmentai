"""
Configuration management for AugmentAI.

Handles settings for LLM providers, backends, and application behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"



class AugmentationBackend(str, Enum):
    """Supported augmentation backends."""
    
    ALBUMENTATIONS = "albumentations"
    KORNIA = "kornia"
    TORCHVISION = "torchvision"


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4o-mini"  # Budget-friendly default
    api_key: str | None = None
    base_url: str | None = None  # For Ollama/LM Studio
    temperature: float = 0.7
    max_tokens: int = 2048
    
    def __post_init__(self) -> None:
        """Load API key from environment if not provided."""
        if self.api_key is None:
            if self.provider == LLMProvider.OPENAI:
                self.api_key = os.environ.get("OPENAI_API_KEY")
            elif self.provider == LLMProvider.GEMINI:
                self.api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            # Ollama and LM Studio don't require API keys
        
        # Set default base URLs for local providers
        if self.base_url is None:
            if self.provider == LLMProvider.OLLAMA:
                self.base_url = "http://localhost:11434/v1"
            elif self.provider == LLMProvider.LMSTUDIO:
                self.base_url = "http://localhost:1234/v1"
            elif self.provider == LLMProvider.GEMINI:
                self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"



@dataclass
class AugmentAIConfig:
    """
    Main configuration for AugmentAI.
    
    Attributes:
        llm: LLM provider configuration
        backend: Default augmentation backend
        domains_dir: Directory containing custom domain YAML files
        output_dir: Default output directory for exported policies
        verbose: Enable verbose logging
    """
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    backend: AugmentationBackend = AugmentationBackend.ALBUMENTATIONS
    domains_dir: Path | None = None
    output_dir: Path = field(default_factory=lambda: Path.cwd())
    verbose: bool = False
    
    @classmethod
    def from_yaml(cls, path: Path | str) -> AugmentAIConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()
        
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AugmentAIConfig:
        """Create configuration from a dictionary."""
        llm_data = data.get("llm", {})
        llm_config = LLMConfig(
            provider=LLMProvider(llm_data.get("provider", "openai")),
            model=llm_data.get("model", "gpt-4o-mini"),
            api_key=llm_data.get("api_key"),
            base_url=llm_data.get("base_url"),
            temperature=llm_data.get("temperature", 0.7),
            max_tokens=llm_data.get("max_tokens", 2048),
        )
        
        backend = data.get("backend", "albumentations")
        if isinstance(backend, str):
            backend = AugmentationBackend(backend)
        
        domains_dir = data.get("domains_dir")
        if domains_dir:
            domains_dir = Path(domains_dir)
        
        output_dir = data.get("output_dir", ".")
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        
        return cls(
            llm=llm_config,
            backend=backend,
            domains_dir=domains_dir,
            output_dir=output_dir,
            verbose=data.get("verbose", False),
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "llm": {
                "provider": self.llm.provider.value,
                "model": self.llm.model,
                "base_url": self.llm.base_url,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "backend": self.backend.value,
            "domains_dir": str(self.domains_dir) if self.domains_dir else None,
            "output_dir": str(self.output_dir),
            "verbose": self.verbose,
        }
    
    def to_yaml(self) -> str:
        """Export configuration to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    @classmethod
    def default_config_path(cls) -> Path:
        """Get the default configuration file path."""
        # Check for config in current directory first
        local_config = Path.cwd() / "augmentai.yaml"
        if local_config.exists():
            return local_config
        
        # Then check home directory
        home_config = Path.home() / ".augmentai" / "config.yaml"
        return home_config
    
    @classmethod
    def load_default(cls) -> AugmentAIConfig:
        """Load configuration from default location."""
        config_path = cls.default_config_path()
        return cls.from_yaml(config_path)

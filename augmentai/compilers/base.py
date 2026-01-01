"""
Base compiler interface for augmentation backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from augmentai.core.policy import Policy


@dataclass
class CompileResult:
    """Result of compiling a policy to a specific backend."""
    
    success: bool
    backend: str
    code: str = ""  # Generated Python code
    config: str = ""  # Generated config (YAML/JSON)
    pipeline: Any = None  # The actual transform pipeline object
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def save_code(self, path: Path | str) -> None:
        """Save the generated code to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.code)
    
    def save_config(self, path: Path | str) -> None:
        """Save the generated config to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.config)


class BaseCompiler(ABC):
    """
    Abstract base class for policy compilers.
    
    Compilers take a Policy and produce executable augmentation
    pipelines for a specific backend (Albumentations, Kornia, etc.).
    """
    
    backend_name: str
    
    @abstractmethod
    def compile(self, policy: Policy) -> CompileResult:
        """
        Compile a policy to this backend.
        
        Args:
            policy: The policy to compile
            
        Returns:
            CompileResult with the generated code/pipeline
        """
        pass
    
    @abstractmethod
    def generate_code(self, policy: Policy) -> str:
        """
        Generate Python code for the policy.
        
        Args:
            policy: The policy to generate code for
            
        Returns:
            Python code as a string
        """
        pass
    
    @abstractmethod
    def generate_config(self, policy: Policy) -> str:
        """
        Generate a configuration file for the policy.
        
        Args:
            policy: The policy to generate config for
            
        Returns:
            YAML or JSON configuration as a string
        """
        pass
    
    @abstractmethod
    def validate_backend_available(self) -> tuple[bool, str]:
        """
        Check if the backend library is available.
        
        Returns:
            Tuple of (is_available, message)
        """
        pass

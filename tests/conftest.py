"""
Pytest configuration and shared fixtures for AugmentAI tests.
"""

import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from augmentai.core.policy import Policy, Transform
from augmentai.core.config import LLMConfig, LLMProvider
from augmentai.llm.client import LLMClient, LLMResponse, Message, MessageRole


# ============================================================================
# Dataset Fixtures
# ============================================================================

@pytest.fixture
def temp_dataset(tmp_path: Path) -> Path:
    """Create a temporary dataset with class folders and sample images."""
    # Create class folders
    for class_name in ["cat", "dog"]:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        
        # Create fake images (just files with image extensions)
        for i in range(5):
            (class_dir / f"img_{i}.jpg").write_bytes(b"fake image data")
    
    return tmp_path


@pytest.fixture
def temp_presplit_dataset(tmp_path: Path) -> Path:
    """Create a temporary pre-split dataset."""
    for split in ["train", "val", "test"]:
        split_dir = tmp_path / split
        split_dir.mkdir()
        
        for class_name in ["cat", "dog"]:
            class_dir = split_dir / class_name
            class_dir.mkdir()
            
            count = 6 if split == "train" else 2
            for i in range(count):
                (class_dir / f"img_{i}.jpg").write_bytes(b"fake")
    
    return tmp_path


@pytest.fixture
def empty_dataset(tmp_path: Path) -> Path:
    """Create an empty dataset directory."""
    return tmp_path


# ============================================================================
# Policy Fixtures
# ============================================================================

@pytest.fixture
def sample_natural_policy() -> Policy:
    """Create a sample policy for natural images."""
    return Policy(
        name="test_natural_policy",
        domain="natural",
        transforms=[
            Transform("HorizontalFlip", 0.5),
            Transform("VerticalFlip", 0.2),
            Transform("Rotate", 0.5, parameters={"limit": 30}),
            Transform("RandomBrightnessContrast", 0.5),
        ]
    )


@pytest.fixture
def sample_medical_policy() -> Policy:
    """Create a sample policy for medical images."""
    return Policy(
        name="test_medical_policy",
        domain="medical",
        transforms=[
            Transform("HorizontalFlip", 0.5),
            Transform("Rotate", 0.3, parameters={"limit": 15}),
            Transform("GaussNoise", 0.2, parameters={"var_limit": (5, 20)}),
        ]
    )


@pytest.fixture
def policy_with_forbidden_transforms() -> Policy:
    """Create a policy with transforms forbidden in medical domain."""
    return Policy(
        name="test_forbidden_policy",
        domain="medical",
        transforms=[
            Transform("HorizontalFlip", 0.5),
            Transform("ElasticTransform", 0.5),  # Forbidden in medical
            Transform("ColorJitter", 0.3),        # Forbidden in medical
        ]
    )


# ============================================================================
# LLM Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_response() -> LLMResponse:
    """Create a mock LLM response with a valid policy JSON."""
    return LLMResponse(
        content='''{
            "reasoning": "Conservative policy for general images",
            "policy_name": "mock_policy",
            "transforms": [
                {"name": "HorizontalFlip", "probability": 0.5},
                {"name": "Rotate", "probability": 0.3, "parameters": {"limit": 15}}
            ],
            "warnings": [],
            "alternatives": []
        }''',
        finish_reason="stop",
        model="mock-model",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )


@pytest.fixture
def mock_llm_client(mock_llm_response: LLMResponse) -> Generator[MagicMock, None, None]:
    """Create a mock LLM client that returns predefined responses."""
    with patch("augmentai.llm.client.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.chat.return_value = mock_llm_response
        instance.test_connection.return_value = True
        instance.config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="mock-model",
        )
        yield instance


@pytest.fixture
def unavailable_llm_client() -> Generator[MagicMock, None, None]:
    """Create a mock LLM client that simulates unavailability."""
    with patch("augmentai.llm.client.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.test_connection.return_value = False
        instance.chat.side_effect = RuntimeError("Connection failed")
        yield instance


# ============================================================================
# CLI Fixtures (for integration tests)
# ============================================================================

@pytest.fixture
def cli_runner():
    """Create a Typer CLI test runner."""
    from typer.testing import CliRunner
    return CliRunner()


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output = tmp_path / "output"
    output.mkdir()
    return output

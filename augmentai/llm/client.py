"""
LLM client abstraction supporting multiple providers.

Supports:
- OpenAI API (GPT-4o-mini, GPT-4o, etc.)
- Ollama (local models)
- LM Studio (local models with OpenAI-compatible API)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator

from augmentai.core.config import LLMConfig, LLMProvider


class MessageRole(str, Enum):
    """Role of a message in the conversation."""
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A message in the conversation history."""
    
    role: MessageRole
    content: str
    
    def to_dict(self) -> dict[str, str]:
        """Convert to OpenAI-compatible message format."""
        return {"role": self.role.value, "content": self.content}


@dataclass
class LLMResponse:
    """Response from the LLM."""
    
    content: str
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    
    @property
    def is_complete(self) -> bool:
        """Check if the response completed normally."""
        return self.finish_reason == "stop"


class LLMClient:
    """
    Unified client for LLM providers.
    
    Provides a consistent interface for OpenAI, Ollama, and LM Studio.
    """
    
    def __init__(self, config: LLMConfig | None = None) -> None:
        """
        Initialize the LLM client.
        
        Args:
            config: LLM configuration. Uses default config if not provided.
        """
        self.config = config or LLMConfig()
        self._client: Any = None
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize the underlying client based on provider."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )
        
        if self.config.provider == LLMProvider.OPENAI:
            if not self.config.api_key:
                raise ValueError(
                    "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                    "or provide api_key in LLMConfig."
                )
            self._client = OpenAI(api_key=self.config.api_key)
        
        elif self.config.provider == LLMProvider.OLLAMA:
            # Ollama uses OpenAI-compatible API
            self._client = OpenAI(
                base_url=self.config.base_url or "http://localhost:11434/v1",
                api_key="ollama",  # Ollama doesn't need a real key
            )
        
        elif self.config.provider == LLMProvider.LMSTUDIO:
            # LM Studio also uses OpenAI-compatible API
            self._client = OpenAI(
                base_url=self.config.base_url or "http://localhost:1234/v1",
                api_key="lm-studio",  # LM Studio doesn't need a real key
            )
    
    def chat(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Send a chat completion request.
        
        Args:
            messages: List of conversation messages
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_mode: Request JSON output format
            
        Returns:
            LLMResponse with the model's reply
        """
        request_params: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        
        if json_mode:
            request_params["response_format"] = {"type": "json_object"}
        
        try:
            response = self._client.chat.completions.create(**request_params)
            
            choice = response.choices[0]
            return LLMResponse(
                content=choice.message.content or "",
                finish_reason=choice.finish_reason or "stop",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                model=response.model,
            )
        
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(
                f"LLM request failed ({self.config.provider.value}): {e}"
            ) from e
    
    def chat_stream(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream a chat completion request.
        
        Args:
            messages: List of conversation messages
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Yields:
            Chunks of the response content as they arrive
        """
        request_params: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": True,
        }
        
        try:
            stream = self._client.chat.completions.create(**request_params)
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            raise RuntimeError(
                f"LLM stream request failed ({self.config.provider.value}): {e}"
            ) from e
    
    def test_connection(self) -> bool:
        """
        Test if the LLM provider is accessible.
        
        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            response = self.chat(
                [Message(MessageRole.USER, "Say 'ok' and nothing else.")],
                max_tokens=10,
            )
            return response.is_complete
        except Exception:
            return False
    
    @classmethod
    def from_provider(
        cls,
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> LLMClient:
        """
        Create a client for a specific provider.
        
        Args:
            provider: Provider name ("openai", "ollama", "lmstudio")
            model: Model name (defaults based on provider)
            api_key: API key (only needed for OpenAI)
            base_url: Base URL for local providers
            
        Returns:
            Configured LLMClient
        """
        provider_enum = LLMProvider(provider.lower())
        
        # Set default models based on provider
        if model is None:
            if provider_enum == LLMProvider.OPENAI:
                model = "gpt-4o-mini"
            elif provider_enum == LLMProvider.OLLAMA:
                model = "llama3.2"
            elif provider_enum == LLMProvider.LMSTUDIO:
                model = "local-model"
        
        config = LLMConfig(
            provider=provider_enum,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        
        return cls(config)

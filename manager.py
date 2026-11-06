import os
from collections.abc import AsyncIterator
from enum import Enum

from anthropic_client import AnthropicClient
from base import BaseLLMClient
from openai_client import OpenAIClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


DEFAULT_MODELS = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-haiku-5-5",
}

API_KEY_VARS = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
}


class AsyncLLMManager:
    """Loads OpenAI or Anthropic from the `provider` argument, or from the
    LLM_PROVIDER environment variable when it is omitted."""

    def __init__(self, provider: Provider | str | None = None):
        raw = provider or os.getenv("LLM_PROVIDER")
        if not raw:
            raise ValueError("No provider given and LLM_PROVIDER is not set")
        try:
            self.provider = Provider(str(getattr(raw, "value", raw)).lower())
        except ValueError:
            valid = ", ".join(p.value for p in Provider)
            raise ValueError(f"Unknown provider '{raw}'. Valid options: {valid}") from None

        key_var = API_KEY_VARS[self.provider]
        api_key = os.getenv(key_var)
        if not api_key:
            raise ValueError(f"{key_var} is not set")

        self.client: BaseLLMClient = (
            OpenAIClient(api_key) if self.provider is Provider.OPENAI else AnthropicClient(api_key)
        )

    def default_config(self, **overrides) -> ModelConfig:
        return ModelConfig(model=DEFAULT_MODELS[self.provider], **overrides)

    async def generate(self, messages: list[ChatMessage], config: ModelConfig | None = None) -> ModelResponse:
        return await self.client.generate(messages, config or self.default_config())

    def generate_stream(self, messages: list[ChatMessage], config: ModelConfig | None = None) -> AsyncIterator[str]:
        return self.client.generate_stream(messages, config or self.default_config())

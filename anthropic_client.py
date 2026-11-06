from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from base import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class AnthropicClient(BaseLLMClient):
    """Note: anthropic SDK 2.x no longer accepts `temperature`, so
    ModelConfig.temperature is ignored by this client."""

    def __init__(self, api_key: str | None = None):
        self.client = AsyncAnthropic(api_key=api_key)

    @staticmethod
    def _build_kwargs(messages: list[ChatMessage], config: ModelConfig) -> dict[str, Any]:
        # Anthropic takes system prompts as a separate parameter.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        kwargs: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
                if m.role != "system"
            ],
        }
        if system:
            kwargs["system"] = system
        return kwargs

    async def _generate(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        message = await self.client.messages.create(**self._build_kwargs(messages, config))
        text = "".join(block.text for block in message.content if block.type == "text")
        return ModelResponse(
            response=text,
            output_tokens=message.usage.output_tokens,
            model_name=config.model,
        )

    async def _stream(self, messages: list[ChatMessage], config: ModelConfig) -> AsyncIterator[str]:
        async with self.client.messages.stream(**self._build_kwargs(messages, config)) as stream:
            async for text in stream.text_stream:
                yield text

    def is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, (anthropic.RateLimitError, anthropic.APIConnectionError)):
            return True
        return isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500

    def describe_error(self, exc: Exception) -> str:
        if isinstance(exc, anthropic.AuthenticationError):
            return "Anthropic authentication failed: check ANTHROPIC_API_KEY"
        if isinstance(exc, anthropic.RateLimitError):
            return "Anthropic rate limit or quota exceeded"
        if isinstance(exc, anthropic.APIConnectionError):  # includes timeouts
            return "Network error while contacting Anthropic"
        if isinstance(exc, anthropic.APIStatusError):
            return f"Anthropic API error ({exc.status_code}): {exc.message}"
        return f"Unexpected error: {exc}"

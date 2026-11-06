from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

from base import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str | None = None):
        self.client = AsyncOpenAI(api_key=api_key)

    @staticmethod
    def _to_api(messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def _generate(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        completion = await self.client.chat.completions.create(
            model=config.model,
            messages=self._to_api(messages),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        return ModelResponse(
            response=completion.choices[0].message.content or "",
            output_tokens=completion.usage.completion_tokens if completion.usage else 0,
            model_name=config.model,
        )

    async def _stream(self, messages: list[ChatMessage], config: ModelConfig) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=config.model,
            messages=self._to_api(messages),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError)):
            return True
        return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500

    def describe_error(self, exc: Exception) -> str:
        if isinstance(exc, openai.AuthenticationError):
            return "OpenAI authentication failed: check OPENAI_API_KEY"
        if isinstance(exc, openai.RateLimitError):
            return "OpenAI rate limit or quota exceeded"
        if isinstance(exc, openai.APIConnectionError):  # includes timeouts
            return "Network error while contacting OpenAI"
        if isinstance(exc, openai.APIStatusError):
            return f"OpenAI API error ({exc.status_code}): {exc.message}"
        return f"Unexpected error: {exc}"

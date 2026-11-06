import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from schemas import ChatMessage, ModelConfig, ModelResponse

MAX_ATTEMPTS = 3
BASE_DELAY_SECONDS = 1.0


class BaseLLMClient(ABC):
    """Common interface. Subclasses implement the provider-specific hooks;
    generate() and generate_stream() add retries and controlled errors."""

    # ---- provider-specific hooks -------------------------------------
    @abstractmethod
    async def _generate(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        ...

    @abstractmethod
    def _stream(self, messages: list[ChatMessage], config: ModelConfig) -> AsyncIterator[str]:
        ...

    @abstractmethod
    def is_retryable(self, exc: Exception) -> bool:
        """True for transient errors (rate limit, network, 5xx)."""

    @abstractmethod
    def describe_error(self, exc: Exception) -> str:
        """Short, human-readable message for an SDK exception."""

    # ---- public API ---------------------------------------------------
    async def generate(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        """Never raises: failures come back in ModelResponse.error."""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return await self._generate(messages, config)
            except Exception as exc:
                if self.is_retryable(exc) and attempt < MAX_ATTEMPTS:
                    await asyncio.sleep(BASE_DELAY_SECONDS * 2 ** (attempt - 1))
                    continue
                return ModelResponse(error=self.describe_error(exc), model_name=config.model)
        raise AssertionError("unreachable")  # pragma: no cover

    async def generate_stream(self, messages: list[ChatMessage], config: ModelConfig) -> AsyncIterator[str]:
        """Yields text chunks. On failure it yields one final '[error] ...'
        chunk instead of raising. No retry: chunks may already be emitted."""
        try:
            async for chunk in self._stream(messages, config):
                yield chunk
        except Exception as exc:
            yield f"[error] {self.describe_error(exc)}"

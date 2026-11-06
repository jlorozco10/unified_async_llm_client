import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from manager import AsyncLLMManager  # noqa: E402  (after load_dotenv so env vars are set)
from schemas import ChatMessage  # noqa: E402

QUESTION = "What is entropy?"


async def run(provider: str | None) -> None:
    manager = AsyncLLMManager(provider)
    messages = [ChatMessage(role="user", content=QUESTION)]
    config = manager.default_config(temperature=0.0, max_tokens=500)
    print(f"=== {manager.provider.value} / {config.model} ===")

    print("\n--- Standard mode ---")
    result = await manager.generate(messages, config)
    if result.ok:
        print(result.response)
        print(f"\n[output_tokens={result.output_tokens}]")
    else:
        print(f"Error: {result.error}")

    print("\n--- Streaming mode ---")
    async for chunk in manager.generate_stream(messages, config):
        print(chunk, end="", flush=True)
    print()


async def main() -> None:
    provider = sys.argv[1] if len(sys.argv) > 1 else None  # falls back to LLM_PROVIDER
    try:
        await run(provider)
    except ValueError as exc:  # bad/missing configuration
        print(f"Configuration error: {exc}")


if __name__ == "__main__":
    asyncio.run(main())

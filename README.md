# Unified Async LLM Client

An asynchronous Python client with one interface for OpenAI and Anthropic.
It supports standard and streaming generation, Pydantic validation, and controlled errors.

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | Yes, unless passed as an argument | `openai` or `anthropic` |
| `OPENAI_API_KEY` | If using OpenAI | OpenAI API key |
| `ANTHROPIC_API_KEY` | If using Anthropic | Anthropic API key |

## Run the test script

`main.py` asks "What is entropy?" in standard mode and then in streaming mode.

```bash
python main.py              # uses LLM_PROVIDER from .env
python main.py anthropic    # overrides the provider
python main.py openai
```

Unit tests: `pytest`

## Usage

```python
from manager import AsyncLLMManager
from schemas import ChatMessage

manager = AsyncLLMManager("openai")          # or None to read LLM_PROVIDER
messages = [ChatMessage(role="user", content="Hi")]
config = manager.default_config(temperature=0.7, max_tokens=300)

result = await manager.generate(messages, config)   # ModelResponse
async for chunk in manager.generate_stream(messages, config):
    print(chunk, end="")
```

## Structure

| File | Purpose |
|---|---|
| `schemas.py` | Pydantic models: `ChatMessage`, `ModelConfig` (temperature 0–2, `max_tokens` > 0), `ModelResponse` |
| `base.py` | `BaseLLMClient` abstract class, retries and error handling |
| `openai_client.py` / `anthropic_client.py` | Provider implementations using `AsyncOpenAI` / `AsyncAnthropic` |
| `manager.py` | `AsyncLLMManager`: picks the provider from config |
| `main.py` | Validation script |

## Error handling

- `generate()` never raises. Failures come back as `ModelResponse(error=...)`, and `result.ok` is `False`.
- Rate-limit, network and 5xx errors are retried up to 3 times with exponential backoff (1s, 2s).
- `generate_stream()` yields a final `[error] ...` chunk instead of raising. It does not retry,
  because chunks may already have been emitted.
- A missing API key or unknown provider raises `ValueError` when `AsyncLLMManager` is created.

## Notes

- The Anthropic SDK 2.x no longer accepts `temperature`, so that client ignores `ModelConfig.temperature`.
  It is applied for OpenAI only.
- `system` messages are passed to Anthropic through its separate `system` parameter.

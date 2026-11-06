from typing import Literal
from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content cannot be blank")
        return v


class ModelConfig(BaseModel):
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0)


class ModelResponse(BaseModel):
    response: str | None = Field(default=None)
    error: str | None = Field(default=None)
    output_tokens: int | None = Field(default=0)
    model_name: str

    @property
    def ok(self) -> bool:
        return self.response is not None

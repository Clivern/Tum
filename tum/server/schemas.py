import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    stream: bool = False


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "tum"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


class ChatCompletionChoice(BaseModel):
    index: int
    message: dict[str, Any]
    finish_reason: Literal["stop", "length"] | None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: dict[str, int]


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: dict[str, Any]
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Literal["stop", "length"] | None


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: dict[str, int]


def new_chat_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4()}"


def new_completion_id() -> str:
    return f"cmpl-{uuid.uuid4()}"


def now_unix() -> int:
    return int(time.time())

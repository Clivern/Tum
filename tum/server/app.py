import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from tum.core.config import GenerateConfig, ServerConfig
from tum.server.schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ModelListResponse,
    ModelObject,
    new_chat_completion_id,
    new_completion_id,
    now_unix,
)

if TYPE_CHECKING:
    from tum.core.session import Session


def _to_generate_config(
    request_max_tokens: int | None,
    request_temperature: float | None,
    defaults: GenerateConfig,
) -> GenerateConfig:
    return GenerateConfig(
        max_tokens=request_max_tokens or defaults.max_tokens,
        temperature=(
            request_temperature if request_temperature is not None else defaults.temperature
        ),
    )


def create_app(
    server_config: ServerConfig | None = None,
    *,
    session: "Session | None" = None,
    load_on_startup: bool = True,
) -> FastAPI:
    config = server_config or ServerConfig()
    if session is None:
        from tum.core.session import Session

        session = Session(config.model)
    defaults = GenerateConfig()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if load_on_startup:
            session.load()
        yield

    app = FastAPI(title="tum", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", response_model=ModelListResponse)
    def list_models() -> ModelListResponse:
        created = now_unix()
        return ModelListResponse(
            data=[
                ModelObject(
                    id=config.model.model_id,
                    created=created,
                )
            ]
        )

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        generate_config = _to_generate_config(
            request.max_tokens,
            request.temperature,
            defaults,
        )
        messages = [message.model_dump() for message in request.messages]

        if request.stream:
            return StreamingResponse(
                _stream_chat(session, request.model, messages, generate_config),
                media_type="text/event-stream",
            )

        text = session.chat(messages, generate_config)
        completion_id = new_chat_completion_id()
        created = now_unix()
        prompt_tokens = _estimate_tokens(messages)
        completion_tokens = _estimate_tokens(text)

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message={"role": "assistant", "content": text},
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        generate_config = _to_generate_config(
            request.max_tokens,
            request.temperature,
            defaults,
        )

        if request.stream:
            return StreamingResponse(
                _stream_completion(session, request.model, request.prompt, generate_config),
                media_type="text/event-stream",
            )

        text = session.generate(request.prompt, generate_config)
        completion_id = new_completion_id()
        created = now_unix()
        prompt_tokens = _estimate_tokens(request.prompt)
        completion_tokens = _estimate_tokens(text)

        return CompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                CompletionChoice(
                    index=0,
                    text=text,
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

    return app


def _estimate_tokens(value: str | list[dict[str, str]]) -> int:
    if isinstance(value, str):
        text = value
    else:
        text = " ".join(message["content"] for message in value)
    return max(len(text.split()), 1)


async def _stream_chat(
    session: "Session",
    model: str,
    messages: list[dict[str, str]],
    config: GenerateConfig,
) -> AsyncIterator[str]:
    completion_id = new_chat_completion_id()
    created = now_unix()

    for chunk_text in session.chat_stream(messages, config):
        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta={"content": chunk_text},
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    final_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta={},
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_completion(
    session: "Session",
    model: str,
    prompt: str,
    config: GenerateConfig,
) -> AsyncIterator[str]:
    completion_id = new_completion_id()
    created = now_unix()

    for chunk_text in session.generate_stream(prompt, config):
        payload = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "text": chunk_text, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload)}\n\n"

    final_payload = {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_payload)}\n\n"
    yield "data: [DONE]\n\n"

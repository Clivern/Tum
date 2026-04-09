from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from tum.core.config import GenerateConfig, ModelConfig, ServerConfig
from tum.server.app import create_app


@pytest.fixture
def model_id() -> str:
    return "test-model"


@pytest.fixture
def session() -> MagicMock:
    mock = MagicMock()
    mock.chat.return_value = "Paris"
    mock.generate.return_value = "Paris"
    mock.chat_stream.return_value = iter(["Par", "is"])
    mock.generate_stream.return_value = iter(["Par", "is"])
    return mock


@pytest.fixture
def client(model_id: str, session: MagicMock) -> TestClient:
    app = create_app(
        ServerConfig(model=ModelConfig(model_id=model_id)),
        session=session,
        load_on_startup=False,
    )
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_models(client: TestClient, model_id: str) -> None:
    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == model_id


def test_chat_completions_handles_unhandled_errors(
    model_id: str,
    session: MagicMock,
) -> None:
    session.chat.side_effect = RuntimeError("generation failed")
    app = create_app(
        ServerConfig(model=ModelConfig(model_id=model_id)),
        session=session,
        load_on_startup=False,
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
            },
        )

    assert response.status_code == 500
    assert response.json() == {"error": "generation failed"}


def test_chat_completions(client: TestClient, model_id: str, session: MagicMock) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == model_id
    assert payload["choices"][0]["message"]["content"] == "Paris"
    assert payload["choices"][0]["finish_reason"] == "stop"
    session.chat.assert_called_once()


def test_text_completions(client: TestClient, model_id: str, session: MagicMock) -> None:
    response = client.post(
        "/v1/completions",
        json={
            "model": model_id,
            "prompt": "What is the capital of France?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "text_completion"
    assert payload["choices"][0]["text"] == "Paris"
    session.generate.assert_called_once()


def test_chat_completions_stream(client: TestClient, model_id: str) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "chat.completion.chunk" in body
    assert "[DONE]" in body


def test_generate_config_defaults() -> None:
    config = GenerateConfig()

    assert config.max_tokens == 128
    assert config.temperature == 0.7


def test_server_config_rejects_invalid_port() -> None:
    with pytest.raises(ValueError, match="port must be between"):
        ServerConfig(port=0)

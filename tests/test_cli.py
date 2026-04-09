import pytest
from click.testing import CliRunner

from tum.cli import cli, main, models, serve
from tum.core.models import (
    BIG_MLX_COMMUNITY_MODELS,
    MLX_COMMUNITY_MODELS,
    SMALL_MLX_COMMUNITY_MODELS,
)


def test_cli_help() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "mlx" in result.output
    assert "models" in result.output
    assert "serve" in result.output


def test_mlx_help() -> None:
    result = CliRunner().invoke(cli, ["mlx", "--help"])

    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--prompt" in result.output
    assert "--max-tokens" in result.output


def test_models_help() -> None:
    result = CliRunner().invoke(models, ["--help"])

    assert result.exit_code == 0
    assert "--query" in result.output
    assert "--limit" in result.output
    assert "--offset" in result.output
    assert "--pager" in result.output
    assert "--size" in result.output


def test_models_filters_by_query() -> None:
    result = CliRunner().invoke(models, ["--query", "whisper", "--limit", "2"])

    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert len(lines) == 2
    assert all("whisper" in line.casefold() for line in lines)


def test_models_applies_offset_and_limit() -> None:
    first_result = CliRunner().invoke(models, ["--limit", "1"])
    second_result = CliRunner().invoke(models, ["--offset", "1", "--limit", "1"])

    assert first_result.exit_code == 0
    assert second_result.exit_code == 0
    assert first_result.stdout.splitlines()[0] != second_result.stdout.splitlines()[0]


def test_models_filters_by_small_size() -> None:
    result = CliRunner().invoke(models, ["--size", "small", "--limit", "1"])

    assert result.exit_code == 0
    assert result.stdout.splitlines()[0] == SMALL_MLX_COMMUNITY_MODELS[0]


def test_models_filters_by_big_size() -> None:
    result = CliRunner().invoke(models, ["--size", "big", "--limit", "1"])

    assert result.exit_code == 0
    assert result.stdout.splitlines()[0] == BIG_MLX_COMMUNITY_MODELS[0]


def test_model_catalog_is_split_by_size() -> None:
    assert MLX_COMMUNITY_MODELS == SMALL_MLX_COMMUNITY_MODELS + BIG_MLX_COMMUNITY_MODELS
    assert set(SMALL_MLX_COMMUNITY_MODELS).isdisjoint(BIG_MLX_COMMUNITY_MODELS)


def test_serve_help() -> None:
    result = CliRunner().invoke(serve, ["--help"])

    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output


def test_serve_rejects_invalid_port() -> None:
    result = CliRunner().invoke(serve, ["--port", "0"])

    assert result.exit_code != 0


def test_main_shows_error_without_traceback(monkeypatch, capsys) -> None:
    def boom(*_args, **_kwargs) -> None:
        raise RuntimeError("model failed to load")

    monkeypatch.setattr("tum.cli.cli", boom)

    with pytest.raises(SystemExit) as exc_info:
        main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.err.strip() == "model failed to load"
    assert "Traceback" not in captured.err


def test_serve_uses_error_log_level(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(_app, **kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("tum.cli.uvicorn.run", fake_run)
    monkeypatch.setattr("tum.cli.create_app", lambda _config: "app")

    result = CliRunner().invoke(serve, [])

    assert result.exit_code == 0
    assert captured["log_level"] == "error"

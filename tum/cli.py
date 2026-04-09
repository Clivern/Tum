import sys

import click
import uvicorn

from tum.core.config import (
    DEFAULT_HOST,
    DEFAULT_MODEL_ID,
    DEFAULT_PORT,
    GenerateConfig,
    ModelConfig,
    ServerConfig,
)
from tum.server.app import create_app


DEFAULT_MODEL_LIST_LIMIT = 50


@click.group()
@click.version_option(package_name="tum")
def cli() -> None:
    """Run OpenAI-compatible LLM servers locally."""


@cli.command("mlx")
@click.option(
    "--model",
    default=DEFAULT_MODEL_ID,
    show_default=True,
    help="Hugging Face model id to load.",
)
@click.option(
    "--prompt",
    default="What is the capital of France?",
    show_default=True,
    help="Prompt to send to the model.",
)
@click.option(
    "--max-tokens",
    default=128,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum tokens to generate.",
)
def mlx(model: str, prompt: str, max_tokens: int) -> None:
    """Run a local MLX model."""
    from tum.core.session import Session

    session = Session(ModelConfig(model_id=model))
    generate_config = GenerateConfig(max_tokens=max_tokens, verbose=True)
    response = session.generate(prompt, config=generate_config)
    click.echo(response)


@cli.command("models")
@click.option(
    "-q",
    "--query",
    help="Case-insensitive text to filter model ids.",
)
@click.option(
    "--limit",
    default=DEFAULT_MODEL_LIST_LIMIT,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum number of matching models to show.",
)
@click.option(
    "--offset",
    default=0,
    show_default=True,
    type=click.IntRange(min=0),
    help="Number of matching models to skip.",
)
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    help="Show every matching model.",
)
@click.option(
    "--pager/--no-pager",
    default=False,
    show_default=True,
    help="Open results in the terminal pager.",
)
@click.option(
    "--size",
    type=click.Choice(("all", "small", "big"), case_sensitive=False),
    default="all",
    show_default=True,
    help="Filter by inferred model size.",
)
def models(
    query: str | None,
    limit: int,
    offset: int,
    show_all: bool,
    pager: bool,
    size: str,
) -> None:
    """Browse bundled mlx-community model ids."""
    from tum.core.models import (
        BIG_MLX_COMMUNITY_MODELS,
        MLX_COMMUNITY_MODELS,
        SMALL_MLX_COMMUNITY_MODELS,
    )

    model_catalog = {
        "all": MLX_COMMUNITY_MODELS,
        "small": SMALL_MLX_COMMUNITY_MODELS,
        "big": BIG_MLX_COMMUNITY_MODELS,
    }[size.casefold()]

    normalized_query = query.casefold() if query else None
    matching_models = [
        model
        for model in model_catalog
        if normalized_query is None or normalized_query in model.casefold()
    ]

    total = len(matching_models)
    selected_models = (
        matching_models[offset:] if show_all else matching_models[offset : offset + limit]
    )

    if not selected_models:
        click.echo("No matching models found.", err=True)
        return

    output = "\n".join(selected_models) + "\n"
    if pager:
        click.echo_via_pager(output)
    else:
        click.echo(output, nl=False)

    start = offset + 1
    end = offset + len(selected_models)
    click.echo(f"Showing {start}-{end} of {total} matching models.", err=True)


@cli.command("serve")
@click.option(
    "--model",
    default=DEFAULT_MODEL_ID,
    show_default=True,
    help="Hugging Face model id to load.",
)
@click.option(
    "--host",
    default=DEFAULT_HOST,
    show_default=True,
    help="Host to bind the server to.",
)
@click.option(
    "--port",
    default=DEFAULT_PORT,
    show_default=True,
    type=click.IntRange(1, 65535),
    help="Port to bind the server to.",
)
def serve(model: str, host: str, port: int) -> None:
    """Run an OpenAI-compatible MLX server with FastAPI."""
    server_config = ServerConfig(
        host=host,
        port=port,
        model=ModelConfig(model_id=model),
    )
    app = create_app(server_config)
    uvicorn.run(
        app,
        host=server_config.host,
        port=server_config.port,
        log_level="error",
    )


def main() -> None:
    try:
        cli(prog_name="tum")
    except click.ClickException as exc:
        exc.show()
        sys.exit(exc.exit_code)
    except Exception as exc:
        click.echo(str(exc) or exc.__class__.__name__, err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

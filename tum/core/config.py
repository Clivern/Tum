from dataclasses import dataclass, field

DEFAULT_MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for loading an MLX model."""

    model_id: str = DEFAULT_MODEL_ID


@dataclass(frozen=True)
class GenerateConfig:
    """Configuration for text generation."""

    max_tokens: int = 128
    temperature: float = 0.7
    verbose: bool = False


@dataclass(frozen=True)
class ServerConfig:
    """Configuration for the OpenAI-compatible HTTP server."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")

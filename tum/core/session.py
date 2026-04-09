from collections.abc import Iterator

from tum.backend.mlx import LoadedModel, MLXBackend
from tum.core.config import GenerateConfig, ModelConfig


class Session:
    """Manages the lifecycle of a loaded MLX model and runs generation."""

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        backend: MLXBackend | None = None,
    ) -> None:
        self.model_config = model_config or ModelConfig()
        self._backend = backend or MLXBackend()
        self._loaded: LoadedModel | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    def load(self) -> None:
        self._loaded = self._backend.load(self.model_config.model_id)

    def generate(
        self,
        prompt: str,
        config: GenerateConfig | None = None,
    ) -> str:
        loaded = self._require_loaded()
        return self._backend.generate(loaded, prompt, config or GenerateConfig())

    def chat(
        self,
        messages: list[dict[str, str]],
        config: GenerateConfig | None = None,
    ) -> str:
        loaded = self._require_loaded()
        prompt = self._backend.format_chat(loaded, messages)
        return self._backend.generate(loaded, prompt, config or GenerateConfig())

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        config: GenerateConfig | None = None,
    ) -> Iterator[str]:
        loaded = self._require_loaded()
        prompt = self._backend.format_chat(loaded, messages)
        yield from self._backend.stream_generate(loaded, prompt, config or GenerateConfig())

    def generate_stream(
        self,
        prompt: str,
        config: GenerateConfig | None = None,
    ) -> Iterator[str]:
        loaded = self._require_loaded()
        yield from self._backend.stream_generate(loaded, prompt, config or GenerateConfig())

    def _require_loaded(self) -> LoadedModel:
        if self._loaded is None:
            self.load()
        return self._loaded

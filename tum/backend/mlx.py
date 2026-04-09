from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from mlx_lm import generate, load, stream_generate
from mlx_lm.sample_utils import make_sampler

from tum.core.config import GenerateConfig


@dataclass
class LoadedModel:
    """A model and tokenizer pair loaded into memory."""

    model: Any
    tokenizer: Any


class MLXBackend:
    """MLX implementation for model loading and generation."""

    def load(self, model_id: str) -> LoadedModel:
        model, tokenizer = load(model_id)
        return LoadedModel(model=model, tokenizer=tokenizer)

    def format_chat(self, loaded: LoadedModel, messages: list[dict[str, str]]) -> str:
        tokenizer = loaded.tokenizer
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

        prompt = ""
        for message in messages:
            role = message["role"].upper()
            prompt += f"{role}: {message['content']}\n"
        prompt += "ASSISTANT:"
        return prompt

    def _sampler(self, config: GenerateConfig):
        return make_sampler(config.temperature)

    def generate(
        self,
        loaded: LoadedModel,
        prompt: str,
        config: GenerateConfig,
    ) -> str:
        return generate(
            loaded.model,
            loaded.tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
            verbose=config.verbose,
            sampler=self._sampler(config),
        )

    def stream_generate(
        self,
        loaded: LoadedModel,
        prompt: str,
        config: GenerateConfig,
    ) -> Iterator[str]:
        for chunk in stream_generate(
            loaded.model,
            loaded.tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
            sampler=self._sampler(config),
        ):
            if chunk.text:
                yield chunk.text

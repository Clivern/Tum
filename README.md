## Tum

Run OpenAI-compatible LLM servers locally with `Tum` and `MLX`.

### Requirements

- Python 3.10+
- macOS with Apple Silicon for MLX model execution
- `uv` for local development commands

### Install

Install from PyPI:

```bash
pip install tum
```

Then run:

```bash
tum --help
```

### Browse Models

List bundled model IDs:

```bash
tum models
```

Filter by name:

```bash
tum models --query Qwen --limit 10
```

Show small or big model catalogs:

```bash
tum models --size small --query Qwen
tum models --size big --query Llama
```

Use `--all` to print every match, or `--pager` for long output.

### Run One Prompt

```bash
tum mlx \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "What is the capital of France?" \
  --max-tokens 128
```

### Serve an OpenAI-Compatible API

```bash
tum serve \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --host 127.0.0.1 \
  --port 8080
```

Then call the chat completions endpoint:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

Stream tokens as server-sent events:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Say hello"}],
    "stream": true
  }'
```

Supported endpoints include:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /chat/completions`
- `POST /v1/completions`

### Development

Run tests:

```bash
uv run pytest
```

Run lint:

```bash
uv run ruff check tum tests
```


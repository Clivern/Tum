#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["openai>=1.0.0"]
# ///

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    messages=[{"role": "user", "content": "How to split a string in Python?"}],
)
print(response.choices[0].message.content)

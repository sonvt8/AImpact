from __future__ import annotations

import json

import httpx

from api.providers import ProviderRegistry


class LLMError(RuntimeError):
    pass


class OpenAICompatibleLLM:
    def __init__(self, registry: ProviderRegistry, timeout: int):
        self.registry = registry
        self.timeout = timeout

    async def stream(self, prompt: str):
        provider = self.registry.active()
        url = f"{provider['base_url'].rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.registry.api_key(provider)}",
            "Content-Type": "application/json",
        }
        body = {
            "model": provider["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "stream": True,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, headers=headers, json=body) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        payload = json.loads(data)
                        token = payload.get("choices", [{}])[0].get("delta", {}).get("content")
                        if token:
                            yield token
        except (httpx.HTTPError, ValueError, KeyError) as error:
            raise LLMError(f"Active provider request failed: {error}") from error

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

import httpx


async def fetch_models(url: str, headers: dict, timeout: int) -> object:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


class ProviderRegistry:
    def __init__(self, state_path: Path, default_threshold: float, timeout: int):
        self.profiles = json.loads(Path(__file__).with_name("providers.json").read_text("utf-8"))
        self.by_id = {profile["id"]: profile for profile in self.profiles}
        self.state_path = state_path
        self.default_threshold = default_threshold
        self.timeout = timeout
        self.lock = Lock()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self._write(
                {
                    "active": self.profiles[0]["id"],
                    "models": {profile["id"]: profile["model"] for profile in self.profiles},
                    "threshold": default_threshold,
                }
            )

    def _read(self) -> dict:
        with self.lock:
            return json.loads(self.state_path.read_text("utf-8"))

    def _write(self, state: dict) -> None:
        with self.lock:
            temporary = self.state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False), "utf-8")
            temporary.replace(self.state_path)

    def _provider(self, provider_id: str) -> dict:
        if provider_id not in self.by_id:
            raise KeyError(provider_id)
        state = self._read()
        provider = dict(self.by_id[provider_id])
        provider["base_url"] = os.getenv(provider.get("base_url_env", ""), provider["base_url"])
        provider["model"] = state.get("models", {}).get(provider_id, provider["model"])
        provider["active"] = state.get("active") == provider_id
        return provider

    def list(self) -> list[dict]:
        return [
            {
                key: provider[key]
                for key in ("id", "label", "base_url", "model", "key_env", "active")
            }
            for provider in (self._provider(profile["id"]) for profile in self.profiles)
        ]

    def active(self) -> dict:
        return self._provider(self._read()["active"])

    def activate(self, provider_id: str) -> dict:
        self._provider(provider_id)
        state = self._read()
        state["active"] = provider_id
        self._write(state)
        return self._provider(provider_id)

    def set_model(self, provider_id: str, model: str) -> dict:
        self._provider(provider_id)
        state = self._read()
        state.setdefault("models", {})[provider_id] = model
        self._write(state)
        return self._provider(provider_id)

    def threshold(self) -> float:
        return float(self._read().get("threshold", self.default_threshold))

    def set_threshold(self, threshold: float) -> float:
        state = self._read()
        state["threshold"] = threshold
        self._write(state)
        return threshold

    def api_key(self, provider: dict) -> str:
        return os.getenv(provider.get("key_env", ""), "") or "not-needed"

    def validate_active(self) -> None:
        self.validate(self.active()["id"])

    def validate(self, provider_id: str) -> None:
        provider = self._provider(provider_id)
        if provider.get("key_required") and not os.getenv(provider["key_env"]):
            raise RuntimeError(f"{provider['key_env']} is required for active provider {provider['id']}")

    async def models(self, provider_id: str) -> list[str]:
        provider = self._provider(provider_id)
        headers = {"Authorization": f"Bearer {self.api_key(provider)}"}
        try:
            payload = await fetch_models(
                f"{provider['base_url'].rstrip('/')}/models",
                headers,
                self.timeout,
            )
        except (httpx.HTTPError, ValueError) as error:
            raise ConnectionError(f"Cannot list models for {provider['label']}: {error}") from error
        values = payload.get("data", payload.get("models", [])) if isinstance(payload, dict) else payload
        models = []
        for value in values if isinstance(values, list) else []:
            model = value if isinstance(value, str) else value.get("id") or value.get("name") or value.get("model")
            if model:
                models.append(str(model))
        return sorted(set(models))

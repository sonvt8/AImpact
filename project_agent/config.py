import os

from dotenv import load_dotenv


load_dotenv()

APP_PASSWORD = os.getenv("APP_PASSWORD") or None

DATA_DIR = os.getenv("DATA_DIR", "data")
HISTORY_DIR = os.getenv("HISTORY_DIR", "history")
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "documents")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/multilingual-e5-small-onnx")

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.84"))

LLM_BACKEND = os.getenv("LLM_BACKEND", "openai_compatible").strip().lower()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
LLM_SKIP_INTERNET_CHECK = os.getenv(
    "LLM_SKIP_INTERNET_CHECK", "true"
).strip().lower() in {"1", "true", "yes", "on"}

EMBEDDING_MODEL_ID = os.getenv(
    "EMBEDDING_MODEL_ID", "intfloat-multilingual-e5-small-onnx-o4-v1"
)
DOMAIN_PROFILE = os.getenv("DOMAIN_PROFILE", "")


def _normalize_backend(backend: str) -> str:
    normalized = backend.strip().lower()
    if normalized == "openai":
        return "openai_compatible"
    if normalized not in {"openai_compatible", "ollama"}:
        raise ValueError(f"Unsupported LLM_BACKEND: {backend}")
    return normalized


def validate() -> None:
    if not APP_PASSWORD:
        raise ValueError("APP_PASSWORD is required")

    backend = _normalize_backend(LLM_BACKEND)
    if backend == "openai_compatible":
        missing = [
            name
            for name, value in (("LLM_BASE_URL", LLM_BASE_URL), ("LLM_MODEL", LLM_MODEL))
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing required configuration for openai_compatible: "
                + ", ".join(missing)
            )


def build_llm_client_kwargs(backend: str) -> dict:
    backend = _normalize_backend(backend)
    if backend == "openai_compatible":
        return {
            "model": LLM_MODEL,
            "base_url": LLM_BASE_URL,
            "api_key": LLM_API_KEY or "not-needed",
            "timeout": LLM_TIMEOUT,
            "temperature": 0.1,
            "streaming": True,
        }
    return {
        "model": OLLAMA_MODEL,
        "base_url": OLLAMA_BASE_URL,
        "temperature": 0.1,
        "streaming": True,
    }

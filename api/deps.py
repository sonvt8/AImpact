from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from api.settings import Settings


CORE_DIR = Path(__file__).resolve().parents[1] / "project_agent"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import config as core_config
import embedding as core_embedding
import index as core_index
import rag as core_rag
import service as core_service
import stats as core_stats
import textnorm as core_textnorm


@lru_cache(maxsize=4)
def get_service(settings: Settings):
    core_config.DATA_DIR = str(settings.data_dir)
    core_config.MODEL_PATH = str(settings.model_path)
    core_config.SIMILARITY_THRESHOLD = settings.similarity_threshold
    embedder = core_embedding.FastTextEmbedder(
        model_path=str(settings.model_path),
        preprocess=core_textnorm.normalize,
    )
    vector_index = core_index.VectorIndex(
        persist_directory=settings.data_dir / "chroma_db",
        key_path=settings.data_dir / "encryption_key.key",
        embedding_model_id=core_config.EMBEDDING_MODEL_ID,
    )
    rag_service = core_service.build_service(
        lambda prompt: "",
        embedder=embedder,
        index_obj=vector_index,
    )
    rag_service.embedder = embedder
    return rag_service


def clear_service_cache() -> None:
    get_service.cache_clear()

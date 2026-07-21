import gc
import hashlib
import sys
from pathlib import Path

import embedding
import index
import rag
import service
import textnorm


WORKBOOK_PATH = Path(__file__).resolve().parents[1] / "Phu luc 1.xlsx"


class FakeModel:
    def get_sentence_vector(self, text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [value + 1 for value in digest[:8]]


def create_service(tmp_path, generator_fn):
    vector_index = index.VectorIndex(
        persist_directory=tmp_path / "chroma",
        key_path=tmp_path / "encryption.key",
        embedding_model_id="test-embedding",
    )
    embedder = embedding.FastTextEmbedder(
        model=FakeModel(),
        preprocess=textnorm.normalize,
    )
    return service.build_service(
        generator_fn,
        embedder=embedder,
        index_obj=vector_index,
    ), vector_index


def test_ingest_then_answer_with_evidence(tmp_path):
    prompts = []

    def generator_fn(prompt):
        prompts.append(prompt)
        return "OK"

    rag_service, vector_index = create_service(tmp_path, generator_fn)
    try:
        ingest_result = rag_service.ingest_path(WORKBOOK_PATH, WORKBOOK_PATH.name)
        answer_result = rag_service.answer(
            "Sự cố điện lưới",
            history="H",
            threshold=0.0,
        )

        assert ingest_result["added"] == 357
        assert answer_result["answer"] == "OK"
        assert answer_result["llm_called"] is True
        assert answer_result["citations"]
        assert len(prompts) == 1
        assert "H" in prompts[0]
        assert answer_result["citations"][0]["locator"] in prompts[0]
    finally:
        del rag_service
        del vector_index
        gc.collect()


def test_answer_no_evidence_no_llm(tmp_path):
    generator_calls = []

    def generator_fn(prompt):
        generator_calls.append(prompt)
        return "unexpected"

    rag_service, vector_index = create_service(tmp_path, generator_fn)
    try:
        rag_service.ingest_path(WORKBOOK_PATH, WORKBOOK_PATH.name)

        result = rag_service.answer("Không có", threshold=1.1)

        assert result == {
            "answer": rag.NO_EVIDENCE_MESSAGE,
            "citations": [],
            "llm_called": False,
        }
        assert generator_calls == []
    finally:
        del rag_service
        del vector_index
        gc.collect()


def test_threshold_defaults_to_config(tmp_path, monkeypatch):
    generator_calls = []

    def generator_fn(prompt):
        generator_calls.append(prompt)
        return "unexpected"

    rag_service, vector_index = create_service(tmp_path, generator_fn)
    try:
        rag_service.ingest_path(WORKBOOK_PATH, WORKBOOK_PATH.name)
        monkeypatch.setattr(service.config, "SIMILARITY_THRESHOLD", 1.1)

        result = rag_service.answer("Không có")

        assert result["llm_called"] is False
        assert result["answer"] == rag.NO_EVIDENCE_MESSAGE
        assert generator_calls == []
    finally:
        del rag_service
        del vector_index
        gc.collect()


def test_service_import_isolation():
    assert "streamlit" not in sys.modules
    assert "langchain" not in sys.modules

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
        structured_workbook=WORKBOOK_PATH,
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

        assert ingest_result["added"] == 127
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


def test_structured_route_skips_embedding_and_llm(tmp_path):
    generator_calls = []
    rag_service, vector_index = create_service(
        tmp_path,
        lambda prompt: generator_calls.append(prompt),
    )
    fail_embed = lambda texts: (_ for _ in ()).throw(
        AssertionError("embedding must not run")
    )
    rag_service.embed_fn = fail_embed
    rag_service.passage_embed_fn = fail_embed
    rag_service.query_embed_fn = fail_embed
    try:
        assert rag_service.route("Có bao nhiêu sự cố AC trong VHKT?") == {
            "answer": 6,
            "citations": [
                {
                    "filename": "Phu luc 1.xlsx",
                    "sheet_name": "Tong hop",
                    "locator": "Tong hop!E3",
                    "content": "6",
                }
            ],
            "llm_called": False,
            "route": "structured",
        }
        assert rag_service.route("Có bao nhiêu sự cố AC?")["route"] == "refuse"
        assert rag_service.route("N6 mất lộ điện nổi thì xử lý thế nào?") is None
        assert generator_calls == []
    finally:
        del rag_service
        del vector_index
        gc.collect()


def test_build_service_routes_passage_and_query_embeddings():
    class DualEmbedder:
        def __init__(self):
            self.calls = []

        def embed_passages(self, texts):
            self.calls.append(("passage", list(texts)))
            return [[1.0]]

        def embed_queries(self, texts):
            self.calls.append(("query", list(texts)))
            return [[1.0]]

        def embed(self, texts):
            raise AssertionError("generic embed must not run")

    class FakeIndex:
        def add_records(self, records, filename, checksum, embed_fn, parser_version):
            self.passage_embed_fn = embed_fn
            embed_fn(["document"])
            return {"added": len(records)}

        def query(self, query, embed_fn, top_k=10, where=None):
            self.query_embed_fn = embed_fn
            embed_fn([query])
            return []

    embedder = DualEmbedder()
    fake_index = FakeIndex()
    rag_service = service.build_service(
        lambda prompt: "unexpected",
        embedder=embedder,
        index_obj=fake_index,
        structured_workbook=WORKBOOK_PATH,
    )

    assert rag_service.ingest_path(WORKBOOK_PATH, WORKBOOK_PATH.name)["added"] == 373
    assert rag_service.answer("Không có", threshold=1.1)["llm_called"] is False
    assert fake_index.passage_embed_fn is rag_service.passage_embed_fn
    assert fake_index.query_embed_fn is rag_service.query_embed_fn
    assert rag_service.embed_fn is rag_service.query_embed_fn
    assert [kind for kind, _ in embedder.calls] == ["passage", "query"]


def test_build_service_default_embedder_follows_model_path(tmp_path, monkeypatch):
    calls = []

    class FakeEmbedder:
        def embed(self, texts):
            return [[1.0] for _ in texts]

    monkeypatch.setattr(
        service.embedding,
        "LocalOnnxEmbedder",
        lambda path, **kwargs: calls.append(("onnx", Path(path), kwargs)) or FakeEmbedder(),
    )
    monkeypatch.setattr(
        service.embedding,
        "FastTextEmbedder",
        lambda **kwargs: calls.append(("fasttext", kwargs)) or FakeEmbedder(),
    )

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    monkeypatch.setattr(service.config, "MODEL_PATH", str(bundle))
    service.build_service(lambda prompt: "", index_obj=object())
    assert calls == [("onnx", bundle, {"max_length": 96, "batch_size": 16})]

    model = tmp_path / "cc.vi.300.bin"
    model.write_bytes(b"model")
    monkeypatch.setattr(service.config, "MODEL_PATH", str(model))
    service.build_service(lambda prompt: "", index_obj=object())
    assert calls[-1][0] == "fasttext"
    assert calls[-1][1]["model_path"] == str(model)


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

from __future__ import annotations

import gc
import hashlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import api.providers as provider_module
from api.deps import (
    _collection_name,
    core_embedding,
    core_index,
    core_service,
    core_textnorm,
)
from api.main import create_app
from tests_api.conftest import WORKBOOK, auth, login, parse_sse


def create_account(client, admin_token, username, password="user-password-strong", role="user"):
    response = client.post(
        "/api/users",
        headers=auth(admin_token),
        json={"username": username, "password": password, "role": role},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_auth_rbac_refresh_logout_all(api_client):
    client, _, _, _ = api_client
    assert client.post("/api/auth/login", json={"username": "admin", "password": "wrong"}).status_code == 401
    admin = login(client)
    create_account(client, admin["access_token"], "operator")
    first = login(client, "operator", "user-password-strong")
    second = login(client, "operator", "user-password-strong")
    assert client.get("/api/users", headers=auth(first["access_token"])).status_code == 403

    rotated = client.post("/api/auth/refresh", json={"refresh_token": first["refresh_token"]})
    assert rotated.status_code == 200
    assert client.post("/api/auth/refresh", json={"refresh_token": first["refresh_token"]}).status_code == 401

    logout = client.post(
        "/api/auth/logout",
        headers=auth(second["access_token"]),
        json={"all": True},
    )
    assert logout.status_code == 200
    assert client.post("/api/auth/refresh", json={"refresh_token": second["refresh_token"]}).status_code == 401
    assert client.post("/api/auth/refresh", json={"refresh_token": rotated.json()["refresh_token"]}).status_code == 401


def test_conversations_are_private(api_client):
    client, _, _, _ = api_client
    admin = login(client)
    create_account(client, admin["access_token"], "alice")
    create_account(client, admin["access_token"], "bobby")
    alice = login(client, "alice", "user-password-strong")
    bobby = login(client, "bobby", "user-password-strong")
    conversation = client.post(
        "/api/conversations",
        headers=auth(alice["access_token"]),
        json={"title": "Riêng tư"},
    ).json()
    assert client.get(
        f"/api/conversations/{conversation['id']}",
        headers=auth(bobby["access_token"]),
    ).status_code == 404


def test_query_gate_never_calls_llm_without_evidence(api_client):
    client, _, service, llm = api_client
    tokens = login(client)
    conversation = client.post(
        "/api/conversations", headers=auth(tokens["access_token"]), json={}
    ).json()
    service.index_obj.hits = [{
        "text_verbatim": "Không đạt ngưỡng",
        "metadata": {"filename": "x.xlsx", "sheet_name": "S", "locator": "S!A1", "stt": "1"},
        "similarity": 0.2,
    }]
    response = client.post(
        "/api/query",
        headers=auth(tokens["access_token"]),
        json={"query": "bí mật-câu-hỏi", "conversation_id": conversation["id"]},
    )
    events = parse_sse(response.text)
    assert events == [{
        "type": "final",
        "text": "Không tìm thấy thông tin phù hợp trong tài liệu.",
        "citations": [],
        "route": "refuse",
    }]
    assert llm.calls == []
    assert service.index_obj.query_calls == 1
    assert service.index_obj.query_embed_fns == [service.query_embed_fn]


def test_query_stream_has_locator_and_verbatim(api_client):
    client, _, service, llm = api_client
    tokens = login(client)
    service.index_obj.hits = [{
        "text_verbatim": "Tên tình huống: Lỗi ACB 2000A tủ MSB4.1 từ ATS3 cấp lên (N6)\nGiải pháp: Reset ACB rồi đóng lại.",
        "metadata": {
            "filename": "Phu luc 1.xlsx", "sheet_name": "2.Ds XLSC", "locator": "2.Ds XLSC!A53:L53",
            "row_index": 53, "stt": "17", "section_path": "Hệ thống AC",
        },
        "similarity": 0.99,
        "lexical_gate": True,
    }]
    response = client.post(
        "/api/query",
        headers=auth(tokens["access_token"]),
        json={"query": "Hướng xử lý sự cố ACB 2000A từ ATS3 cấp lên tại N6", "threshold": 0.78},
    )
    events = parse_sse(response.text)
    final = events[-1]
    assert final["type"] == "final"
    assert final["route"] == "retrieval"
    assert final["citations"][0]["locator"] == "2.Ds XLSC!A53:L53"
    assert "Reset ACB" in final["citations"][0]["content"]
    assert service.index_obj.query_calls == 1
    assert len(llm.calls) == 1


def test_keyword_overlap_without_gated_evidence_refuses(api_client):
    client, _, service, llm = api_client
    tokens = login(client)
    record = {
        "text_verbatim": "Reset ACB MSB4 ATS3 N6",
        "metadata": {
            "filename": "Phu luc 1.xlsx",
            "sheet_name": "2.Ds XLSC",
            "locator": "2.Ds XLSC!A53:L53",
        },
    }
    service.index_obj.records = [record]
    service.index_obj.hits = [{
        **record,
        "similarity": 0.99,
        "lexical_gate": False,
    }]

    response = client.post(
        "/api/query",
        headers=auth(tokens["access_token"]),
        json={"query": "Reset ACB MSB4 ATS3 N6"},
    )

    assert parse_sse(response.text)[-1] == {
        "type": "final",
        "text": "Không tìm thấy thông tin phù hợp trong tài liệu.",
        "citations": [],
        "route": "refuse",
    }
    assert service.index_obj.query_calls == 1
    assert service.index_obj.hits[0]["similarity"] == 0.99
    assert llm.calls == []


def test_structured_query_and_refusal_skip_retrieval_and_llm(api_client):
    client, _, service, llm = api_client
    tokens = login(client)

    response = client.post(
        "/api/query",
        headers=auth(tokens["access_token"]),
        json={"query": "Có bao nhiêu sự cố AC trong vhkt?"},
    )
    final = parse_sse(response.text)[-1]
    assert final == {
        "type": "final",
        "text": "6",
        "citations": [
            {
                "filename": "Phu luc 1.xlsx",
                "sheet_name": "Tong hop",
                "locator": "Tong hop!E3",
                "content": "6",
            }
        ],
        "route": "structured",
    }

    response = client.post(
        "/api/query",
        headers=auth(tokens["access_token"]),
        json={"query": "Có bao nhiêu sự cố AC?"},
    )
    final = parse_sse(response.text)[-1]
    assert final["route"] == "refuse"
    assert final["text"] == "Không tìm thấy thông tin phù hợp trong tài liệu."
    assert final["citations"] == []
    assert service.index_obj.query_calls == 0
    assert service.embed_calls == 0
    assert llm.calls == []


class FakeModel:
    def get_sentence_vector(self, text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [value + 1 for value in digest[:8]]


def test_collection_name_is_deterministic_safe_and_new():
    name = _collection_name("intfloat/multilingual-e5-small@onnx")

    assert name == _collection_name("intfloat/multilingual-e5-small@onnx")
    assert name != "viettel_docs"
    assert len(name) <= 63
    assert name.replace("-", "").isalnum()


def test_upload_example_workbook_skips_startup_seeded_checksum(settings):
    vector_index = core_index.VectorIndex(
        persist_directory=settings.data_dir / "upload-chroma",
        key_path=settings.data_dir / "upload.key",
        embedding_model_id="api-test-embedding",
    )
    embedder = core_embedding.FastTextEmbedder(model=FakeModel(), preprocess=core_textnorm.normalize)
    service = core_service.build_service(
        lambda prompt: "",
        embedder=embedder,
        index_obj=vector_index,
        structured_workbook=settings.stats_workbook,
    )
    app = create_app(settings)
    app.state.service_getter = lambda: service
    try:
        with TestClient(app) as client:
            tokens = login(client)
            with WORKBOOK.open("rb") as source:
                response = client.post(
                    "/api/documents",
                    headers=auth(tokens["access_token"]),
                    files={"file": (WORKBOOK.name, source, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                )
            assert response.status_code == 200, response.text
            assert response.json() == {"status": "skipped_same_checksum", "added": 0}
            assert vector_index.count() == 127
    finally:
        del service
        del vector_index
        gc.collect()


def test_stats_filters_index_column(api_client):
    client, _, _, _ = api_client
    tokens = login(client)
    response = client.get("/api/stats", headers=auth(tokens["access_token"]))
    assert response.status_code == 200
    assert response.json()["totals"] == {"UCTT": 61, "XLSC": 50, "VHKT": 6, "TỔNG": 117}


def test_provider_runtime_models_and_502(api_client, monkeypatch):
    client, _, _, _ = api_client
    tokens = login(client)
    profiles = client.get("/api/providers", headers=auth(tokens["access_token"]))
    assert profiles.status_code == 200
    serialized = json.dumps(profiles.json())
    assert "not-needed" not in serialized and "secret" not in serialized

    async def models(*args):
        return {"data": [{"id": "model-b"}, {"id": "model-a"}]}

    monkeypatch.setattr(provider_module, "fetch_models", models)
    response = client.get("/api/providers/ollama/models", headers=auth(tokens["access_token"]))
    assert response.json() == {"models": ["model-a", "model-b"]}
    assert client.post(
        "/api/providers/active",
        headers=auth(tokens["access_token"]),
        json={"id": "ollama"},
    ).status_code == 200
    assert client.post(
        "/api/providers/ollama/model",
        headers=auth(tokens["access_token"]),
        json={"model": "model-b"},
    ).json()["model"] == "model-b"

    async def unavailable(*args):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(provider_module, "fetch_models", unavailable)
    response = client.get("/api/providers/ollama/models", headers=auth(tokens["access_token"]))
    assert response.status_code == 502
    assert "Cannot list models" in response.json()["detail"]


def test_audit_contains_metadata_not_sensitive_content(api_client):
    client, _, service, llm = api_client
    llm.answer = "bí-mật-câu-trả-lời"
    service.index_obj.hits = [{
        "text_verbatim": "bằng chứng",
        "metadata": {"filename": "x.xlsx", "sheet_name": "S", "locator": "S!A1"},
        "similarity": 0.99,
    }]
    tokens = login(client)
    client.post(
        "/api/query",
        headers=auth(tokens["access_token"]),
        json={"query": "bí-mật-câu-hỏi"},
    )
    response = client.get("/api/audit", headers=auth(tokens["access_token"]))
    text = json.dumps(response.json(), ensure_ascii=False)
    assert '"action": "query"' in text
    assert "bí-mật-câu-hỏi" not in text
    assert "bí-mật-câu-trả-lời" not in text
    assert "bằng chứng" not in text


def test_startup_does_not_call_legacy_config_validate(settings, monkeypatch):
    monkeypatch.setattr("api.deps.core_config.validate", lambda: (_ for _ in ()).throw(AssertionError("legacy")))
    app = create_app(settings)
    app.state.service_getter = lambda: StartupService(count=1)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200


class StartupService:
    def __init__(self, *, count=0, fail_on=None):
        self.events = []
        self.fail_on = fail_on
        self.embedder = type("Embedder", (), {"load": lambda embedder: self.events.append("load")})()
        self.index_obj = type(
            "Index",
            (),
            {
                "count": lambda index: count,
                "reset": lambda index: self.events.append("reset"),
            },
        )()

    def ingest_path(self, path, filename):
        self.events.append(filename)
        if filename == self.fail_on:
            raise RuntimeError("migration failed")


def test_startup_prewarms_configured_service(settings):
    service = StartupService(count=1)
    app = create_app(settings)
    app.state.service_getter = lambda: service

    with TestClient(app):
        pass

    assert service.events == ["load"]


def test_startup_migrates_supported_documents_in_deterministic_order(settings):
    app = create_app(settings)
    for filename in ("b.PDF", "A.txt", "c.csv", "d.docx", "e.xlsx"):
        (settings.documents_dir / filename).write_text("test", encoding="utf-8")
    service = StartupService()
    app.state.service_getter = lambda: service

    with TestClient(app):
        pass

    assert service.events == [
        "load",
        "A.txt",
        "b.PDF",
        "c.csv",
        "d.docx",
        "e.xlsx",
        settings.stats_workbook.name,
    ]


def test_startup_falls_back_to_stats_workbook(settings):
    service = StartupService()
    app = create_app(settings)
    app.state.service_getter = lambda: service

    with TestClient(app):
        pass

    assert service.events == ["load", settings.stats_workbook.name]


def test_startup_uses_case_insensitive_document_workbook_once(settings):
    app = create_app(settings)
    filename = settings.stats_workbook.name.swapcase()
    (settings.documents_dir / filename).write_text("test", encoding="utf-8")
    service = StartupService()
    app.state.service_getter = lambda: service

    with TestClient(app):
        pass

    assert service.events == ["load", filename]


def test_startup_skips_hidden_unsupported_and_non_files(settings):
    app = create_app(settings)
    (settings.documents_dir / ".hidden.txt").write_text("test", encoding="utf-8")
    (settings.documents_dir / "unsupported.md").write_text("test", encoding="utf-8")
    (settings.documents_dir / "folder.txt").mkdir()
    (settings.documents_dir / "visible.txt").write_text("test", encoding="utf-8")
    service = StartupService()
    app.state.service_getter = lambda: service

    with TestClient(app):
        pass

    assert service.events == ["load", "visible.txt", settings.stats_workbook.name]


def test_startup_resets_partial_collection_and_fails(settings):
    app = create_app(settings)
    for filename in ("a.txt", "b.txt", "c.txt"):
        (settings.documents_dir / filename).write_text("test", encoding="utf-8")
    service = StartupService(fail_on="b.txt")
    app.state.service_getter = lambda: service

    with pytest.raises(RuntimeError, match="migration failed"):
        with TestClient(app):
            pass

    assert service.events == ["load", "a.txt", "b.txt", "reset"]


def test_health_supports_onnx_loaded_flag(api_client):
    client, app, service, _ = api_client
    service.embedder = type("Embedder", (), {"_loaded": True})()

    async def models(provider_id):
        return []

    app.state.providers.models = models
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["fasttext_loaded"] is True

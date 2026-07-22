from __future__ import annotations

import gc
import hashlib
import json

import httpx
from fastapi.testclient import TestClient

import api.providers as provider_module
from api.deps import core_embedding, core_index, core_service, core_textnorm
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
    }]
    assert llm.calls == []


def test_query_stream_has_locator_and_verbatim(api_client):
    client, _, service, llm = api_client
    tokens = login(client)
    service.index_obj.hits = [{
        "text_verbatim": "Nội dung nguyên văn",
        "metadata": {
            "filename": "Phu luc 1.xlsx", "sheet_name": "VHKT", "locator": "VHKT!A12:F12",
            "stt": "6", "section_path": "Điện lưới",
        },
        "similarity": 0.91,
    }]
    response = client.post(
        "/api/query",
        headers=auth(tokens["access_token"]),
        json={"query": "Sự cố VHKT", "threshold": 0.78},
    )
    events = parse_sse(response.text)
    final = events[-1]
    assert final["type"] == "final"
    assert final["citations"][0]["locator"] == "VHKT!A12:F12"
    assert final["citations"][0]["content"] == "Nội dung nguyên văn"
    assert len(llm.calls) == 1


class FakeModel:
    def get_sentence_vector(self, text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [value + 1 for value in digest[:8]]


def test_upload_example_workbook_adds_357(settings):
    vector_index = core_index.VectorIndex(
        persist_directory=settings.data_dir / "upload-chroma",
        key_path=settings.data_dir / "upload.key",
        embedding_model_id="api-test-embedding",
    )
    embedder = core_embedding.FastTextEmbedder(model=FakeModel(), preprocess=core_textnorm.normalize)
    service = core_service.build_service(lambda prompt: "", embedder=embedder, index_obj=vector_index)
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
            assert response.json()["added"] == 357
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
    app.state.service_getter = lambda: type("Service", (), {"index_obj": type("Index", (), {"count": lambda self: 0})(), "embedder": type("Embedder", (), {"_model": None})()})()
    with TestClient(app) as client:
        assert client.get("/").status_code == 200

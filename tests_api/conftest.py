from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import Settings


WORKBOOK = Path(__file__).resolve().parents[1] / "project_agent" / "Phu luc 1.xlsx"


class FakeIndex:
    def __init__(self):
        self.hits = []
        self.filenames = ["Phu luc 1.xlsx"]

    def query(self, query, embed_fn, top_k=10, where=None):
        return self.hits[:top_k]

    def list_filenames(self):
        return list(self.filenames)

    def delete_file(self, filename):
        if filename in self.filenames:
            self.filenames.remove(filename)

    def count(self):
        return len(self.hits)


class FakeService:
    def __init__(self):
        self.index_obj = FakeIndex()
        self.embed_fn = lambda texts: [[1.0] for _ in texts]
        self.embedder = type("Embedder", (), {"_model": object()})()

    def ingest_path(self, path, filename):
        if filename not in self.index_obj.filenames:
            self.index_obj.filenames.append(filename)
        return {"status": "added", "added": 357}


@dataclass
class FakeLLM:
    answer: str = "Câu trả lời kiểm chứng"
    calls: list[str] = field(default_factory=list)

    async def stream(self, prompt: str):
        self.calls.append(prompt)
        for token in self.answer.split(" "):
            yield token + " "


@pytest.fixture
def settings(tmp_path):
    model = tmp_path / "models" / "cc.vi.300.bin"
    model.parent.mkdir()
    model.write_bytes(b"test-model")
    workbook = tmp_path / "Phu luc 1.xlsx"
    shutil.copy2(WORKBOOK, workbook)
    return Settings(
        root_dir=tmp_path,
        data_dir=tmp_path / "data",
        history_dir=tmp_path / "history",
        documents_dir=tmp_path / "documents",
        model_path=model,
        stats_workbook=workbook,
        database_path=tmp_path / "data" / "app.db",
        providers_state_path=tmp_path / "data" / "providers.state.json",
        frontend_dist=tmp_path / "dist",
        jwt_secret="test-secret-that-is-longer-than-thirty-two-characters",
        jwt_access_expire_min=15,
        jwt_refresh_expire_days=7,
        admin_username="admin",
        admin_password="admin-password-strong",
        frontend_origin="http://testserver",
        similarity_threshold=0.78,
        llm_timeout=3,
        max_upload_mb=50,
        app_port=8000,
    )


@pytest.fixture
def api_client(settings):
    app = create_app(settings)
    service = FakeService()
    llm = FakeLLM()
    app.state.service_getter = lambda: service
    app.state.llm = llm
    with TestClient(app) as client:
        yield client, app, service, llm


def login(client: TestClient, username="admin", password="admin-password-strong"):
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()


def auth(token: str):
    return {"Authorization": f"Bearer {token}"}


def parse_sse(text: str):
    return [
        __import__("json").loads(line[5:].strip())
        for block in text.strip().split("\n\n")
        for line in block.splitlines()
        if line.startswith("data:")
    ]

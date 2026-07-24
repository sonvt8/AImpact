from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.auth import LoginRateLimiter, hash_password
from api.db import Database
from api.deps import core_rag, core_stats, get_service
from api.llm import OpenAICompatibleLLM
from api.providers import ProviderRegistry
from api.routes import router
from api.settings import Settings, get_settings


SUPPORTED_DOCUMENT_SUFFIXES = {".xlsx", ".pdf", ".docx", ".txt", ".csv"}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_directories()
    database = Database(settings.database_path)
    providers = ProviderRegistry(
        settings.providers_state_path,
        settings.similarity_threshold,
        settings.llm_timeout,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.validate()
        database.initialize()
        if database.user_count() == 0:
            if not settings.admin_username or len(settings.admin_password) < 10:
                raise RuntimeError(
                    "ADMIN_USERNAME and ADMIN_PASSWORD (at least 10 characters) are required for first startup"
                )
            database.create_user(
                settings.admin_username,
                hash_password(settings.admin_password),
                "admin",
            )
        providers.validate_active()
        service = app.state.service_getter()
        load_embedder = getattr(getattr(service, "embedder", None), "load", None)
        if callable(load_embedder):
            load_embedder()
        if service.index_obj.count() == 0:
            try:
                documents = sorted(
                    (
                        path
                        for path in settings.documents_dir.iterdir()
                        if path.is_file()
                        and not path.name.startswith(".")
                        and path.suffix.lower() in SUPPORTED_DOCUMENT_SUFFIXES
                    ),
                    key=lambda path: (path.name.casefold(), path.name),
                )
                if not any(
                    path.name.casefold() == settings.stats_workbook.name.casefold()
                    for path in documents
                ):
                    documents.append(settings.stats_workbook)
                for path in documents:
                    service.ingest_path(path, path.name)
            except Exception:
                service.index_obj.reset()
                raise
        yield

    app = FastAPI(title="AImpact API", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.db = database
    app.state.providers = providers
    app.state.llm = OpenAICompatibleLLM(providers, settings.llm_timeout)
    app.state.login_limiter = LoginRateLimiter()
    app.state.service_getter = lambda: get_service(settings)
    app.state.rag = core_rag
    app.state.stats = core_stats
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.include_router(router)

    if settings.frontend_dist.is_dir() and (settings.frontend_dist / "index.html").is_file():
        assets = settings.frontend_dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            candidate = settings.frontend_dist / path
            if path and candidate.is_file() and settings.frontend_dist in candidate.resolve().parents:
                return FileResponse(candidate)
            return FileResponse(settings.frontend_dist / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        async def root():
            return {"name": "AImpact API", "docs": "/docs", "health": "/api/health"}

    return app


app = create_app()

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import unicodedata
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from api.auth import (
    current_user,
    hash_password,
    issue_tokens,
    require_roles,
    rotate_refresh,
    token_hash,
    verify_password,
)
from api.llm import LLMError
from api.models import (
    ConversationCreate,
    LoginRequest,
    LogoutRequest,
    ProviderActivate,
    ProviderModelUpdate,
    QueryRequest,
    RefreshRequest,
    ThresholdUpdate,
    UserCreate,
    UserUpdate,
)


router = APIRouter(prefix="/api")
all_roles = require_roles("admin", "user", "viewer")
admin_only = require_roles("admin")


def _sse(event_type: str, **payload) -> str:
    return "data: " + json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n\n"


def _public_user(user: dict) -> dict:
    return {key: user[key] for key in ("id", "username", "role", "created_at")}


@router.post("/auth/login")
async def login(payload: LoginRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    rate_key = f"{client_ip}|{payload.username.casefold()}"
    if not request.app.state.login_limiter.allow(rate_key, time.monotonic()):
        raise HTTPException(status_code=429, detail="Too many login attempts; retry in one minute")
    user = request.app.state.db.get_user_by_username(payload.username)
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    request.app.state.login_limiter.clear(rate_key)
    tokens = issue_tokens(user, request.app.state.db, request.app.state.settings)
    request.app.state.db.add_audit(user, "login")
    return {**tokens, "role": user["role"], "username": user["username"]}


@router.post("/auth/refresh")
async def refresh(payload: RefreshRequest, request: Request):
    tokens, _ = rotate_refresh(payload.refresh_token, request.app.state.db, request.app.state.settings)
    return tokens


@router.post("/auth/logout")
async def logout(payload: LogoutRequest, request: Request, user: dict = Depends(current_user)):
    if payload.all:
        request.app.state.db.revoke_all_refresh(user["id"])
    elif payload.refresh_token:
        request.app.state.db.revoke_refresh(token_hash(payload.refresh_token))
    else:
        raise HTTPException(status_code=400, detail="refresh_token is required unless all=true")
    return {"status": "logged_out"}


@router.get("/auth/me")
async def me(user: dict = Depends(current_user)):
    return {"username": user["username"], "role": user["role"]}


@router.get("/users")
async def list_users(request: Request, _: dict = Depends(admin_only)):
    return request.app.state.db.list_users()


@router.post("/users", status_code=201)
async def create_user(payload: UserCreate, request: Request, admin: dict = Depends(admin_only)):
    try:
        user = request.app.state.db.create_user(
            payload.username,
            hash_password(payload.password),
            payload.role,
        )
    except sqlite3.IntegrityError as error:
        raise HTTPException(status_code=409, detail="Username already exists") from error
    request.app.state.db.add_audit(admin, "user_create", str(user["id"]))
    return user


@router.patch("/users/{user_id}")
async def update_user(user_id: int, payload: UserUpdate, request: Request, admin: dict = Depends(admin_only)):
    existing = request.app.state.db.get_user(user_id)
    if not existing:
        raise HTTPException(status_code=404, detail="User not found")
    if (
        existing["role"] == "admin"
        and payload.role is not None
        and payload.role != "admin"
        and request.app.state.db.count_role("admin") <= 1
    ):
        raise HTTPException(status_code=409, detail="The last administrator must remain an admin")
    if payload.role is None and payload.password is None:
        raise HTTPException(status_code=400, detail="No changes supplied")
    user = request.app.state.db.update_user(
        user_id,
        role=payload.role,
        password_hash=hash_password(payload.password) if payload.password else None,
    )
    request.app.state.db.revoke_all_refresh(user_id)
    request.app.state.db.add_audit(admin, "user_update", str(user_id))
    return user


@router.delete("/users/{user_id}")
async def delete_user(user_id: int, request: Request, admin: dict = Depends(admin_only)):
    try:
        request.app.state.db.delete_user(user_id, admin["id"])
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    request.app.state.db.add_audit(admin, "user_delete", str(user_id))
    return {"status": "deleted"}


@router.get("/conversations")
async def list_conversations(request: Request, user: dict = Depends(all_roles)):
    return request.app.state.db.list_conversations(user["id"])


@router.post("/conversations", status_code=201)
async def create_conversation(
    payload: ConversationCreate,
    request: Request,
    user: dict = Depends(all_roles),
):
    title = (payload.title or "Hội thoại mới").strip() or "Hội thoại mới"
    return request.app.state.db.create_conversation(user["id"], title)


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request, user: dict = Depends(all_roles)):
    conversation = request.app.state.db.conversation(conversation_id, user["id"])
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, request: Request, user: dict = Depends(all_roles)):
    if not request.app.state.db.delete_conversation(conversation_id, user["id"]):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted"}


@router.post("/query")
async def query(payload: QueryRequest, request: Request, user: dict = Depends(all_roles)):
    history = ""
    if payload.conversation_id:
        history = request.app.state.db.conversation_history(payload.conversation_id, user["id"])
        if history is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

    rag_service = request.app.state.service_getter()
    hits = await asyncio.to_thread(
        rag_service.index_obj.query,
        payload.query,
        rag_service.embed_fn,
        payload.top_k,
    )
    threshold = payload.threshold if payload.threshold is not None else request.app.state.providers.threshold()
    prompt, citations = request.app.state.rag.answer_or_refuse(
        payload.query,
        hits,
        threshold,
        history,
        "Trợ lý kỹ thuật",
    )
    if payload.conversation_id:
        request.app.state.db.add_message(payload.conversation_id, "user", payload.query)
    request.app.state.db.add_audit(user, "query", payload.conversation_id)

    async def events():
        if prompt is None:
            answer = request.app.state.rag.NO_EVIDENCE_MESSAGE
            if payload.conversation_id:
                request.app.state.db.add_message(payload.conversation_id, "assistant", answer)
            yield _sse("final", text=answer, citations=[])
            return

        answer_parts = []
        try:
            async for token in request.app.state.llm.stream(prompt):
                answer_parts.append(token)
                yield _sse("token", text=token)
        except LLMError:
            yield _sse("error", detail="Active LLM provider is unavailable")
            return
        answer = "".join(answer_parts)
        if payload.conversation_id:
            request.app.state.db.add_message(
                payload.conversation_id,
                "assistant",
                answer,
                citations,
            )
        yield _sse("final", citations=citations)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/documents")
async def list_documents(request: Request, _: dict = Depends(all_roles)):
    return request.app.state.service_getter().index_obj.list_filenames()


@router.get("/documents/{filename}")
async def open_document(filename: str, request: Request, _: dict = Depends(all_roles)):
    safe_name = Path(filename).name
    path = request.app.state.settings.documents_dir / safe_name
    if safe_name != filename or not path.is_file():
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(path, filename=safe_name)


@router.post("/documents")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    admin: dict = Depends(admin_only),
):
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() not in {".xlsx", ".pdf", ".docx", ".txt", ".csv"}:
        raise HTTPException(status_code=400, detail="Supported document types: xlsx, pdf, docx, txt, csv")
    destination = request.app.state.settings.documents_dir / filename
    temporary = destination.with_name(f".{destination.stem}-{uuid4().hex}{destination.suffix}")
    size = 0
    limit = request.app.state.settings.max_upload_mb * 1024 * 1024
    try:
        with temporary.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail="Document exceeds upload size limit")
                output.write(chunk)
        result = await asyncio.to_thread(
            request.app.state.service_getter().ingest_path,
            temporary,
            filename,
        )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    request.app.state.db.add_audit(admin, "upload", filename)
    return result


@router.delete("/documents/{filename}")
async def delete_document(filename: str, request: Request, admin: dict = Depends(admin_only)):
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    await asyncio.to_thread(request.app.state.service_getter().index_obj.delete_file, safe_name)
    (request.app.state.settings.documents_dir / safe_name).unlink(missing_ok=True)
    request.app.state.db.add_audit(admin, "delete_document", safe_name)
    return {"status": "deleted"}


def _index_column(column: object) -> bool:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFD", str(column)).casefold()
        if unicodedata.category(character) != "Mn" and character.isalnum()
    )
    return normalized in {"stt", "sothutu", "index", "no"}


@router.get("/stats")
async def stats(request: Request, sheet: str = Query(default="Tong hop"), _: dict = Depends(all_roles)):
    rows = await asyncio.to_thread(
        request.app.state.stats.read_table,
        request.app.state.settings.stats_workbook,
        sheet,
    )
    totals = request.app.state.stats.column_totals(rows)
    return {
        "sheet": sheet,
        "rows": rows,
        "totals": {str(key): value for key, value in totals.items() if not _index_column(key)},
    }


@router.get("/providers")
async def providers(request: Request, _: dict = Depends(current_user)):
    return request.app.state.providers.list()


@router.post("/providers/active")
async def activate_provider(
    payload: ProviderActivate,
    request: Request,
    admin: dict = Depends(admin_only),
):
    try:
        request.app.state.providers.validate(payload.id)
        provider = request.app.state.providers.activate(payload.id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Provider not found") from error
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    request.app.state.db.add_audit(admin, "provider_switch", payload.id)
    return provider


@router.post("/providers/{provider_id}/model")
async def set_provider_model(
    provider_id: str,
    payload: ProviderModelUpdate,
    request: Request,
    _: dict = Depends(admin_only),
):
    try:
        return request.app.state.providers.set_model(provider_id, payload.model)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Provider not found") from error


@router.get("/providers/{provider_id}/models")
async def provider_models(provider_id: str, request: Request, _: dict = Depends(current_user)):
    try:
        return {"models": await request.app.state.providers.models(provider_id)}
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Provider not found") from error
    except ConnectionError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.get("/settings/threshold")
async def get_threshold(request: Request, _: dict = Depends(current_user)):
    return {"threshold": request.app.state.providers.threshold()}


@router.post("/settings/threshold")
async def set_threshold(payload: ThresholdUpdate, request: Request, _: dict = Depends(admin_only)):
    return {"threshold": request.app.state.providers.set_threshold(payload.threshold)}


@router.get("/audit")
async def audit(request: Request, _: dict = Depends(admin_only)):
    return request.app.state.db.list_audit()


@router.get("/health")
async def health(request: Request):
    index_count = 0
    fasttext_loaded = False
    try:
        rag_service = request.app.state.service_getter()
        index_count = rag_service.index_obj.count()
        fasttext_loaded = rag_service.embedder._model is not None
    except Exception:
        pass
    active = request.app.state.providers.active()
    try:
        await asyncio.wait_for(request.app.state.providers.models(active["id"]), timeout=3)
        provider_reachable = True
    except Exception:
        provider_reachable = False
    return {
        "fasttext_loaded": fasttext_loaded,
        "index_count": index_count,
        "active_provider": active["id"],
        "provider_reachable": provider_reachable,
    }

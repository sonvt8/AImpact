# Changelog

## 2026-07-21 — Web UI / FastAPI

### Added

- FastAPI API với auth Argon2, access/refresh JWT xoay server-side, RBAC và login rate limit.
- SQLite store cho users, refresh tokens, conversations, messages và metadata-only audit.
- Query SSE giữ nguyên gate/citation/verbatim của lõi RAG; nhánh no-evidence không gọi LLM.
- Upload/list/open/delete document, deterministic stats và lọc totals cột STT ở lớp API.
- Provider registry 9Router/Ollama/OpenAI/Grok, model discovery, switch/model/threshold runtime state.
- React/Vite/Tailwind UI gồm Login, Chat, Citation Panel, Documents, Stats và Admin.
- Docker multi-stage, host volumes và env mẫu không chứa secret.
- 9 test API cho auth/RBAC/rotation/logout, conversation isolation, query gate, citation, upload 357, stats, providers và audit.
- Bộ tài liệu kiến trúc, API, env, vận hành, người dùng, bảo mật và roadmap.

### Preserved

- Không sửa hành vi/file lõi `project_agent`.
- 56 test lõi/robustness vẫn xanh.
- `APP_PASSWORD` không còn là dependency startup của web app.

### Deviations / Blockers

- Repository gốc báo `fatal: unable to write new index file`; công việc được thực hiện trên clean clone nhánh `feature/web-ui` theo fallback đã quy định.
- Đặc tả yêu cầu `ChatOpenAI` nhưng đồng thời cấm `langchain*` trong image. Triển khai dùng `httpx` streaming OpenAI-compatible để giữ hành vi mà không thêm LangChain.
- `chromadb==0.5.0` bắt buộc/import `onnxruntime` transitively. Không thể loại package khỏi image mà vẫn import Chroma; code AImpact không dùng ONNX embedding.

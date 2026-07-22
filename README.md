# AImpact Web

AImpact Web là giao diện nội bộ thay Streamlit cho hệ thống RAG vận hành kỹ thuật. Backend FastAPI bọc nguyên lõi trong `project_agent/`; frontend React/Vite/Tailwind ưu tiên tốc độ, khả năng truy vết và trạng thái “không đủ bằng chứng” rõ ràng.

## Đặc tính chính

- Gate bằng chứng dùng trực tiếp `rag.answer_or_refuse`; nếu không đạt ngưỡng thì không gọi LLM.
- Trích dẫn giữ nguyên `filename`, `sheet_name`, `locator`, `similarity` và nội dung verbatim.
- RBAC `viewer` / `user` / `admin`, Argon2, access JWT ngắn hạn và refresh token xoay phía server.
- Hội thoại tách biệt tuyệt đối theo chủ sở hữu; admin không đọc hội thoại người khác qua API thường.
- Provider runtime: 9Router, Ollama, OpenAI, xAI Grok; key chỉ đến từ biến môi trường.
- SQLite cho tài khoản, token, hội thoại và audit; Chroma/FastText vẫn do lõi RAG quản lý.
- Frontend route-split, initial bundle khoảng 56 kB gzip, không dùng UI framework nặng.

## Yêu cầu

- Python 3.11.
- Node.js 22+ cho phát triển frontend.
- FastText Vietnamese model tại `models/cc.vi.300.bin` (khoảng 4.5 GB sau giải nén).
- Một provider OpenAI-compatible đang chạy hoặc API key tương ứng.
- Docker Desktop + Docker Compose nếu triển khai container.

## Chạy nhanh một lệnh (DEV)

Đặt prerequisite ngoài image tại `models/cc.vi.300.bin`, sau đó chạy từ **gốc repo**:

```powershell
docker compose up --build
```

Không cần tạo `.env`. Ứng dụng mở tại `http://localhost:8000`; health check là `http://localhost:8000/api/health`.

- Admin DEV lần đầu: username `admin`, password `ChangeMe-Dev-2026`. Các giá trị này chỉ seed khi bảng `users` còn rỗng; restart không đổi tài khoản đã tạo.
- Khi `JWT_SECRET` trống, entrypoint sinh secret ngẫu nhiên 64 ký tự hex, lưu ở `data/.jwt_secret` với quyền `600` và đọc lại sau restart. Secret không được in ra log.
- Nếu thiếu model, container thoát mã `78` và hướng dẫn đặt `./models/cc.vi.300.bin` thay vì để FastAPI in stacktrace khó hiểu.
- Cảnh báo DEV luôn nhắc đổi mật khẩu và đặt `JWT_SECRET` riêng trước khi dùng production.

## Chạy DEV không Docker

### 1. Cài backend

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Trên macOS/Linux, thay lệnh kích hoạt bằng `python3.11 -m venv .venv` và dùng `.venv/bin/python`.

### 2. Cấu hình tối thiểu

Điền `.env`:

```env
JWT_SECRET=<chuỗi-ngẫu-nhiên-tối-thiểu-32-ký-tự>
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<mật-khẩu-tối-thiểu-10-ký-tự>
MODEL_PATH=./models/cc.vi.300.bin
SIMILARITY_THRESHOLD=0.78
FRONTEND_ORIGIN=http://localhost:5173
```

`ADMIN_USERNAME` và `ADMIN_PASSWORD` chỉ được dùng khi bảng `users` còn rỗng. Không cần `APP_PASSWORD`; API không gọi `project_agent.config.validate()`.

### 3. Chạy backend và frontend

Terminal 1:

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000
```

Terminal 2:

```powershell
Set-Location web
npm install
npm run dev
```

Mở `http://localhost:5173`. OpenAPI ở `http://localhost:8000/docs`.

## Production

```powershell
Copy-Item .env.example .env
# Điền JWT_SECRET (>=32 ký tự), ADMIN_USERNAME, ADMIN_PASSWORD (>=10 ký tự)
# và FRONTEND_ORIGIN thật, ví dụ https://aimpact.example.internal
docker compose --env-file .env -f docker-compose.yml -f docker/docker-compose.prod.yml up --build -d
```

Overlay production không có default cho bốn biến bắt buộc trên: thiếu hoặc để rỗng thì Docker Compose fail ngay trước khi tạo container. Không đặt secret trong compose, image hay log; dùng secret store của môi trường triển khai hoặc file `.env` chỉ lưu cục bộ và đã được Git ignore.

Cả DEV và PROD đều bind-mount `models/` read-only cùng `data/`, `history/`, `documents/` writable. Workbook thống kê dùng file có sẵn trong image tại `/app/project_agent/Phu luc 1.xlsx` qua biến `STATS_WORKBOOK`.

## Test và build

```powershell
$env:PYTHONPATH=(Resolve-Path project_agent).Path
.venv\Scripts\python.exe -m pytest project_agent\tests -q
.venv\Scripts\python.exe -m pytest tests_api -q
npm --prefix web run build
npm --prefix web audit
```

Kỳ vọng hiện tại: `65 passed`; frontend initial JS khoảng 56 kB gzip.

## Cấu trúc

```text
api/              FastAPI, auth, DB, provider runtime, adapter lõi RAG
web/              React/Vite/Tailwind
docker/           Dockerfile, entrypoint và overlay production
tests_api/        Test hợp đồng API và bảo mật
docs/             Kiến trúc, API, vận hành, bảo mật, hướng dẫn
project_agent/    Lõi RAG hiện có, không thay đổi hành vi
```

## Giới hạn hiện tại

Pipeline chỉ đọc text và bảng; trang chỉ có sơ đồ, biểu đồ hoặc ảnh không được diễn giải. OCR/vision, hybrid retrieval và mở rộng golden set được ghi trong `docs/ROADMAP.md`.

Xem thêm: `docs/ARCHITECTURE.md`, `docs/API_REFERENCE.md`, `docs/ENV.md`, `docs/OPERATIONS.md`, `docs/SECURITY.md`.

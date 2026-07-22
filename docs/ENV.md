# Biến môi trường

API đọc biến môi trường khi `create_app()` được gọi. Đường dẫn tương đối được resolve từ `AIMPACT_ROOT` (mặc định là thư mục chạy tiến trình).

| Biến | Bắt buộc | Mặc định | Ý nghĩa |
|---|---:|---|---|
| `AIMPACT_ROOT` | Không | thư mục hiện tại | Gốc để resolve đường dẫn tương đối |
| `JWT_SECRET` | PROD: Có | DEV Compose: tự sinh | Secret HS256, tối thiểu 32 ký tự; DEV lưu ổn định tại `data/.jwt_secret` |
| `JWT_ACCESS_EXPIRE_MIN` | Không | `15` | Thời hạn access token, phút |
| `JWT_REFRESH_EXPIRE_DAYS` | Không | `7` | Thời hạn refresh token, ngày |
| `ADMIN_USERNAME` | PROD: Có | DEV Compose: `admin` | Tài khoản admin seed khi bảng users rỗng |
| `ADMIN_PASSWORD` | PROD: Có | DEV Compose: mật khẩu cảnh báo | Mật khẩu seed, tối thiểu 10 ký tự; chỉ dùng khi users rỗng |
| `FRONTEND_ORIGIN` | PROD: Có | DEV Compose: `http://localhost:8000` | Origin duy nhất được CORS cho phép |
| `APP_PORT` | Không | `8000` | Cổng host trong Docker Compose |
| `DATA_DIR` | Không | `data` | Chroma, encryption key và runtime data |
| `HISTORY_DIR` | Không | `history` | Volume history tương thích |
| `DOCUMENTS_DIR` | Không | `documents` | Nơi lưu tệp upload |
| `MODEL_PATH` | Có | `models/cc.vi.300.bin` | FastText model; startup yêu cầu file tồn tại |
| `STATS_WORKBOOK` | Có | `project_agent/Phu luc 1.xlsx` | Workbook cho `/api/stats` |
| `DATABASE_PATH` | Không | `data/app.db` | SQLite ứng dụng |
| `PROVIDERS_STATE_PATH` | Không | `data/providers.state.json` | Provider/model/threshold runtime |
| `FRONTEND_DIST` | Không | `web/dist` | Vite build được FastAPI serve |
| `SIMILARITY_THRESHOLD` | Không | `0.78` | Ngưỡng seed ban đầu; sau đó runtime state giữ giá trị |
| `LLM_TIMEOUT` | Không | `60` | Timeout HTTP tới provider, giây |
| `MAX_UPLOAD_MB` | Không | `50` | Giới hạn upload mỗi tệp |
| `NROUTER_API_KEY` | Tùy provider | rỗng | Key 9Router; profile cho phép `not-needed` |
| `NROUTER_BASE_URL` | Không | Direct: `http://127.0.0.1:20128/v1`; Compose: `http://host.docker.internal:20128/v1` | Base URL 9Router theo vị trí tiến trình gọi proxy |
| `OLLAMA_BASE_URL` | Không | `http://localhost:11434/v1` | Override base URL Ollama |
| `OPENAI_BASE_URL` | Không | `https://api.openai.com/v1` | Override endpoint OpenAI-compatible |
| `XAI_BASE_URL` | Không | `https://api.x.ai/v1` | Override endpoint xAI-compatible |
| `OPENAI_API_KEY` | Khi dùng OpenAI | rỗng | Key OpenAI, không trả ra client |
| `XAI_API_KEY` | Khi dùng Grok | rỗng | Key xAI, không trả ra client |

## Thư mục host cho Docker Compose

Các biến sau chỉ được Docker Compose dùng để chọn thư mục bind-mount trên host; API trong container vẫn đọc các đường dẫn `/app/...`:

| Biến | Mặc định | Vai trò |
|---|---|---|
| `AIMPACT_MODELS_DIR` | `./models` | Thư mục host chứa trực tiếp `cc.vi.300.bin`; đặt đường dẫn tuyệt đối nếu model nằm ngoài repo; mount read-only vào `/app/models` |
| `AIMPACT_DATA_DIR` | `./data` | SQLite, Chroma, encryption key, provider state và JWT secret DEV; mount writable vào `/app/data` |
| `AIMPACT_HISTORY_DIR` | `./history` | Dữ liệu history tương thích; mount writable vào `/app/history` |
| `AIMPACT_DOCUMENTS_DIR` | `./documents` | Tài liệu upload; mount writable vào `/app/documents` |

Nếu `AIMPACT_MODELS_DIR` trỏ đúng thư mục chứa model, không cần đổi `MODEL_PATH`: Compose mount thư mục đó vào `/app/models` và mặc định vẫn là `/app/models/cc.vi.300.bin`.

## Mạng 9Router

- Chạy bằng Docker Compose: container gọi 9Router đang chạy trên host qua `http://host.docker.internal:20128/v1`; compose đã đặt default này.
- Chạy DEV trực tiếp, không Docker: backend và 9Router cùng ở host nên dùng `http://127.0.0.1:20128/v1`.

## Ví dụ DEV

```powershell
docker compose up --build
```

Docker DEV không cần `.env`. Nếu chạy backend trực tiếp ngoài container, đặt `JWT_SECRET`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `FRONTEND_ORIGIN` và các đường dẫn như trước.

## Production

Production bắt buộc đặt tường minh `JWT_SECRET`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `FRONTEND_ORIGIN`. Overlay `docker/docker-compose.prod.yml` fail ngay khi biến thiếu hoặc rỗng.

## Sinh JWT secret

PowerShell:

```powershell
[Convert]::ToBase64String([Security.Cryptography.RandomNumberGenerator]::GetBytes(48))
```

Python:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Không dùng `APP_PASSWORD`: đó là biến di sản của Streamlit và không thuộc startup contract mới.

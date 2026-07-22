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
| `NROUTER_BASE_URL` | Không | `http://127.0.0.1:20128/v1` | Override base URL 9Router; Docker dùng `host.docker.internal` |
| `OLLAMA_BASE_URL` | Không | `http://localhost:11434/v1` | Override base URL Ollama |
| `OPENAI_BASE_URL` | Không | `https://api.openai.com/v1` | Override endpoint OpenAI-compatible |
| `XAI_BASE_URL` | Không | `https://api.x.ai/v1` | Override endpoint xAI-compatible |
| `OPENAI_API_KEY` | Khi dùng OpenAI | rỗng | Key OpenAI, không trả ra client |
| `XAI_API_KEY` | Khi dùng Grok | rỗng | Key xAI, không trả ra client |

Các biến `AIMPACT_MODELS_DIR`, `AIMPACT_DATA_DIR`, `AIMPACT_HISTORY_DIR`, `AIMPACT_DOCUMENTS_DIR` chỉ dành cho Docker Compose để đổi thư mục bind-mount trên host; mặc định lần lượt là `./models`, `./data`, `./history`, `./documents`.

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

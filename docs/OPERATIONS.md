# Vận hành

## Chạy nhanh một lệnh (DEV)

1. Đặt FastText model ngoài image tại `models/cc.vi.300.bin` (khoảng 4.5 GB).
2. Từ gốc repo, chạy:

```powershell
docker compose up --build
```

Không cần tạo `.env`. Compose dùng cổng `8000`, admin DEV `admin` / `ChangeMe-Dev-2026`, và bind-mount `models/`, `data/`, `history/`, `documents/`. Admin chỉ được seed khi bảng `users` rỗng; đổi biến sau đó không đổi mật khẩu của tài khoản đã tồn tại.

Khi `JWT_SECRET` trống, entrypoint sinh chuỗi ngẫu nhiên 64 ký tự hex, ghi `data/.jwt_secret` với quyền `600`, rồi đọc lại file này ở các lần restart. Giá trị secret không được in ra log. Nếu đang dùng mật khẩu DEV mặc định, startup in cảnh báo đổi mật khẩu và đặt `JWT_SECRET` riêng cho production nhưng không in mật khẩu.

## Production

Tạo `.env` cục bộ hoặc inject từ secret store, rồi đặt tường minh:

- `JWT_SECRET`: tối thiểu 32 ký tự.
- `ADMIN_USERNAME`: không để rỗng.
- `ADMIN_PASSWORD`: tối thiểu 10 ký tự.
- `FRONTEND_ORIGIN`: origin HTTPS thật của frontend, không dùng giá trị localhost mẫu.

```powershell
Copy-Item .env.example .env
# Điền bốn biến production bắt buộc trong .env
docker compose --env-file .env -f docker-compose.yml -f docker/docker-compose.prod.yml up --build -d
```

Overlay `docker/docker-compose.prod.yml` dùng phép kiểm tra `${VAR:?message}`. Thiếu hoặc để rỗng bất kỳ biến bắt buộc nào thì Compose fail trước khi tạo container với tên biến rõ ràng. Không commit `.env`, không ghi secret vào compose/image, và không đưa secret vào lệnh hoặc log CI.

## Kiểm tra sau deploy

```powershell
Invoke-RestMethod http://localhost:8000/api/health
```

Sau login, kiểm:

- `/api/documents` có workbook đã ingest.
- `/api/stats?sheet=Tong%20hop` trả UCTT 61, XLSC 50, VHKT 6, TỔNG 117.
- Query ngoài phạm vi trả thông báo không bằng chứng và citation rỗng.
- Admin UI tải được danh sách model của provider active.

## Backup

Phải backup cùng thời điểm các phần sau:

- `data/app.db`
- `data/chroma_db/`
- `data/encryption_key.key`
- `data/.jwt_secret` nếu DEV đang dùng secret tự sinh
- `data/providers.state.json`
- `documents/`
- `history/`
- `models/` nếu không có nguồn tải lại đáng tin cậy

PowerShell:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
New-Item -ItemType Directory -Force "backup\$stamp" | Out-Null
Copy-Item data,history,documents -Destination "backup\$stamp" -Recurse
Copy-Item models -Destination "backup\$stamp" -Recurse
Compress-Archive -Path "backup\$stamp\*" -DestinationPath "backup\aimpact-$stamp.zip"
```

Nên dừng container trước backup để SQLite/Chroma nhất quán:

```powershell
docker compose stop app
# backup
docker compose start app
```

## Restore

1. Dừng app.
2. Giữ bản sao thư mục hiện tại.
3. Restore đồng bộ `data/`, `history/`, `documents/`, `models/`.
4. Kiểm quyền đọc/ghi của tài khoản container.
5. Khởi động và gọi `/api/health`.

Không restore Chroma mà thiếu đúng `encryption_key.key`; document text đã mã hóa sẽ không giải mã được.

## Xoay JWT secret

1. Sinh secret mới tối thiểu 32 ký tự.
2. Dừng app.
3. Với PROD, cập nhật `JWT_SECRET` trong secret store/`.env`. Với DEV tự sinh, xóa `data/.jwt_secret` để entrypoint tạo secret mới.
4. Khởi động lại.
5. Người dùng đăng nhập lại; mọi access/refresh JWT cũ mất hiệu lực chữ ký.
6. Admin có thể logout-all các tài khoản nhạy cảm trước khi xoay nếu cần thu hồi phía database ngay.

## Provider và model

- Switch provider: Admin → LLM runtime → Kích hoạt.
- Tải model động: bấm “Tải model”; API gọi `{base_url}/models`.
- Đổi model: chọn model trong dropdown; state được ghi `data/providers.state.json`.
- Local provider trong Docker: dùng `NROUTER_BASE_URL=http://host.docker.internal:20128/v1` hoặc `OLLAMA_BASE_URL=http://host.docker.internal:11434/v1`.
- OpenAI/Grok: đặt key trong environment rồi restart; không nhập key qua UI.

Thêm provider mới yêu cầu thêm profile không-secret vào `api/providers.json`, chỉ định `key_env`/`base_url_env`, rebuild image, sau đó cấu hình key qua environment.

## Threshold

- Mặc định: `0.78`.
- Admin UI thay đổi runtime mà không sửa code/restart.
- Mức thận trọng tối đa khuyến nghị: `0.84`; có thể chặn gần toàn bộ câu ngoài phạm vi nhưng từ chối nhầm khoảng 1/10 câu hợp lệ theo hiệu chỉnh hiện tại.
- State runtime ưu tiên hơn env sau lần tạo `providers.state.json`. Muốn reset theo env, dừng app, backup rồi xóa trường `threshold` hoặc file state.

## Troubleshooting

### Startup báo MODEL_PATH không tồn tại

Kiểm file và mount:

```powershell
Test-Path models\cc.vi.300.bin
docker compose config --quiet
```

### Provider không reachable

- Gọi `/api/health` và `/api/providers/{id}/models`.
- Với local provider trong Docker, không dùng `localhost`; dùng `host.docker.internal` qua biến base URL.
- Kiểm provider process đang listen đúng `/v1` và endpoint `/models`.
- Kiểm firewall/proxy và key env.

### Stats lỗi workbook

Kiểm `STATS_WORKBOOK` tồn tại và sheet đúng tên. Mặc định container dùng workbook đã copy từ repo tại `/app/project_agent/Phu luc 1.xlsx`.

### Upload lỗi

Kiểm extension, `MAX_UPLOAD_MB`, quyền ghi `documents/`, dung lượng disk và log exception của uvicorn. File upload dở dang được xóa khi ingest thất bại.

### PDF scan/OCR

Pipeline hiện không OCR ảnh. PDF chỉ có ảnh có thể không tạo record hữu ích; đây không phải lỗi provider. Chuyển tài liệu thành PDF có text hoặc chờ roadmap OCR/vision.

### Chroma/ONNX footprint

`chromadb==0.5.0` import default embedding module phụ thuộc `onnxruntime` dù AImpact dùng FastText embedding riêng. Không gỡ package thủ công khỏi image hiện tại; việc đó làm import Chroma thất bại.

# API Reference

Base path: `/api`. Trừ `/api/query` (SSE) và tải file, response là JSON. Lỗi chuẩn: `{"detail":"..."}`.

Header xác thực:

```http
Authorization: Bearer <access_token>
```

## Ma trận quyền

| Nhóm | viewer | user | admin |
|---|:---:|:---:|:---:|
| Query, stats, GET documents, hội thoại của mình | ✓ | ✓ | ✓ |
| Upload/xóa tài liệu, provider/model, threshold write, users, audit | — | — | ✓ |

Sai/mất token trả 401; đúng token nhưng sai role trả 403. Hội thoại không thuộc user luôn trả 404.

## Auth

### `POST /api/auth/login` — public

Request:

```json
{"username":"admin","password":"strong-password"}
```

Response 200:

```json
{"access_token":"...","refresh_token":"...","role":"admin","username":"admin"}
```

Sai credential trả 401. Quá 5 lần/phút cho cùng IP + username trả 429.

### `POST /api/auth/refresh` — public

```json
{"refresh_token":"..."}
```

Response trả cặp access/refresh mới và thu hồi refresh cũ. Token hết hạn/đã thu hồi trả 401.

### `POST /api/auth/logout` — authenticated

Thu hồi một refresh:

```json
{"refresh_token":"...","all":false}
```

Thu hồi mọi refresh của user:

```json
{"all":true}
```

### `GET /api/auth/me` — authenticated

Response: `{"username":"operator","role":"user"}`.

## Users — admin

### `GET /api/users`

Danh sách `id`, `username`, `role`, `created_at`. Không bao giờ có `password_hash`.

### `POST /api/users`

```json
{"username":"viewer01","password":"at-least-10-characters","role":"viewer"}
```

Trùng username trả 409.

### `PATCH /api/users/{id}`

Ít nhất một trường:

```json
{"role":"user","password":"new-strong-password"}
```

Đổi role/mật khẩu sẽ thu hồi toàn bộ refresh token của tài khoản. Không cho hạ role admin cuối cùng.

### `DELETE /api/users/{id}`

Không cho tự xóa hoặc xóa admin cuối cùng; trả 409 với lý do.

## Query — viewer/user/admin

### `POST /api/query`

Request:

```json
{
  "query":"Khi mất một lộ điện lưới tại N6 cần làm gì?",
  "conversation_id":"optional-uuid",
  "top_k":10,
  "threshold":0.78
}
```

`threshold` bỏ trống sẽ dùng giá trị runtime trong `providers.state.json`. API retrieve từ Chroma rồi gọi `rag.answer_or_refuse`.

Response media type `text/event-stream`.

Token event:

```text
data: {"type":"token","text":"Thực hiện "}
```

Final có bằng chứng:

```text
data: {"type":"final","citations":[{"filename":"Phu luc 1.xlsx","sheet_name":"VHKT","locator":"VHKT!A12:F12","stt":"6","section_path":"Điện lưới","similarity":0.91,"content":"Nội dung nguyên văn"}]}
```

Final không đủ bằng chứng:

```text
data: {"type":"final","text":"Không tìm thấy thông tin phù hợp trong tài liệu.","citations":[]}
```

Ở nhánh không bằng chứng, LLM không được gọi. Nếu provider lỗi sau khi stream bắt đầu, API gửi `{"type":"error","detail":"Active LLM provider is unavailable"}`.

## Documents

### `GET /api/documents` — viewer/user/admin

Response: `['Phu luc 1.xlsx', 'manual.pdf']`.

### `GET /api/documents/{filename}` — viewer/user/admin

Tải/mở tệp nguồn. Client phải gửi Bearer token; frontend tải blob rồi mở tab mới.

### `POST /api/documents` — admin

Multipart field `file`; hỗ trợ XLSX, PDF, DOCX, TXT, CSV. Giới hạn mặc định 50 MB.

```bash
curl -H "Authorization: Bearer $TOKEN" -F "file=@Phu luc 1.xlsx" http://localhost:8000/api/documents
```

Response: `{"status":"added","added":357,"version":1}` hoặc trạng thái do lõi trả về.

### `DELETE /api/documents/{filename}` — admin

Xóa khỏi Chroma và thư mục documents. Response `{"status":"deleted"}`.

## Stats — viewer/user/admin

### `GET /api/stats?sheet=Tong%20hop`

Response:

```json
{
  "sheet":"Tong hop",
  "rows":[{"STT":1,"SỰ CỐ":"AC","UCTT":28,"XLSC":27,"VHKT":6,"TỔNG":61}],
  "totals":{"UCTT":61,"XLSC":50,"VHKT":6,"TỔNG":117}
}
```

Rows giữ dữ liệu gốc; `totals` loại cột chỉ mục như `STT`, `Số thứ tự`, `index`, `no`.

## Conversations — viewer/user/admin

### `GET /api/conversations`

Chỉ liệt kê hội thoại của user hiện tại.

### `POST /api/conversations`

Request `{"title":"Mất điện N6"}`; title trống dùng `Hội thoại mới`.

### `GET /api/conversations/{id}`

Response gồm metadata và `messages[]` với `role`, `content`, `citations`, `created_at`. Sai owner trả 404 kể cả khi caller là admin.

### `DELETE /api/conversations/{id}`

Chỉ owner xóa được; sai owner trả 404.

## Providers

### `GET /api/providers` — authenticated

Mỗi profile có `id`, `label`, `base_url`, `model`, `key_env`, `active`. `key_env` chỉ là tên biến; giá trị key không được trả.

### `POST /api/providers/active` — admin

```json
{"id":"ollama"}
```

Provider cần key nhưng biến môi trường rỗng trả 400. Thành công ghi audit `provider_switch`.

### `POST /api/providers/{id}/model` — admin

```json
{"model":"llama3.2"}
```

### `GET /api/providers/{id}/models` — authenticated

Gọi `{base_url}/models`, chuẩn hóa OpenAI `data[].id` hoặc `models[]`. Lỗi HTTP/kết nối trả 502 và detail rõ provider nào lỗi.

## Threshold runtime

### `GET /api/settings/threshold` — authenticated

Response `{"threshold":0.78}`.

### `POST /api/settings/threshold` — admin

Request `{"threshold":0.84}`. Giá trị hợp lệ 0..1 và được lưu trong `providers.state.json`.

## Audit — admin

### `GET /api/audit`

Tối đa 500 bản ghi mới nhất:

```json
[{"id":12,"username":"admin","action":"provider_switch","resource_id":"ollama","timestamp":"2026-07-21T10:20:30+00:00"}]
```

Action: `login`, `query`, `upload`, `delete_document`, `provider_switch`, `user_create`, `user_update`, `user_delete`. Không chứa query, answer, document content, token hoặc secret.

## Health — public

### `GET /api/health`

```json
{"fasttext_loaded":false,"index_count":357,"active_provider":"9router","provider_reachable":true}
```

`fasttext_loaded=false` trước lần embed đầu không đồng nghĩa model path thiếu; startup đã kiểm file tồn tại.

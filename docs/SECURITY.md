# Bảo mật

## RBAC

| Role | Khả năng |
|---|---|
| `viewer` | Query, stats, đọc/tải tài liệu, hội thoại của mình |
| `user` | Giống viewer; dành cho người vận hành chuẩn |
| `admin` | Tất cả quyền đọc + upload/xóa tài liệu, provider/model/threshold, user, audit |

Mọi dependency role được enforce tại API; ẩn menu frontend không phải cơ chế bảo mật.

## Password

- Hash bằng Argon2 qua `argon2-cffi`.
- API không có mật khẩu mặc định.
- Password tạo/cập nhật tối thiểu 10 ký tự.
- Hash không xuất hiện trong response hoặc audit.
- Đổi password thu hồi toàn bộ refresh token hiện có của user.

## Token lifecycle

1. Login thành công cấp access token 15 phút và refresh token 7 ngày theo mặc định.
2. Access JWT ký HS256 bằng `JWT_SECRET`; client gửi Bearer token.
3. Refresh token được lưu phía server dưới dạng SHA-256, không lưu raw token.
4. `/auth/refresh` thu hồi token cũ rồi cấp cặp mới; reuse token cũ trả 401.
5. `/auth/logout` thu hồi một refresh; `{"all":true}` thu hồi mọi refresh của user.
6. Xóa user dùng foreign key cascade để xóa refresh/conversation/message.

Xoay `JWT_SECRET` làm mọi JWT hiện tại mất hiệu lực; quy trình ở `OPERATIONS.md`.

## Lưu token phía frontend

- Access token chỉ ở biến JavaScript in-memory.
- Refresh token ở `sessionStorage`, không dùng `localStorage`; đóng tab/browser sẽ xóa theo phiên.
- `sessionStorage` vẫn đọc được nếu có XSS. Vì vậy Markdown phải qua DOMPurify, citation render text, CSP/reverse proxy nên được bật ở môi trường production.
- Thiết kế cookie HttpOnly chưa dùng vì hợp đồng hiện trả refresh token JSON; nên chuyển sang cookie khi triển khai qua domain HTTPS ổn định và có CSRF protection.

## Rate limiting

Login giới hạn 5 attempt/phút theo `IP + username`. Limiter hiện in-process để không thêm hạ tầng. Khi chạy nhiều replica, thay bằng Redis/shared gateway limiter; nếu không, mỗi replica có quota riêng.

## Cách ly hội thoại

Mọi query SQL conversation đều có `user_id = current_user.id`. API trả 404 thay vì 403 cho tài nguyên không thuộc user để không tiết lộ ID có tồn tại. Admin không có endpoint bypass để đọc conversation của user khác.

## Audit

Audit chỉ lưu:

- `user_id`/`username`
- `action`
- `resource_id` không nhạy cảm (conversation UUID, filename, provider ID, user ID)
- timestamp UTC

Audit tuyệt đối không lưu question, answer, citation content, document content, JWT, refresh token, password hoặc provider key.

## Provider secret

`providers.json` chỉ có `key_env`; giá trị đọc từ environment đúng lúc gọi provider. `/api/providers` trả tên biến, không trả giá trị. Không đặt key trong `.env.example`, image, runtime state hoặc log.

## Upload

- Chỉ nhận XLSX/PDF/DOCX/TXT/CSV.
- Dùng `Path(filename).name` chặn traversal.
- Giới hạn kích thước trước khi ingest; file dở dang bị xóa khi lỗi.
- Nội dung chỉ được parser lõi xử lý, không thực thi macro/script.

## CORS và transport

CORS chỉ cho `FRONTEND_ORIGIN`. Ở production, đặt reverse proxy HTTPS, HSTS, giới hạn request body và security headers. Không expose API qua HTTP công cộng.

## Checklist triển khai

- `JWT_SECRET` ngẫu nhiên >= 32 ký tự.
- Admin seed mạnh và đổi qua quy trình nội bộ sau lần đầu.
- File `.env` chỉ quyền đọc cho tài khoản chạy service.
- Backup đồng thời `app.db`, Chroma và `encryption_key.key`.
- Không copy model/data/documents vào image.
- Kiểm `npm audit`, test auth/isolation/gate trước release.

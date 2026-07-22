# Hướng dẫn người dùng

## Đăng nhập

Mở URL nội bộ do quản trị viên cung cấp, nhập username và password. Nếu nhập sai quá nhiều lần, hệ thống khóa thử đăng nhập trong một phút. Đóng phiên bằng nút **Đăng xuất**; đóng tab cũng xóa refresh token của phiên trình duyệt.

## Tra cứu

1. Chọn **Tra cứu**.
2. Tạo **Hội thoại mới** hoặc mở hội thoại cũ.
3. Viết câu hỏi cụ thể, có đối tượng/sự cố/vị trí nếu biết.
4. Nhấn Enter hoặc **Gửi câu hỏi**.
5. Theo dõi câu trả lời stream và panel nguồn.

Ví dụ tốt:

- “Khi mất một lộ điện lưới tại N6 cần thực hiện các bước nào?”
- “Cảnh báo HP của điều hòa được xử lý ra sao?”
- “Quy trình xử lý ACB 4000A tủ LV1 cấp ATS1?”

`Top K` là số kết quả retrieval trước gate; để mặc định 10 trừ khi quản trị viên hướng dẫn.

## Đọc trích dẫn

Mỗi citation hiển thị:

- Tên file.
- Sheet và locator ô/dải ô hoặc trang/đoạn.
- Điểm tương đồng.
- Nội dung verbatim đúng như tài liệu.

Bấm dòng nguồn để mở/tải file. Hãy đối chiếu verbatim trước khi áp dụng thao tác có rủi ro vận hành.

## “Không đủ bằng chứng”

Thông báo **“Không tìm thấy thông tin phù hợp trong tài liệu.”** có nghĩa không có hit đạt ngưỡng. Khi đó hệ thống không gọi LLM và không cố đoán câu trả lời.

Cách xử lý:

1. Viết lại câu hỏi cụ thể hơn bằng thuật ngữ có trong tài liệu.
2. Kiểm tra tài liệu liên quan đã được upload.
3. Liên hệ admin nếu tài liệu thiếu hoặc retrieval thường bỏ sót câu hợp lệ.

Không diễn giải trạng thái này thành “không có sự cố”; nó chỉ nói kho bằng chứng hiện tại chưa đủ.

## Hội thoại riêng tư

Bạn chỉ thấy hội thoại của chính mình. User khác và admin không đọc được nội dung qua API hội thoại thường. Xóa hội thoại bằng nút `×` ở sidebar khi không còn cần.

## Thống kê

Trang **Thống kê** đọc deterministic từ workbook, không dùng LLM. Sheet mặc định `Tong hop`. Bảng totals loại cột STT; workbook mẫu trả:

- UCTT: 61
- XLSC: 50
- VHKT: 6
- TỔNG: 117

## Quản lý tài liệu — admin

1. Mở **Tài liệu**.
2. Thả tệp hoặc chọn file XLSX/PDF/DOCX/TXT/CSV.
3. Chờ ingest hoàn tất và kiểm số record `added`.
4. Xóa file khi chắc chắn không còn cần; thao tác xóa cả index và bản lưu trong documents.

Workbook mẫu `Phu luc 1.xlsx` tạo 357 record khi ingest mới.

## Quản trị — admin

- **LLM runtime:** tải model, chọn model, kích hoạt provider.
- **Ngưỡng bằng chứng:** chỉnh 0..1; mặc định 0.78, khuyến nghị không vượt 0.84 nếu chưa eval lại.
- **Người dùng:** tạo tài khoản, đổi role/password, xóa tài khoản.
- **Audit:** xem ai thực hiện action nào; audit không chứa nội dung câu hỏi/trả lời.

## Giới hạn nội dung hình

Tài liệu chỉ có sơ đồ, biểu đồ hoặc ảnh không được hiểu ở phiên bản này. Hệ thống chỉ dùng text/bảng parser lấy được; không yêu cầu LLM suy đoán nội dung hình.

# Roadmap

Các hạng mục sau được hoãn có chủ đích; không nằm trong phiên bản web hiện tại.

## 1. Hiểu sơ đồ, biểu đồ và ảnh

Hiện parser thuần text/bảng. Trang PDF chỉ có ảnh, sơ đồ điện, biểu đồ hoặc ảnh chụp không được diễn giải.

Hướng phát triển:

- OCR có đo confidence và giữ bounding box/page locator.
- Vision model cho sơ đồ/biểu đồ, nhưng phải tạo citation tới vùng ảnh gốc.
- Gate riêng cho evidence vision; không trộn kết luận vision không kiểm chứng vào text citation.
- Eval trên tài liệu scan thực tế trước khi bật production.

## 2. Nâng retrieval

FastText tổng quát có giới hạn recall và khả năng tách in-scope/out-of-scope. Hướng nâng cấp:

- Đánh giá embedding tiếng Việt/domain mạnh hơn.
- Hybrid BM25 + vector, sau đó rerank.
- Query expansion có kiểm soát cho thuật ngữ viết tắt kỹ thuật.
- Calibrate threshold theo từng collection/source type nếu dữ liệu đủ.
- Giữ nguyên no-answer gate và citation verbatim khi đổi retrieval.

## 3. Mở rộng golden set

Golden set hiện cần thêm dữ liệu thực tế:

- Câu hỏi thật đã ẩn thông tin nhạy cảm từ người vận hành.
- Tài liệu PDF và DOCX, không chỉ workbook mẫu.
- Bộ out-of-scope khó, câu mơ hồ và câu gần miền nhưng không có bằng chứng.
- Metric recall@k, gate precision/recall, citation locator accuracy và verbatim fidelity.
- Regression chạy bắt buộc trước khi đổi embedding, parser, threshold hoặc provider prompt.

## Điều kiện triển khai roadmap

Chỉ đưa vào production khi có test tự động, tài liệu vận hành, đánh giá rò rỉ dữ liệu và bằng chứng rằng thay đổi không làm giảm no-answer safety.

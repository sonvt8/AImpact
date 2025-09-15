# Hệ thống RAG xử lý sự cố Điện/Viễn thông (Hoàng Hoa Thám)

> **Lưu ý quan trọng về phiên bản Python**  
> Dự án và các thư viện đi kèm **chỉ hoạt động ổn định với Python ≤ 3.11**.  
> **Tác giả sử dụng Python 3.11.4** trong môi trường phát triển và kiểm thử. Nếu bạn dùng Python 3.12 trở lên (đặc biệt trên Windows), một số gói như `fasttext`, `onnxruntime`, v.v. có thể không có wheel tương thích và phát sinh lỗi khi cài đặt/chạy.

## 1) Mục đích
Hệ thống **Retrieval-Augmented Generation (RAG)** hỗ trợ kỹ sư kỹ thuật điện/viễn thông tra cứu nhanh **phương án xử lý sự cố** (ƯCTT/XLSC/VHKT) theo tài liệu nội bộ — trọng tâm là file **`Phu luc 1.xlsx`** chứa quy trình tại **Trạm trung tâm Hoàng Hoa Thám**. Ứng dụng ưu tiên tiếng Việt, hiểu các **thuật ngữ chuyên ngành** (ví dụ: MPĐ, ATS, Interlock), và trích dẫn lại nguồn đáp án.

Hệ thống có thể dùng như **mẫu (blueprint)** để triển khai các RAG khác cho tài liệu miền chuyên biệt tương tự.

## 2) Tính năng chính
- **Nạp & đánh chỉ mục tài liệu**: hỗ trợ `.txt`, `.pdf`, `.docx`, `.xlsx`, `.csv`. Hệ thống chia nhỏ (chunk) và chuẩn hóa tiếng Việt trước khi lập chỉ mục.
- **Embedding tiếng Việt bằng FastText** (`cc.vi.300.bin`) – tải tự động bằng `load_model.py` nếu chưa có.
- **Kho tri thức bền vững với ChromaDB (persistent)** – lưu/đọc lâu dài trên đĩa.
- **Mã hóa nội dung** bằng khóa Fernet (tự sinh nếu chưa có) khi lưu trữ; giải mã khi hiển thị.
- **Chọn LLM linh hoạt**:
  - **Ollama – `llama3.2`** (chạy cục bộ, offline);
  - **OpenAI – `gpt-4o-mini`** (online), cần `OPENAI_API_KEY`.
  Ứng dụng tự kiểm tra trạng thái Ollama/Kết nối internet và **fallback** phù hợp.
- **Thống kê từ Excel**: tự nhận diện sheet **`Tong hop`** để hiển thị số lượng ƯCTT/XLSC/VHKT/TỔNG.
- **Lịch sử truy vấn & trích dẫn**: lưu và xuất xem lại truy vấn, câu trả lời, nguồn bằng Streamlit UI.

## 3) Yêu cầu hệ thống
- **Python ≤ 3.11 (khuyến nghị 3.11.4)**.  
- Hệ điều hành: Windows / macOS / Linux.
- Dung lượng trống ~**2GB** cho file model `cc.vi.300.bin`.
- (Tùy chọn) **Tesseract OCR** nếu muốn trích xuất chữ từ PDF scan.
- **LLM**:
  - **Ollama**: cài và chạy `ollama`, có sẵn model `llama3.2`;
  - **Hoặc** cấu hình **OpenAI** với `OPENAI_API_KEY` để dùng `gpt-4o-mini`.

> **Khuyến nghị cho Windows**: Nếu gặp lỗi khi cài các gói native (như `fasttext`, `onnxruntime`), hãy đảm bảo dùng **Python 3.11.x**, hoặc cân nhắc **Conda**/**WSL**. Với trường hợp buộc phải build, cần cài **Microsoft C++ Build Tools** và (nếu cần) toolchain Rust/maturin.

## 4) Cài đặt

### 4.1. Tạo môi trường & cài phụ thuộc
```bash
# Clone dự án
git clone <repository-url>
cd <repository-directory>

# Tạo và kích hoạt venv (Python 3.11.x)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Cài gói phụ thuộc
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
> `requirements.txt` đã liệt kê các gói cốt lõi: `streamlit`, `chromadb`, `openai`, `langchain`, `langgraph`, `pdfplumber`, `pytesseract`, `fasttext`, v.v.

### 4.2. Tải & giải nén model FastText (tự động/hoặc thủ công)
- **Tự động**: Lần chạy đầu, nếu chưa có `./models/cc.vi.300.bin`, ứng dụng sẽ tự gọi script tải model.
- **Thủ công** (tùy chọn):
  ```bash
  python load_model.py
  ```
  Sau khi hoàn tất, thư mục `models/` sẽ chứa `cc.vi.300.bin`.

## 5) Cấu hình `.env`
Tạo file `.env` ở thư mục gốc (cùng cấp `hht_rag_system.py`), ví dụ:
```dotenv
# Thư mục dữ liệu/lịch sử/tài liệu
DATA_DIR=./data
HISTORY_DIR=./history
DOCUMENTS_DIR=./documents

# Đường dẫn model FastText
MODEL_PATH=./models/cc.vi.300.bin

# Mật khẩu truy cập giao diện
APP_PASSWORD=T0mmy

# Ngưỡng tương đồng cosine (0..1)
SIMILARITY_THRESHOLD=0.5

# (Tuỳ chọn) OpenAI API cho chế độ online
OPENAI_API_KEY=sk-...
```
> Nếu không đặt `.env`, chương trình dùng mặc định: `DATA_DIR=data`, `HISTORY_DIR=history`, `DOCUMENTS_DIR=documents`, `MODEL_PATH=./models/cc.vi.300.bin`, `APP_PASSWORD=T0mmy`, `SIMILARITY_THRESHOLD=0.5`.

## 6) Chạy ứng dụng
```bash
streamlit run hht_rag_system.py
# hoặc (giảm reload tệp lớn trên Windows)
streamlit run hht_rag_system.py --server.fileWatcherType=none
```
- Truy cập `http://localhost:8501` → nhập mật khẩu `APP_PASSWORD`.
- Sidebar: chọn **LLM** (`Ollama`/`OpenAI`), mức chi tiết, ngưỡng tương đồng.
- Tải tài liệu (`Phu luc 1.xlsx` …) hoặc đặt vào thư mục `documents/` rồi bấm lập chỉ mục.
- Đặt câu hỏi tiếng Việt; hệ thống trả lời kèm **trích dẫn nguồn**. Nếu có sheet `Tong hop`, phần **“Thống kê sự cố”** sẽ hiển thị.

## 7) Cách sử dụng (khuyến nghị)
- Hỏi theo **ngữ cảnh miền**: “Sự cố 1 lộ điện lưới lộ nổi, hệ thống tự động interlock… xử lý thế nào?”  
- Hỏi theo **thiết bị/khối**: “ACB 4000A tủ LV1 cấp tới ATS1 bị lỗi → bước xử lý?”.  
- Hỏi **thống kê**: “Trong năm vừa qua có bao nhiêu sự cố AC/XLSC/VHKT?” (nếu `Tong hop` có dữ liệu).

## 8) Cấu trúc & các tệp chính
- **`hht_rag_system.py`**: Entry UI Streamlit & pipeline RAG (tiền xử lý, mã hóa/giải mã, truy vấn, trích dẫn, thống kê Excel, chọn LLM/fallback).
- **`load_model.py`**: Tải và giải nén Vietnamese FastText `cc.vi.300.bin`.
- **`Phu luc 1.xlsx`**: Tài liệu phụ lục (nội bộ) dùng để thử nghiệm/triển khai.
- **`requirements.txt`**: Danh sách phụ thuộc Python cho dự án.

## 9) Sự cố thường gặp
- **Không cài được gói trên Windows** (đặc biệt Python 3.12): hãy dùng **Python 3.11.4**. Nếu vẫn cần build native, cài **MSVC Build Tools**, và thử lại.  
- **`onnxruntime` báo lỗi DLL**: cài/ cập nhật **Microsoft Visual C++ Redistributable (x64)** và dùng phiên bản ORT tương thích với NumPy hiện tại; khuyến nghị bám Python 3.11.x.
- **OCR không hoạt động**: cài Tesseract và đảm bảo `pytesseract` nhìn thấy `tesseract.exe` (Windows) hoặc binary (macOS/Linux).
- **Không kết nối được Ollama**: đảm bảo dịch vụ đang chạy (`ollama serve`), model `llama3.2` đã pull (`ollama run llama3.2`).

## 10) Mở rộng
- Thêm bộ quy tắc chuẩn hóa thuật ngữ cho các miền khác (y tế, sản xuất…).
- Thay embedding (VD: Sentence Transformers) hoặc fine-tune tương thích tiếng Việt.
- Tích hợp giám sát real-time từ API/SCADA và phát hiện bất thường.

## 11) Giấy phép & liên hệ
- **License**: MIT (cập nhật theo file `LICENSE` của bạn).  
- **Liên hệ/Báo lỗi**: Tạo issue trong repository hoặc liên hệ maintainer.

---

**Ghi chú cuối**: Để có trải nghiệm mượt nhất, **hãy dùng Python 3.11.4**, tạo venv sạch, cài `requirements.txt`, chạy `load_model.py`, sau đó `streamlit run hht_rag_system.py`. Khi cần chế độ online, đặt `OPENAI_API_KEY`; khi offline, đảm bảo **Ollama** đang chạy với model `llama3.2`.
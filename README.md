
# RAG System for Telecom Incident Handling

## Purpose
This Retrieval-Augmented Generation (RAG) system is designed to assist telecom data center technicians in quickly resolving electrical and mechanical incidents. It is optimized for querying the `Phu luc 1.xlsx` document, which contains detailed incident handling procedures for the Hoàng Hoa Thám central station. The system processes Vietnamese technical documents, leveraging specialized telecom terminology (e.g., MPĐ, ATS, Interlock) to provide accurate, context-aware responses. By streamlining access to predefined solutions, it enables technicians to address issues efficiently.

The system can serve as a blueprint for building similar RAG systems tailored to other domain-specific documents or extended to handle additional scenarios using advanced techniques.

## Prerequisites
- **Python Version**: Use Python < 3.11 to ensure compatibility with required libraries.
- **Operating System**: Compatible with Linux, macOS, or Windows.
- Sufficient disk space for model files (~2GB for `cc.vi.300.bin`).
- **LLM Requirement**: Either start an Ollama server locally (run `ollama run llama3.2`) or provide an OpenAI API key in the `.env` file for online LLM access.

## Installation Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**
   Install the required Python libraries listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```
   Note: Ensure you have pip for Python < 3.11. If issues arise with `underthesea`, consider installing it separately:
   ```bash
   pip install underthesea==6.8.0
   ```

3. **Download and Extract FastText Model**
   Run the `load_model.py` script to download and decompress the Vietnamese FastText model (`cc.vi.300.bin`):
   ```bash
   python load_model.py
   ```
   This creates a `models` directory with the `cc.vi.300.bin` file.

4. **Configure Environment Variables**
   Create a `.env` file by copying the provided `.env.example`:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to set the following:
   ```env
   # API TOKENS
   OPENAI_API_KEY=<your-openai-api-key>  # Optional, leave empty for Ollama

   # LOCAL PARAMETERS
   DATA_DIR=data
   HISTORY_DIR=history
   DOCUMENTS_DIR=documents
   MODEL_PATH=./models/cc.vi.300.bin
   APP_PASSWORD=your_password
   SIMILARITY_THRESHOLD=0.5
   ```
   Replace `your_password` with a secure password for authentication. If using OpenAI, provide a valid API key; otherwise, ensure the Ollama server is running.

5. **Run the Application**
   Start the Streamlit application and redirect logs to `log.txt`:
   ```bash
   streamlit run hht_rag_system.py --server.fileWatcherType=none > log.txt 2>&1
   ```
   The system creates directories (`data`, `history`, `documents`) and log files as needed.  
   Access the web interface at [http://localhost:8501](http://localhost:8501).

## Usage
- **Authentication**: Enter the password set in `APP_PASSWORD` to access the system.
- **Upload Documents**: Upload `Phu luc 1.xlsx` via the sidebar to initialize the database.
- **Querying**: Submit queries in Vietnamese using telecom-specific terms (e.g., "Sự cố mất một lộ điện lưới lộ nổi ở N6, cần làm gì?" or "Lỗi ACB 4000A tủ LV1 cấp tới ATS1, cách xử lý?").
   The system retrieves relevant procedures from `Phu luc 1.xlsx` and generates detailed responses, including:
   - **Tình huống**: Incident description (building, system).
   - **Dấu hiệu nhận biết**: Specific indicators.
   - **Giải pháp thực hiện**: Step-by-step solutions.
   - **Nguyên nhân, Mức độ sự cố, Nguồn vật tư, Ghi chú**: Additional details if available.
- **View Summaries**: Check incident statistics (e.g., total AC, UPS incidents) in the "Xem thống kê sự cố" section.
- **Export History**: Download query history as a PDF from the "Lịch sử truy vấn" section.

### Example Queries
- "Sự cố mất một lộ điện lưới lộ nổi ở N6, cần làm gì?"
- "Lỗi ACB 4000A tủ LV1 cấp tới ATS1, cách xử lý?"
- "Điều hòa đẩy cảnh báo HP, các bước khắc phục?"
- "Trong năm vừa qua có bao nhiêu sự cố AC?"

## Notes
- **Language**: All queries and documents must be in Vietnamese for optimal performance.
- **Terminology**: The system is optimized for telecom terms (e.g., MPĐ, ĐHCX, UDB/PDU) defined in `hht_rag_system.py`.
- **LLM Setup**: Ensure the Ollama server is running (`ollama run llama3.2`) or an OpenAI API key is configured in `.env` before starting the application.
- **Limitations**: Currently tailored for `Phu luc 1.xlsx`. Other documents may require preprocessing to align with the system’s schema.

## Extending the System
The system is a foundation for domain-specific RAG applications. To adapt it:
- **New Documents**: Modify `DocumentProcessor` in `hht_rag_system.py` to handle different document structures.
- **Domains**: Update `TELECOM_KEYWORDS` and `TERMINOLOGY_MAPPING` for other fields (e.g., healthcare, manufacturing).

### Enhancements:
- Integrate advanced LLMs (e.g., Grok 3.5 when available).
- Add multi-modal support (e.g., image-based incident reports).
- Implement real-time monitoring via APIs for live incident data.
- Enhance embeddings with domain-specific fine-tuning.

## Troubleshooting
- **Library Installation Issues**: If `underthesea` fails, try installing it in a virtual environment with Python 3.10.
- **Model Download Errors**: Ensure internet connectivity and sufficient disk space for `cc.vi.300.bin`.
- **Query Failures**: Verify that `Phu luc 1.xlsx` is uploaded and use precise telecom terminology.
- **LLM Errors**: Check that the Ollama server is running or the OpenAI API key is valid.
- **Logs**: Check `log.txt` for detailed error messages.

## License
This project is licensed under the MIT License. See LICENSE for details.

## Contact
For support, open an issue on the repository or contact the maintainers.

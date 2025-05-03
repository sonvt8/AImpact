import os
import sys
import streamlit as st
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List
import pdfplumber
import shutil
from datetime import datetime
from functools import lru_cache
import hashlib
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import SQLChatMessageHistory
from cryptography.fernet import Fernet
import requests
import warnings
import torch
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import io

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

if sys.version_info >= (3, 13):
    st.error("Python 3.13 không được hỗ trợ. Vui lòng sử dụng Python 3.11 hoặc 3.12.")
    st.stop()

if torch.__version__ < "2.3.1":
    st.error(f"Phiên bản torch ({torch.__version__}) không đủ mới. Vui lòng cài torch>=2.3.1.")
    st.stop()

def check_ollama():
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        return response.status_code == 200
    except:
        return False

DATA_DIR = "data"
HISTORY_DIR = "history"
DOCUMENTS_DIR = "Documents"
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
ENCRYPTION_KEY_PATH = os.path.join(DATA_DIR, "encryption_key.key")
HISTORY_DB_PATH = os.path.join(HISTORY_DIR, "query_history.db")
MODEL_PATH = "./models/all-MiniLM-L6-v2"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

def display_message(message, message_type="error"):
    if "message_placeholder" not in st.session_state:
        st.session_state.message_placeholder = st.empty()

    if "last_message" not in st.session_state or st.session_state.last_message != message:
        st.session_state.last_message = message
        message_id = f"message_{hashlib.md5(message.encode()).hexdigest()}"

        color = {
            "error": "#ff4b4b",
            "warning": "#ffcc00",
            "info": "#28a745"
        }.get(message_type, "#ff4b4b")

        html_message = f"""
        <div id="{message_id}" style="padding: 10px; margin-bottom: 10px; border-radius: 5px; color: white; 
        background-color: {color};">
            {message}
        </div>
        <script>
            setTimeout(function() {{
                var elem = document.getElementById("{message_id}");
                if (elem) {{
                    elem.parentNode.removeChild(elem);
                }}
            }}, 5000);
        </script>
        """

        st.session_state.message_placeholder.markdown(html_message, unsafe_allow_html=True)

def check_existing_data(persist_directory: str = CHROMA_DB_PATH):
    return os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

    def load_and_split(self, pdf_file) -> List[str]:
        saved_filename = pdf_file.name
        saved_filepath = os.path.join(DOCUMENTS_DIR, saved_filename)

        if not os.path.exists(saved_filepath):
            with open(saved_filepath, "wb") as f:
                f.write(pdf_file.read())

        chunks = []
        try:
            with pdfplumber.open(saved_filepath) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        chunks.append(text.strip())
                    tables = page.extract_tables()
                    for table in tables:
                        table_text = "\n".join([",".join(str(cell) if cell is not None else "" for cell in row) for row in table])
                        if table_text.strip():
                            chunks.append(table_text.strip())
        except Exception as e:
            display_message(f"Lỗi xử lý PDF: {str(e)}", "error")
            return []

        splits = self.text_splitter.create_documents(chunks)
        return [chunk.page_content.strip() for chunk in splits if chunk.page_content.strip()]

class SentenceEmbedding:
    def __init__(self, model_path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if not os.path.exists(model_path):
            display_message(f"Mô hình 'all-MiniLM-L6-v2' không tìm thấy tại {model_path}.", "error")
            display_message("Vui lòng chạy: from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2'); model.save('./models/all-MiniLM-L6-v2')", "info")
            st.stop()
        try:
            self.model = SentenceTransformer(model_path)
        except Exception as e:
            display_message(f"Lỗi tải mô hình sentence-transformers: {str(e)}", "error")
            st.stop()

    def __call__(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.model.encode(texts, show_progress_bar=False).tolist()
        except Exception as e:
            display_message(f"Lỗi tạo embedding: {str(e)}", "error")
            return []

class ChromaDBManager:
    def __init__(self, persist_directory: str = CHROMA_DB_PATH):
        self.key_path = ENCRYPTION_KEY_PATH
        os.makedirs(os.path.dirname(self.key_path), exist_ok=True)
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as f:
                self.key = f.read()
        else:
            self.key = Fernet.generate_key()
            with open(self.key_path, "wb") as f:
                f.write(self.key)
        self.cipher = Fernet(self.key)
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="viettel_docs")

    def add(self, texts: List[str], embeddings: List[List[float]], filename: str):
        encrypted_texts = [self.cipher.encrypt(text.encode()).decode() for text in texts]
        ids = [f"doc_{hashlib.md5((filename + str(i)).encode()).hexdigest()}" for i in range(len(texts))]
        metas = [{"source": f"Đoạn {i+1}", "filename": filename, "upload_date": datetime.now().isoformat()} for i in range(len(texts))]
        self.collection.add(
            ids=ids,
            documents=encrypted_texts,
            metadatas=metas,
            embeddings=embeddings,
        )

    def query(self, query_embedding: List[float], top_k: int = 3):
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        res["documents"] = [[self.cipher.decrypt(doc.encode()).decode() for doc in docs] for docs in res["documents"]]
        return res

class AnswerGenerator:
    def __init__(self, model_type: str, role: str = "Expert"):
        self.model_type = model_type
        self.role = role
        if model_type == "openai":
            try:
                self.model = ChatOpenAI(model="gpt-4o-mini")
            except Exception as e:
                display_message(f"Lỗi kết nối OpenAI: {str(e)}. Vui lòng kiểm tra API key hoặc kết nối internet.", "error")
                st.stop()
        elif model_type == "ollama":
            if not check_ollama():
                display_message("Ollama server không hoạt động. Vui lòng chạy 'ollama run llama3.2'.", "error")
                st.stop()
            self.model = ChatOllama(
                base_url="http://localhost:11434",
                model="llama3.2",
                temperature=0.3,
                max_tokens=2000
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def generate_answer(self, question: str, context: str, citations: str) -> str:
        prompt = PromptTemplate(
            input_variables=["question", "context", "citations", "role"],
            template="""You are an expert in telecommunications and Data Center operations, answering at a {role} level. Given the following question, context, and citations, provide a concise, accurate, and technical answer using domain-specific terminology:

            Question: {question}

            Context: {context}

            Citations: {citations}

            Answer:"""
        )
        try:
            formatted_prompt = prompt.format(question=question, context=context, citations=citations, role=self.role)
            response = self.model.invoke(formatted_prompt)
            return response.content.strip()
        except Exception as e:
            display_message(f"Error generating answer: {str(e)}", "error")
            return ""

def get_session_history(session_id: str):
    return SQLChatMessageHistory(session_id=session_id, connection=f"sqlite:///{HISTORY_DB_PATH}")

def wrap_text(text, width, canvas_obj, font_name="Helvetica", font_size=12):
    """Hàm để bọc văn bản trong PDF nếu vượt quá chiều rộng."""
    lines = []
    current_line = ""
    canvas_obj.setFont(font_name, font_size)
    for word in text.split():
        test_line = current_line + word + " "
        if canvas_obj.stringWidth(test_line, font_name, font_size) <= width:
            current_line = test_line
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.strip())
    return lines

def main():
    st.set_page_config(page_title="RAG Điện Viễn Thông (Nội bộ)", layout="wide")

    if "message_placeholder" not in st.session_state:
        st.session_state.message_placeholder = st.empty()

    st.title("📄 Hệ thống truy vấn tài liệu thông minh")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        password = st.text_input("Nhập mật khẩu:", type="password", value="")
        if st.button("Xác thực"):
            if password == "T0mmy":
                st.session_state.authenticated = True
                st.rerun()
            else:
                display_message("Mật khẩu không đúng. Vui lòng thử lại.", "error")
        return

    if "processor" not in st.session_state:
        st.session_state.processor = None
    if "embedder" not in st.session_state:
        st.session_state.embedder = None
    if "chroma" not in st.session_state:
        st.session_state.chroma = None
    if "query_history" not in st.session_state:
        st.session_state.query_history = get_session_history("user_default")
    if "has_db_notified" not in st.session_state:
        st.session_state.has_db_notified = False

    st.sidebar.header("Cấu hình")
    llm_type = st.sidebar.radio(
        "Chọn mô hình LLM:",
        ["ollama", "openai"],
        format_func=lambda x: "Ollama Llama3.2 (Offline)" if x == "ollama" else "OpenAI GPT-4o-mini (Online)",
    )
    role = st.sidebar.radio("Mức độ chi tiết:", ["Beginner", "Expert", "PhD"])

    has_db = check_existing_data()
    if has_db:
        display_message("CSDL đã có dữ liệu. Bạn có thể bắt đầu truy vấn.", "info")
        # st.sidebar.info("CSDL đã có dữ liệu. Bạn có thể bắt đầu truy vấn.")
        if st.session_state.chroma is None:
            st.session_state.chroma = ChromaDBManager(persist_directory=CHROMA_DB_PATH)
        if st.session_state.embedder is None:
            st.session_state.embedder = SentenceEmbedding(model_path=MODEL_PATH)
        st.session_state.has_db_notified = False
    else:
        if not st.session_state.has_db_notified:
            display_message("Chưa có dữ liệu. Vui lòng tải lên tài liệu PDF.", "warning")
            st.session_state.has_db_notified = True

    st.sidebar.subheader("Quản lý tài liệu")
    pdf_file = st.sidebar.file_uploader("Tải lên file PDF", type="pdf")
    if pdf_file:
        with st.spinner("Đang xử lý tài liệu…"):
            try:
                proc = DocumentProcessor()
                chunks = proc.load_and_split(pdf_file)
                if not chunks:
                    display_message("Không thể trích xuất văn bản từ PDF.", "error")
                    return
                embedder = SentenceEmbedding(model_path=MODEL_PATH)
                embeddings = embedder(chunks)
                chroma = ChromaDBManager(persist_directory=CHROMA_DB_PATH)
                chroma.add(chunks, embeddings, filename=pdf_file.name)
                st.session_state.processor = proc
                st.session_state.embedder = embedder
                st.session_state.chroma = chroma
                display_message("Đã xử lý và lưu tài liệu thành công!", "info")
                st.session_state.has_db_notified = False
            except Exception as e:
                display_message(f"Lỗi xử lý tài liệu: {str(e)}", "error")

    if has_db and st.sidebar.button("Xóa tất cả tài liệu"):
        try:
            st.session_state.chroma.client.delete_collection("viettel_docs")
            st.session_state.chroma = None
            st.session_state.embedder = None
            st.session_state.processor = None
            for file in os.listdir(DOCUMENTS_DIR):
                file_path = os.path.join(DOCUMENTS_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            display_message("Đã xóa tất cả tài liệu!", "info")
            st.session_state.has_db_notified = False
        except Exception as e:
            display_message(f"Lỗi xóa tài liệu: {str(e)}", "error")

    query = st.text_input("Nhập câu hỏi (tiếng Việt):")
    if query:
        if st.session_state.chroma is None or st.session_state.embedder is None:
            display_message("Chưa có dữ liệu. Vui lòng tải lên tài liệu PDF.", "warning")
            return
        with st.spinner("Đang tạo câu trả lời…"):
            try:
                @lru_cache(maxsize=100)
                def cached_query(query: str, top_k: int) -> tuple:
                    emb = st.session_state.embedder([query])[0]
                    res = st.session_state.chroma.query(emb, top_k)
                    return res["documents"][0], res["metadatas"][0], res["distances"][0]

                docs, metas, dists = cached_query(query, top_k=3)
                context = " ".join(docs)
                citations = "\n".join([f"**{meta['source']}** ({meta['filename']}, score={dist:.4f}):\n> {doc}" for meta, doc, dist in zip(metas, docs, dists)])

                answer_generator = AnswerGenerator(model_type=llm_type, role=role)
                final_answer = answer_generator.generate_answer(query, context, citations)

                if final_answer.startswith("Error generating answer"):
                    display_message(final_answer, "error")
                    return

                st.session_state.query_history.add_user_message(query)
                st.session_state.query_history.add_ai_message(final_answer)

                st.markdown("### Câu trả lời cuối cùng:")
                st.write(final_answer)

                with st.expander("📚 Xem trích dẫn nguồn"):
                    st.markdown(citations, unsafe_allow_html=True)

                if st.button("Xuất câu trả lời"):
                    buffer = io.BytesIO()
                    c = canvas.Canvas(buffer, pagesize=letter)
                    width, height = letter

                    # Kiểm tra font DejaVuSans
                    font_name = "Helvetica"  # Font mặc định nếu không tìm thấy DejaVuSans
                    font_available = os.path.exists("DejaVuSans.ttf")
                    if font_available:
                        try:
                            pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))
                            font_name = "DejaVuSans"
                        except Exception as e:
                            display_message(f"Lỗi tải font DejaVuSans: {str(e)}. Sẽ sử dụng font mặc định (Helvetica). Tiếng Việt có thể không hiển thị đúng.", "warning")
                    else:
                        display_message(
                            "Không tìm thấy file font DejaVuSans.ttf. Sẽ sử dụng font mặc định (Helvetica). "
                            "Tiếng Việt có thể không hiển thị đúng. Vui lòng tải font DejaVuSans.ttf và đặt vào thư mục dự án.",
                            "warning"
                        )

                    y_position = height - inch  # Vị trí bắt đầu từ trên xuống

                    # Tiêu đề: Câu hỏi
                    c.setFont(font_name, 16)
                    c.drawString(inch, y_position, "Câu hỏi:")
                    y_position -= 0.5 * inch

                    # Nội dung câu hỏi
                    c.setFont(font_name, 12)
                    query_lines = wrap_text(query, width - 2 * inch, c, font_name, 12)
                    for line in query_lines:
                        if y_position < inch:
                            c.showPage()
                            c.setFont(font_name, 12)
                            y_position = height - inch
                        c.drawString(inch, y_position, line)
                        y_position -= 0.3 * inch

                    # Tiêu đề: Câu trả lời
                    y_position -= 0.5 * inch
                    if y_position < inch:
                        c.showPage()
                        c.setFont(font_name, 16)
                        y_position = height - inch
                    c.setFont(font_name, 16)
                    c.drawString(inch, y_position, "Câu trả lời:")
                    y_position -= 0.5 * inch

                    # Nội dung câu trả lời
                    c.setFont(font_name, 12)
                    answer_lines = wrap_text(final_answer, width - 2 * inch, c, font_name, 12)
                    for line in answer_lines:
                        if y_position < inch:
                            c.showPage()
                            c.setFont(font_name, 12)
                            y_position = height - inch
                        c.drawString(inch, y_position, line)
                        y_position -= 0.3 * inch

                    # Tiêu đề: Trích dẫn
                    y_position -= 0.5 * inch
                    if y_position < inch:
                        c.showPage()
                        c.setFont(font_name, 16)
                        y_position = height - inch
                    c.setFont(font_name, 16)
                    c.drawString(inch, y_position, "Trích dẫn:")
                    y_position -= 0.5 * inch

                    # Nội dung trích dẫn
                    c.setFont(font_name, 12)
                    citations_lines = wrap_text(citations, width - 2 * inch, c, font_name, 12)
                    for line in citations_lines:
                        if y_position < inch:
                            c.showPage()
                            c.setFont(font_name, 12)
                            y_position = height - inch
                        c.drawString(inch, y_position, line)
                        y_position -= 0.3 * inch

                    c.save()
                    buffer.seek(0)
                    st.download_button("Tải PDF", buffer, file_name="answer.pdf", mime="application/pdf")
            except Exception as e:
                display_message(f"Lỗi truy vấn: {str(e)}", "error")

    with st.expander("📜 Lịch sử truy vấn"):
        messages = st.session_state.query_history.messages
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                user_msg = messages[i]
                ai_msg = messages[i + 1]
                user_content = user_msg.content.replace("\n", "<br>")
                ai_content = ai_msg.content.replace("\n", "<br>")
                st.markdown(
                    """
                    <div style="background-color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        👤 <strong>Người dùng:</strong> {user_content}
                    </div>
                    """.format(user_content=user_content),
                    unsafe_allow_html=True
                )
                st.markdown(
                    """
                    <div style="background-color: #d3d3d3; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        🤖 <strong>Trợ lý:</strong> {ai_content}
                    </div>
                    """.format(ai_content=ai_content),
                    unsafe_allow_html=True
                )
                if i + 2 < len(messages):
                    st.markdown("---")
            else:
                user_msg = messages[i]
                user_content = user_msg.content.replace("\n", "<br>")
                st.markdown(
                    """
                    <div style="background-color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        👤 <strong>Người dùng:</strong> {user_content}
                    </div>
                    """.format(user_content=user_content),
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()
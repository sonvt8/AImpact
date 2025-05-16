import os
import sys
import asyncio
import streamlit as st
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
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
import re
import numpy as np
import logging
import unicodedata
import pytesseract
from PIL import Image
from retry import retry
from dotenv import load_dotenv
import aiohttp
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = os.getenv("DATA_DIR", "data")
HISTORY_DIR = os.getenv("HISTORY_DIR", "history")
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "Documents")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
ENCRYPTION_KEY_PATH = os.path.join(DATA_DIR, "encryption_key.key")
HISTORY_DB_PATH = os.path.join(HISTORY_DIR, "query_history.db")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/multilingual-e5-large")
PASSWORD = os.getenv("APP_PASSWORD", "secure_default_password")
SIMILARITY_THRESHOLD_DEFAULT = float(os.getenv("SIMILARITY_THRESHOLD", 0.6))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# Load telecom keywords from file
TELECOM_KEYWORDS_FILE = os.getenv("TELECOM_KEYWORDS_FILE", "telecom_keywords.txt")
if os.path.exists(TELECOM_KEYWORDS_FILE):
    with open(TELECOM_KEYWORDS_FILE, "r", encoding="utf-8") as f:
        TELECOM_KEYWORDS = [line.strip() for line in f if line.strip()]
else:
    TELECOM_KEYWORDS = [
        "băng thông", "hạ tầng mạng", "định tuyến", "giao thức", "tần số", "kênh truyền",
        "đồng bộ", "độ trễ", "tốc độ truyền", "mạng 5G", "IP", "MPLS", "VPN", "firewall",
        "đám mây", "trung tâm dữ liệu", "Data Center", "cáp quang", "viễn thông", "trạm BTS",
        "ACB", "tủ điện", "máy phát", "SYN", "ATS", "LVSB", "LVGB"
    ]

# Setup logging
logging.basicConfig(level=logging.INFO, filename='rag_system.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Version checks
if sys.version_info >= (3, 13):
    st.error("Python 3.13 không được hỗ trợ. Vui lòng sử dụng Python 3.11 hoặc 3.12.")
    st.stop()
if torch.__version__ < "2.3.1":
    st.error(f"Phiên bản torch ({torch.__version__}) không đủ mới. Vui lòng cài torch>=2.3.1.")
    st.stop()

# Directory setup
for dir_path in [DATA_DIR, HISTORY_DIR, DOCUMENTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

async def check_ollama():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434", timeout=5) as response:
                return response.status == 200
    except Exception as e:
        logger.error(f"Error checking Ollama server: {str(e)}")
        return False

def display_message(message: str, message_type: str = "error"):
    if "message_placeholder" not in st.session_state:
        st.session_state.message_placeholder = st.empty()
    if "last_message" not in st.session_state or st.session_state.last_message != message:
        st.session_state.last_message = message
        message_id = f"message_{hashlib.md5(message.encode()).hexdigest()}"
        color = {"error": "#ff4b4b", "warning": "#ffcc00", "info": "#28a745"}.get(message_type, "#ff4b4b")
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

def check_existing_data(persist_directory: str = CHROMA_DB_PATH) -> bool:
    try:
        if not os.path.exists(persist_directory):
            return False
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_or_create_collection(name="viettel_docs")
        return collection.count() > 0
    except Exception as e:
        logger.error(f"Error checking existing data: {str(e)}")
        return False

def preprocess_text(text: str) -> str:
    original_text = text
    text = re.sub(r'\b[-=]{3,}\b', ' ', text)
    text = re.sub(r'\b\d{1,3}(?:,\d{1,3})+(?:\.\d+)?\b', ' ', text)
    text = re.sub(r'\|{2,}', ' ', text)
    matches = re.findall(r'(\b\w+\b(?:\s+\b\w+\b)*\s*\\?&?\s*\b\w+\b)(?:\s*\1)+', text)
    for match in matches:
        text = re.sub(r'(\b' + re.escape(match) + r'\s*)+', match + ' ', text)
    text = unicodedata.normalize('NFC', text)
    for keyword in TELECOM_KEYWORDS:
        placeholder = f"__{hashlib.md5(keyword.encode()).hexdigest()}__"
        text = text.replace(keyword, placeholder)
    replacements = {
        r'\bCHÉ PÓ\b': 'CHẾ ĐỘ', r'\bVIĖC\b': 'VIỆC', r'\bTÙ\b': 'TỪ',
        r'\be\b': 'ệ', r'\bo\b': 'ộ', r'\bO\b': 'Ộ', r'\bu\b': 'ụ',
        r'\bd\b': 'đ', r'\bD\b': 'Đ', r'\ba\b': 'ạ', r'\bi\b': 'ị',
    }
    for wrong, correct in replacements.items():
        text = re.sub(wrong, correct, text)
    for keyword in TELECOM_KEYWORDS:
        placeholder = f"__{hashlib.md5(keyword.encode()).hexdigest()}__"
        text = text.replace(placeholder, keyword)
    if text != original_text:
        logger.info(f"Before preprocessing: {original_text[:100]}...")
        logger.info(f"After preprocessing: {text[:100]}...")
    return text.strip()

class DocumentProcessor:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

    async def load_and_split(self, pdf_file) -> Tuple[List[str], List[int]]:
        saved_filename = pdf_file.name
        saved_filepath = os.path.join(DOCUMENTS_DIR, saved_filename)
        if not os.path.exists(saved_filepath):
            with open(saved_filepath, "wb") as f:
                f.write(pdf_file.read())
        full_text = ""
        page_boundaries = []
        current_pos = 0
        try:
            with pdfplumber.open(saved_filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if not text.strip():
                        logger.info(f"No text extracted from page {page_num}, trying OCR...")
                        try:
                            image = page.to_image(resolution=300).original.convert("RGB")
                            text = pytesseract.image_to_string(image, lang='vie').strip()
                            logger.info(f"OCR success for page {page_num}: {text[:100]}...")
                        except Exception as e:
                            logger.error(f"OCR error for page {page_num}: {str(e)}")
                            text = ""
                    tables = page.extract_tables()
                    table_text = "\n".join([",".join(str(cell) if cell is not None else "" for cell in row) for table in tables for row in table]) if tables else ""
                    page_content = (text + "\n" + table_text).strip()
                    if page_content:
                        page_content = preprocess_text(page_content)
                        full_text += page_content + "\n"
                        page_boundaries.append((current_pos, page_num))
                        current_pos += len(page_content) + 1
                    else:
                        logger.warning(f"No content on page {page_num} after processing.")
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            display_message(f"PDF processing error: {str(e)}", "error")
            return [], []
        if not full_text.strip():
            logger.error("No text extracted from PDF.")
            return [], []
        splits = self.text_splitter.create_documents([full_text])
        chunks = [chunk.page_content.strip() for chunk in splits if chunk.page_content.strip()]
        page_numbers = []
        for chunk in splits:
            chunk_text = chunk.page_content.strip()
            if not chunk_text:
                continue
            chunk_start = full_text.index(chunk_text)
            chunk_page = page_boundaries[0][1]
            for pos, page_num in page_boundaries:
                if chunk_start >= pos:
                    chunk_page = page_num
                else:
                    break
            page_numbers.append(chunk_page)
        return chunks, page_numbers

class SentenceEmbedding:
    def __init__(self, model_path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if not os.path.exists(model_path):
            display_message(f"Model not found at {model_path}.", "error")
            st.stop()
        try:
            self.model = SentenceTransformer(model_path)
        except Exception as e:
            display_message(f"Error loading sentence-transformers model: {str(e)}", "error")
            st.stop()

    def __call__(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True, batch_size=32)
            return embeddings.tolist()
        except Exception as e:
            display_message(f"Embedding generation error: {str(e)}", "error")
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

    def add(self, texts: List[str], embeddings: List[List[float]], filename: str, page_numbers: List[int]):
        encrypted_texts = [self.cipher.encrypt(text.encode()).decode() for text in texts]
        ids = [f"doc_{hashlib.md5((filename + str(i)).encode()).hexdigest()}" for i in range(len(texts))]
        metas = [{"source": f"Đoạn {i+1}", "filename": filename, "page_number": page_numbers[i], "upload_date": datetime.now().isoformat()} for i in range(len(texts))]
        self.collection.add(ids=ids, documents=encrypted_texts, metadatas=metas, embeddings=embeddings)

    def query(self, query_embedding: List[float], top_k: int = 7):
        res = self.collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "metadatas", "distances"])
        res["documents"] = [[self.cipher.decrypt(doc.encode()).decode() for doc in docs] for docs in res["documents"]]
        distances = np.array(res["distances"])
        similarities = 1 - distances
        res["distances"] = similarities.tolist()
        logger.info(f"Cosine similarities for query: {res['distances']}")
        return res

    def list_documents(self) -> List[str]:
        try:
            results = self.collection.get(include=["metadatas"])
            return list(set(meta["filename"] for meta in results["metadatas"]))
        except Exception as e:
            display_message(f"Error listing documents: {str(e)}", "error")
            return []

class AnswerGenerator:
    def __init__(self, role: str = "Expert"):
        self.role = role
        self.model = None
        self.model_type = None

    async def initialize(self, model_type: str):
        self.model_type = model_type
        if model_type == "openai":
            try:
                self.model = ChatOpenAI(model="gpt-4o-mini")
            except Exception as e:
                display_message(f"OpenAI connection error: {str(e)}. Check API key or internet.", "error")
                st.stop()
        elif model_type == "ollama":
            if not await check_ollama():
                display_message("Ollama server not running. Run 'ollama run llama3.2'.", "error")
                st.stop()
            self.model = ChatOllama(base_url="http://localhost:11434", model="llama3.2", temperature=0.3, max_tokens=2000)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @retry(tries=3, delay=1, backoff=2)
    def generate_answer(self, question: str, context: str, citations: str) -> str:
        prompt = PromptTemplate(
            input_variables=["question", "context", "citations", "role"],
            template="""Bạn là một chuyên gia trong lĩnh vực viễn thông và vận hành Data Center, trả lời ở mức độ {role}. Dựa trên câu hỏi, ngữ cảnh và trích dẫn sau đây, hãy cung cấp câu trả lời ngắn gọn, chính xác và mang tính kỹ thuật, sử dụng thuật ngữ chuyên ngành viễn thông nếu có:

            Câu hỏi: {question}

            Ngữ cảnh: {context}

            Trích dẫn: {citations}

            Nếu ngữ cảnh hoặc trích dẫn không chứa thông tin liên quan trực tiếp đến câu hỏi, hoặc thông tin không đủ để trả lời chính xác, hãy trả lời rõ ràng: "Không tìm thấy thông tin phù hợp trong tài liệu để trả lời câu hỏi này." Đừng cố gắng suy diễn hoặc trả lời dựa trên thông tin không rõ ràng.

            Nếu có thông tin lặp lại trong ngữ cảnh hoặc trích dẫn, hãy chỉ sử dụng một lần và trình bày câu trả lời mạch lạc.

            Trả lời:"""
        )
        try:
            formatted_prompt = prompt.format(question=question, context=context, citations=citations, role=self.role)
            response = self.model.invoke(formatted_prompt)
            return response.content.strip()
        except Exception as e:
            display_message(f"Answer generation error: {str(e)}", "error")
            return ""

def get_session_history(conversation_id: str):
    return SQLChatMessageHistory(session_id=conversation_id, connection=f"sqlite:///{HISTORY_DB_PATH}")

def load_conversations():
    """Tải danh sách các conversation_id từ cơ sở dữ liệu SQLite."""
    try:
        engine = create_engine(f"sqlite:///{HISTORY_DB_PATH}")
        with engine.connect() as connection:
            result = connection.execute(text("SELECT DISTINCT session_id FROM chat_messages ORDER BY session_id"))
            conversation_ids = [row[0] for row in result]
        return conversation_ids
    except Exception as e:
        logger.error(f"Error loading conversations: {str(e)}")
        return []

def check_table_exists():
    """Kiểm tra xem bảng chat_messages có tồn tại trong cơ sở dữ liệu không."""
    try:
        engine = create_engine(f"sqlite:///{HISTORY_DB_PATH}")
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_messages'")
            )
            return result.fetchone() is not None
    except Exception as e:
        logger.error(f"Error checking table existence: {str(e)}")
        return False

def wrap_text(text: str, width: float, canvas_obj, font_name: str = "Helvetica", font_size: int = 12) -> List[str]:
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

def export_history_to_pdf(history, font_name: str = "Helvetica", font_size: int = 12) -> io.BytesIO:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    font_available = os.path.exists("DejaVuSans.ttf")
    if font_available:
        try:
            pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))
            font_name = "DejaVuSans"
        except Exception as e:
            display_message(f"Error loading DejaVuSans font: {str(e)}. Using default font.", "warning")
    y_position = height - inch
    c.setFont(font_name, 16)
    c.drawString(inch, y_position, "Lịch sử truy vấn")
    y_position -= 0.5 * inch
    c.setFont(font_name, font_size)
    for i in range(0, len(history), 2):
        if i + 1 < len(history):
            user_msg = history[i].content
            ai_msg = history[i + 1].content
            c.setFont(font_name, font_size)
            c.drawString(inch, y_position, "Câu hỏi:")
            y_position -= 0.3 * inch
            for line in wrap_text(user_msg, width - 2 * inch, c, font_name, font_size):
                if y_position < inch:
                    c.showPage()
                    c.setFont(font_name, font_size)
                    y_position = height - inch
                c.drawString(inch, y_position, line)
                y_position -= 0.3 * inch
            y_position -= 0.3 * inch
            c.setFont(font_name, font_size)
            c.drawString(inch, y_position, "Câu trả lời:")
            y_position -= 0.3 * inch
            for line in wrap_text(ai_msg, width - 2 * inch, c, font_name, font_size):
                if y_position < inch:
                    c.showPage()
                    c.setFont(font_name, font_size)
                    y_position = height - inch
                c.drawString(inch, y_position, line)
                y_position -= 0.3 * inch
            y_position -= 0.5 * inch
            if y_position < inch:
                c.showPage()
                y_position = height - inch
    c.save()
    buffer.seek(0)
    return buffer

def setup_session_state():
    if "message_placeholder" not in st.session_state:
        st.session_state.message_placeholder = st.empty()
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "processor" not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if "embedder" not in st.session_state:
        st.session_state.embedder = SentenceEmbedding(model_path=MODEL_PATH)
    if "chroma" not in st.session_state:
        st.session_state.chroma = ChromaDBManager(persist_directory=CHROMA_DB_PATH)
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}  # Lưu các conversation: {conversation_id: SQLChatMessageHistory}
    if "conversation_order" not in st.session_state:
        st.session_state.conversation_order = load_conversations()  # Tải conversation từ DB
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    if "has_db_notified" not in st.session_state:
        st.session_state.has_db_notified = False
    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = False
    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False  # Trạng thái để làm trống ô nhập
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None  # Lưu câu trả lời cuối cùng
    if "last_citations" not in st.session_state:
        st.session_state.last_citations = None  # Lưu trích dẫn cuối cùng

def handle_authentication():
    if not st.session_state.authenticated:
        password = st.text_input("Nhập mật khẩu:", type="password", value="")
        if st.button("Xác thực"):
            if password == PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                display_message("Mật khẩu không đúng. Vui lòng thử lại.", "error")
        return False
    return True

async def process_pdf(pdf_file):
    try:
        chunks, page_numbers = await st.session_state.processor.load_and_split(pdf_file)
        if not chunks:
            display_message("Không thể trích xuất văn bản từ PDF.", "error")
            return
        total_length = sum(len(chunk) for chunk in chunks)
        if total_length > 1000000:
            display_message("Nội dung tài liệu có thể bị cắt bớt do kích thước lớn.", "warning")
        embeddings = st.session_state.embedder(chunks)
        st.session_state.chroma.add(chunks, embeddings, filename=pdf_file.name, page_numbers=page_numbers)
        display_message("Đã xử lý và lưu tài liệu thành công!", "info")
        st.session_state.has_db_notified = False
    except Exception as e:
        display_message(f"Lỗi xử lý tài liệu: {str(e)}", "error")

def display_documents():
    if "chroma" in st.session_state:
        documents = st.session_state.chroma.list_documents()
        if documents:
            for doc in documents:
                st.sidebar.write(f"- {doc}")
        else:
            st.sidebar.write("Chưa có tài liệu nào.")
    else:
        st.sidebar.write("Chưa khởi tạo ChromaDB.")

def delete_all_documents():
    try:
        if "chroma" in st.session_state:
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
        else:
            display_message("Chưa khởi tạo ChromaDB.", "error")
    except Exception as e:
        display_message(f"Lỗi xóa tài liệu: {str(e)}", "error")

def create_new_conversation(confirmed=False):
    if len(st.session_state.conversation_order) >= 5 and not confirmed:
        oldest_conversation = st.session_state.conversation_order[0]
        st.session_state.confirm_delete = True
        display_message(
            f"Đã đạt giới hạn 5 cuộc trò chuyện. Xóa cuộc trò chuyện cũ nhất ({oldest_conversation.split('_')[1]}) để tạo mới?",
            "warning"
        )
        return False

    if confirmed and len(st.session_state.conversation_order) >= 5:
        oldest_conversation = st.session_state.conversation_order.pop(0)
        if check_table_exists():
            try:
                engine = create_engine(f"sqlite:///{HISTORY_DB_PATH}")
                with engine.connect() as connection:
                    connection.execute(
                        text("DELETE FROM chat_messages WHERE session_id = :conversation_id"),
                        {"conversation_id": oldest_conversation}
                    )
                    connection.commit()
                del st.session_state.conversations[oldest_conversation]
                display_message(f"Đã xóa cuộc trò chuyện cũ nhất ({oldest_conversation.split('_')[1]}).", "info")
            except Exception as e:
                display_message(f"Lỗi xóa cuộc trò chuyện cũ: {str(e)}", "error")
                return False
        else:
            del st.session_state.conversations[oldest_conversation]
            display_message(f"Không có dữ liệu lịch sử, đã xóa cuộc trò chuyện cũ nhất ({oldest_conversation.split('_')[1]}).", "info")

    conversation_id = f"conversation_{datetime.now().isoformat()}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()}"
    st.session_state.conversations[conversation_id] = get_session_history(conversation_id)
    st.session_state.conversation_order.append(conversation_id)
    st.session_state.current_conversation_id = conversation_id
    st.session_state.confirm_delete = False
    st.session_state.clear_input = True  # Yêu cầu làm trống ô nhập
    st.session_state.last_answer = None
    st.session_state.last_citations = None
    display_message("Đã tạo cuộc trò chuyện mới!", "info")
    return True

def display_conversation_list():
    st.sidebar.subheader("Danh sách cuộc trò chuyện")
    if st.session_state.conversation_order:
        for conversation_id in reversed(st.session_state.conversation_order):
            conversation_time = conversation_id.split("_")[1]
            if st.sidebar.button(f"Cuộc trò chuyện - {conversation_time}", key=conversation_id):
                st.session_state.current_conversation_id = conversation_id
                st.session_state.confirm_delete = False
                st.session_state.clear_input = True  # Yêu cầu làm trống ô nhập
                st.session_state.last_answer = None
                st.session_state.last_citations = None
                display_message(f"Đã chuyển sang cuộc trò chuyện {conversation_time}", "info")
    else:
        st.sidebar.write("Chưa có cuộc trò chuyện nào.")

async def handle_query(query: str, llm_type: str, role: str, similarity_threshold: float):
    if not st.session_state.current_conversation_id:
        create_new_conversation()
    if "chroma" not in st.session_state or st.session_state.chroma is None or "embedder" not in st.session_state:
        display_message("Chưa có dữ liệu. Vui lòng tải lên tài liệu PDF.", "warning")
        return
    try:
        @lru_cache(maxsize=100)
        def cached_query(query: str, top_k: int) -> Tuple[List[str], List[dict], List[float]]:
            query = preprocess_text(query)
            emb = st.session_state.embedder([query])[0]
            res = st.session_state.chroma.query(emb, top_k)
            return res["documents"][0], res["metadatas"][0], res["distances"][0]

        docs, metas, dists = cached_query(query, top_k=7)
        relevant_docs = []
        relevant_metas = []
        relevant_dists = []
        for doc, meta, dist in zip(docs, metas, dists):
            if dist >= similarity_threshold:
                relevant_docs.append(doc)
                relevant_metas.append(meta)
                relevant_dists.append(dist)

        citations_list = []
        if relevant_docs:
            context = " ".join(relevant_docs)
            for meta, doc, dist in zip(relevant_metas, relevant_docs, relevant_dists):
                citations_list.append({
                    "source": meta["source"],
                    "filename": meta["filename"],
                    "page_number": meta.get("page_number", "Không xác định"),
                    "score": dist,
                    "content": doc
                })
            citations = "Có các trích dẫn liên quan."
        else:
            context = "Không có thông tin phù hợp trong tài liệu."
            citations = f"Không có trích dẫn nào đạt ngưỡng tương đồng (cosine similarity >= {similarity_threshold})."

        answer_generator = AnswerGenerator(role=role)
        await answer_generator.initialize(model_type=llm_type)
        final_answer = answer_generator.generate_answer(query, context, citations)

        if final_answer.startswith("Error"):
            display_message(final_answer, "error")
            return

        current_history = st.session_state.conversations[st.session_state.current_conversation_id]
        current_history.add_user_message(query)
        current_history.add_ai_message(final_answer)

        # Lưu câu trả lời và trích dẫn, yêu cầu làm trống ô nhập
        st.session_state.last_answer = final_answer
        st.session_state.last_citations = citations_list
        st.session_state.clear_input = True

    except Exception as e:
        display_message(f"Lỗi truy vấn: {str(e)}", "error")

def display_query_history():
    if not st.session_state.current_conversation_id:
        st.write("Chưa có cuộc trò chuyện nào được chọn.")
        return
    messages = st.session_state.conversations[st.session_state.current_conversation_id].messages
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            user_msg = messages[i]
            ai_msg = messages[i + 1]
            user_content = user_msg.content.replace("\n", "<br>")
            ai_content = ai_msg.content.replace("\n", "<br>")
            st.markdown(
                f"""
                <div style="background-color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    👤 <strong>Người dùng:</strong> {user_content}
                </div>
                <div style="background-color: #d3d3d3; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    🤖 <strong>Trợ lý:</strong> {ai_content}
                </div>
                {"---" if i + 2 < len(messages) else ""}
                """,
                unsafe_allow_html=True
            )
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
    if messages and st.button("Xuất lịch sử truy vấn"):
        pdf_buffer = export_history_to_pdf(messages)
        st.download_button(
            "Tải lịch sử truy vấn (PDF)",
            pdf_buffer,
            file_name=f"query_history_{st.session_state.current_conversation_id}.pdf",
            mime="application/pdf"
        )

async def main():
    st.set_page_config(page_title="RAG Điện Viễn Thông (Nội bộ)", layout="wide")
    setup_session_state()
    if not handle_authentication():
        return

    st.title("📄 Hệ thống truy vấn tài liệu thông minh")
    has_db = check_existing_data()

    st.sidebar.header("Cấu hình")
    llm_type = st.sidebar.radio(
        "Chọn mô hình LLM:",
        ["ollama", "openai"],
        format_func=lambda x: "Ollama Llama3.2 (Offline)" if x == "ollama" else "OpenAI GPT-4o-mini (Online)",
    )
    role = st.sidebar.radio("Mức độ chi tiết:", ["Beginner", "Expert", "PhD"])
    similarity_threshold = st.sidebar.slider("Ngưỡng độ tương đồng (cosine similarity)", 0.0, 1.0, SIMILARITY_THRESHOLD_DEFAULT, step=0.05)

    if has_db:
        display_message("CSDL đã có dữ liệu. Bạn có thể bắt đầu truy vấn.", "info")
        st.session_state.has_db_notified = False
    else:
        if not st.session_state.has_db_notified:
            display_message("Chưa có dữ liệu. Vui lòng tải lên tài liệu PDF.", "warning")
            st.session_state.has_db_notified = True

    st.sidebar.subheader("Quản lý tài liệu")
    pdf_file = st.sidebar.file_uploader("Tải lên file PDF", type="pdf")
    if pdf_file:
        with st.spinner("Đang xử lý tài liệu…"):
            await process_pdf(pdf_file)

    if has_db:
        st.sidebar.subheader("Tài liệu đã tải")
        display_documents()
        if st.sidebar.button("Xóa tất cả tài liệu"):
            delete_all_documents()

    st.sidebar.subheader("Quản lý cuộc trò chuyện")
    if st.sidebar.button("Trò chuyện mới", help="Bắt đầu một cuộc trò chuyện mới", key="new_conversation", type="primary"):
        create_new_conversation()
        st.rerun()

    if st.session_state.confirm_delete:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Đồng ý xóa"):
                if create_new_conversation(confirmed=True):
                    st.rerun()
        with col2:
            if st.button("Hủy"):
                st.session_state.confirm_delete = False
                display_message("Đã hủy tạo cuộc trò chuyện mới.", "info")
                st.rerun()

    display_conversation_list()

    # Làm trống ô nhập nếu được yêu cầu
    input_value = "" if st.session_state.clear_input else st.session_state.get("query_input", "")
    query_input = st.text_input("Nhập câu hỏi (tiếng Việt):", value=input_value, key="query_input")
    if st.session_state.clear_input:
        st.session_state.clear_input = False  # Đặt lại trạng thái sau khi làm trống

    if st.button("Gửi câu hỏi"):
        if query_input:
            with st.spinner("Đang tạo câu trả lời…"):
                await handle_query(query_input, llm_type, role, similarity_threshold)

    # Hiển thị câu trả lời và trích dẫn nếu có
    if st.session_state.last_answer:
        st.markdown("### Câu trả lời cuối cùng:")
        st.write(st.session_state.last_answer)

        with st.expander("📚 Xem trích dẫn nguồn"):
            if st.session_state.last_citations:
                for idx, citation in enumerate(st.session_state.last_citations, 1):
                    st.markdown(f"#### Trích dẫn {idx}:")
                    st.markdown(f"- **Nguồn**: {citation['source']}")
                    st.markdown(f"- **Tài liệu**: {citation['filename']}")
                    st.markdown(f"- **Trang**: {citation['page_number']}")
                    st.markdown(f"- **Độ tương đồng (cosine similarity)**: {citation['score']:.4f}")
                    st.markdown(f"- **Nội dung**:")
                    st.write(citation['content'])
                    st.markdown("---")
            else:
                st.write(f"Không có trích dẫn nào đạt ngưỡng tương đồng (cosine similarity >= {similarity_threshold}).")

    with st.expander("📜 Lịch sử truy vấn"):
        display_query_history()

if __name__ == "__main__":
    asyncio.run(main())
import os
import sys
import asyncio
import streamlit as st
import chromadb
import io
import re
import numpy as np
import logging
import unicodedata
import pdfplumber
import hashlib
import warnings
import pytesseract
import aiohttp
import fasttext
import pandas as pd
import openpyxl
from sqlalchemy import create_engine, text
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from chromadb.config import Settings
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, CSVLoader,
    Docx2txtLoader
)
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from underthesea import word_tokenize
from typing import List, Tuple
from datetime import datetime
from functools import lru_cache
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import SQLChatMessageHistory
from cryptography.fernet import Fernet
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from retry import retry
from dotenv import load_dotenv

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

# Bật tùy chọn để tránh cảnh báo FutureWarning từ pandas
pd.set_option('future.no_silent_downcasting', True)

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = os.getenv("DATA_DIR", "data")
HISTORY_DIR = os.getenv("HISTORY_DIR", "history")
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "Documents")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
ENCRYPTION_KEY_PATH = os.path.join(DATA_DIR, "encryption_key.key")
HISTORY_DB_PATH = os.path.join(HISTORY_DIR, "query_history.db")
PASSWORD = os.getenv("APP_PASSWORD", "secure_default_password")
SIMILARITY_THRESHOLD_DEFAULT = float(os.getenv("SIMILARITY_THRESHOLD", 0.5))
MODEL_PATH = os.getenv("MODEL_PATH", "./models/cc.vi.300.bin.gz")
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
SUPPORTED_FILE_TYPES = [".txt", ".pdf", ".docx", ".xlsx", ".csv"]
CONTEXT_WINDOW_SIZE = 5
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Define TELECOM_KEYWORDS
TELECOM_KEYWORDS = [
    "ACB", "ATS", "MPĐ", "HĐB", "ĐHCX", "UPS", "DC", "UDB", "PDU", "UDB/PDU",
    "LV1", "LV2", "LV3", "LV4", "ESDB2", "ESDB3", "ACDB2_ext", "ACDB3_ext", "MSB4",
    "AMTS1", "AMTS3", "Interlock", "MCCB"
]
KEYWORD_MAP = {keyword: f"KEYWORD_{hashlib.md5(keyword.encode('utf-8')).hexdigest()}" for keyword in TELECOM_KEYWORDS}

# Ánh xạ thuật ngữ
TERMINOLOGY_MAPPING = {
    "máy cắt không khí": "ACB",
    "tủ LV1": "tủ điện hạ thế 1",
    "ATS1": "hệ thống chuyển nguồn tự động 1",
    "AMTS1": "hệ thống chuyển nguồn tự động 1",
    "máy phát điện": "MPĐ",
    "điều hòa chính xác": "ĐHCX",
    "tủ phân phối nguồn": "UDB/PDU",
    "sự cố": "lỗi",
    "mất điện": "sự cố lộ điện",
    "mất một lộ điện": "sự cố 1 lộ điện lưới",
    "tự động vận hành": "tự động interlock",
    "xác nhận tình huống": "báo cáo tình huống ƯCTT",
    "nhân viên cơ điện": "NV1 NV2",
    "tòa nhà N6": "N6",
    "tòa nhà N4": "N4",
    "lộ nổi": "lộ nổi",
    "lộ ngầm": "lộ ngầm",
    "máy cắt khối": "MCCB",
    "main power distribution": "MPĐ"
}

# Setup logging
logging.basicConfig(level=logging.INFO, filename='rag_system.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

async def check_internet():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.openai.com/v1/", timeout=5) as response:
                return response.status == 200
    except Exception:
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

def preprocess_text(text: str, restore: bool = False, for_embedding: bool = False) -> str:
    if not text or not isinstance(text, str):
        return ""

    # Normalize text to NFC form
    text = unicodedata.normalize('NFC', text)
    text = text.encode('utf-8', 'ignore').decode('utf-8')

    # Remove excessive dots and invalid characters, but preserve Vietnamese characters
    text = re.sub(r'\.\.\.+', ' ', text)
    # Preserve telecom-specific terms with numbers (e.g., "4000A", "3200A")
    for keyword in TELECOM_KEYWORDS:
        text = re.sub(rf'\b{keyword}\s*\d+[aA]\b', lambda m: m.group(0), text, flags=re.IGNORECASE)
    
    # Remove unwanted characters, but keep Vietnamese characters and basic punctuation
    text = re.sub(r'[^\w\s,.?!:;()\-/áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ\d]', ' ', text)
    text = re.sub(r'_+', ' ', text)
    text = re.sub(r'\b[-=]{3,}\b', ' ', text)
    text = re.sub(r'\|{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Apply terminology mapping for standardization
    for term, standard in TERMINOLOGY_MAPPING.items():
        text = re.sub(rf'\b{term}\b', standard, text, flags=re.IGNORECASE)

    if for_embedding:
        tokens = word_tokenize(text, format="text").split()
        max_tokens = 500
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        text = " ".join(tokens)
    elif restore:
        for keyword, placeholder in KEYWORD_MAP.items():
            text = text.replace(placeholder, keyword)
        logger.info(f"Restored keywords in text: {text[:100]}...")

    return text

class DocumentProcessor:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

    def find_column(self, headers, possible_names):
        for name in possible_names:
            for header in headers:
                if name.lower() in str(header).lower():
                    return header
        return None

    def format_row(self, row, headers, sheet_name: str):
        stt_col = self.find_column(headers, ['Stt', 'Số thứ tự', 'STT'])
        if not stt_col:
            logger.warning(f"Không tìm thấy cột Stt trong headers: {headers}")
            return None
        
        stt_value = row.get(stt_col)
        if pd.isna(stt_value):
            logger.info(f"Hàng bị bỏ qua do thiếu giá trị Stt: {row.to_dict()}")
            return None
        
        if sheet_name.lower() == "tong hop":
            su_co_col = self.find_column(headers, ['SỰ CỐ'])
            if not su_co_col:
                logger.warning(f"Không tìm thấy cột SỰ CỐ trong sheet Tong hop: {headers}")
                return None
            
            su_co_value = row.get(su_co_col)
            if pd.isna(su_co_value):
                logger.info(f"Hàng bị bỏ qua do thiếu giá trị SỰ CỐ: {row.to_dict()}")
                return None
            
            formatted = f"Sự cố {su_co_value}: "
            for header in headers:
                if header in ['STT', 'SỰ CỐ']:
                    continue
                value = str(row.get(header, '')) if pd.notna(row.get(header)) else '0'
                formatted += f"{header} {value}, "
            formatted = formatted.rstrip(", ")
            return formatted.strip()
        
        situation_col = self.find_column(headers, ['Tình huống', 'Tình huống sự cố', 'Tình huống xử lý', 'Tình huống ƯCTT', 'Tên tình huống', 'Tên lỗi, sự cố'])
        if not situation_col:
            situation_col = self.find_column(headers, ['Mức độ ảnh hưởng', 'Phương án', 'Tình huống sự cố', 'SỰ CỐ'])
            if not situation_col:
                logger.warning(f"Không tìm thấy cột Tình huống hoặc cột thay thế trong headers: {headers}")
                return None
        
        situation_value = row.get(situation_col)
        if pd.isna(situation_value):
            logger.info(f"Hàng bị bỏ qua do thiếu giá trị Tình huống: {row.to_dict()}")
            return None
        
        formatted = f"Tình huống số: {stt_value}\n"
        for header in headers:
            if 'Unnamed' in str(header):
                continue
            value = str(row.get(header, '')) if pd.notna(row.get(header)) else ''
            formatted += f"{header}: {value}\n"
        return formatted.strip()

    def split_long_row(self, row_text, max_chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        sub_chunks = []
        if len(row_text) <= max_chunk_size:
            sub_chunks.append(row_text)
        else:
            start = 0
            while start < len(row_text):
                end = start + max_chunk_size
                if end < len(row_text):
                    while end > start and row_text[end] not in ['\n', ' ']:
                        end -= 1
                sub_chunk = row_text[start:end]
                sub_chunks.append(sub_chunk)
                start = end - overlap
        return sub_chunks

    def find_header_row(self, filepath, sheet_name):
        workbook = openpyxl.load_workbook(filepath, data_only=True)
        sheet = workbook[sheet_name]
        
        skip_rows = 0
        for row in sheet.iter_rows(min_row=1, max_row=10):
            for cell in row:
                if cell.value and 'stt' in str(cell.value).lower():
                    return skip_rows
            skip_rows += 1
        return skip_rows

    def process_excel_with_openpyxl(self, filepath, sheet_name):
        skip_rows = self.find_header_row(filepath, sheet_name)
        logger.info(f"Sheet {sheet_name}: Skipping {skip_rows} rows to find header row.")
        
        df = pd.read_excel(filepath, sheet_name=sheet_name, engine='openpyxl', skiprows=skip_rows)
        
        workbook = openpyxl.load_workbook(filepath, data_only=True)
        sheet = workbook[sheet_name]
        merged_cells = sheet.merged_cells.ranges
        
        df = df.astype(object)
        
        for merged_range in merged_cells:
            min_row, min_col, max_row, max_col = merged_range.bounds
            value = sheet.cell(min_row, min_col).value
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    row_idx = row - skip_rows - 1
                    col_idx = col - 1
                    if 0 <= row_idx < len(df) and 0 <= col_idx < len(df.columns):
                        df.iloc[row_idx, col_idx] = value if pd.isna(df.iloc[row_idx, col_idx]) else df.iloc[row_idx, col_idx]
        
        # Standardize column names
        df.columns = [f"Column_{i}" if str(col).startswith("Unnamed") else col for i, col in enumerate(df.columns)]
        df = df.dropna(how='all').dropna(axis=1, how='all')
        df = df.ffill().infer_objects(copy=False)
        
        # Ensure 'Stt' column exists
        if 'Stt' not in df.columns and '1' in df.columns:
            df = df.rename(columns={'1': 'Stt'})
        
        return df

    async def load_and_split(self, file, filename: str) -> List[Tuple[List[str], List[int], List[str], str]]:
        saved_filepath = os.path.join(DOCUMENTS_DIR, filename)
        if not os.path.exists(saved_filepath):
            with open(saved_filepath, "wb") as f:
                f.write(file.read())
        
        loaders = []
        ext = os.path.splitext(filename)[1].lower()
        try:
            if ext == ".txt":
                loaders.append(TextLoader(saved_filepath))
            elif ext == ".pdf":
                return [(await self._process_pdf(saved_filepath), filename)]
            elif ext == ".xlsx":
                try:
                    xl = pd.ExcelFile(saved_filepath, engine='openpyxl')
                    results = []
                    for sheet_name in xl.sheet_names:
                        df = self.process_excel_with_openpyxl(saved_filepath, sheet_name)
                        df = df.dropna(how='all').dropna(axis=1, how='all')
                        df = df.ffill().infer_objects(copy=False)
                        df.columns = df.columns.astype(str).fillna("")
                        if '1' in df.columns:
                            df = df.rename(columns={'1': 'Stt'})
                        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                        if df.empty:
                            logger.info(f"Sheet {sheet_name} is empty after processing.")
                            continue
                        headers = df.columns.tolist()
                        logger.info(f"Headers in sheet {sheet_name}: {headers}")
                        chunks = []
                        situation_ids = []
                        page_numbers = []
                        for _, row in df.iterrows():
                            logger.info(f"Raw row data: {row.to_dict()}")
                            chunk = self.format_row(row, headers, sheet_name)
                            if not chunk:
                                continue
                            logger.info(f"Raw chunk: {chunk[:100]}...")
                            chunk = preprocess_text(chunk, restore=False)
                            logger.info(f"Processed chunk: {chunk[:100]}...")
                            sub_chunks = self.split_long_row(chunk)
                            chunks.extend(sub_chunks)
                            stt_col = self.find_column(headers, ['Stt', 'Số thứ tự', 'STT'])
                            situation_ids.extend([str(row.get(stt_col, ""))] * len(sub_chunks))
                            page_numbers.extend([1] * len(sub_chunks))
                        logger.info(f"Processed sheet {sheet_name}: {len(chunks)} chunks")
                        results.append((chunks, page_numbers, situation_ids, sheet_name))
                    return results
                except Exception as e:
                    logger.error(f"Error processing Excel file {filename}: {str(e)}")
                    display_message(f"Không thể xử lý file Excel {filename}: {str(e)}", "error")
                    return []
            elif ext == ".csv":
                loaders.append(CSVLoader(saved_filepath))
            else:
                logger.error(f"Unsupported file type: {ext}")
                return []
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            return []

        if loaders:
            merged_loader = MergedDataLoader(loaders=loaders)
            docs = merged_loader.load()
            full_text = " ".join(doc.page_content for doc in docs)
            full_text = preprocess_text(full_text, restore=False)
            splits = self.text_splitter.create_documents([full_text])
            chunks = [chunk.page_content.strip() for chunk in splits if chunk.page_content.strip()]
            page_numbers = [1] * len(chunks)
            situation_ids = [""] * len(chunks)
            return [(chunks, page_numbers, situation_ids, filename)]
        return []

    async def _process_pdf(self, filepath: str) -> Tuple[List[str], List[int], List[str]]:
        full_text = ""
        page_boundaries = []
        current_pos = 0
        try:
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if not text.strip():
                        logger.info(f"No text extracted from page {page_num}, trying OCR...")
                        try:
                            image = page.to_image(resolution=300).original.convert("RGB")
                            text = pytesseract.image_to_string(image, lang='vie').strip()
                        except Exception as e:
                            logger.error(f"OCR error for page {page_num}: {str(e)}")
                            text = ""
                    tables = page.extract_tables()
                    table_text = "\n".join([",".join(str(cell) if cell is not None else "" for cell in row) for table in tables for row in table]) if tables else ""
                    page_content = (text + "\n" + table_text).strip()
                    if page_content:
                        page_content = preprocess_text(page_content, restore=False)
                        full_text += page_content + "\n"
                        page_boundaries.append((current_pos, page_num))
                        current_pos += len(page_content) + 1
                    else:
                        logger.warning(f"No content on page {page_num} after processing.")
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            display_message(f"PDF processing error: {str(e)}", "error")
            return [], [], []
        if not full_text.strip():
            logger.error("No text extracted from PDF.")
            return [], [], []
        splits = self.text_splitter.create_documents([full_text])
        chunks = [chunk.page_content.strip() for chunk in splits if chunk.page_content.strip()]
        page_numbers = []
        for chunk in splits:
            chunk_text = chunk.page_content.strip()
            if not chunk_text:
                continue
            chunk_start = full_text.index(chunk_text)
            chunk_page = page_boundaries[0][1] if page_boundaries else 1
            for pos, page_num in page_boundaries:
                if chunk_start >= pos:
                    chunk_page = page_num
                else:
                    break
            page_numbers.append(chunk_page)
        situation_ids = [""] * len(chunks)
        return chunks, page_numbers, situation_ids

class SentenceEmbedding:
    def __init__(self, model_path: str = MODEL_PATH):
        try:
            if not os.path.exists(model_path):
                display_message(f"Model path {model_path} does not exist. Please ensure the FastText model is downloaded.", "error")
                st.stop()
            self.model = fasttext.load_model(model_path)
        except Exception as e:
            logger.error(f"Error loading FastText model from {model_path}: {str(e)}")
            display_message(f"Error loading FastText model: {str(e)}.", "error")
            st.stop()

    def __call__(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        embeddings = []
        valid_texts = []
        valid_indices = []
        for idx, text in enumerate(texts):
            if not text or not isinstance(text, str):
                logger.warning(f"Skipping empty or invalid text chunk: {text}")
                continue
            try:
                text = preprocess_text(text, for_embedding=True)
                embedding = self.model.get_sentence_vector(text)
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding.tolist())
                valid_texts.append(text)
                valid_indices.append(idx)
            except Exception as e:
                logger.error(f"Failed to generate embedding for text: {text[:100]}... Error: {str(e)}")
                continue
        if is_query:
            if not embeddings:
                raise ValueError("Failed to generate embedding for query text")
            return embeddings[0]
        return embeddings, valid_texts, valid_indices

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
        try:
            self.collection = self.client.get_or_create_collection(name="viettel_docs")
            doc_count = self.collection.count()
            logger.info(f"Initialized ChromaDB collection 'viettel_docs' with {doc_count} documents")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collection: {str(e)}")
            raise

    def verify_collection(self):
        try:
            doc_count = self.collection.count()
            logger.info(f"Collection verification: 'viettel_docs' has {doc_count} documents")
            return doc_count > 0
        except Exception as e:
            logger.error(f"Error verifying collection: {str(e)}")
            return False

    def reset_collection(self):
        try:
            self.client.delete_collection("viettel_docs")
            self.collection = self.client.get_or_create_collection(name="viettel_docs")
            logger.info("Collection 'viettel_docs' has been reset")
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            raise

    def add(self, texts: List[str], embeddings: List[List[float]], filename: str, page_numbers: List[int], sheet_name: str = "", situation_ids: List[str] = None):
        if not texts or not embeddings or not page_numbers:
            logger.error("Empty lists provided to ChromaDBManager.add")
            return
        if len(texts) != len(embeddings) or len(texts) != len(page_numbers) or len(texts) != len(situation_ids or []):
            logger.error(f"Mismatched lengths: texts={len(texts)}, embeddings={len(embeddings)}, page_numbers={len(page_numbers)}, situation_ids={len(situation_ids or [])}")
            return
        self.collection = self.client.get_or_create_collection(name="viettel_docs")
        existing_metas = self.collection.get(include=["metadatas"])["metadatas"]
        existing_docs = {(meta["filename"], meta["sheet_name"]) for meta in existing_metas if isinstance(meta, dict)}
        if (filename, sheet_name) in existing_docs:
            logger.info(f"Skipping duplicate document: {filename}, sheet {sheet_name}")
            return
        existing_ids = set(self.collection.get(include=[])["ids"])
        new_ids = []
        new_texts = []
        new_embeddings = []
        new_metas = []
        for i, (text, emb, sid) in enumerate(zip(texts, embeddings, situation_ids or [""] * len(texts))):
            doc_id = f"doc_{hashlib.md5((filename + sheet_name + sid + str(i) + str(len(existing_ids))).encode()).hexdigest()}"
            logger.info(f"Generated doc_id: {doc_id} for sheet {sheet_name}, sid {sid}, index {i}")
            if doc_id in existing_ids:
                logger.warning(f"Duplicate doc_id found: {doc_id}, skipping...")
                continue
            new_ids.append(doc_id)
            new_texts.append(self.cipher.encrypt(text.encode('utf-8')).decode('utf-8'))
            new_embeddings.append(emb)
            meta = {
                "source": f"Tình huống số {i+1}",
                "filename": filename,
                "page_number": page_numbers[i],
                "upload_date": datetime.now().isoformat(),
                "sheet_name": sheet_name,
                "situation_id": sid,
                "keywords": ",".join([kw for kw in TELECOM_KEYWORDS if kw in text])
            }
            if not isinstance(meta, dict):
                logger.error(f"Invalid metadata format: meta={meta}, type={type(meta)}")
                continue
            new_metas.append(meta)
            logger.info(f"Added metadata: {meta}")
            existing_ids.add(doc_id)
        if new_ids:
            self.collection.add(ids=new_ids, documents=new_texts, metadatas=new_metas, embeddings=new_embeddings)
            logger.info(f"Added {len(new_ids)} new chunks to ChromaDB")
        else:
            logger.warning("No new chunks added to ChromaDB")

    def query(self, query_embedding: List[float], top_k: int, query_text: str = "", is_summary_query: bool = False):
        # Verify collection state
        if not self.verify_collection():
            logger.info("Collection is empty or inaccessible, returning empty result")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        # Perform the query
        try:
            res = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"Error executing ChromaDB query: {str(e)}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        # Safely check for empty or malformed results
        if not res.get("documents") or not isinstance(res["documents"], list) or len(res["documents"]) == 0:
            logger.info("No documents found in ChromaDB for query (empty or malformed result)")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        # Check if the first list of documents is empty
        if not res["documents"][0]:
            logger.info("No documents found in ChromaDB for query (empty document list)")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        distances = res["distances"][0]

        # Validate lengths of returned lists
        if not (len(docs) == len(metas) == len(distances)):
            logger.error(f"Mismatch in query result lengths: docs={len(docs)}, metas={len(metas)}, distances={len(distances)}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        # Validate metadata format
        valid_docs = []
        valid_metas = []
        valid_distances = []
        for doc, meta, dist in zip(docs, metas, distances):
            if not isinstance(meta, dict):
                logger.error(f"Invalid metadata format: meta={meta}, type={type(meta)}")
                continue
            valid_docs.append(doc)
            valid_metas.append(meta)
            valid_distances.append(dist)

        if not valid_docs:
            logger.info("No valid documents with correct metadata format found")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        docs = valid_docs
        metas = valid_metas
        distances = valid_distances

        distances = np.array(distances)
        similarities = 1 - distances

        # Extract building and system from query
        query_lower = query_text.lower()
        building = "N4" if "n4" in query_lower else "N6" if "n6" in query_lower else None
        systems = [kw for kw in TELECOM_KEYWORDS if kw.lower() in query_lower]
        query_keywords = set(word.lower() for word in query_text.split() if word.lower() in [kw.lower() for kw in TELECOM_KEYWORDS])

        filtered_docs = []
        filtered_metas = []
        filtered_similarities = []

        for i, (doc, meta, sim) in enumerate(zip(docs, metas, similarities)):
            sheet_name = meta.get("sheet_name", "").lower()
            # Filter for summary queries
            if is_summary_query and sheet_name != "tong hop":
                continue
            # Filter for error resolution queries
            if not is_summary_query and sheet_name not in ["1.ds uctt", "form xlsc"]:
                continue

            # Check building match
            doc_content = self.cipher.decrypt(doc.encode('utf-8')).decode('utf-8').lower()
            building_match = True
            if building and building.lower() not in doc_content and building.lower() not in meta.get("situation_id", "").lower():
                logger.info(f"Document filtered out due to building mismatch: Query building={building}, Document={doc_content[:100]}...")
                building_match = False

            # Check system match with partial matching, preserving numerical differences
            doc_has_system = False
            if systems:
                for system in systems:
                    core_system = re.sub(r'\d+a', '', system.lower())  # Remove numerical part (e.g., "4000a" -> "acb")
                    full_system = system.lower()  # Keep full term (e.g., "acb 4000a")
                    if (core_system and core_system in doc_content) or (full_system in doc_content):
                        doc_has_system = True
                        logger.info(f"System match found: Query system={system}, Core system={core_system}, Full system={full_system}, Document={doc_content[:100]}...")
                        break
                if not doc_has_system:
                    logger.info(f"Document filtered out due to system mismatch: Query systems={systems}, Document={doc_content[:100]}...")
            else:
                doc_has_system = True  # No system specified in query, so pass filter

            if building_match and doc_has_system:
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                # Keyword-based boosting
                doc_keywords = set(word.lower() for word in doc_content.split())
                meta_keywords = set(kw.lower() for kw in meta.get("keywords", "").split(",") if kw)
                matched_keywords = query_keywords & (doc_keywords | meta_keywords)
                boost = 0.1 * len(matched_keywords)
                filtered_similarities.append(sim + boost)

        if not filtered_docs:
            logger.info("No documents passed building/system filters after final check")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        # Sort by similarity and limit to top_k
        sorted_indices = np.argsort(-np.array(filtered_similarities))[:top_k]
        restored_docs = [
            preprocess_text(
                self.cipher.decrypt(filtered_docs[i].encode('utf-8')).decode('utf-8'),
                restore=True
            ) for i in sorted_indices
        ]
        res = {
            "documents": [restored_docs],
            "metadatas": [[filtered_metas[i] for i in sorted_indices]],
            "distances": [[filtered_similarities[i] for i in sorted_indices]]
        }

        logger.info(f"Cosine similarities for query (after boosting): {res['distances']}")
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
        if model_type == "ollama":
            if not await check_ollama():
                display_message("Ollama server not running. Run 'ollama run llama3.2'.", "error")
                st.stop()
            self.model = ChatOllama(
                base_url="http://localhost:11434",
                model="llama3.2",
                temperature=0.1,
                max_tokens=2000,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
        elif model_type == "openai":
            if not ChatOpenAI:
                display_message("LangChain OpenAI module not installed. Install 'langchain-openai'.", "error")
                st.stop()
            if not OPENAI_API_KEY:
                display_message("OpenAI API key not set. Set OPENAI_API_KEY in .env.", "error")
                st.stop()
            if not await check_internet():
                display_message("No internet connection. Falling back to Ollama.", "warning")
                if not await check_ollama():
                    display_message("Ollama server not running. Run 'ollama run llama3.2'.", "error")
                    st.stop()
                self.model_type = "ollama"
                self.model = ChatOllama(
                    base_url="http://localhost:11434",
                    model="llama3.2",
                    temperature=0.1,
                    max_tokens=2000,
                    streaming=True,
                    callbacks=[StreamingStdOutCallbackHandler()]
                )
            else:
                self.model = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=OPENAI_API_KEY,
                    temperature=0.1,
                    max_tokens=2000,
                    streaming=True,
                    callbacks=[StreamingStdOutCallbackHandler()]
                )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @retry(tries=3, delay=1, backoff=2)
    async def generate_answer_stream(self, question: str, context: str, citations: str, conversation_history: str, is_summary_query: bool = False):
        if is_summary_query:
            prompt = PromptTemplate(
                input_variables=["conversation_history", "question", "context", "citations", "role"],
                template="""Bạn là một chuyên gia trong lĩnh vực viễn thông và vận hành Data Center, trả lời ở mức độ {role}. Dựa trên câu hỏi và ngữ cảnh từ tài liệu sau, hãy cung cấp câu trả lời chính xác về thống kê sự cố trong năm vừa qua.

                    Câu hỏi: {question}

                    Ngữ cảnh từ tài liệu: {context}

                    Hướng dẫn:
                    - Chỉ sử dụng thông tin từ ngữ cảnh tài liệu để trả lời.
                    - Trả lời câu hỏi về số lượng sự cố (e.g., ƯCTT, XLSC, VHKT, TỔNG) dựa trên dữ liệu từ sheet "Tong hop".
                    - Nếu không tìm thấy thông tin phù hợp, trả lời: "Không tìm thấy thông tin thống kê phù hợp trong tài liệu."
                    - Đảm bảo câu trả lời rõ ràng, ngắn gọn và đúng với dữ liệu trong ngữ cảnh.
                    - Không suy diễn hoặc tự tạo thông tin ngoài tài liệu.

                    Trả lời:"""
            )
        else:
            prompt = PromptTemplate(
                input_variables=["conversation_history", "question", "context", "citations", "role"],
                template="""Bạn là một chuyên gia trong lĩnh vực viễn thông và vận hành Data Center, trả lời ở mức độ {role}. Dựa trên câu hỏi và ngữ cảnh từ tài liệu sau, hãy cung cấp câu trả lời chính xác, sử dụng thông tin từ tài liệu. Trả lời theo định dạng:

                    - **Tình huống**: [Mô tả tình huống sự cố, bao gồm tòa nhà (N4/N6) và hệ thống (MPĐ, AC, UPS, v.v.)]
                    - **Dấu hiệu nhận biết**: [Dấu hiệu cụ thể của sự cố]
                    - **Giải pháp thực hiện**: [Các bước cụ thể, liệt kê từng bước rõ ràng]
                    - **Nguyên nhân**: [Nguyên nhân nếu được nêu trong tài liệu]
                    - **Mức độ sự cố**: [Lớn/Trung bình/Nhỏ nếu có]
                    - **Nguồn vật tư**: [Nơi lấy vật tư thay thế nếu có]
                    - **Ghi chú**: [Thông tin bổ sung nếu có]

                    Câu hỏi: {question}

                    Ngữ cảnh từ tài liệu: {context}

                    Hướng dẫn:
                    - PHẢI sử dụng thông tin từ ngữ cảnh tài liệu để trả lời.
                    - Đảm bảo câu trả lời liên quan trực tiếp đến tòa nhà (N4/N6) và hệ thống (MPĐ, AC, UPS, v.v.) được hỏi.
                    - Nếu ngữ cảnh không chứa thông tin liên quan, trả lời: "Không tìm thấy thông tin phù hợp trong tài liệu. Vui lòng thử lại với các thuật ngữ cụ thể như MPĐ, ATS, Interlock hoặc kiểm tra các sheet khác như 1.Ds UCTT hoặc Form XLSC."
                    - Không suy diễn hoặc tự tạo thông tin ngoài tài liệu.
                    - Nếu có nhiều tình huống liên quan, chọn tình huống phù hợp nhất với tòa nhà và hệ thống được hỏi.

                    Trả lời:"""
            )

        try:
            if "Không có thông tin từ tài liệu" in context:
                yield "Không tìm thấy thông tin phù hợp trong tài liệu. Vui lòng thử lại với các thuật ngữ cụ thể như MPĐ, ATS, Interlock hoặc kiểm tra các sheet khác như 1.Ds UCTT hoặc Form XLSC."
                return
            formatted_prompt = prompt.format(
                conversation_history=conversation_history,
                question=question,
                context=context,
                citations=citations,
                role=self.role
            )
            async for chunk in self.model.astream(formatted_prompt):
                yield chunk.content
        except Exception as e:
            display_message(f"Answer generation error: {str(e)}", "error")
            yield ""

    @retry(tries=3, delay=1, backoff=2)
    def generate_answer(self, question: str, context: str, citations: str, conversation_history: str, is_summary_query: bool = False) -> str:
        if is_summary_query:
            prompt = PromptTemplate(
                input_variables=["conversation_history", "question", "context", "citations", "role"],
                template="""Bạn là một chuyên gia trong lĩnh vực viễn thông và vận hành Data Center, trả lời ở mức độ {role}. Dựa trên câu hỏi và ngữ cảnh từ tài liệu sau, hãy cung cấp câu trả lời chính xác về thống kê sự cố trong năm vừa qua.

                    Câu hỏi: {question}

                    Ngữ cảnh từ tài liệu: {context}

                    Hướng dẫn:
                    - Chỉ sử dụng thông tin từ ngữ cảnh tài liệu để trả lời.
                    - Trả lời câu hỏi về số lượng sự cố (e.g., ƯCTT, XLSC, VHKT, TỔNG) dựa trên dữ liệu từ sheet "Tong hop".
                    - Nếu không tìm thấy thông tin phù hợp, trả lời: "Không tìm thấy thông tin thống kê phù hợp trong tài liệu."
                    - Đảm bảo câu trả lời rõ ràng, ngắn gọn và đúng với dữ liệu trong ngữ cảnh.
                    - Không suy diễn hoặc tự tạo thông tin ngoài tài liệu.

                    Trả lời:"""
            )
        else:
            prompt = PromptTemplate(
                input_variables=["conversation_history", "question", "context", "citations", "role"],
                template="""Bạn là một chuyên gia trong lĩnh vực viễn thông và vận hành Data Center, trả lời ở mức độ {role}. Dựa trên câu hỏi và ngữ cảnh từ tài liệu sau, hãy cung cấp câu trả lời chính xác, sử dụng thông tin từ tài liệu. Trả lời theo định dạng:

                    - **Tình huống**: [Mô tả tình huống sự cố, bao gồm tòa nhà (N4/N6) và hệ thống (MPĐ, AC, UPS, v.v.)]
                    - **Dấu hiệu nhận biết**: [Dấu hiệu cụ thể của sự cố]
                    - **Giải pháp thực hiện**: [Các bước cụ thể, liệt kê từng bước rõ ràng]
                    - **Nguyên nhân**: [Nguyên nhân nếu được nêu trong tài liệu]
                    - **Mức độ sự cố**: [Lớn/Trung bình/Nhỏ nếu có]
                    - **Nguồn vật tư**: [Nơi lấy vật tư thay thế nếu có]
                    - **Ghi chú**: [Thông tin bổ sung nếu có]

                    Câu hỏi: {question}

                    Ngữ cảnh từ tài liệu: {context}

                    Hướng dẫn:
                    - PHẢI sử dụng thông tin từ ngữ cảnh tài liệu để trả lời.
                    - Đảm bảo câu trả lời liên quan trực tiếp đến tòa nhà (N4/N6) và hệ thống (MPĐ, AC, UPS, v.v.) được hỏi.
                    - Nếu ngữ cảnh không chứa thông tin liên quan, trả lời: "Không tìm thấy thông tin phù hợp trong tài liệu. Vui lòng thử lại với các thuật ngữ cụ thể như MPĐ, ATS, Interlock hoặc kiểm tra các sheet khác như 1.Ds UCTT hoặc Form XLSC."
                    - Không suy diễn hoặc tự tạo thông tin ngoài tài liệu.
                    - Nếu có nhiều tình huống liên quan, chọn tình huống phù hợp nhất với tòa nhà và hệ thống được hỏi.

                    Trả lời:"""
            )
        
        try:
            if "Không có thông tin từ tài liệu" in context:
                return "Không tìm thấy thông tin phù hợp trong tài liệu. Vui lòng thử lại với các thuật ngữ cụ thể như MPĐ, ATS, Interlock hoặc kiểm tra các sheet khác như 1.Ds UCTT hoặc Form XLSC."
            formatted_prompt = prompt.format(
                conversation_history=conversation_history,
                question=question,
                context=context,
                citations=citations,
                role=self.role
            )
            response = self.model.invoke(formatted_prompt)
            response = response.content.strip()
            return response
        except Exception as e:
            display_message(f"Answer generation error: {str(e)}", "error")
            return ""

def get_session_history(conversation_id: str):
    try:
        engine = create_engine(f"sqlite:///{HISTORY_DB_PATH}")
        with engine.connect() as connection:
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            connection.commit()
        history = SQLChatMessageHistory(session_id=conversation_id, connection=f"sqlite:///{HISTORY_DB_PATH}")
        return history
    except Exception as e:
        logger.error(f"Error initializing session history: {str(e)}")
        return SQLChatMessageHistory(session_id=conversation_id, connection=f"sqlite:///{HISTORY_DB_PATH}")

def load_conversations():
    try:
        engine = create_engine(f"sqlite:///{HISTORY_DB_PATH}")
        with engine.connect() as connection:
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            connection.commit()
            result = connection.execute(text("SELECT DISTINCT session_id FROM chat_messages ORDER BY session_id"))
            conversation_ids = [row[0] for row in result]
        return conversation_ids
    except Exception as e:
        logger.error(f"Error loading conversations: {str(e)}")
        return []

def check_table_exists():
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
        st.session_state.embedder = SentenceEmbedding()
    if "chroma" not in st.session_state:
        st.session_state.chroma = ChromaDBManager(persist_directory=CHROMA_DB_PATH)
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "conversation_order" not in st.session_state:
        st.session_state.conversation_order = load_conversations()
    if "current_conversation_id" not in st.session_state:
        conversation_id = f"conversation_{datetime.now().isoformat()}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()}"
        st.session_state.conversations[conversation_id] = get_session_history(conversation_id)
        st.session_state.conversation_order.append(conversation_id)
        st.session_state.current_conversation_id = conversation_id
        logger.info(f"Initialized conversation ID: {conversation_id}")
    if "has_db_notified" not in st.session_state:
        st.session_state.has_db_notified = False
    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = False
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None
    if "last_citations" not in st.session_state:
        st.session_state.last_citations = None
    if "query_input_value" not in st.session_state:
        st.session_state.query_input_value = ""

def handle_authentication():
    if not st.session_state.authenticated:
        password = st.text_input("Nhập mật khẩu:", type="password", value="")
        if st.button("Xác thực"):
            if password == PASSWORD:
                st.session_state.authenticated = True
                logger.info("Authentication successful, triggering rerun.")
                st.rerun()
            else:
                display_message("Mật khẩu không đúng. Vui lòng thử lại.", "error")
        return False
    return True

async def process_files(uploaded_files):
    try:
        for uploaded_file in uploaded_files:
            results = await st.session_state.processor.load_and_split(uploaded_file, uploaded_file.name)
            for chunks, page_numbers, situation_ids, sheet_name in results:
                if not chunks:
                    display_message(f"Không thể trích xuất văn bản từ {uploaded_file.name}, sheet {sheet_name}.", "error")
                    continue
                total_length = sum(len(chunk) for chunk in chunks)
                if total_length > 1000000:
                    display_message(f"Nội dung {uploaded_file.name} (sheet {sheet_name}) có thể bị cắt bớt do kích thước lớn.", "warning")
                embeddings, valid_chunks, valid_indices = st.session_state.embedder(chunks)
                if not embeddings or not valid_chunks:
                    display_message(f"Không thể tạo embeddings cho tài liệu {uploaded_file.name} (sheet {sheet_name}).", "error")
                    continue
                valid_page_numbers = [page_numbers[i] for i in valid_indices]
                valid_situation_ids = [situation_ids[i] for i in valid_indices]
                st.session_state.chroma.add(
                    texts=valid_chunks,
                    embeddings=embeddings,
                    filename=uploaded_file.name,
                    page_numbers=valid_page_numbers,
                    sheet_name=sheet_name,
                    situation_ids=valid_situation_ids
                )
        display_message("Đã xử lý và lưu tài liệu thành công!", "info")
        st.session_state.has_db_notified = False
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
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
            st.session_state.chroma.reset_collection()  # Reset the collection
            st.session_state.chroma = ChromaDBManager(persist_directory=CHROMA_DB_PATH)  # Reinitialize ChromaDB
            st.session_state.embedder = SentenceEmbedding()  # Reinitialize embedder
            st.session_state.processor = DocumentProcessor()  # Reinitialize processor
            for file in os.listdir(DOCUMENTS_DIR):
                file_path = os.path.join(DOCUMENTS_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            display_message("Đã xóa tất cả tài liệu và khởi tạo lại bộ sưu tập! Vui lòng tải lại tài liệu.", "info")
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
    st.session_state.last_answer = None
    st.session_state.last_citations = None
    display_message("Đã tạo cuộc trò chuyện mới!", "info")
    logger.info(f"Created new conversation ID: {conversation_id}, triggering rerun.")
    st.rerun()
    return True

def display_conversation_list():
    st.sidebar.subheader("Danh sách cuộc trò chuyện")
    if st.session_state.conversation_order:
        for conversation_id in reversed(st.session_state.conversation_order):
            conversation_time = conversation_id.split("_")[1]
            if st.sidebar.button(f"Cuộc trò chuyện - {conversation_time}", key=conversation_id):
                st.session_state.current_conversation_id = conversation_id
                st.session_state.confirm_delete = False
                st.session_state.last_answer = None
                st.session_state.last_citations = None
                display_message(f"Đã chuyển sang cuộc trò chuyện {conversation_time}", "info")
                logger.info(f"Switched to conversation ID: {conversation_id}")
    else:
        st.sidebar.write("Chưa có cuộc trò chuyện nào.")

def get_conversation_context(conversation_id: str) -> str:
    if not conversation_id or conversation_id not in st.session_state.conversations:
        return "Không có lịch sử trò chuyện."
    
    messages = st.session_state.conversations[conversation_id].messages
    if not messages:
        return "Không có lịch sử trò chuyện."
    
    recent_messages = messages[-CONTEXT_WINDOW_SIZE:] if len(messages) > CONTEXT_WINDOW_SIZE else messages
    context = ""
    for i in range(0, len(recent_messages), 2):
        if i + 1 < len(recent_messages):
            user_msg = recent_messages[i].content
            ai_msg = recent_messages[i + 1].content
            context += f"Người dùng: {user_msg}\nTrợ lý: {ai_msg}\n"
        else:
            user_msg = recent_messages[i].content
            context += f"Người dùng: {user_msg}\n"
    return context.strip()

async def handle_query(query: str, llm_type: str, role: str, similarity_threshold: float):
    if "chroma" not in st.session_state or st.session_state.chroma is None or "embedder" not in st.session_state:
        display_message("Chưa có dữ liệu. Vui lòng tải lên tài liệu.", "warning")
        return

    try:
        current_conversation_id = st.session_state.current_conversation_id
        logger.info(f"Handling query with conversation ID: {current_conversation_id}")

        conversation_history = get_conversation_context(current_conversation_id)
        logger.info(f"Conversation context: {conversation_history}")

        is_summary_query = any(keyword in query.lower() for keyword in ["bao nhiêu sự cố", "tổng số sự cố", "thống kê sự cố", "trong năm vừa qua"])
        expanded_query = query
        for term, standard in TERMINOLOGY_MAPPING.items():
            if term.lower() in query.lower():
                expanded_query += f" {standard}"
        logger.info(f"Expanded query: {expanded_query}")

        answer_generator = AnswerGenerator(role=role)
        await answer_generator.initialize(model_type=llm_type)
        
        context = "Không có thông tin từ tài liệu."
        citations = "Không có trích dẫn từ tài liệu."
        citations_list = []
        
        logger.info("Conversation history insufficient, querying database.")
        @lru_cache(maxsize=100)
        def cached_query(query: str, top_k: int) -> Tuple[List[str], List[dict], List[float]]:
            query = preprocess_text(query, restore=False)
            emb = st.session_state.embedder([query], is_query=True)
            res = st.session_state.chroma.query(emb, top_k, query_text=query, is_summary_query=is_summary_query)
            logger.info(f"Query result from ChromaDB: documents={len(res['documents']) if res.get('documents') else 0}, metadatas={len(res['metadatas']) if res.get('metadatas') else 0}, distances={len(res['distances']) if res.get('distances') else 0}")
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            if not isinstance(docs, list) or not isinstance(metas, list) or not isinstance(dists, list):
                logger.error(f"Invalid data format from ChromaDB: docs={type(docs)}, metas={type(metas)}, dists={type(dists)}")
                return [], [], []
            if len(docs) != len(metas) or len(docs) != len(dists):
                logger.error(f"Mismatched lengths: docs={len(docs)}, metas={len(metas)}, dists={len(dists)}")
                return [], [], []
            return docs, metas, dists

        docs, metas, dists = cached_query(expanded_query, top_k=10)
        logger.info(f"Retrieved {len(docs)} documents from cached_query")
        
        if not docs or not isinstance(docs, list):
            context = "Không có thông tin phù hợp trong tài liệu."
            citations = f"Không tìm thấy tài liệu nào liên quan đến '{query}'. Vui lòng sử dụng các thuật ngữ cụ thể như MPĐ, ATS, Interlock hoặc kiểm tra các sheet như 1.Ds UCTT, Form XLSC."
        else:
            relevant_docs = []
            relevant_metas = []
            relevant_dists = []
            
            for doc, meta, dist in zip(docs, metas, dists):
                if not isinstance(doc, str) or not isinstance(meta, dict) or not isinstance(dist, (int, float)):
                    logger.warning(f"Invalid document data: doc={type(doc)}, meta={type(meta)}, dist={type(dist)}")
                    continue
                if dist >= similarity_threshold:
                    relevant_docs.append(doc)
                    relevant_metas.append(meta)
                    relevant_dists.append(dist)

            if relevant_docs:
                sorted_pairs = sorted(zip(relevant_docs, relevant_metas, relevant_dists), key=lambda x: x[2], reverse=True)[:5]
                context_parts = []
                for doc, meta, dist in sorted_pairs:
                    if not isinstance(meta, dict):
                        logger.error(f"Invalid metadata format: meta={meta}, type={type(meta)}")
                        continue
                    context_parts.append(
                        f"Document (Sheet: {meta.get('sheet_name', 'Unknown')}, Situation ID: {meta.get('situation_id', 'Unknown')}, Similarity: {dist:.4f}):\n{doc}\n"
                    )
                context = "\n".join(context_parts) if context_parts else "Không có thông tin phù hợp trong tài liệu."
                if context_parts:
                    for meta, doc, dist in sorted_pairs:
                        if not isinstance(meta, dict):
                            logger.error(f"Invalid metadata format in citations: meta={meta}, type={type(meta)}")
                            continue
                        citations_list.append({
                            "source": meta.get("source", "Unknown"),
                            "filename": meta.get("filename", "Unknown"),
                            "sheet_name": meta.get("sheet_name", "Không xác định"),
                            "situation_id": meta.get("situation_id", "Không xác định"),
                            "page_number": meta.get("page_number", "Không xác định"),
                            "score": dist,
                            "content": preprocess_text(doc, restore=True)
                        })
                    citations = "Có các trích dẫn liên quan."
                else:
                    context = "Không có thông tin phù hợp trong tài liệu."
                    citations = f"Không tìm thấy tài liệu nào đạt ngưỡng tương đồng (cosine similarity >= {similarity_threshold}). Vui lòng sử dụng các thuật ngữ như MPĐ, ATS, Interlock hoặc kiểm tra các sheet như 1.Ds UCTT, Form XLSC."
            else:
                context = "Không có thông tin phù hợp trong tài liệu."
                citations = f"Không tìm thấy tài liệu nào đạt ngưỡng tương đồng (cosine similarity >= {similarity_threshold}). Vui lòng sử dụng các thuật ngữ như MPĐ, ATS, Interlock hoặc kiểm tra các sheet như 1.Ds UCTT, Form XLSC."

        st.markdown("### Câu trả lời:")
        response_container = st.empty()
        response_text = ""
        async for chunk in answer_generator.generate_answer_stream(query, context, citations, conversation_history, is_summary_query):
            response_text += chunk
            response_container.markdown(response_text)

        if st.session_state.current_conversation_id != current_conversation_id:
            logger.warning(f"Conversation ID changed during query from {current_conversation_id} to {st.session_state.current_conversation_id}")
            current_conversation_id = st.session_state.current_conversation_id

        current_history = st.session_state.conversations[current_conversation_id]
        current_history.add_user_message(query)
        current_history.add_ai_message(response_text)
        engine = create_engine(f"sqlite:///{HISTORY_DB_PATH}")
        with engine.connect() as connection:
            connection.commit()
        st.session_state.last_answer = response_text
        st.session_state.last_citations = citations_list
        st.session_state.query_input_value = ""
        logger.info(f"Query handled successfully, conversation ID: {current_conversation_id}, input cleared.")

    except Exception as e:
        error_message = f"Lỗi truy vấn: {str(e)}. Vui lòng kiểm tra lại câu hỏi, đảm bảo sử dụng các thuật ngữ viễn thông như ACB, ATS, MPĐ. Nếu lỗi vẫn xảy ra, thử xóa tất cả tài liệu và tải lại hoặc liên hệ quản trị viên."
        display_message(error_message, "error")
        logger.error(f"Query error: {str(e)}")

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

def display_summary_data():
    if "chroma" not in st.session_state:
        st.write("Chưa khởi tạo ChromaDB.")
        return
    
    try:
        results = st.session_state.chroma.collection.get(include=["documents", "metadatas"])
        summary_docs = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            if meta.get("sheet_name", "").lower() == "tong hop":
                decrypted_doc = st.session_state.chroma.cipher.decrypt(doc.encode('utf-8')).decode('utf-8')
                restored_doc = preprocess_text(decrypted_doc, restore=True)
                logger.info(f"Restored document for display: {restored_doc[:100]}...")
                summary_docs.append(restored_doc)
        
        if summary_docs:
            st.markdown("### Thống kê sự cố trong năm vừa qua (Sheet Tong hop):")
            for doc in summary_docs:
                st.write(doc)
        else:
            st.write("Không tìm thấy dữ liệu thống kê từ sheet Tong hop.")
    except Exception as e:
        st.write(f"Lỗi hiển thị dữ liệu thống kê: {str(e)}")

async def main():
    st.set_page_config(page_title="RAG Điện Viễn Thông (Nội bộ)", layout="wide")
    setup_session_state()
    if not handle_authentication():
        return

    st.title("📄 Hệ thống truy vấn tài liệu thông minh")
    has_db = check_existing_data()

    # Log the current document count in the collection
    if has_db and "chroma" in st.session_state:
        doc_count = st.session_state.chroma.collection.count()
        logger.info(f"Current document count in collection: {doc_count}")

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
            display_message("Chưa có dữ liệu. Vui lòng tải lên tài liệu.", "warning")
            st.session_state.has_db_notified = True

    st.sidebar.subheader("Quản lý tài liệu")
    uploaded_files = st.sidebar.file_uploader(
        "Tải lên tài liệu",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True
    )
    if uploaded_files:
        with st.spinner("Đang xử lý tài liệu…"):
            await process_files(uploaded_files)

    if has_db:
        st.sidebar.subheader("Tài liệu đã tải")
        display_documents()
        if st.sidebar.button("Xóa tất cả tài liệu"):
            delete_all_documents()

    st.sidebar.subheader("Quản lý cuộc trò chuyện")
    if st.sidebar.button("Trò chuyện mới", help="Bắt đầu một cuộc trò chuyện mới", key="new_conversation", type="primary"):
        create_new_conversation()
        logger.info("New conversation button clicked, rerun triggered.")

    if st.session_state.confirm_delete:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Đồng ý xóa"):
                if create_new_conversation(confirmed=True):
                    logger.info("Confirmed delete, rerun triggered.")
        with col2:
            if st.button("Hủy"):
                st.session_state.confirm_delete = False
                display_message("Đã hủy tạo cuộc trò chuyện mới.", "info")
                logger.info("Canceled delete, rerun triggered.")
                st.rerun()

    display_conversation_list()

    with st.expander("📖 Gợi ý câu hỏi"):
        st.markdown("""
        Để nhận được câu trả lời chính xác, hãy sử dụng các thuật ngữ viễn thông như MPĐ, ATS, Interlock. Dưới đây là một số câu hỏi mẫu:
        - Sự cố mất một lộ điện lưới lộ nổi ở N6, cần làm gì?
        - Lỗi ACB 4000A tủ LV1 cấp tới ATS1, cách xử lý?
        - Điều hòa đẩy cảnh báo HP, các bước khắc phục?
        - Sự cố 2 lộ điện, MPĐ lỗi ở N4, giải pháp là gì?
        - UPS hỏng module công suất, cách xử lý?
        - Trong năm vừa qua có bao nhiêu sự cố AC?
        """)

    query_input = st.text_input("Nhập câu hỏi (tiếng Việt):", value=st.session_state.query_input_value, key="query_input")
    st.session_state.query_input_value = query_input

    if st.button("Gửi câu hỏi"):
        if query_input:
            with st.spinner("Đang tạo câu trả lời…"):
                await handle_query(query_input, llm_type, role, similarity_threshold)

    with st.expander("📊 Xem thống kê sự cố"):
        display_summary_data()

    with st.expander("📚 Xem trích dẫn nguồn"):
        if st.session_state.last_citations:
            for idx, citation in enumerate(st.session_state.last_citations, 1):
                st.markdown(f"#### Trích dẫn {idx}:")
                st.markdown(f"- **Nguồn**: {citation['source']}")
                st.markdown(f"- **Tài liệu**: {citation['filename']}")
                st.markdown(f"- **Sheet**: {citation['sheet_name']}")
                st.markdown(f"- **Situation ID**: {citation['situation_id']}")
                st.markdown(f"- **Trang**: {citation['page_number']}")
                st.markdown(f"- **Độ tương đồng (cosine similarity)**: {citation['score']:.4f}")
                st.markdown(f"- **Nội dung**:")
                st.write(citation['content'])
                st.markdown("---")
        else:
            st.write(f"Không có trích dẫn nào đạt ngưỡng tương đồng (cosine similarity >= {similarity_threshold}). Vui lòng sử dụng thuật ngữ như MPĐ, ATS, Interlock.")

    with st.expander("📜 Lịch sử truy vấn"):
        display_query_history()

if __name__ == "__main__":
    asyncio.run(main())
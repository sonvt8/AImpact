import os
import tempfile
import streamlit as st
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted
from typing import List
import joblib
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# === 1. Kiểm tra cơ sở dữ liệu trong thư mục ChromaDB và vectorizer ===
def check_existing_data(persist_directory: str = "./chroma_db", vectorizer_path: str = "./data/tfidf_vectorizer.pkl"):
    # Kiểm tra cả thư mục ChromaDB và file vectorizer trong thư mục data
    return (os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0 and 
            os.path.exists(vectorizer_path))

# === 2. Xử lý PDF và chia nhỏ văn bản ===
class DocumentProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

    def load_and_split(self, pdf_file) -> List[str]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        splits = self.text_splitter.split_documents(docs)
        return [chunk.page_content.strip() for chunk in splits if chunk.page_content.strip()]

# === 3. TF-IDF embedding function ===
class TFIDFEmbedding:
    def __init__(self, vectorizer_path: str = "./data/tfidf_vectorizer.pkl"):
        self.vectorizer_path = vectorizer_path
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        
        # Kiểm tra thư mục data, nếu không có thì tạo thư mục
        os.makedirs(os.path.dirname(self.vectorizer_path), exist_ok=True)

        # Nếu mô hình đã được lưu, tải lại
        if os.path.exists(self.vectorizer_path):
            self.vectorizer = joblib.load(self.vectorizer_path)

    def is_fitted(self) -> bool:
        try:
            check_is_fitted(self.vectorizer)
            return True
        except:
            return False

    def fit(self, texts: List[str]):
        if texts and any(t.strip() for t in texts):
            self.vectorizer.fit(texts)
            joblib.dump(self.vectorizer, self.vectorizer_path)
        else:
            raise ValueError("Cannot fit TfidfVectorizer with empty or invalid texts")

    def __call__(self, texts: List[str]) -> List[List[float]]:
        if not self.is_fitted():
            raise ValueError("TfidfVectorizer is not fitted. Please process a document first.")
        return self.vectorizer.transform(texts).toarray().tolist()

# === 4. Quản lý ChromaDB ===
class ChromaDBManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="viettel_docs")

    def add(self, texts: List[str], embeddings: List[List[float]]):
        ids = [f"doc_{i}" for i in range(len(texts))]
        metas = [{"source": f"Đoạn {i+1}"} for i in range(len(texts))]
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metas,
            embeddings=embeddings,
        )

    def query(self, query_embedding: List[float], top_k: int = 5):
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return res

# === 5. Tạo câu trả lời cuối cùng từ mô hình LLM và kết hợp trích dẫn ===
class AnswerGenerator:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model = ChatOpenAI(model=model_name)

    def generate_answer(self, question: str, context: str, citations: str) -> str:
        prompt = PromptTemplate(
            input_variables=["question", "context", "citations"],
            template="""Given the following question, context, and citations, please provide a detailed and accurate answer:

            Question: {question}

            Context: {context}

            Citations: {citations}

            Answer:"""
        )
        formatted_prompt = prompt.format(question=question, context=context, citations=citations)
        response = self.model.invoke(formatted_prompt)
        return response.content.strip()

# === 6. Streamlit UI ===
def main():
    st.set_page_config(page_title="RAG Điện Viễn Thông (Nội bộ)", layout="wide")
    st.title("📄 Hệ thống truy vấn tài liệu thông minh")

    # Khởi tạo session state
    if "processor" not in st.session_state:
        st.session_state.processor = None
    if "tfidf" not in st.session_state:
        st.session_state.tfidf = None
    if "chroma" not in st.session_state:
        st.session_state.chroma = None

    # Kiểm tra nếu cơ sở dữ liệu và vectorizer đã tồn tại
    has_db = check_existing_data()

    if has_db:
        st.sidebar.info("CSDL đã có dữ liệu. Bạn có thể bắt đầu truy vấn câu hỏi ngay.")
        # Khởi tạo ChromaDB và TFIDFEmbedding từ dữ liệu hiện có
        if st.session_state.chroma is None:
            st.session_state.chroma = ChromaDBManager(persist_directory="./chroma_db")
        if st.session_state.tfidf is None:
            st.session_state.tfidf = TFIDFEmbedding(vectorizer_path="./data/tfidf_vectorizer.pkl")
    else:
        st.sidebar.warning("Không có dữ liệu trong CSDL hoặc mô hình TF-IDF chưa được tạo. Vui lòng tải lên tài liệu PDF để xử lý.")
        pdf_file = st.file_uploader("1. Tải lên file Document_hht.pdf", type="pdf")
        if pdf_file:
            with st.spinner("Đang trích xuất và chia nhỏ văn bản…"):
                # Xử lý tài liệu
                proc = DocumentProcessor()
                chunks = proc.load_and_split(pdf_file)
                if not chunks:
                    st.error("Không thể trích xuất văn bản từ PDF. Vui lòng kiểm tra file.")
                    return

                # Khởi tạo và fit TFIDF
                tfidf = TFIDFEmbedding(vectorizer_path="./data/tfidf_vectorizer.pkl")
                tfidf.fit(chunks)
                embeddings = tfidf(chunks)

                # Lưu vào ChromaDB
                chroma = ChromaDBManager(persist_directory="./chroma_db")
                chroma.add(chunks, embeddings)

                # Lưu vào session
                st.session_state.processor = proc
                st.session_state.tfidf = tfidf
                st.session_state.chroma = chroma
            st.success("✅ Đã xử lý xong tài liệu và lưu vào ChromaDB!")

    # --- Nhập câu hỏi và truy vấn ---
    query = st.text_input("2. Nhập câu hỏi (tiếng Việt):")
    if query:
        if st.session_state.chroma is None or st.session_state.tfidf is None:
            st.error("Chưa có dữ liệu hoặc mô hình TF-IDF. Vui lòng tải lên tài liệu PDF trước.")
            return
        try:
            with st.spinner("Đang truy vấn và tạo câu trả lời…"):
                emb = st.session_state.tfidf([query])[0]
                res = st.session_state.chroma.query(emb, top_k=3)
                docs = res["documents"][0]
                metas = res["metadatas"][0]
                dists = res["distances"][0]
                
                context = " ".join(docs)
                citations = "\n".join([f"**{meta['source']}** (score={dist:.4f}):\n> {doc}" for meta, doc, dist in zip(metas, docs, dists)])

                answer_generator = AnswerGenerator()
                final_answer = answer_generator.generate_answer(query, context, citations)
            
            st.markdown("### Câu trả lời cuối cùng:")
            st.write(final_answer)

            with st.expander("📚 Xem các trích dẫn nguồn"):
                for citation in citations.split("\n"):
                    st.markdown(citation)
        except ValueError as e:
            st.error(f"Lỗi: {str(e)}")

if __name__ == "__main__":
    main()
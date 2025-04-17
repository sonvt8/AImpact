import os
import re
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    DirectoryLoader,
)

from langchain.prompts import PromptTemplate
import shutil

import streamlit as st  # Optional for visualization

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

class EmbeddingModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.embedding_fn = OpenAIEmbeddings(api_key=openai_api_key,model="text-embedding-3-small")
        elif model_type == "nomic":
            self.embedding_fn = OllamaEmbeddings(model="nomic-embed-text")

class LLMModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = ChatOpenAI(api_key=openai_api_key,model="gpt-4o-mini")
        else:
            self.client = ChatOllama(
                            model='llama3.2',
                            temperature=0.3,
                        )
        
class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

    def load_files(self, files_directory: str) -> List[Document]:
        """Load documents from a directory."""
        try:
            directory_path = os.path.join(os.getcwd(), files_directory)
            loaders = []
            
            if not os.path.exists(directory_path):
                st.error(f"Directory {directory_path} does not exist")
                return []
            
            # For multi documents with different file types
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                try:
                    if filename.endswith(".txt"):
                        loaders.append(TextLoader(file_path))
                    elif filename.endswith(".md"):
                        loaders.append(UnstructuredMarkdownLoader(file_path))
                    elif filename.endswith(".pdf"):
                        loaders.append(PyPDFLoader(file_path))
                    elif filename.endswith(".csv"):
                        loaders.append(CSVLoader(file_path))
                    else:
                        continue
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    
            merged_loader = MergedDataLoader(loaders=loaders)
            documents = merged_loader.load()

            st.success(f"Loaded {len(documents)} documents")
            return documents
        except Exception as e:
            st.error(f"Error loading PDFs: {str(e)}")
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        try:
            chunks = self.text_splitter.split_documents(documents)
            st.info(f"Split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            st.error(f"Error splitting documents: {str(e)}")
            return documents

class QueryExpansionRAG:
    """
    RAG system with query expansion capabilities.
    """

    def __init__(self, documents, embeddings):
        self.retriever = FAISS.from_documents(documents, embeddings).as_retriever(
            search_kwargs={"k": 3},
            search_type="similarity"
        )
        
    def retrieve_with_expansion(
        self, question: str
    ) -> List[Document]:
        docs = self.retriever.invoke(question)
        return docs

### === Add the Answer Generator ===  Final part###
# Add this new class for final answer generation
class AnswerGenerator:
    """
    Generates final answer from multiple document sources using LLM with proper citations.
    """

    def __init__(self, llm_type: str, temperature: float = 0):
        self.llm = LLMModel(llm_type)
        self.answer_generation_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are a highly knowledgeable assistant tasked with providing 
            comprehensive answers based on multiple document sources. Your goal is to 
            synthesize information accurately and provide well-structured responses.

            Question: {question}
            
            Below are relevant excerpts from different documents, each with a citation ID:
            {formatted_context}

            Please provide a detailed response that:
            1. Directly answers the question
            2. Uses specific citations in the format [CitationID] when referencing information
            3. Synthesizes information from multiple sources when relevant
            4. Highlights any contradictions between sources
            5. Uses quoted snippets from the sources when particularly relevant

            Format your response as follows:

            DIRECT ANSWER:
            [Concise answer with citations]

            DETAILED EXPLANATION:
            [Detailed explanation with citations and quoted snippets when relevant]

            KEY POINTS:
            - [Point 1 with citation]
            - [Point 2 with citation]
            - [Point 3 with citation]

            SOURCES CITED:
            [List the citation IDs used and their key contributions]

            Answer:""",
        )    

    def _prepare_citation_chunks(
        self, documents: List[Document], max_chunk_length: int = 250
    ) -> Tuple[str, Dict[str, Dict[str, str]]]:
        """
        Prepare context with citations and create a citation map.

        Args:
            documents: list of relevant documents
            max_chunk_length: Maximum length for document chunks

        Returns:
            Tuple of (formatted_context, citation)
        """
        citation_id = 1
        citation_chunks = []
        citation_map = {}

        for doc in documents:
            # Create a truncated chunk with context
            content = doc.page_content
            truncated_content = content[:max_chunk_length]
            if len(content) > max_chunk_length:
                truncated_content += "..."
            # Store the citation
            citation_ref = f"[Citation{citation_id}]"
            citation_chunks.append(f"{citation_ref}:\n{truncated_content}\n")
            citation_map[citation_ref] = {
                "content": truncated_content,
                "full_content": content
            }
            citation_id += 1
            

        formatted_context = "\n".join(citation_chunks)
        return formatted_context, citation_map

    def generate_answer(
        self, question: str, documents: List[Document]
    ) -> Dict[str, Any]:
        """
        Generate final answer from multiple search results with citations.

        Args:
            question: Original question
            results: Dictionary of query->documents mappings

        Returns:
            Dictionary containing answer and citation information
        """
        try:
            # Prepare context with citations
            formatted_context, citation_map = self._prepare_citation_chunks(documents)

            # Generate answer using LLM
            response = self.llm.client.invoke(
                self.answer_generation_prompt.format(
                    question=question, formatted_context=formatted_context
                )
            )

            return {
                "answer": response.content,
                "citations": citation_map,
                "formatted_context": formatted_context,
            }

        except Exception as e:
            st.error(f"Error generating final answer: {str(e)}")
            return {
                "answer": "Failed to generate answer due to an error.",
                "citations": {},
                "formatted_context": "",
            }


def main():
    st.set_page_config(page_title="RAG Query Expansion", layout="wide")

    st.title("RAG System with Query retrieving from Documents")

    # Initialize paths with absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_directory = os.path.join(current_dir, "data")
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None

    # Create directories if they don't exist
    os.makedirs(file_directory, exist_ok=True)

    # Display directory information
    st.sidebar.title("System Information")
    st.sidebar.info(f"Files Directory: {file_directory}")
    
    llm_type = st.sidebar.radio(
        "Select LLM Model:",
        ["openai", "ollama"],
        format_func=lambda x: "OpenAI GPT-4" if x == "openai" else "Ollama Llama2",
    )

    embedding_type = st.sidebar.radio(
        "Select Embedding Model:",
        ["openai", "nomic"],
        format_func=lambda x: {
            "openai": "OpenAI Embeddings",
            "nomic": "Nomic Embed Text (Ollama)",
        }[x],
    )

    # Sidebar controls
    st.sidebar.title("Controls")

    # File uploader for PDFs
    uploaded_files = st.sidebar.file_uploader(
        "Upload accessible files", 
        type=["pdf", "txt", "md", "csv"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.sidebar.success(f"Uploaded {len(uploaded_files)} files")
        # Save uploaded files to pdf_directory
        for uploaded_file in uploaded_files:
            with open(os.path.join(file_directory, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getvalue())

    # Process documents when button is clicked
    if st.sidebar.button("Process Documents"):
        with st.spinner("Processing documents..."):
            doc_processor = DocumentProcessor()
            embeddings = EmbeddingModel(embedding_type)
            documents = doc_processor.load_files(file_directory)
            if documents:
                splits = doc_processor.split_documents(documents)
                st.session_state.rag_system = QueryExpansionRAG(splits, embeddings.embedding_fn)
                if st.session_state.rag_system:
                    st.sidebar.success("Documents processed successfully!")
                else:
                    st.sidebar.error("Failed to process documents")
            else:
                st.sidebar.error("No documents found")
            

    # Main query interface
    st.header("Query Interface")

    # Text input for query
    query = st.text_input("Enter your question:")

    # Number of results slider
    k = st.slider("Number of results to return", min_value=1, max_value=10, value=3)

    ## What are the key findings presented?
    # Search button
    if st.button("Search"):
        if query:
            with st.spinner("Processing query..."):
                try:
                    # Initialize query expander and answer generator
                    answer_generator = AnswerGenerator(llm_type)
                    
                    docs = st.session_state.rag_system.retrieve_with_expansion(query)

                    # Generate final answer with citations
                    st.subheader("📝 Detailed Analysis")
                    with st.spinner("Generating comprehensive answer..."):
                        response_data = answer_generator.generate_answer(
                            query, docs
                        )

                        # Display the answer
                        st.markdown(response_data["answer"])

                        # Display citations
                        st.subheader("📚 Source Citations")
                        for citation_id, citation_data in response_data["citations"].items():
                            with st.expander(f"{citation_id} - Click to view source"):
                                st.markdown("**Excerpt:**")
                                st.markdown(f"```\n{citation_data['content']}\n```")

                                # Option to view full content
                                if st.button(f"View Full Content for {citation_id}"):
                                    st.markdown("**Full Content:**")
                                    st.markdown(
                                        f"```\n{citation_data['full_content']}\n```"
                                    )

                    # Option to view all search results
                    with st.expander("🔎 View All Search Results"):
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"**Query:** {query}")
                            st.markdown(f"*Document {i}:*")
                            st.markdown(f"```\n{doc.page_content[:500]}...\n```")
                            st.markdown("---")

                    # Generate final synthesized answer
                    st.subheader("🎯 Final Answer")
                    with st.spinner("Synthesizing final answer..."):
                        final_prompt = PromptTemplate(
                            input_variables=["question", "detailed_answer"],
                            template="""Based on the detailed analysis provided, generate a clear, 
                            concise final answer to the original question. Focus on the most important 
                            points while maintaining accuracy.

                            Original Question: {question}

                            Detailed Analysis:
                            {detailed_answer}

                            Please provide a final answer that:
                            1. Directly addresses the question
                            2. Summarizes the key points
                            3. Is clear and concise
                            4. Maintains the crucial citations

                            Final Answer:""",
                        )
                        
                        llm_model = LLMModel(llm_type)
                        final_response = llm_model.client.invoke(
                            final_prompt.format(
                                question=query, detailed_answer=response_data["answer"]
                            )
                        )

                        # Display final answer in a highlighted box
                        st.markdown("---")
                        st.markdown("### 💡 Summary")
                        st.markdown(
                            f"""
                            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                            {final_response.content}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
                    st.error("Full error:", exception=e)
        else:
            if not query:
                st.error("Please enter a query")

if __name__ == "__main__":
    main()
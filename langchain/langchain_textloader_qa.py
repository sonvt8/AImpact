from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)
from dotenv import load_dotenv
import pprint
import re
import os

load_dotenv()
loaders = []

# Data cleaning function
def clean_text(text):
    # Remove unwanted characters (e.g., digits, special characters)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Convert to lowercase
    text = text.lower()

    return text

directory_path = os.path.join(os.getcwd(), "data")
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

# Merge loaders
merged_loader = MergedDataLoader(loaders=loaders)

# Tải các tài liệu từ thư mục
documents = merged_loader.load()

# documents = TextLoader(os.path.join(directory_path, "news-report.txt")).load()
# print(document[:10])

# Split the text into characters
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# texts = text_splitter.split_text(cleaned_documents[0])
docs = text_splitter.split_documents(documents)

# cleanup the text
texts = [clean_text(doc.page_content) for doc in docs]

# print(texts)

# Load the OpenAI embeddings to vectorize the text
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# create the retriever from the loaded embeddings and documents
retriever = FAISS.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 2})


# Query the retriever
# query = "what did Martin Luther King Jr. dream about?"
query = "What specific actions have Vietnamese companies like FPT and Vingroup taken to develop AI infrastructure, and how do their efforts contribute to the national strategy for innovation?"
docs = retriever.invoke(query)

# pprint.pprint(f" => DOCS: {docs}:")

# (" => DOCS: [Document(id='a5e82bb0-0ceb-4c04-b319-3ae0480272e6', metadata={}, "
#  "page_content='artificial intelligence ai has evolved beyond its scientific "
#  'and technological origins into a transformative force in everyday life from '
#  'the way we interact with technology to how industries function ai is '
#  'seamlessly integrated into our daily experiences it is reshaping how we live '
#  'work and make decisions with the potential to revolutionize entire sectors '
#  'the purpose of this report is to explore the practical applications of ai in '
#  "various aspects of daily life examine its benefits and'), "
#  "Document(id='4cdc6976-f8fa-4e66-9ba5-319c6ebb66d3', metadata={}, "
#  "page_content='explore the practical applications of ai in various aspects of "
#  'daily life examine its benefits and challenges and look at the future of ai '
#  "in everyday activities')]:")

# Chat with the model and our docs


# # Create the chat prompt
prompt = ChatPromptTemplate.from_template(
    "Please use the following docs {docs},and answer the following question {query}. If you don't know, just say so!",
)

# # Create a chat model
# model = ChatOpenAI(model="gpt-4o-mini")
model = ChatOllama(
        model='llama3.2',
        temperature=0.3,
    )

chain = prompt | model | StrOutputParser()

response = chain.invoke({"docs": docs, "query": query})
print(f"Model Response::: \n \n{response}")
import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    DirectoryLoader,
)
from langchain_community.document_loaders.merge import MergedDataLoader

directory_path = os.path.join(os.getcwd(), "data")
docs = []
loaders = []

# # Create loaders for each file type
# text_loader = TextLoader(os.path.join(directory_path,"random_content.txt"))  
# md_loader = UnstructuredMarkdownLoader(os.path.join(directory_path,"random_content.md"))
# pdf_loader = PyPDFLoader(os.path.join(directory_path,"cnfgRX2530M5.pdf"))

# Duyệt qua tất cả file trong thư mục
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
docs = merged_loader.load()

# Hiển thị số lượng tài liệu đã tải được
print(f"Đã tải {len(docs)} tài liệu từ thư mục {directory_path}")

# In ra nội dung của các tài liệu (chỉ in nội dung của 5 tài liệu đầu tiên để kiểm tra)
for i, doc in enumerate(docs[:5]):
    print(f"Document {i + 1}:")
    print(doc.page_content[:500])  # Chỉ in ra 500 ký tự đầu của tài liệu
    print("-" * 50)
    
    
# dir_loader = DirectoryLoader(directory_path, glob="**/*.txt")
# dir_documents = dir_loader.load()

# print("Directory Text Documents:", dir_documents)

#[Document(metadata={'source': 'C:\\Private\\sonvt8\\Coding\\project\\python-AI-Udemy\\data\\random_content.txt'}, page_content='Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.')]
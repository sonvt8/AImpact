import csv
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")

class EmbeddingModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI()
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key,
                model_name="text-embedding-3-small",
            )
        elif model_type == "chroma":
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        elif model_type == "nomic":
            # using Ollama nomic-embed-text model
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key="ollama",
                api_base="http://localhost:11434/v1",
                model_name="nomic-embed-text",
            )
            
class LLMModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = "gpt-4o-mini"
        else:
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            self.model_name = "llama3.2"

    def generate_completion(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,  # 0.0 is deterministic
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
        
def select_models():
    # Select LLM Model
    print("\nSelect LLM Model:")
    print("1. OpenAI GPT-4")
    print("2. Ollama Llama2")
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            llm_type = "openai" if choice == "1" else "ollama"
            break
        print("Please enter either 1 or 2")

    # Select Embedding Model
    print("\nSelect Embedding Model:")
    print("1. OpenAI Embeddings")
    print("2. Chroma Default")
    print("3. Nomic Embed Text (Ollama)")
    while True:
        choice = input("Enter choice (1, 2, or 3): ").strip()
        if choice in ["1", "2", "3"]:
            embedding_type = {"1": "openai", "2": "chroma", "3": "nomic"}[choice]
            break
        print("Please enter 1, 2, or 3")

    return llm_type, embedding_type

def generate_csv():
    facts = [
        {"id": 1, "fact": "Global gold prices rose 15% in 2025 due to inflation and geopolitical tensions."},
        {"id": 2, "fact": "Vietnam's GDP is projected to grow 6.8% in 2025, driven by high-tech exports."},
        {"id": 3, "fact": "Middle East peace talks in 2025 remain stalled over resource disputes."},
        {"id": 4, "fact": "Vietnam holds local elections in May 2025, focusing on sustainable development."},
        {"id": 5, "fact": "Lithium-sulfur battery tech could double electric vehicle range by 2026."},
        {"id": 6, "fact": "NASA found evidence of liquid water on Mars, hinting at possible alien life."},
        {"id": 7, "fact": "20% more young Vietnamese adopted minimalism in 2025 due to financial pressures."},
        {"id": 8, "fact": "40% of Vietnamese companies maintain hybrid work models in 2025."},
        {"id": 9, "fact": "Many Vietnamese couples prefer pets over having children early."},
        {"id": 10, "fact": "Raising a child in Vietnam's big cities costs $12,000 annually on average."},
    ]

    with open("daily_facts.csv", mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "fact"])
        writer.writeheader()
        writer.writerows(facts)

    print("CSV file 'nowadays_facts.csv' created successfully!")
    
def load_csv():
    df = pd.read_csv("daily_facts.csv")
    documents = df["fact"].tolist()
    print("\nLoaded documents:")
    for doc in documents:
        print(f"- {doc}")
    return documents

def setup_chromadb(documents, embedding_model):
    client = chromadb.Client()

    try:
        client.delete_collection("daily_facts")
    except:
        pass

    collection = client.create_collection(
        name="daily_facts", embedding_function=embedding_model.embedding_fn
    )

    collection.add(documents=documents, ids=[str(i) for i in range(len(documents))])

    print("\nDocuments added to ChromaDB collection successfully!")
    return collection

def find_related_chunks(query, collection, top_k=2):
    results = collection.query(query_texts=[query], n_results=top_k)

    print("\nRelated chunks found:")
    for doc in results["documents"][0]:
        print(f"- {doc}")

    return list(
        zip(
            results["documents"][0],
            (
                results["metadatas"][0]
                if results["metadatas"][0]
                else [{}] * len(results["documents"][0])
            ),
        )
    )
    
def augment_prompt(query, related_chunks):
    context = "\n".join([chunk[0] for chunk in related_chunks])
    augmented_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    print("\nAugmented prompt: ⤵️")
    print(augmented_prompt)

    return augmented_prompt

def rag_pipeline(query, collection, llm_model, top_k=2):
    print(f"\nProcessing query: {query}")

    related_chunks = find_related_chunks(query, collection, top_k)
    augmented_prompt = augment_prompt(query, related_chunks)

    response = llm_model.generate_completion(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant who answers questions based solely on provided facts. If the answer is unclear, respond with 'I don't know.'"
            },
            {"role": "user", "content": augmented_prompt},
        ]
    )

    print("\nGenerated response:")
    print(response)

    references = [chunk[0] for chunk in related_chunks]
    return response, references

def main():
    print("Starting the RAG pipeline demo...")

    # Select models
    llm_type, embedding_type = select_models()

    # Initialize models
    llm_model = LLMModel(llm_type)
    embedding_model = EmbeddingModel(embedding_type)

    print(f"\nUsing LLM: {llm_type.upper()}")
    print(f"Using Embeddings: {embedding_type.upper()}")

    # Generate and load data
    generate_csv()
    documents = load_csv()

    # Setup ChromaDB
    collection = setup_chromadb(documents, embedding_model)

    # Run queries
    queries = [
        "What are the key issues in Vietnam's local elections this year?",
        "Why are peace talks in the Middle East facing challenges in 2025",
    ]

    for query in queries:
        print("\n" + "=" * 50)
        print(f"Processing query: {query}")
        response, references = rag_pipeline(query, collection, llm_model)

        print("\nFinal Results:")
        print("-" * 30)
        print("Response:", response)
        print("\nReferences used:")
        for ref in references:
            print(f"- {ref}")
        print("=" * 50)
        
if __name__ == "__main__":
    main()
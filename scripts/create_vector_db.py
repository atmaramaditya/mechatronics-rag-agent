import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# 1. Initialize the Embedding Model (The "Translator")
# This model turns English sentences into lists of numbers
model = SentenceTransformer('all-MiniLM-L6-v2')

def build_database():
    # 2. Setup ChromaDB (Local Storage)
    # This creates a folder named 'vector_db' to save your data permanently
    client = chromadb.PersistentClient(path="./vector_db")
    collection = client.get_or_create_collection(name="engineering_docs")

    pdf_folder = "./data"
    
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            print(f"Adding {filename} to database...")
            doc = fitz.open(os.path.join(pdf_folder, filename))
            
            # Extract and Chunk (Same logic as Phase 1)
            full_text = "".join([page.get_text() for page in doc])
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(full_text)

            # 3. Create Embeddings and Store
            # We give each chunk a unique ID based on the filename and index
            for i, chunk in enumerate(chunks):
                vector = model.encode(chunk).tolist()
                collection.add(
                    ids=[f"{filename}_{i}"],
                    embeddings=[vector],
                    documents=[chunk],
                    metadatas=[{"source": filename}]
                )
    
    print("\n✅ Database built successfully! Your AI now has 'Permanent Memory'.")

if __name__ == "__main__":
    build_database()
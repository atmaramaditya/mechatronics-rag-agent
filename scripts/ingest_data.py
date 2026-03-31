import fitz  # PyMuPDF
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_process_pdfs(folder_path):
    # Check if data folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found!")
        return

    # Loop through all files in the data folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            print(f"\n--- Processing: {filename} ---")
            
            try:
                # 1. Extract Text
                doc = fitz.open(os.path.join(folder_path, filename))
                full_text = ""
                for page in doc:
                    full_text += page.get_text()
                
                # 2. Semantic Chunking
                # We split text into 1000 character pieces with 100 char overlap
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )
                chunks = splitter.split_text(full_text)
                
                print(f"Result: Successfully split into {len(chunks)} chunks.")
                if chunks:
                    print(f"Preview of first chunk: {chunks[0][:150]}...")
            
            except Exception as e:
                print(f"Could not process {filename}: {e}")

if __name__ == "__main__":
    # This assumes your PDFs are in a folder named 'data' in the root directory
    load_and_process_pdfs("./data")
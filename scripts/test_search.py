import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def ask_my_docs(question):
    client = chromadb.PersistentClient(path="./vector_db")
    collection = client.get_collection(name="engineering_docs")

    query_vector = model.encode(question).tolist()

    # UPGRADE: Increased n_results to 5 to find the "hidden" correct answer
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=5 
    )

    print(f"\n🔍 Question: {question}")
    print("=" * 50)
    
    for i, doc in enumerate(results['documents'][0]):
        source = results['metadatas'][0][i]['source']
        # Highlight if we found the right source!
        if "arduino" in source.lower():
            print(f"🌟 TOP MATCH (Source: {source}):")
        else:
            print(f"MATCH {i+1} (Source: {source}):")
            
        print(f"{doc[:400]}") # Show more text to find the voltage info
        print("-" * 30)

if __name__ == "__main__":
    # Let's try a more specific question to help the vector search
    user_query = "What are the input voltage limits for the Arduino Uno board?"
    ask_my_docs(user_query)
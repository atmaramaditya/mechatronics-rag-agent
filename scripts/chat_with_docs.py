import os
import chromadb
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 1. Setup Environment and API
load_dotenv()
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
model_embed = SentenceTransformer('all-MiniLM-L6-v2')

def get_context(question):
    # Connect to your vector_db from Phase 2
    client_db = chromadb.PersistentClient(path="./vector_db")
    collection = client_db.get_collection(name="engineering_docs")
    
    # Search for the top 3 matches
    query_vector = model_embed.encode(question).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=3)
    
    # Combine the results into one block of text
    return "\n".join(results['documents'][0])

def chat():
    print("🤖 Engineering AI Assistant Active! (Type 'exit' to quit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit': break

        # A. Get relevant info from your PDFs
        context = get_context(user_input)
        
        print(f"\n[DEBUG] AI is reading this context: {context[:300]}...")

        # B. Send context + question to Groq (Llama 3)
        prompt = f"""
        You are a professional Mechatronics Assistant. 
        
        INSTRUCTIONS:
        1. Use the provided TECHNICAL CONTEXT as your primary source of truth.
        2. If the context contains the answer, cite the source (e.g., "According to the Arduino Datasheet...").
        3. If the context is missing a specific detail but you know the answer from your general engineering training, provide the answer but clarify that it is general knowledge.
        4. Be technical and precise.

        TECHNICAL CONTEXT:
        {context}

        USER QUESTION: 
        {user_input}
        """

        chat_completion = client_groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant", # Using a fast, smart open-source model
        )

        print(f"\nAI: {chat_completion.choices[0].message.content}")

if __name__ == "__main__":
    chat()
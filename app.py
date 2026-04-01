import streamlit as st
import os
import chromadb
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.tools import DuckDuckGoSearchRun
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# 1. Setup
load_dotenv()
st.set_page_config(page_title="Mechatronics AI Agent", page_icon="🤖")
search = DuckDuckGoSearchRun()
model_embed = SentenceTransformer('all-MiniLM-L6-v2')
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 2. Helper Functions
def get_local_context(query):
    try:
        client_db = chromadb.PersistentClient(path="./vector_db")
        collection = client_db.get_collection(name="engineering_docs")
        query_vector = model_embed.encode(query).tolist()
        results = collection.query(query_embeddings=[query_vector], n_results=3)
        return "\n".join(results['documents'][0])
    except:
        return ""

# 3. Streamlit UI
st.title("🤖 Mechatronics Engineering Agent")
st.markdown("Querying your **PDFs** + **Live Web Search**")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about Arduino, Sensors, or Latest Tech..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("🔍 Searching Internal PDFs & Web...") as status:
            # A. Get Local Data
            local_info = get_local_context(prompt)
            # B. Get Web Data
            web_info = search.run(prompt)
            status.update(label="✅ Information Found! Generating Answer...", state="complete")

        # C. Generate Final Answer
        sys_prompt = f"""
        You are an expert Mechatronics Engineer. 
        INTERNAL PDF INFO: {local_info}
        WEB INFO: {web_info}
        
        Combine both sources to answer the user. If the PDF has the answer, prioritize it. 
        If it's a new tech question, use the Web Info.
        """
        
        response = client_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": sys_prompt}, 
                      {"role": "user", "content": prompt}]
        )
        
        answer = response.choices[0].message.content
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

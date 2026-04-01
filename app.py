import streamlit as st
import os
import chromadb
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.tools import DuckDuckGoSearchRun

# 0. Global Fixes
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# 1. Setup & Branding
load_dotenv()
st.set_page_config(
    page_title="Aditya's Mechatronics AI", 
    page_icon="⚙️", 
    layout="wide"
)

# Professional Gradient Background & Glassmorphism UI
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: #f8fafc;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Chat Input Styling */
    .stChatInput {
        background-color: rgba(30, 41, 59, 0.7) !important;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* Message Bubbles */
    .stChatMessage {
        background-color: rgba(51, 65, 85, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }

    /* Headings */
    h1, h2, h3, p {
        color: #f8fafc !important;
    }

    .stStatus {
        background-color: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid #38bdf8 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar - Personal Branding
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1998/1998614.png", width=80) # Replace with your photo URL
    st.title("Aditya Atmaram")
    st.markdown("""
    **Mechatronics Engineer & Data Scientist**
    *B.Tech @ Mukesh Patel (MPSTME)*
    *Diploma in AI @ BIA*
    
    ---
    ### 🛠️ Agent Capabilities
    - **PDF RAG:** Accesses internal engineering docs.
    - **Live Web:** Browses for 2026 tech trends.
    - **Logic:** Powered by Llama 3.1 & Groq.
    
    ---
    ### 🔗 Quick Links
    [GitHub](https://github.com/atmaramaditya) | [LinkedIn](#)
    """)
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# 3. Helper Functions
def get_local_context(query):
    try:
        client_db = chromadb.PersistentClient(path="./vector_db")
        collection = client_db.get_collection(name="engineering_docs")
        query_vector = model_embed.encode(query).tolist()
        results = collection.query(query_embeddings=[query_vector], n_results=3)
        return "\n".join(results['documents'][0])
    except Exception as e:
        return ""

# 4. Main UI Logic
st.title("🤖 Aditya's Mechatronics Agent")
st.caption("Fusing Mechanical Precision with AI Intelligence")

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
        # Step A & B: Gathering Intelligence
        with st.status("🚀 Engineering Intelligence Hub Active...", expanded=False) as status:
            st.write("Reading internal PDF vectors...")
            local_info = get_local_context(prompt)
            
            st.write("Querying DuckDuckGo for live updates...")
            try:
                web_info = search.run(prompt)
            except:
                web_info = "Search temporarily unavailable."
                
            status.update(label="✅ Context Synthesized!", state="complete")

        # C. Generate Final Answer
        # Note: We include your identity in the system prompt so the AI knows its creator
        sys_prompt = f"""
        You are the Personal AI Assistant of Aditya Atmaram, a Mechatronics Engineer and Data Scientist.
        Your goal is to provide expert technical advice.
        
        ADITYA'S KNOWLEDGE BASE (PDF): {local_info}
        LIVE WEB UPDATES: {web_info}
        
        RULES:
        1. If the PDF has the answer, prioritize it as "Aditya's Verified Data."
        2. Use Web Info for current events or missing specs.
        3. Use LaTeX for math/formulas (e.g., $V = IR$).
        4. Be concise but highly technical.
        """
        
        response = client_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": sys_prompt}, 
                      {"role": "user", "content": prompt}]
        )
        
        answer = response.choices[0].message.content
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

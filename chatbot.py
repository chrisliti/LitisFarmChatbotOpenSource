import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

# --- 1. CONFIGURATION & UI SETUP ---
INF_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

st.set_page_config(
    page_title="LitisFarm Intelligence", 
    page_icon="🌿", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Header with Clear Chat Button
header_col, button_col = st.columns([3, 1])

with header_col:
    st.title("🌿 LitisFarm AI")
    # st.caption(f"Engine: {INF_MODEL}")

with button_col:
    st.write("##") 
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        # Clear LangChain memory if it exists
        if "qa_chain" in st.session_state:
            st.session_state.qa_chain.memory.clear()
        st.rerun()
# --- NEW: Short Context Expander ---
with st.expander("📖 About LitisFarm Chatbot", expanded=True):
    st.markdown("""
    Welcome to your digital agronomist! This AI is connected to the **LitisFarm Knowledge Base** to help you manage your crops more effectively.
    
    """)

st.divider()

st.divider()

# --- 2. BACKEND LOGIC ---
@st.cache_resource
def setup_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if not os.path.exists("faiss_index"):
        st.error("Vector database 'faiss_index' not found. Run ingest.py first!")
        st.stop()
        
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    llm = ChatGroq(
        temperature=0.3, 
        model_name=INF_MODEL, 
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key="answer"
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = setup_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Chat with Liti'sFarm AI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Scanning LitisFarm knowledge base..."):
            response = st.session_state.qa_chain.invoke({"question": prompt})
            answer = response["answer"]
            st.markdown(answer)
            
            if response["source_documents"]:
                with st.expander("Reference Sources"):
                    for doc in response["source_documents"]:
                        st.write(f"📍 {doc.metadata.get('source', 'Unknown source')}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
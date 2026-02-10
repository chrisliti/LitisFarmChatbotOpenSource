import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def create_vector_db():
    # 1. Load documents from 'data' folder
    loader = DirectoryLoader('docs/', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # 3. Create Embeddings (using a lightweight HuggingFace model)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 4. Create and Save FAISS index
    vector_db = FAISS.from_documents(texts, embeddings)
    vector_db.save_local("faiss_index")
    print("Vector database created and saved to 'faiss_index'")

if __name__ == "__main__":
    create_vector_db()
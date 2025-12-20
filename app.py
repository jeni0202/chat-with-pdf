"""Main Streamlit application for Chat with PDF"""
import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from utils.document_loader import PDFDocumentLoader
from utils.vector_store import VectorStoreManager
from utils.rag_chain import RAGChain
import tempfile

# Load environment variables from .env file in the current directory
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Do not require OpenAI API key — use local HF embeddings/generation by default to avoid quota errors
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.sidebar.info("No OPENAI_API_KEY found — using local HuggingFace models for embeddings and generation by default.")

# Configure Streamlit page - minimal for speed
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize Streamlit session state"""
    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None


def load_pdf(uploaded_file, store_type: str = "faiss", embedding_type: str = "hf", hf_model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 400, k: int = 2):
    """Load and process PDF file"""
    try:
        with st.spinner("Processing PDF..."):
            # Save uploaded file to temporary location
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and split document with configurable chunk size for speed
            loader = PDFDocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_size//4)
            documents = loader.load_and_split(temp_path)
            
            # Create vector store using local HF embeddings by default to avoid OpenAI quota issues
            vector_store_manager = VectorStoreManager(
                store_type=store_type,
                persist_directory="vector_store",
                embedding_type=embedding_type,
                hf_model_name=hf_model_name,
            )
            vector_store_manager.create_vector_store(documents)
            
            # Create RAG chain using requested number of context docs (k)
            rag_chain = RAGChain(
                vector_store=vector_store_manager.vector_store,
                k=k
            )
            
            st.session_state.vector_store_manager = vector_store_manager
            st.session_state.rag_chain = rag_chain
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.messages = []
            
            st.success(f"✅ Successfully loaded '{uploaded_file.name}'")
            
    except Exception as e:
        st.error(f"❌ Error loading PDF: {str(e)}")


def chat_interface():
    """Display chat interface"""
    if st.session_state.rag_chain is None:
        st.info("👈 Upload a PDF in the sidebar to get started!")
        return
    
    st.subheader(f"📄 Chatting with: {st.session_state.pdf_name}")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the PDF..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag_chain.query(prompt)
                    answer = result["answer"]
                    source_docs = result["source_documents"]
                    
                    st.markdown(answer)
                    
                    # Display source documents
                    with st.expander("📚 Source Documents"):
                        for i, doc in enumerate(source_docs, 1):
                            st.markdown(f"**Document {i}:**")
                            st.markdown(f"*Page: {doc.metadata.get('page', 'N/A')}*")
                            st.markdown(f"```\n{doc.page_content[:500]}...\n```")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


def sidebar():
    """Sidebar configuration"""
    st.sidebar.title("⚙️ Configuration")
    
    # API Key check
    if not os.getenv("OPENAI_API_KEY"):
        st.sidebar.warning(
            "🔑 OpenAI API Key not found. Please set OPENAI_API_KEY in .env file"
        )
    
    # Vector store type (hidden) — using FAISS by default
    store_type = "faiss"
    # Embedding source: using local HF models by default to avoid OpenAI quota errors
    embedding_source = "hf"
    hf_model = st.sidebar.text_input("HF model name (embeddings)", value="all-MiniLM-L6-v2", help="Lighter models: all-MiniLM-L6-v2 (fast), all-distilroberta-v1 (faster)")
    
    # Fixed chunk size for faster, consistent processing
    CHUNK_SIZE = 400

    # Context docs (k) - keep this configurable for tradeoff between speed/accuracy
    k = st.sidebar.slider(
        "Context Docs",
        min_value=1,
        max_value=5,
        value=2,
        help="Fewer = faster (recommended: 1-2)"
    )

    # PDF upload
    st.sidebar.markdown("---")
    st.sidebar.subheader("📤 Upload PDF")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF to chat with"
    )

    if uploaded_file is not None:
        load_pdf(
            uploaded_file,
            store_type=store_type,
            embedding_type=embedding_source,
            hf_model_name=(hf_model or "all-MiniLM-L6-v2"),
            chunk_size=CHUNK_SIZE,
            k=k,
        )
    
    # Model configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Model Settings")
    st.sidebar.markdown("Deterministic extractive QA (no temperature control).")
    
    # Clear chat history
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.success("Chat history cleared!")
    
    # Information
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info(
        """
        **Chat with PDF** uses RAG (Retrieval-Augmented Generation) to:
        - Load and process PDF documents
        - Create embeddings using OpenAI
        - Store them in FAISS or Chroma
        - Answer questions based on document content
        """
    )


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.title("📄 Chat with Your PDF")
    st.markdown("*Powered by LangChain, OpenAI, and Vector Databases*")
    
    # Layout
    sidebar()
    chat_interface()


if __name__ == "__main__":
    main()

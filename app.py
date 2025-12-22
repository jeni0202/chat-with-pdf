"""Main Streamlit application for Chat with PDF"""
import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from utils.document_loader import PDFDocumentLoader
from utils.vector_store import VectorStoreManager
from utils.rag_chain import RAGChain
import tempfile
import hashlib
from typing import List, Dict

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

# Themed frontend (light professional) matching provided screenshot
st.markdown(
    """
    <style>
    /* Light professional theme */
    html, body, [class*="css"] { background: #f8fafc !important; color: #0f172a !important; }
    .stApp, .main, .block-container { background: transparent !important; color: #0f172a !important; }
    .stSidebar { background: #f1f5f9 !important; color: #0f172a !important; padding: 18px 16px !important; }
    .stSidebar .stMarkdown, .stSidebar .stText { color: #0f172a !important; }
    .stButton>button, button[kind] { background: #ffffff !important; color: #0f172a !important; border: 1px solid #e6eef8 !important; border-radius: 10px !important; }
    .card, .streamlit-expanderHeader { background: #ffffff; color: #0f172a; border-radius:12px; box-shadow: 0 2px 8px rgba(15,23,42,0.06); padding: 14px; border: 1px solid #e6eef8; }
    .stAlert { background: #ecfdf5 !important; color: #065f46 !important; border-left: 4px solid #bbf7d0 !important; border-radius:8px !important; padding:10px 12px !important; }
    .stExpander { background: #fff !important; border-radius: 8px; }
    [data-testid="stFileUploader"] { background:#fff; border-radius:8px; border:1px solid #e6eef8; padding:12px; }
    .stTextInput>div, .stTextArea>div { background: #fbfdff !important; color: #0f172a !important; border: 1px solid #e6eef8 !important; border-radius: 999px !important; padding: 10px 14px !important; }
    .stChatInput>div { background:#f1f5f9 !important; border-radius:999px !important; padding:8px 12px !important; }
    a { color: #2563eb !important; }
    .small-muted { color: #64748b !important; }
    .title { font-weight:700; color:#0f172a; font-size:28px; }
    /* Make success messages and toasts match screenshot */
    .stSuccess { background: #ecfdf5 !important; color: #065f46 !important; }
    /* Tweak markdown code blocks */
    pre, code { background: #f8fafc; color: #0f172a; border: 1px solid #e6eef8; padding:8px; border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
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
    if "_split_cache" not in st.session_state:
        st.session_state["_split_cache"] = {}


def load_pdf(uploaded_file, store_type: str = "faiss", embedding_type: str = "hf", hf_model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 400, k: int = 2):
    """Load and process PDF file"""
    try:
        with st.spinner("Processing PDF..."):
            # Save uploaded file to temporary location
            file_bytes = uploaded_file.getbuffer().tobytes()
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            # Use cached split if available to avoid re-processing same file
            cache_key = hashlib.sha256(file_bytes).hexdigest()
            if cache_key in st.session_state.get("_split_cache", {}):
                documents_serialized = st.session_state["_split_cache"][cache_key]
            else:
                documents_serialized = _cached_split_pdf(file_bytes, uploaded_file.name, chunk_size)
                # store in session cache (small memory cost)
                st.session_state["_split_cache"][cache_key] = documents_serialized
            # Reconstruct Documents
            documents = [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in documents_serialized]
            
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
        st.error("Check the Streamlit terminal for detailed logs.")


def chat_interface():
    """Display chat interface"""
    if st.session_state.rag_chain is None:
        st.info("👈 Upload a PDF in the sidebar to get started!")
        return
    
    st.subheader(f"📄 Chatting with: {st.session_state.pdf_name}")
    
    # Display chat messages
    ui_mode = st.session_state.get("ui_mode", "Professional")
    if ui_mode == "Classic":
        recent = st.session_state.messages
    else:
        # Limit to last 20 in Professional mode to avoid rendering lag
        recent = st.session_state.messages[-20:]

    for message in recent:
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
                            if ui_mode == "Classic":
                                st.markdown(f"**Document {i}:**")
                                st.markdown(f"*Page: {doc.metadata.get('page', 'N/A')}*")
                                st.markdown(f"```\n{doc.page_content[:500]}...\n```")
                            else:
                                st.markdown(f"<div class='card'>**Document {i}:**<div class='small-muted'>Page: {doc.metadata.get('page', 'N/A')}</div>\n\n{doc.page_content[:800]}...</div>", unsafe_allow_html=True)
                    
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
            hf_model_name=hf_model,
            chunk_size=CHUNK_SIZE,
            k=k,
        )


def main():
    """Main application"""
    initialize_session_state()

    ui_mode = st.session_state.get("ui_mode", "Professional")

    # Inject CSS only for Professional mode
    if ui_mode == "Professional":
        st.markdown(
            """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
            html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
            .stApp { background-color: #f7fafc; }
            .title { color: #0f172a; font-weight:600; }
            .card { background: white; border-radius:10px; padding:16px; box-shadow: 0 2px 6px rgba(15,23,42,0.06); }
            .small-muted { color: #64748b; font-size:0.9rem }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Professional header
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("<div style='font-size:32px'>📄</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='title' style='font-size:20px'>Chat with Your PDF</div>", unsafe_allow_html=True)
            st.markdown("<div class='small-muted'>Answer questions directly from your PDF — fast, deterministic, and grounded in your document.</div>", unsafe_allow_html=True)
    else:
        # Classic header (simple)
        st.title("📄 Chat with Your PDF")
        st.markdown("Ask questions and get extractive answers from your document.")

    # Layout
    sidebar()
    # Main content
    chat_interface()


@st.cache_data(max_entries=10)
def _cached_split_pdf(file_bytes: bytes, filename: str, chunk_size: int) -> List[Dict]:
    """Write bytes to temp file, split using PDFDocumentLoader and return serializable list of dicts."""
    import tempfile as _temp
    tmp = _temp.mkdtemp()
    path = os.path.join(tmp, filename)
    with open(path, "wb") as f:
        f.write(file_bytes)

    loader = PDFDocumentLoader(chunk_size=chunk_size, chunk_overlap=max(50, chunk_size // 8))
    docs = loader.load_and_split(path)
    serial = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
    return serial


if __name__ == "__main__":
    main()

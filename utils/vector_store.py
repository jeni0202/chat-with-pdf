"""Vector store management module"""
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings
from typing import Iterable
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBT = True
except Exception:
    HAS_SBT = False


class HFLocalEmbeddings:
    """Simple local embeddings wrapper using SentenceTransformers.

    Exposes `embed_documents` and `embed_query` for compatibility with
    LangChain vectorstore constructors.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not HAS_SBT:
            raise ImportError("sentence-transformers is not installed")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        arr = self.model.encode(list(texts), show_progress_bar=False)
        return [list(map(float, v)) for v in np.array(arr).tolist()]

    def embed_query(self, text: str) -> list[float]:
        v = self.model.encode([text], show_progress_bar=False)
        return list(map(float, np.array(v).tolist()[0]))

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector store operations with FAISS or Chroma"""
    
    def __init__(self, store_type: str = "faiss", persist_directory: str = "vector_store", embedding_type: str = "openai", hf_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector store manager.
        
        Args:
            store_type: Type of vector store ("faiss" or "chroma")
            persist_directory: Directory to persist the vector store
        """
        self.store_type = store_type.lower()
        self.persist_directory = persist_directory
        self.embedding_type = embedding_type.lower()
        self.hf_model_name = hf_model_name
        if self.embedding_type == "openai":
            self.embeddings = OpenAIEmbeddings()
        elif self.embedding_type == "hf":
            # Use local SentenceTransformer wrapper
            self.embeddings = HFLocalEmbeddings(model_name=hf_model_name)
        else:
            raise ValueError(f"Unsupported embedding_type: {embedding_type}")
        self.vector_store = None
        
        if self.store_type not in ["faiss", "chroma"]:
            raise ValueError(f"Unsupported store type: {store_type}")
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of documents to add to the vector store
        """
        try:
            if self.store_type == "faiss":
                self.vector_store = FAISS.from_documents(
                    documents, self.embeddings
                )
                self.save_faiss()
                logger.info(f"Created FAISS vector store with {len(documents)} documents")
            
            elif self.store_type == "chroma":
                os.makedirs(self.persist_directory, exist_ok=True)
                self.vector_store = Chroma.from_documents(
                    documents,
                    self.embeddings,
                    persist_directory=self.persist_directory
                )
                logger.info(f"Created Chroma vector store with {len(documents)} documents")
        
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to existing vector store.
        
        Args:
            documents: Documents to add
        """
        if self.vector_store is None:
            self.create_vector_store(documents)
        else:
            try:
                if self.store_type == "faiss":
                    self.vector_store.add_documents(documents)
                    self.save_faiss()
                elif self.store_type == "chroma":
                    self.vector_store.add_documents(documents)
                    self.vector_store.persist()
                logger.info(f"Added {len(documents)} documents to vector store")
            except Exception as e:
                logger.error(f"Error adding documents: {e}")
                raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Load or create one first.")
        
        try:
            # Different vectorstore versions expose differently named methods.
            if hasattr(self.vector_store, "similarity_search_with_scores"):
                return self.vector_store.similarity_search_with_scores(query, k=k)
            if hasattr(self.vector_store, "similarity_search_with_score"):
                return self.vector_store.similarity_search_with_score(query, k=k)
            if hasattr(self.vector_store, "similarity_search"):
                docs = self.vector_store.similarity_search(query, k=k)
                return [(d, None) for d in docs]
            raise AttributeError("No compatible similarity_search method found on vector store")
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise
    
    def save_faiss(self) -> None:
        """Save FAISS vector store to disk"""
        if self.store_type == "faiss" and self.vector_store:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vector_store.save_local(self.persist_directory)
            logger.info(f"Saved FAISS vector store to {self.persist_directory}")
    
    def load_faiss(self) -> None:
        """Load FAISS vector store from disk"""
        if self.store_type == "faiss":
            try:
                self.vector_store = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings
                )
                logger.info(f"Loaded FAISS vector store from {self.persist_directory}")
            except Exception as e:
                logger.warning(f"Could not load FAISS vector store: {e}")
    
    def load_chroma(self) -> None:
        """Load Chroma vector store from disk"""
        if self.store_type == "chroma":
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info(f"Loaded Chroma vector store from {self.persist_directory}")
            except Exception as e:
                logger.warning(f"Could not load Chroma vector store: {e}")
    
    def load_vector_store(self) -> None:
        """Load existing vector store from disk"""
        if self.store_type == "faiss":
            self.load_faiss()
        elif self.store_type == "chroma":
            self.load_chroma()
    
    def delete_vector_store(self) -> None:
        """Delete vector store from disk"""
        import shutil
        try:
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                logger.info(f"Deleted vector store at {self.persist_directory}")
                self.vector_store = None
        except Exception as e:
            logger.error(f"Error deleting vector store: {e}")

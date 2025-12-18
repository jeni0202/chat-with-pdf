"""RAG chain module for question answering"""
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import logging
import os
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    HAS_HF_GEN = True
except Exception:
    HAS_HF_GEN = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChain:
    """RAG chain for document-based question answering"""
    
    def __init__(
        self,
        vector_store,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        k: int = 4,
        generation_provider: str = "auto",
    ):
        """
        Initialize RAG chain.
        
        Args:
            vector_store: Vector store instance with retriever capability
            model_name: LLM model to use
            temperature: Temperature for generation
            k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.temperature = temperature
        self.k = k
        
        # Only initialize OpenAI LLM if explicitly requested and API key exists
        self.generation_provider = generation_provider
        self.llm = None
        if generation_provider in ("openai", "auto") and os.getenv("OPENAI_API_KEY"):
            try:
                self.llm = ChatOpenAI(
                    model_name=model_name,
                    temperature=temperature
                )
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI LLM: {e}")
        
        self.retriever = vector_store.as_retriever(search_kwargs={"k": k})
        self.qa_chain = self._create_qa_chain()
        # Prepare local HF generator as fallback
        self.hf_generator = None
        if HAS_HF_GEN:
            try:
                model_name = os.getenv("HF_FALLBACK_MODEL", "google/flan-t5-small")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.hf_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
                logger.info(f"Local HF generator loaded: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load local HF generator: {e}")
    
    def _create_qa_chain(self):
        """Create the QA chain with custom prompt"""
        # Only create OpenAI-based chain if LLM is available
        if not self.llm:
            return None
        
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        def format_docs(docs):
            """Format documents into a string"""
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create the chain
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def query(self, question: str) -> dict:
        """
        Query the RAG chain.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and source documents
        """
        # First retrieve source documents
        try:
            source_docs = self.retriever.invoke(question)
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise

        # Build prompt context
        context = "\n\n".join(d.page_content for d in source_docs)
        prompt = (
            "Use the following pieces of context to answer the question at the end. "
            "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

        # Try OpenAI LLM first if available
        if self.qa_chain and self.llm:
            try:
                answer = self.qa_chain.invoke(question)
                logger.info(f"Query processed (OpenAI): {question[:50]}...")
                return {"answer": answer, "source_documents": source_docs}
            except Exception as e:
                logger.warning(f"OpenAI generation failed: {e}")

        # Use local HF generator (primary if llm is None)
        try:
            if not self.hf_generator:
                raise RuntimeError("No LLM or HF generator available")
            gen = self.hf_generator(prompt, max_length=512)
            text = gen[0]["generated_text"] if isinstance(gen, list) and gen else str(gen)
            logger.info("Query processed (HF local generation)")
            answer = text
            return {"answer": answer, "source_documents": source_docs}
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def get_relevant_documents(self, question: str) -> List[Document]:
        """
        Get relevant documents for a question without generating an answer.
        
        Args:
            question: User question
            
        Returns:
            List of relevant documents
        """
        try:
            docs = self.retriever.invoke(question)
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise

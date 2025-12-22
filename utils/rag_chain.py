"""Robust extractive RAG chain using HuggingFace QA pipeline."""
from typing import List, Tuple, Optional
from langchain_core.documents import Document
import logging
import os

try:
    from transformers import pipeline
    HAS_HF_GEN = True
except Exception:
    HAS_HF_GEN = False

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class RAGChain:
    """RAG chain that performs extractive QA over retrieved document chunks.

    Implementation notes:
    - Uses HF `question-answering` pipeline when available.
    - Queries each retrieved chunk separately and selects the best-scoring answer.
    - Returns the answer and the source chunks used.
    """

    def __init__(self, vector_store, k: int = 2, hf_model: Optional[str] = None):
        self.vector_store = vector_store
        self.k = k
        self.retriever = vector_store.as_retriever(search_kwargs={"k": k})
        self.hf_generator = None

        if HAS_HF_GEN:
            model_name = hf_model or os.getenv("HF_FALLBACK_MODEL", "distilbert-base-uncased-distilled-squad")
            try:
                # device=-1 forces CPU; if GPU available, set CUDA device index externally
                self.hf_generator = pipeline("question-answering", model=model_name, device=-1)
                logger.info(f"Loaded HF QA model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load HF QA model '{model_name}': {e}")
                self.hf_generator = None

    def _score_and_extract(self, question: str, context: str) -> Tuple[Optional[str], float]:
        """Run QA on a single context chunk and return (answer, score).

        Returns (None, 0.0) when no valid answer found or pipeline errors.
        """
        if not self.hf_generator:
            return None, 0.0

        try:
            res = self.hf_generator(question=question, context=context, top_k=1)
            # Normalize results: list or dict
            candidates = []
            if isinstance(res, dict):
                candidates = [res]
            elif isinstance(res, list):
                candidates = res

            best_ans = None
            best_score = 0.0
            for c in candidates:
                # Different transformer versions use different keys
                ans = c.get("answer") or c.get("answer_text") or c.get("label") or None
                score = c.get("score") or c.get("probability") or c.get("confidence") or 0.0
                try:
                    score = float(score)
                except Exception:
                    score = 0.0

                if ans and (best_ans is None or score > best_score):
                    best_ans = ans
                    best_score = score

            return best_ans, best_score
        except Exception as e:
            logger.error(f"QA pipeline error on chunk: {e}")
            return None, 0.0

    def query(self, question: str) -> dict:
        """Retrieve top-k chunks and return the best extractive answer found."""
        try:
            docs = self.retriever.invoke(question)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise

        if not docs:
            return {"answer": "No relevant information found.", "source_documents": []}

        best_answer = None
        best_score = -1.0
        best_chunk_idx = -1

        # Try QA on each retrieved chunk (stop early if very high score)
        for idx, doc in enumerate(docs):
            context = doc.page_content
            ans, score = self._score_and_extract(question, context)
            if ans:
                # Prefer answers with higher score
                if score > best_score:
                    best_answer = ans
                    best_score = score
                    best_chunk_idx = idx
                    # early exit for extremely confident answers
                    if best_score >= 0.98:
                        break

        if best_answer:
            # Return best answer and the chunks around it (limit to k)
            sources = docs[: self.k]
            return {"answer": best_answer, "source_documents": sources}

        # No extractive answer found — return clear message and include source snippets for manual inspection
        snippets = []
        for d in docs[: self.k]:
            snippets.append(d.page_content[:400])

        message = "I cannot find a concise answer in the document. Here are the top snippets:"
        message += "\n\n" + "\n\n---\n\n".join(snippets)
        return {"answer": message, "source_documents": docs[: self.k]}

    def get_relevant_documents(self, question: str) -> List[Document]:
        try:
            return self.retriever.invoke(question)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise

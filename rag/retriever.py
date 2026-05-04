"""
NextGen Tutor AI – Retriever
==============================
Provides a configurable similarity-search retriever over the ChromaDB
vector store, including score-aware retrieval for confidence checking.
"""

from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from utils.config import DEFAULT_SCORE_THRESHOLD, DEFAULT_TOP_K


def get_retriever(
    vectorstore: Chroma,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
):
    """Return a LangChain retriever with similarity-score filtering.

    Parameters
    ----------
    vectorstore : Chroma
        The ChromaDB vector store instance.
    top_k : int
        Number of chunks to retrieve.
    score_threshold : float
        Minimum similarity score (0-1) for a chunk to be included.
    """
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": top_k,
            "score_threshold": score_threshold,
        },
    )


def retrieve_with_scores(
    query: str,
    vectorstore: Chroma,
    top_k: int = DEFAULT_TOP_K,
) -> List[Tuple[Document, float]]:
    """Retrieve documents together with their similarity scores.

    Returns a list of (Document, score) tuples sorted by descending
    relevance.  This is used for:
      - confidence checking (low scores → "I don't know" behaviour)
      - logging retrieval hit-rate metrics
    """
    results = vectorstore.similarity_search_with_relevance_scores(
        query, k=top_k
    )
    # Sort by score descending (most relevant first)
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def format_context(docs_with_scores: List[Tuple[Document, float]]) -> str:
    """Format retrieved documents into a context string for the LLM prompt.

    Includes source metadata so the LLM can produce proper citations.
    """
    parts: list[str] = []
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        meta = doc.metadata
        source = meta.get("source", "Unknown")
        page = meta.get("page", "?")
        parts.append(
            f"[Source {i}] {source} — Page {page} "
            f"(relevance: {score:.2f})\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)

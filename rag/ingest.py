"""
NextGen Tutor AI – Document Ingestion Pipeline
================================================
Handles PDF loading, text cleaning, chunking, metadata attachment,
and vector-store construction with ChromaDB.

Educators can customise:
  - chunk_size / chunk_overlap  (via UI or config.py)
  - metadata fields              (edit _attach_metadata)
  - header/footer cleaning regex (edit _clean_text)
"""

import hashlib
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from utils.config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    EMBEDDING_MODEL,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _file_hash(data: bytes) -> str:
    """SHA-256 hash for de-duplication / cache invalidation."""
    return hashlib.sha256(data).hexdigest()


def _clean_text(text: str) -> str:
    """Remove common noisy artefacts from extracted PDF text.

    Strips repeated headers/footers that typically appear as short lines at
    page boundaries, excess whitespace, and page-number-only lines.
    """
    # Remove lines that are just page numbers
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse 3+ spaces into 1
    text = re.sub(r" {3,}", " ", text)
    return text.strip()


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared HuggingFace embedding model instance.

    Uses a local model (no API key required). The model is downloaded
    on first use and cached automatically by HuggingFace.
    Cached via lru_cache so the model loads only ONCE per process.
    """
    if not hasattr(_get_embeddings, "_instance"):
        _get_embeddings._instance = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _get_embeddings._instance


# ── Core Functions ───────────────────────────────────────────────────────────

def load_pdfs(
    uploaded_files,
    course_id: str = "default",
    uploaded_by: str = "learner",
) -> tuple[list[Document], dict[str, str]]:
    """Load uploaded Streamlit file objects and return LangChain Documents.

    Also returns a dict mapping filename → SHA-256 hash so that callers
    can skip re-ingestion of unchanged files.
    """
    documents: list[Document] = []
    file_hashes: dict[str, str] = {}

    for uf in uploaded_files:
        raw = uf.read()
        fhash = _file_hash(raw)
        file_hashes[uf.name] = fhash
        uf.seek(0)  # reset for potential re-read

        # Write to a temp file so PyPDFLoader can read it
        tmp_path = Path("/tmp") / f"nexgen_{fhash}.pdf"
        tmp_path.write_bytes(raw)

        loader = PyPDFLoader(str(tmp_path))
        pages = loader.load()

        for page_doc in pages:
            page_doc.page_content = _clean_text(page_doc.page_content)
            page_doc.metadata.update(
                {
                    "source": uf.name,
                    "page": page_doc.metadata.get("page", 0) + 1,  # 1-based
                    "title": uf.name.replace(".pdf", ""),
                    "section": "",
                    "uploaded_by": uploaded_by,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "course_id": course_id,
                    "file_hash": fhash,
                }
            )
            documents.append(page_doc)

        # Clean up temp file
        tmp_path.unlink(missing_ok=True)

    return documents, file_hashes


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """Split documents into retrieval-friendly chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(documents)


def build_vector_store(
    chunks: List[Document],
    existing_hashes: Optional[set] = None,
) -> Chroma:
    """Embed chunks and persist them in ChromaDB.

    If *existing_hashes* is provided, chunks whose ``file_hash`` metadata is
    already present will be skipped (cache-aware ingestion).

    Note: Embeddings are computed locally — no API key needed.
    """
    if existing_hashes:
        chunks = [
            c for c in chunks if c.metadata.get("file_hash") not in existing_hashes
        ]

    embeddings = _get_embeddings()

    if chunks:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=CHROMA_COLLECTION,
            persist_directory=CHROMA_PERSIST_DIR,
        )
    else:
        # No new chunks – just open the existing store
        vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )

    return vectorstore


def load_existing_vector_store() -> Optional[Chroma]:
    """Re-open an existing persisted ChromaDB store, or return None."""
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return None
    embeddings = _get_embeddings()
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def get_existing_hashes(vectorstore: Chroma) -> set[str]:
    """Return the set of file hashes already present in the vector store."""
    try:
        results = vectorstore.get(include=["metadatas"])
        hashes = {
            m.get("file_hash")
            for m in (results.get("metadatas") or [])
            if m and m.get("file_hash")
        }
        return hashes
    except Exception:
        return set()


def clear_vector_store() -> None:
    """Delete the ChromaDB persistent directory entirely."""
    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)

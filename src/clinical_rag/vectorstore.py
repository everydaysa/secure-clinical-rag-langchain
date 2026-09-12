"""Chroma vector store: build the index and open it again later.

The index directory is per provider. Azure embeddings are 1536-dimensional and
the offline mock is 256-dimensional, and mixing the two in one collection would
fail at query time in a confusing way.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from .config import Settings, get_settings
from .llm import get_embeddings

COLLECTION_NAME = "clinical_policies"


def index_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return Path(settings.index_dir) / settings.llm_provider


def get_vectorstore(settings: Settings | None = None) -> VectorStore:
    """Open the existing index (creating an empty one if absent)."""
    settings = settings or get_settings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(settings),
        persist_directory=str(index_path(settings)),
    )


def build_index(
    documents: list[Document],
    settings: Settings | None = None,
    reset: bool = True,
) -> VectorStore:
    """Embed documents and persist them. By default, replace any existing index."""
    settings = settings or get_settings()
    path = index_path(settings)

    if reset and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    store = get_vectorstore(settings)
    store.add_documents(documents)
    return store

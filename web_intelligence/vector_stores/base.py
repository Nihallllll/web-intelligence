"""
Abstract interface for all vector store backends.
"""

from typing import List, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class BaseVectorStore(Protocol):
    """
    Protocol that every vector store must satisfy.

    Methods required:
        add, search, count, clear, list_documents, get_document,
        delete_document, delete_by_url
    """

    def add(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict],
        ids: List[str],
        documents: Optional[List[str]] = None,
    ) -> None:
        """Add vectors with metadata (and optional raw documents) to the store."""
        ...

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        where_filter: Optional[Dict] = None,
        min_score: float = 0.0,
    ) -> List[Dict]:
        """
        Semantic search. Returns ranked list of result dicts.

        Each dict has keys: id, text, source, score, metadata.
        """
        ...

    def list_documents(self) -> List[Dict]:
        """List all indexed documents (grouped by doc_id)."""
        ...

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get all chunks for a specific document."""
        ...

    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks belonging to a document. Returns True if found."""
        ...

    def delete_by_url(self, url: str) -> int:
        """Delete all chunks from a URL. Returns number deleted."""
        ...

    def count(self) -> int:
        """Total number of chunks in the store."""
        ...

    def clear(self) -> None:
        """Delete all data."""
        ...

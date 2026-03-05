"""
ChromaDB vector store backend (persistent, production-grade).

Install:  pip install chromadb
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional

logger = logging.getLogger("web_intelligence.vector_stores.chroma")


class ChromaVectorStore:
    """
    Persistent vector store backed by ChromaDB.

    Args:
        persist_directory: Path to store ChromaDB data on disk.
        collection_name: Name of the ChromaDB collection.
    """

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "web_content",
    ):
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "ChromaDB backend requires 'chromadb'. "
                "Install with:  pip install chromadb"
            ) from exc

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(collection_name)
        logger.info("ChromaDB store ready (dir=%s, collection=%s)", persist_directory, collection_name)

    def add(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict],
        ids: List[str],
        documents: Optional[List[str]] = None,
    ) -> None:
        """Add vectors with metadata. Uses ChromaDB ``documents`` field for text."""
        kwargs: Dict = {
            "embeddings": vectors,
            "metadatas": metadatas,
            "ids": ids,
        }
        if documents:
            kwargs["documents"] = documents
        self.collection.add(**kwargs)

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        where_filter: Optional[Dict] = None,
        min_score: float = 0.0,
    ) -> List[Dict]:
        kwargs: Dict = {
            "query_embeddings": [query_vector],
            "n_results": limit,
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)

        formatted = []
        for i in range(len(results["ids"][0])):
            # ChromaDB distance → similarity score
            score = 1 - results["distances"][0][i]
            if score < min_score:
                continue

            meta = results["metadatas"][0][i]
            # Text can come from documents field or metadata fallback
            text = ""
            if results.get("documents") and results["documents"][0]:
                text = results["documents"][0][i] or ""
            if not text:
                text = meta.get("text", "")

            formatted.append({
                "id": results["ids"][0][i],
                "text": text,
                "source": meta.get("url", ""),
                "score": score,
                "metadata": meta,
            })

        return formatted

    def list_documents(self) -> List[Dict]:
        all_data = self.collection.get()
        if not all_data["ids"]:
            return []

        docs: Dict[str, Dict] = {}
        for meta in all_data["metadatas"]:
            doc_id = meta.get("doc_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "url": meta.get("url", ""),
                    "title": meta.get("title", "Untitled"),
                    "indexed_at": meta.get("indexed_at", ""),
                    "chunk_count": 0,
                    "total_words": 0,
                }
            docs[doc_id]["chunk_count"] += 1
            docs[doc_id]["total_words"] += meta.get("word_count", 0)

        return sorted(docs.values(), key=lambda d: d["indexed_at"], reverse=True)

    def get_document(self, doc_id: str) -> Optional[Dict]:
        results = self.collection.get(where={"doc_id": doc_id})
        if not results["ids"]:
            return None

        chunks = []
        for i, id_ in enumerate(results["ids"]):
            text = ""
            if results.get("documents") and results["documents"][i]:
                text = results["documents"][i]
            if not text:
                text = results["metadatas"][i].get("text", "")

            chunks.append({
                "id": id_,
                "text": text,
                "chunk_index": results["metadatas"][i].get("chunk_index", 0),
                "word_count": results["metadatas"][i].get("word_count", 0),
            })
        chunks.sort(key=lambda c: c["chunk_index"])

        meta = results["metadatas"][0]
        return {
            "doc_id": doc_id,
            "url": meta.get("url", ""),
            "title": meta.get("title", "Untitled"),
            "indexed_at": meta.get("indexed_at", ""),
            "chunk_count": len(chunks),
            "full_text": "\n\n".join(c["text"] for c in chunks),
            "chunks": chunks,
        }

    def delete_document(self, doc_id: str) -> bool:
        results = self.collection.get(where={"doc_id": doc_id})
        if not results["ids"]:
            return False
        self.collection.delete(ids=results["ids"])
        return True

    def delete_by_url(self, url: str) -> int:
        results = self.collection.get(where={"url": url})
        if not results["ids"]:
            return 0
        count = len(results["ids"])
        self.collection.delete(ids=results["ids"])
        return count

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(self.collection_name)

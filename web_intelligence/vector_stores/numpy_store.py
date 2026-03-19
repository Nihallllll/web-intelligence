from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger("web_intelligence.vector_stores.numpy")


class NumpyVectorStore:
    def __init__(
        self,
        persist_path: Optional[str] = None,
        collection_name: str = "web_content",
    ):
        self.persist_path = Path(persist_path) if persist_path else None
        self.collection_name = collection_name

        self._vectors: np.ndarray | None = None
        self._ids: list[str] = []
        self._metadatas: list[Dict] = []
        self._documents: list[str] = []

        if self.persist_path and self.persist_path.exists():
            self._load()
            logger.info("Loaded %d chunks from %s", len(self._ids), self.persist_path)
        else:
            logger.info("NumpyVectorStore initialized (empty)")

    def _save(self) -> None:
        if self.persist_path is None:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vectors": self._vectors,
            "ids": self._ids,
            "metadatas": self._metadatas,
            "documents": self._documents,
        }
        with open(self.persist_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load(self) -> None:
        with open(self.persist_path, "rb") as f:
            data = pickle.load(f)
        self._vectors = data["vectors"]
        self._ids = data["ids"]
        self._metadatas = data["metadatas"]
        self._documents = data.get("documents", [""] * len(self._ids))

    def add(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict],
        ids: List[str],
        documents: Optional[List[str]] = None,
    ) -> None:
        new_vecs = np.array(vectors, dtype=np.float32)

        if self._vectors is None:
            self._vectors = new_vecs
        else:
            self._vectors = np.vstack([self._vectors, new_vecs])

        self._ids.extend(ids)
        self._metadatas.extend(metadatas)
        self._documents.extend(documents or [""] * len(ids))
        self._save()

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        where_filter: Optional[Dict] = None,
        min_score: float = 0.0,
    ) -> List[Dict]:
        if self._vectors is None or len(self._ids) == 0:
            return []

        q = np.array(query_vector, dtype=np.float32)

        norms = np.linalg.norm(self._vectors, axis=1)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        similarities = (self._vectors @ q) / (norms * q_norm + 1e-10)

        mask = np.ones(len(self._ids), dtype=bool)
        if where_filter:
            for key, value in where_filter.items():
                mask &= np.array([m.get(key) == value for m in self._metadatas])
            similarities = np.where(mask, similarities, -1.0)

        top_k = min(limit, len(self._ids))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < min_score:
                break
            text = self._documents[idx] or self._metadatas[idx].get("text", "")
            results.append({
                "id": self._ids[idx],
                "text": text,
                "source": self._metadatas[idx].get("url", ""),
                "score": score,
                "metadata": self._metadatas[idx],
            })

        return results

    def list_documents(self) -> List[Dict]:
        docs: Dict[str, Dict] = {}
        for meta in self._metadatas:
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
        chunks = []
        for i, meta in enumerate(self._metadatas):
            if meta.get("doc_id") == doc_id:
                text = self._documents[i] or meta.get("text", "")
                chunks.append({
                    "id": self._ids[i],
                    "text": text,
                    "chunk_index": meta.get("chunk_index", 0),
                    "word_count": meta.get("word_count", 0),
                })

        if not chunks:
            return None

        chunks.sort(key=lambda c: c["chunk_index"])
        meta0 = next(m for m in self._metadatas if m.get("doc_id") == doc_id)
        return {
            "doc_id": doc_id,
            "url": meta0.get("url", ""),
            "title": meta0.get("title", "Untitled"),
            "indexed_at": meta0.get("indexed_at", ""),
            "chunk_count": len(chunks),
            "full_text": "\n\n".join(c["text"] for c in chunks),
            "chunks": chunks,
        }

    def delete_document(self, doc_id: str) -> bool:
        indices = [i for i, m in enumerate(self._metadatas) if m.get("doc_id") == doc_id]
        if not indices:
            return False
        self._remove_indices(indices)
        return True

    def delete_by_url(self, url: str) -> int:
        indices = [i for i, m in enumerate(self._metadatas) if m.get("url") == url]
        if not indices:
            return 0
        self._remove_indices(indices)
        return len(indices)

    def count(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self._vectors = None
        self._ids = []
        self._metadatas = []
        self._documents = []
        self._save()

    def _remove_indices(self, indices: List[int]) -> None:
        keep = sorted(set(range(len(self._ids))) - set(indices))
        self._ids = [self._ids[i] for i in keep]
        self._metadatas = [self._metadatas[i] for i in keep]
        self._documents = [self._documents[i] for i in keep]
        if self._vectors is not None and len(keep) > 0:
            self._vectors = self._vectors[keep]
        else:
            self._vectors = None
        self._save()

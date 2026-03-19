import chromadb
from typing import List, Dict, Optional

import warnings

warnings.warn(
    "web_intelligence.vector_store is deprecated. "
    "Use web_intelligence.vector_stores.ChromaVectorStore instead.",
    DeprecationWarning,
    stacklevel=2,
)


class VectorStore:
    def __init__(self, persist_directory: str = "./data/chroma",
                 collection_name: str = "web_content"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, vectors: List[List[float]], metadatas: List[Dict], ids: List[str]):
        self.collection.add(
            embeddings=vectors,
            metadatas=metadatas,
            ids=ids,
        )

    def search(self, query_vector: List[float], limit: int = 5,
               filter: Optional[Dict] = None,
               min_score: float = 0.0) -> List[Dict]:
        kwargs = {
            "query_embeddings": [query_vector],
            "n_results": limit,
        }
        if filter:
            kwargs["where"] = filter

        results = self.collection.query(**kwargs)

        formatted = []
        for i in range(len(results["ids"][0])):
            score = 1 - results["distances"][0][i]
            if score < min_score:
                continue
            formatted.append({
                "id": results["ids"][0][i],
                "text": results["metadatas"][0][i].get("text", ""),
                "source": results["metadatas"][0][i].get("url", ""),
                "score": score,
                "metadata": results["metadatas"][0][i],
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
            chunks.append({
                "id": id_,
                "text": results["metadatas"][i].get("text", ""),
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

    def clear(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(self.collection_name)
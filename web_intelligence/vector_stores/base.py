from typing import List, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class BaseVectorStore(Protocol):
    def add(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict],
        ids: List[str],
        documents: Optional[List[str]] = None,
    ) -> None: ...

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        where_filter: Optional[Dict] = None,
        min_score: float = 0.0,
    ) -> List[Dict]: ...

    def list_documents(self) -> List[Dict]: ...

    def get_document(self, doc_id: str) -> Optional[Dict]: ...

    def delete_document(self, doc_id: str) -> bool: ...

    def delete_by_url(self, url: str) -> int: ...

    def count(self) -> int: ...

    def clear(self) -> None: ...
